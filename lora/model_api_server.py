import os
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import json
import time

# ================= 配置区域 =================
# 指向你刚才合并好的模型路径
MODEL_PATH = "./qwen1.5-1.8b-merged-final-v0.3.4"
# 服务端口
PORT = 8000
# 显卡设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 加载模型 =================
print(f"正在加载模型: {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,
    torch_dtype=torch.float16,  # 2060S 使用 FP16
    trust_remote_code=True
)
model.eval()  # 切换到评估模式
print("模型加载完成！")

# ================= 定义 OpenAI 格式的数据结构 =================
app = FastAPI(title="Qwen OpenAI API")

# 允许跨域 (方便前端调用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


# ================= 核心逻辑 =================

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 1. 构建 Prompt
    # 将 Pydantic 对象转换为字典列表
    messages = [msg.dict() for msg in request.messages]

    # 使用 Qwen 的 Chat Template 转换输入
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 2. 配置生成参数
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=True if request.temperature > 0 else False,
    )

    # ================= 处理流式请求 (Stream=True) =================
    if request.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        # 在新线程中运行生成，主线程返回流
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        async def stream_generator():
            request_id = f"chatcmpl-{int(time.time())}"
            created_time = int(time.time())

            for new_text in streamer:
                if new_text:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": new_text},
                            "finish_reason": None
                        }]
                    }
                    yield json.dumps(chunk, ensure_ascii=False)

            # 发送结束标志
            yield json.dumps({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }, ensure_ascii=False)
            yield "[DONE]"

        return EventSourceResponse(stream_generator(), media_type="text/event-stream")

    # ================= 处理普通请求 (Stream=False) =================
    else:
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        # 只取新生成的 token
        new_tokens = outputs[0][len(input_ids[0]):]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": len(new_tokens),
                "total_tokens": len(input_ids[0]) + len(new_tokens)
            }
        }


if __name__ == "__main__":
    print(f"服务启动在: http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)