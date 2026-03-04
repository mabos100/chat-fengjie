from unsloth import FastLanguageModel  # 关键：使用 Unsloth
import os
import uvicorn
import torch
import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sse_starlette.sse import EventSourceResponse
from transformers import TextIteratorStreamer

from threading import Thread

from datetime import datetime
import logging

# 这是0.0.4版本，增加token计算功能！！！！！！！每次修改前都需要修改次版本号。
# 📊 [STREAM] | Tokens: 1234 | Elapsed: 7.71s | Speed: 159.99 tokens/s | TTFT: 0.018s    print生效 logging不生效
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.token_count = 0
        self.first_token_time = None
        self.start_time = None

    def put(self, value):
        if self.start_time is None:
            self.start_time = time.time()

        # 记录首个 token 产生的时间 (TTFT)
        if self.token_count == 0 and value.numel() > 0:
            self.first_token_time = time.time()

        self.token_count += value.numel()
        super().put(value)


def generate_with_metrics(model, tokenizer, input_ids, gen_kwargs, streamer: Optional[EnhancedStreamer] = None):
    prompt_len = input_ids.shape[1]
    mode = "stream" if streamer is not None else "non-stream"

    torch.cuda.synchronize()
    start_ts = time.time()

    try:
        if streamer is not None:
            # 注入开始时间以便计算 TTFT
            streamer.start_time = start_ts
            model.generate(input_ids=input_ids, streamer=streamer, **gen_kwargs)
            completion_tokens = streamer.token_count
            ttft = (streamer.first_token_time - start_ts) if streamer.first_token_time else 0
        else:
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, **gen_kwargs)
            completion_tokens = outputs.shape[1] - prompt_len
            ttft = 0  # 非流式 TTFT 意义不大，等同于总耗时

        torch.cuda.synchronize()
        end_ts = time.time()
        elapsed = end_ts - start_ts
        speed = completion_tokens / elapsed if elapsed > 0 else 0

        # 统一结构化日志
        log_msg = (f"📊 [{mode.upper()}] | Tokens: {completion_tokens} | "
                   f"Elapsed: {elapsed:.2f}s | Speed: {speed:.2f} tokens/s | TTFT: {ttft:.3f}s")
        print(log_msg, flush=True)  # 👈 确保这行存在
        logger.info(log_msg)

        return (outputs if streamer is None else None), completion_tokens, elapsed

    except Exception as e:
        logger.error(f"❌ Generation Error: {str(e)}")
        if streamer:
            streamer.end()  # 确保生成器能退出
        return None, 0, 0


os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
# ================= 配置区域 =================
# 指向你合并后的模型路径（或直接指向 checkpoint 路径也可以）
MODEL_PATH = "/data/qlora-fengjie/Qwen3-8B-Unsloth-out-checkpoint-500-final-v0.5.1"
PORT = 8000

# ================= 加载 Unsloth 模型 =================
print(f"🚀 正在以 Unsloth 模式加载模型: {MODEL_PATH} ...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,  # 2060S 建议设置为 2048 或更低
    load_in_4bit=True,  # 8G 显存必须保持 4bit 加载以留出推理空间
    device_map={"": 0},
)

# 开启推理加速 (Unsloth 核心优化)
FastLanguageModel.for_inference(model)

print("✅ 模型加载完成，推理加速已激活！")

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
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    # enable_thinking: Optional[bool] = False
    user: Optional[str] = None  # OpenAI 官方支持的字段


# ================= 核心逻辑 =================

def predict(query, history, system, temperature, top_p, max_tokens):
    # 这里我们直接利用 tokenizer.apply_chat_template 处理所有 messages
    # 所以在下面的接口逻辑里，我们不手动拆分 history
    pass


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 1. 构建 Prompt
    # 将 Pydantic 对象转换为字典列表
    messages = [msg.model_dump() for msg in request.messages]
    print('-' * 30)
    # 获取当前时间并格式化为字符串
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    # 示例输出: 2026-02-10 14:25:30
    print(request)

    # 使用 Qwen 的 Chat Template 转换输入
    encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=(request.user == "True") if request.user is not None else False,
    )
    input_ids = encoding.to(model.device)
    # 手动创建 attention_mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)

    # 2. 配置生成参数
    # 基础生成参数（不包含 input_ids 和 streamer）
    base_gen_kwargs = dict(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        attention_mask=attention_mask,  # 传入这个参数
        do_sample=True if request.temperature > 0 else False,
    )

    # ================= 处理流式请求 (Stream=True) =================
    if request.stream:
        streamer = EnhancedStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        # base_gen_kwargs["streamer"] = streamer

        # 在新线程中运行生成，主线程返回流
        # 🚀 核心修改：调用包装好的 generate_with_metrics 函数
        thread = Thread(
            target=generate_with_metrics,
            args=(model, tokenizer, input_ids, base_gen_kwargs, streamer)
        )
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
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield json.dumps(final_chunk, ensure_ascii=False)
            yield "[DONE]"

        return EventSourceResponse(stream_generator(), media_type="text/event-stream")

    # ================= 处理普通请求 (Stream=False) =================
    else:
        # 使用 EnhancedStreamer 捕获生成内容（不实际流式输出）
        streamer = EnhancedStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=160.0  # 防止卡死
        )

        # 在主线程中同步调用 generate_with_metrics（不再开新线程）
        generate_with_metrics(model, tokenizer, input_ids, base_gen_kwargs, streamer)

        # 收集所有生成的 token 文本
        generated_text = "".join(streamer)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": streamer.token_count,  # 假设 EnhancedStreamer 记录了生成 token 数
                "total_tokens": len(input_ids[0]) + streamer.token_count
            }
        }


if __name__ == "__main__":
    print(f"服务启动在: http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)