这份指南旨在展示如何将经典文学人物“王熙凤”通过大语言模型微调技术，转化为一个具备其性格特征的对话 AI。本项目不仅是人物的复刻，更是一套**个性化 AI 角色生成框架**，适用于任何小说或剧本角色的数字化重生。

---

# 🎭 Chat-凤姐：从语料到灵魂的 AI 炼金术

**Chat-凤姐** 是基于《红楼梦》原著所有与王熙凤相关的台词、对话与行为描写，结合大语言模型，通过 LoRA 与 QLoRA 高效微调技术打造的个性化聊天 AI。

王熙凤，人称“凤辣子”，精明强干、口齿伶俐，是贾府内务的掌管者。本项目通过提取其特有的语言风格——泼辣犀利、巧言令色或暗藏机锋，使用户能与这位“脂粉队里的英雄”直接对话。

---

## 第一阶段：训练数据准备

构建个性化 AI 的第一步是**语料清洗**。由于原始文本包含大量旁白和他人对话，我们需要精准“提纯”出凤姐的专属语段。

### 1.1 基础环境准备 (时间节点 2026.2)

选择稳定的系统架构是后续高性能计算的基础。

- **操作系统**：Ubuntu 24.04.3 LTS
- **Python 版本**：3.13.9
- **conda 版本**：25.11.0

### 1.2 原始语料获取

以《红楼梦》白话文小说作为基础。文本中包含大量叙述性描写，需作为背景参考。

Shell

```
第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀

此开卷第一回也。

作者自云：因曾历过一番梦幻之后，故将真事隐去，而借“通灵”之说，撰此<<石头记>>一书也。

故曰“甄士隐”云云。

但书中所记何事何人？自又云：“今风尘碌碌，一事无成，忽念及当日所有之女子，一一细考较去，觉其行止见识，皆出于我之上。

何我堂堂须眉，诚不若彼裙钗哉？实愧则有余，悔又无益之大无可如何之日也！
```

### 1.3 对话内容提取策略

为了从密集的长文中提取王熙凤的台词，我们采用\*\*“上下文回溯法”\*\*：

1. **识别对话边界**：定位所有带中文引号（“”）的内容。
2. **关键词回溯**：对每段对话向前回溯 50 个字符，检索“王熙凤”、“凤姐”、“琏二奶奶”等核心称呼。
3. **结构化存储**：将匹配成功的对话存入 `fengjie.jsonl`，保留 ID 方便后续追溯。

提取后的数据格式如下，通过 ID 标识对话顺序：

Shell

```
【ID:305】 我都问了，没什么紧事，我就叫他们散了！ 太太说了，今日不得闲，二奶奶陪着便是一样。多谢费心想着。白来逛逛呢便罢，若有甚说的，只管告诉二奶奶，都是一样！
【ID:310】 东府里的小大爷进来了！ 不必说了！
【ID:312】 你蓉大爷在那里呢？ 你只管坐着，这是我侄儿！
```

### 1.4 对话提取脚本实现

该 Python 脚本通过正则匹配与逻辑判断，实现了自动化的角色对话提取。

> **逻辑说明**：`feng_aliases` 定义了角色的所有称谓。脚本先通过正则 `r'“(.*?)”'` 找到所有引语，再通过 `pre_context` 检查引语前是否出现了角色名。

Python

```
def extract_feng_dialogue_chain_strict(file_path):
    # --- 0. 配置 ---
    # 王熙凤的常见称呼
    feng_aliases = ["王熙凤", "凤姐", "凤姐儿", "凤丫头", "琏二奶奶", "凤辣子"]

    # ============================
    # 步骤 1: 处理文本 (清洗)
    # ============================
    try:
        with open(file_path, 'r', encoding='ANSI') as f:
            raw_content = f.read()

        # 去掉换行、回车、空格，使文本成为连续的一条长龙
        text = raw_content.replace('\n', '').replace('\r', '').replace(' ', '')
        print(f"步骤1完成：文本清洗完毕，总长度 {len(text)} 字。")

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return []

    # ============================
    # 步骤 2: 找到所有对话并标记
    # ============================
    # 我们使用一个列表来存储所有对话的元数据，这相当于给每个对话打上了序号(ID)
    dialogues_list = []

    # 正则：匹配引号内的内容
    pattern = re.compile(r'“(.*?)”')

    # finditer 返回所有匹配对象的迭代器
    for match in pattern.finditer(text):
        dialogues_list.append({
            "id": len(dialogues_list),  # 序号标记 (0, 1, 2, ...)
            "start": match.start(),  # 引号开始在全文的绝对位置
            "end": match.end(),  # 引号结束在全文的绝对位置
            "content": match.group(1)  # 对话内容
        })

    print(f"步骤2完成：共标记了 {len(dialogues_list)} 处对话。")

    # ============================
    # 步骤 3: 找到王熙凤的话(n)，输出 (n-1) 到 (n)
    # ============================
    results = []

    # 从第 1 个开始遍历 (因为我们需要 n-1，所以不能从 0 开始)
    for i in range(1, len(dialogues_list)):
        dialogue_n = dialogues_list[i]  # 当前对话 (n)
        dialogue_prev = dialogues_list[i - 1]  # 上一句对话 (n-1)

        # --- A. 判断 n 是否为王熙凤 ---
        # 逻辑：查看当前引号前面的一段文字(前文语境)，看有没有王熙凤的名字
        # 我们向前看 50 个字，通常"凤姐道"紧挨着引号
        lookback_range = 50
        search_start = max(0, dialogue_n['start'] - lookback_range)
        # 截取引语部分 (从 n-1 结束 到 n 开始 之间的文字)
        # 注意：为了防止误判，我们最好只检查 n-1 结束之后的部分
        check_start = max(dialogue_prev['end'], search_start)
        pre_context = text[check_start: dialogue_n['start']]

        is_feng = False
        for name in feng_aliases:
            if name in pre_context:
                is_feng = True
                break

        # --- B. 如果是王熙凤，执行提取 ---
        if is_feng:
            # 我们要找提取的【起点】和【终点】

            # 终点很简单：就是 n 的结束位置
            extract_end = dialogue_n['end']

            # 起点比较讲究：
            # 题目要求包括 n-1。但为了阅读通顺，我们需要找到 n-1 的“引语”开始的地方。
            # 方法：从 n-1 的引号开始位置，向前找最近的一个句号/感叹号/问号。

            # 向前回溯 100 字找标点
            search_limit_prev = 100
            search_text_start = max(0, dialogue_prev['start'] - search_limit_prev)
            text_before_prev = text[search_text_start: dialogue_prev['start']]

            # 找最后一个结束标点
            last_punct_match = None
            for m in re.finditer(r'[。！？]', text_before_prev):
                last_punct_match = m

            if last_punct_match:
                # 起点设为标点之后
                # search_text_start + 标点位置 + 1
                extract_start = search_text_start + last_punct_match.end()
            else:
                # 没找到标点，就直接从 n-1 引号前一点截取 (兜底)
                extract_start = max(0, dialogue_prev['start'] - 20)

            # --- C. 截取并保存 ---
            full_segment = text[extract_start: extract_end]

            results.append({
                "seq_id": i,  # 记录这是全书第几句对话
                "text": full_segment,
                "n_1_content": dialogue_prev['content'],  # 方便调试看上一句是啥
                "n_content": dialogue_n['content']  # 方便调试看这句是啥
            })

    return results
```

### 1.5 训练数据格式化 (ChatML)

为适配 **Qwen1.5-1.8B** 模型，需要将语料转化为 JSONL 格式。关键在于 `system prompt` 的设定，它是 AI 行为的准则。

> **关键逻辑**：使用 `system` 角色注入人物画像（泼辣、精明、言语爽利）；`user` 为对话上下文；`assistant` 为凤姐的回答。

JSON

```
{
  "messages": [
    {
      "role": "system",
      "content": "你现在是《红楼梦》中的王熙凤（人称凤姐、琏二奶奶）。你的性格泼辣张狂、精明强干、言语爽利，说话常带三分笑，善于察言观色，同时也有些心狠手辣。请根据前文情境，模仿她的语气进行回答。"
    },
    {
      "role": "user",
      "content": "怎么还不请进来？"
    },
    {
      "role": "assistant",
      "content": "周姐姐，快搀起来，别拜罢，请坐。我年轻，不大认得，可也不知是什么辈数，不敢称呼！"
    }
  ]
}
```

---

## 第二阶段：LoRA 参数高效微调

微调阶段是将静态语料转化为动态生成能力的内核。我们利用 **LoRA (Low-Rank Adaptation)**  技术，在 8GB 显存的显卡上实现大模型的能力注入。

### 2.1 硬件与环境配置

- **显卡需求**：RTX 2060 Super (8GB 显存) 即可稳定运行。
- **核心库安装**：使用 ModelScope 下载 Qwen 基础权重。

Bash

```
pip install modelscope
modelscope download --model Qwen/Qwen1.5-1.8B
# 虚拟环境创建
conda create -n qwen_lora python=3.10 -y
conda activate qwen_lora
```

### 2.2 核心微调脚本 (`train_lora.py`)

该脚本负责加载模型并开启训练。注意代码中对 **Mixed Precision (混合精度)**  的修正逻辑，这是解决显存溢出与梯度不稳定的关键。

> **变量说明**：
>
> - `r=8`: LoRA 的秩。
> - `target_modules`: 指定微调的注意力机制层（如 `q_proj`, `k_proj`）。
> - `fp16=True`: 使用半精度训练以加速。

Python

```
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType

# === 配置路径 ===
MODEL_ID = "/root/.cache/modelscope/hub/models/Qwen/Qwen1.5-1.8B"  # 会自动下载，或者填你本地权重的绝对路径
DATA_PATH = "feng_chatml3.1.2.jsonl"
OUTPUT_DIR = "./Qwen1.5-1.8B-Chat-lora-out-v0.3.4"

# ================= 1. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ================= 2. 加载模型 (FP16) =================
print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_cache=False
)

# 显存优化
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# ================= 3. LoRA 配置 =================
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# 🔥🔥🔥【关键修正】🔥🔥🔥
# 将所有可训练参数(LoRA)强制转为 FP32
# 这能解决 "Attempting to unscale FP16 gradients" 错误
# 同时 LoRA 参数很少，转为 FP32 几乎不增加显存占用
for name, param in model.named_parameters():
    if "lora" in name or param.requires_grad:
        param.data = param.data.to(torch.float32)

print("已将 LoRA 参数转换为 FP32 以匹配 Mixed Precision 训练。")
model.print_trainable_parameters()

# ================= 4. 数据处理 =================
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_chat_template(row):
    row["text"] = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return row

dataset = dataset.map(format_chat_template)

# ================= 5. 训练参数 =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=f"{OUTPUT_DIR}/logs",
    per_device_train_batch_size=1,  # 2060S 8G 显存保险起见设为1
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=6,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="adamw_torch",
    report_to="tensorboard",
    remove_unused_columns=True
)

# ================= 6. 开始训练 =================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,         # trl 0.9.6 需要在这里传参
    tokenizer=tokenizer,        # trl 0.9.6 需要在这里传参
    args=training_args,
    packing=False,
)

print("开始训练...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
print(f"训练完成！模型已保存至 {OUTPUT_DIR}")
```

### 2.3 部署 API 与 Web 交互

通过 FastAPI 封装 OpenAI 兼容的 API 接口，并使用 Streamlit 构建前端界面。

#### 后端 API 服务 (`start.model.py`)

该服务支持**流式输出 (Streaming)** ，能让对话体验更流畅。

Python

```
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
```

#### 前端交互界面 (`client.py`)

利用 Streamlit 的 `st.chat_message` 和 `st.write_stream` 轻松复现 ChatGPT 般的对话体验。

Python

```
import streamlit as st
from openai import OpenAI

# ================= 配置区域 =================
st.set_page_config(page_title="红楼梦·凤姐聊天室", page_icon="🌶️")
st.title("🌶️ 和凤姐姐聊天吧 (AI版 v0.3.2)")

# API 配置（指向本地推理服务）
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"  # 本地服务无需真实密钥

# ================= 侧边栏设置 =================
with st.sidebar:
    st.header("🎭 角色设定")
    system_prompt = st.text_area(
        "系统提示词 (System Prompt)",
        value=(
            "你扮演《红楼梦》中的王熙凤（人称凤姐、琏二奶奶）。"
            "你的性格泼辣张狂、精明强干、言语爽利，说话常带三分笑，"
            "善于察言观色，同时也有些心狠手辣。请根据前文情境，"
            "模仿她的语气进行回答。"
        ),
        height=160,
        help="修改此提示可调整凤姐的性格表现"
    )

    st.divider()

    st.header("⚙️ 生成参数")
    temperature = st.slider("活跃度 (Temperature)", 0.0, 1.0, 0.7, 0.1,
                           help="值越高，回复越随机；越低则越确定")
    max_tokens = st.slider("最大回复长度", 128, 1024, 512, 64,
                          help="控制单次回复的最大字数")

    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ================= 初始化客户端与会话状态 =================
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 聊天主界面 =================

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入处理
if prompt := st.chat_input("请给凤奶奶请安..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 构造含 system prompt 的完整请求（不显示在界面上）
    messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    # 流式生成 AI 回复
    with st.chat_message("assistant"):
        stream = st.session_state.client.chat.completions.create(
            model="qwen-lora",
            messages=messages_to_send,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        response = st.write_stream(stream)  # 自动处理流式输出

    # 保存助手回复到历史
    st.session_state.messages.append({"role": "assistant", "content": response})
```

### 2.4 模型合并与固化

当微调版本趋于稳定，可使用 `merge_and_unload()` 将 LoRA 权重注入基座模型。这一步是为了摆脱训练环境的依赖，生成一个原生的单模型文件。

Python

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# ================= 配置区域 =================
# 1. 基座模型名称 (必须和你训练时用的一样)
BASE_MODEL_ID = "/root/.cache/modelscope/hub/models/Qwen/Qwen1.5-1.8B-Chat"

# 2. 你的 LoRA 输出目录 (训练完成后的 output_dir)
LORA_PATH = "./Qwen1.5-1.8B-Chat-lora-out-v0.3.4"

# 3. 合并后的完整模型保存路径 (自定义)
MERGED_OUTPUT_PATH = "./qwen1.5-1.8b-merged-final-v0.3.4"

# ================= 开始合并 =================
print(f"正在加载基座模型: {BASE_MODEL_ID} ...")

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True
)

# 加载基座模型 (使用 FP16 以节省显存和存储空间)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print(f"正在加载 LoRA 权重: {LORA_PATH} ...")
# 将 LoRA 挂载到基座模型上
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("正在执行合并操作 (Merge and Unload)...")
# 核心步骤：将 LoRA 权重加权合并到基座权重中，并卸载 LoRA 结构
merged_model = model.merge_and_unload()

# 切换回评估模式
merged_model.eval()

print(f"正在保存合并后的模型至: {MERGED_OUTPUT_PATH} ...")
# 1. 保存模型权重 (默认保存为 safetensors 格式，更安全更快)
merged_model.save_pretrained(
    MERGED_OUTPUT_PATH,
    safe_serialization=True,
    max_shard_size="2GB" # 分片大小，方便拷贝
)

# 2. 保存 Tokenizer (非常重要，否则以后加载会报错)
tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

print("✅ 模型合并完成！")
print(f"你可以直接使用 '{MERGED_OUTPUT_PATH}' 进行推理，不再需要 LoRA 文件。")
```

---

## 结语：让文学在数字世界永生

通过“语料清洗—格式化—微调—固化—前端交互”这一全链路流程，我们不仅得到了一个能言善辩的“凤姐”，更掌握了将任何文化 IP AI 化的能力。

**下一步建议**：

- 尝试在 `system prompt` 中加入不同的情绪约束，观察凤姐回复的变化。
- 增加“宝黛”语料，训练多人物交互模型。

希望这篇指南能助你构建出属于自己的“红楼梦”虚拟世界。如果你在使用过程中遇到显存溢出或其他报错，请务必检查 `fp16` 设置与 `batch size` 的平衡点。
