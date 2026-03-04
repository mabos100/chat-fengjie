import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# === 配置路径 ===
MODEL_ID = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
OUTPUT_DIR = "./Qwen3-8B-QLoRA-out-v0.5.1"
DATA_PATH = "/data/train_data/feng_chatml3.1.2.jsonl"

# ================= 1. 加载 Tokenizer =================
# Qwen3 建议先尝试 use_fast=True，如果报错再切回 False
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False
)
# 确保 Padding Token 正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ================= 2. 加载模型 (QLoRA 4-bit) =================
print("正在以 4-bit 模式加载 Qwen3-8B (优化显存模式)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # 二次量化
    bnb_4bit_quant_type="nf4",           # 精度更高
    bnb_4bit_compute_dtype=torch.float16 # 2060S 务必使用 fp16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,           # 2060S 必须
    low_cpu_mem_usage=True,              # 减小内存峰值占用
    use_cache=False                      # 训练模式必须关闭
)

# 梯度检查点开启，节省显存的关键
model.gradient_checkpointing_enable()
# QLoRA 预处理
model = prepare_model_for_kbit_training(model)

# ================= 3. LoRA 配置 =================
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# 稳定性修正：LoRA 参数转 FP32


print(f"可训练参数量：")
model.print_trainable_parameters()

# ================= 4. 数据处理 =================
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_chat_template(row):
    # 使用 Qwen3 官方标准的 ChatML 模板
    row["text"] = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return row

dataset = dataset.map(format_chat_template)

# ================= 5. 训练参数优化 (针对 2060S 8G) =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,      # 8G 只能设为 1
    gradient_accumulation_steps=32,     # 累积 16 步，等效 Batch Size = 16
    learning_rate=5e-5,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    fp16=True,                          # 开启混合精度训练
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",          # 核心：显存不足时自动溢出到内存
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, # 减少显存峰值
    max_grad_norm=0.3,                  # 梯度裁剪防止 FP16 溢出导致 NaN
    warmup_ratio=0.03,
    remove_unused_columns=True,
    report_to="none"                    # 暂时关闭 wandb 等，专注训练
)

# ================= 6. 开始训练 =================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=256,                 # 2060S 的极限，如果崩了请改 384
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

print("正在启动训练器...")
trainer.train()

# 保存 LoRA 权重
trainer.save_model(OUTPUT_DIR)
print(f"训练完成！LoRA 权重已保存至 {OUTPUT_DIR}")