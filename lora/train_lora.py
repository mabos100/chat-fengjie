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
MODEL_ID = "/root/.cache/modelscope/hub/models/Qwen/Qwen1.5-1.8B-Chat"  # 会自动下载，或者填你本地权重的绝对路径
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