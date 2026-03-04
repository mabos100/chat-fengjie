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