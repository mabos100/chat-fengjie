import os
from unsloth import FastLanguageModel
import torch

os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
# 关键点：直接指向你的训练产物（LoRA 适配器路径）
CHECKPOINT_PATH = "/data/qlora-fengjie/Qwen3-8B-Unsloth-out/checkpoint-500"
FINAL_MODEL_DIR = "/data/qlora-fengjie/Qwen3-8B-Unsloth-out-checkpoint-500-final-v0.5.1"

# 1. 加载已经挂载了 LoRA 的模型
# Unsloth 会自动读取 checkpoint 里的 adapter_config.json 找到 base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = CHECKPOINT_PATH,  # 这里改用适配器路径
    max_seq_length = 256,
    load_in_4bit = True,           # 2060S 必须用 4bit 加载来节省合并前的显存
    device_map = {"": 0},
)

print("正在执行 16bit 全量合并导出...")

# 2. 执行合并
# 注意：在 8G 显存上执行 merged_16bit，CPU 内存建议在 32G 以上
# 如果合并时死机或报错，请改用 save_method = "merged_4bit" 导出到本地测试
model.save_pretrained_merged(
    FINAL_MODEL_DIR,
    tokenizer,
    save_method = "merged_16bit",
)

print(f"✅ 合并完成！含有‘王熙凤’语气的完整模型已保存至: {FINAL_MODEL_DIR}")

#虽然代码修正了，但你在执行 merged_16bit 时极大概率会遇到以下问题：

#显存/内存炸裂：把 4bit 模型还原成 16bit 模型是一个非常吃内存的操作。8B 模型合并成 16bit 约需要 16GB 显存 或 32GB 内存。

#如果报错 OOM：请将 save_method 改为 "merged_4bit"。

#好处：merged_4bit 导出的模型文件夹可以直接被 vLLM 或 Unsloth 再次加载，且体积小（约 5.5GB），对 2060S 非常友好。
