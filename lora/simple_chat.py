import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= 配置 =================
# 1. 基座模型路径
BASE_MODEL_ID = "/root/.cache/modelscope/hub/models/Qwen/Qwen1.5-1.8B-Chat"
# 2. 你刚才训练保存的路径
LORA_PATH = "./Qwen1.5-1.8B-Chat-lora-out-v0.3.1"

# ================= 加载模型 =================
print("正在加载基座模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("正在挂载微调后的 LoRA 适配器...")
# 将训练好的 LoRA 权重与基座模型合并
model = PeftModel.from_pretrained(base_model, LORA_PATH)


# ================= 对话测试函数 =================
def chat_with_model(query):
    # 构造标准 ChatML 格式的输入
    messages = [
        {"role": "system",
         "content": "你现在是《红楼梦》中的王熙凤（人称凤姐、琏二奶奶）。你的性格泼辣张狂、精明强干、言语爽利，说话常带三分笑，善于察言观色，同时也有些心狠手辣。请根据前文情境，模仿她的语气进行回答。"},
        # 可以改成你训练数据里的 system prompt
        {"role": "user", "content": query}
    ]

    # 转换为模型输入 ID
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成回答
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,  # 回答最大长度
            temperature=0.7,  # 0.7 比较平衡，想更有创造力设 0.9，想更严谨设 0.1
            top_p=0.3
        )

    # 解码结果
    # 只要生成的新部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"凤姐: {response}")
    print("-" * 40)


# ================= 开始测试 =================
# 请在这里输入几个你训练集里有的问题，或者是类似风格的问题
while True:
    try:
        user_input = input("你说：").strip()

        resp = chat_with_model(user_input)


    except KeyboardInterrupt:
        print("\n对话结束 👋")
        break