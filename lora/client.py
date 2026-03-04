import streamlit as st
from openai import OpenAI

# ================= 配置区域 =================
# 页面标题
st.set_page_config(page_title="红楼梦-凤姐聊天室", page_icon="🌶️")
st.title("🌶️ 和凤姐姐聊天吧 (AI版v0.3.2)")

# API 配置 (指向你刚才启动的 api_server.py)
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"  # 本地服务不需要真 Key

# ================= 侧边栏设置 =================
with st.sidebar:
    st.header("🎭 角色设定")
    system_prompt = st.text_area(
        "系统提示词 (System Prompt)",
        value="你扮演《红楼梦》中的王熙凤（人称凤姐、琏二奶奶）。你的性格泼辣张狂、精明强干、言语爽利，说话常带三分笑，善于察言观色。请根据前文情境，模仿她的语气进行回答。",
        height=150
    )

    st.divider()

    st.header("⚙️ 参数调节")
    temperature = st.slider("活跃度 (Temperature)", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("最大回复长度", 128, 1024, 512, 64)

    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# ================= 初始化客户端 =================
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# 初始化消息历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 聊天界面主逻辑 =================

# 1. 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. 处理用户输入
if prompt := st.chat_input("请给凤奶奶请安..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. 生成 AI 回复
    with st.chat_message("assistant"):
        # 构建包含 System Prompt 的完整请求列表
        # 注意：System Prompt 不存入历史记录显示在界面上，但每次请求都会带上
        messages_to_send = [
                               {"role": "system", "content": system_prompt}
                           ] + st.session_state.messages

        stream = st.session_state.client.chat.completions.create(
            model="qwen-lora",
            messages=messages_to_send,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        # Streamlit 自动处理流式输出
        response = st.write_stream(stream)

    # 存入历史
    st.session_state.messages.append({"role": "assistant", "content": response})