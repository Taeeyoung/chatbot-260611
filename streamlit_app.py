import streamlit as st
from openai import OpenAI

MAX_MESSAGES = 20  # 최근 N개 메시지만 API에 전송

st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)

    # --- 사이드바: 설정 ---
    with st.sidebar:
        st.header("Settings")

        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful assistant.",
            height=120,
            help="챗봇의 역할과 성격을 정의합니다.",
        )

        st.divider()

        st.caption(f"메시지 수 제한: 최근 {MAX_MESSAGES}개")
        if "messages" in st.session_state:
            st.caption(f"현재 대화: {len(st.session_state.messages)}개")

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- 세션 초기화 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- 대화 출력 ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- 입력 처리 ---
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 최근 MAX_MESSAGES개만 잘라서 전송
        recent = st.session_state.messages[-MAX_MESSAGES:]

        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}] + [
                {"role": m["role"], "content": m["content"]} for m in recent
            ],
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # MAX_MESSAGES 초과 시 오래된 메시지 경고
        if len(st.session_state.messages) > MAX_MESSAGES:
            st.toast(
                f"대화가 {MAX_MESSAGES}개를 넘었습니다. 오래된 메시지는 AI 컨텍스트에서 제외됩니다.",
                icon="⚠️",
            )
