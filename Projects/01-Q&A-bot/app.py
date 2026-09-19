from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="AI Q&A Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0d1117;
        color: #ffffff;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem;
        max-width: 900px;
    }

    /* Header */
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }

    .subtitle {
        text-align: center;
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 12px;
    }

    /* User message */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #111827;
    }

    /* Input box */
    [data-testid="stChatInput"] {
        background: #161b22;
    }

    [data-testid="stChatInput"] textarea {
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #090d12;
        border-right: 1px solid #21262d;
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }

    .sidebar-text {
        color: #8b949e;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Status card */
    .status-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 14px;
        margin-top: 20px;
    }

    .status-online {
        color: #3fb950;
        font-weight: 600;
    }

    /* Clear button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #30363d;
        background: #161b22;
        color: white;
    }

    .stButton > button:hover {
        border-color: #58a6ff;
        color: #58a6ff;
    }

    /* Welcome box */
    .welcome {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 30px;
    }

    .welcome h3 {
        margin-bottom: 8px;
        color: white;
    }

    .welcome p {
        color: #8b949e;
    }

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        max_tokens=300
    )


llm = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🤖 AI Q&A</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-text">'
        'A simple conversational AI chatbot powered by '
        '<b>LangChain + Groq</b>.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ⚙️ Model")

    st.code("openai/gpt-oss-120b", language="text")

    st.markdown(
        """
        <div class="status-card">
            <div>🟢 <span class="status-online">Model Online</span></div>
            <br>
            <div style="color:#8b949e;">
                Provider: Groq<br>
                Framework: LangChain
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.history = []

        st.rerun()

    st.divider()

    st.markdown(
        """
        <div style="color:#8b949e; font-size:0.8rem;">
            <b>Built with</b><br><br>
            🦜 LangChain<br>
            ⚡ Groq<br>
            🎈 Streamlit<br>
            🐍 Python
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="main-title">🤖 AI Q&A Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask anything and have a conversation with AI'
    '</div>',
    unsafe_allow_html=True
)

if len(st.session_state.history) == 0:

    st.markdown(
        """
        ### 👋 Welcome!

        I'm your AI assistant. Ask me a question, start a conversation,
        or explore a topic.

        **Try asking:**

        💡 Explain how APIs work

        🤖 What is machine learning?

        🧠 Explain recursion in simple terms
        """,
        unsafe_allow_html=True
    )

for message in st.session_state.history:

    if message["role"] == "user":

        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])

    elif message["role"] == "assistant":

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])

query = st.chat_input("💬 Ask me anything...")


if query:

    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # Add user message to history
    st.session_state.history.append(
        {
            "role": "user",
            "content": query
        }
    )

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner("Thinking..."):

            try:

                response = llm.invoke(
                    st.session_state.history
                )

                answer = response.content

                st.markdown(answer)

                # Save AI response
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    "Something went wrong while generating the response."
                )

                st.caption(str(e))