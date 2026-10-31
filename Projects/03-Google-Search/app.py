# Google Search + Real Time Weather Agent

# app.py

import streamlit as st

from agent import ask_agent


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Search & Weather AI Agent",
    page_icon="🌐",
    layout="centered"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        color: #f0f6fc;
    }

    .subtitle {
        text-align: center;
        color: #8b949e;
        margin-bottom: 2rem;
    }

    [data-testid="stChatMessage"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
    }

    section[data-testid="stSidebar"] {
        background: #090d12;
        border-right: 1px solid #30363d;
    }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("## 🌐 Search & Weather AI")

    st.caption(
        "An AI agent that searches the live web "
        "and retrieves current weather conditions."
    )

    st.divider()

    st.markdown("### 🛠️ Available tools")

    st.markdown("""
    **🔎 Web Search**

    Search the internet for current information.

    **🌦️ Weather**

    Retrieve current weather for a city.
    """)

    st.divider()

    st.markdown("### 💡 Try asking")

    st.caption("What's the weather in Kolkata?")

    st.caption("Find the latest AI news.")

    st.caption(
        "What's the weather in London, "
        "and find recent news about London?"
    )

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# 5. MAIN HEADER
# ==========================================

st.markdown(
    '<div class="main-title">'
    '🌐 Search & Weather AI Agent'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions. Search the live web. Get current weather.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# 6. WELCOME SCREEN
# ==========================================

if not st.session_state.messages:
    st.markdown("### 👋 How can I help you?")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "🔎 **Web Search**\n\n"
            "Find recent news, technical documentation, "
            "and information from online sources."
        )

    with col2:
        st.info(
            "🌦️ **Real-Time Weather**\n\n"
            "Check temperature, humidity, wind speed, "
            "and current weather conditions."
        )


# ==========================================
# 7. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    if message["role"] not in ("user", "assistant"):
        continue

    avatar = "👤" if message["role"] == "user" else "🤖"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):
        st.markdown(message["content"])


# ==========================================
# 8. CHAT INPUT
# ==========================================

prompt = st.chat_input(
    "Ask about anything or check the weather..."
)


# ==========================================
# 9. PROCESS USER QUERY
# ==========================================

if prompt:

    # Display and store the user's message.
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    previous_messages = list(st.session_state.messages)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Call the AI agent.
    with st.chat_message("assistant", avatar="🤖"):

        try:
            with st.spinner(
                "Thinking and checking available tools..."
            ):
                answer = ask_agent(
                    question=prompt,
                    chat_history=previous_messages
                )

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:
            st.error("The agent encountered an error.")
            st.caption(str(error))

    st.rerun()