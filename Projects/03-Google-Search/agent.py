# agent.py

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from tools import tools

load_dotenv()

MODEL = os.getenv("MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not MODEL:
    raise ValueError(
        "MODEL is missing from your .env file."
    )

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from your .env file."
    )


# ==========================================
# 1. INITIALIZE THE LLM
# ==========================================

llm = ChatGroq(
    model=MODEL,
    temperature=0.2,
    max_tokens=1500
)


# ==========================================
# 2. AGENT INSTRUCTIONS
# ==========================================

SYSTEM_PROMPT = """
You are a helpful AI assistant with access to two tools:

1. web_search
   Searches the live web for information, recent news,
   documentation, and other online resources.

2. get_current_weather
   Retrieves current weather conditions for a city.

Your responsibilities:

- Answer the user's question clearly and accurately.
- Use web_search when the question requires current
  information from the internet.
- Use get_current_weather when the user asks about
  current weather conditions in a city.
- Use both tools when the user asks a question that
  requires both web research and weather information.
- Do not invent search results, URLs, or weather data.
- If a tool fails, explain the limitation honestly.
- Use the returned search URLs when providing sources.
- Distinguish retrieved facts from your own explanations.
- Keep answers concise unless the user requests detail.
- If the city is ambiguous, ask the user to clarify it.
- Do not call tools unnecessarily for general questions.

Examples:

User: What is the weather in Kolkata?
Action: Call get_current_weather("Kolkata").

User: Find the latest news about AI agents.
Action: Call web_search with a relevant query.

User: What's the weather in London, and find recent
technology news from the city?
Action: Call both tools.

Always base current-information answers on tool results.
"""


# ==========================================
# 3. CREATE THE AGENT
# ==========================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


# ==========================================
# 4. HELPER FUNCTION
# ==========================================

def ask_agent(question: str, chat_history=None) -> str:
    """
    Send a question to the agent.

    chat_history may contain previous LangChain messages.
    """

    if not question.strip():
        return "Please enter a question."

    messages = list(chat_history or [])

    messages.append({
        "role": "user",
        "content": question
    })

    result = agent.invoke({
        "messages": messages
    })

    # The final message is normally the agent's answer.
    final_message = result["messages"][-1]

    return final_message.content