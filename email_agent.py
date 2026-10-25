
import os
import smtplib

from dotenv import load_dotenv
from email.message import EmailMessage

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")


# Validate environment variables
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
    raise ValueError("Email credentials are missing from .env")


# Initialize the LLM
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY
)


# Define the email-sending too
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email using Gmail after collecting all required details.

    Args:
        to: Valid recipient email address.
        subject: Non-empty email subject.
        body: Non-empty email body.
    """

    # Validate inputs BEFORE displaying the preview
    to = to.strip()
    subject = subject.strip()
    body = body.strip()

    if not to or "@" not in to or "." not in to.split("@")[-1]:
        return "ERROR: A valid recipient email address is required."

    if not subject:
        return "ERROR: Email subject is required. Ask the user for it."

    if not body:
        return "ERROR: Email body is required. Ask the user for it."

    # Show preview only after validation
    print("\n" + "=" * 45)
    print("EMAIL PREVIEW")
    print("=" * 45)
    print("From:", EMAIL_ADDRESS)
    print("To:", to)
    print("Subject:", subject)
    print("\nBody:\n", body)
    print("=" * 45)

    confirmation = input(
        "\nSend this email? Type YES to confirm: "
    ).strip()

    if confirmation != "YES":
        return "Email cancelled. Nothing was sent."

    message = EmailMessage()
    message["From"] = EMAIL_ADDRESS
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    try:
        with smtplib.SMTP_SSL(
            "smtp.gmail.com", 465, timeout=30
        ) as server:
            server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            server.send_message(message)

        return f"Email successfully sent to {to}."

    except (smtplib.SMTPException, OSError) as error:
        return f"Email sending failed: {error}"


# Create the agent
agent = create_agent(
    model=llm,
    tools=[send_email],
    system_prompt = """
        You are an email assistant.

        Rules:
        1. Collect the recipient email address, subject, and body.
        2. Ask the user for any missing details.
        3. Remember details provided in previous messages.
        4. Never invent a recipient address or email content.
        5. Once all three fields are available, call send_email immediately.
        6. Do not ask "Would you like me to send this email?"
        The send_email tool already displays a preview and asks
        the user for explicit confirmation.
        7. Never claim an email was sent unless the tool confirms success.
        8. If sending is cancelled or fails, explain that clearly.
    """
)

# Conversation loop
chat_history = []

print("Email AI Agent is ready!")
print("Enter 0 to exit.\n")

while True:
    try:
        user_input = input("You: ").strip()

        if user_input == "0":
            print("Agent: Goodbye!")
            break

        if not user_input:
            continue

        # Add the new user message
        from langchain_core.messages import HumanMessage

        chat_history.append(
            HumanMessage(content=user_input)
        )

        # Invoke agent with the complete conversation
        result = agent.invoke({
            "messages": chat_history
        })

        # Preserve assistant messages and tool results
        chat_history = result["messages"]

        # Display the latest assistant response
        for message in reversed(chat_history):
            if message.type == "ai" and message.content:
                print("\nAgent:", message.content, "\n")
                break

    except KeyboardInterrupt:
        print("\nAgent: Goodbye!")
        break

    except Exception as error:
        print(f"\nError: {error}\n")
