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


# Define the email-sending tool
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email using Gmail.

    Args:
        to: Recipient's email address.
        subject: Subject line of the email.
        body: Complete email message.
    """

    # Show the email before sending
    print("\n" + "=" * 45)
    print("EMAIL PREVIEW")
    print("=" * 45)
    print("From:", EMAIL_ADDRESS)
    print("To:", to)
    print("Subject:", subject)
    print("\nBody:\n", body)
    print("=" * 45)

    # Require explicit user confirmation
    confirmation = input(
        "\nSend this email? Type YES to confirm: "
    ).strip()

    if confirmation != "YES":
        return "Email cancelled. Nothing was sent."

    # Construct the email
    message = EmailMessage()
    message["From"] = EMAIL_ADDRESS
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    try:
        # Connect to Gmail SMTP
        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465,
            timeout=30
        ) as server:

            server.login(
                EMAIL_ADDRESS,
                EMAIL_APP_PASSWORD
            )

            server.send_message(message)

        return f"Email successfully sent to {to}."

    except (smtplib.SMTPException, OSError) as error:
        return f"Email sending failed: {error}"


# Create the AI agent
agent = create_agent(
    model=llm,
    tools=[send_email],
    system_prompt="""
    You are an email assistant.

    Your responsibilities:
    1. Understand the user's email-sending request.
    2. Ask for missing recipient, subject, or message details.
    3. Draft a clear and professional email.
    4. Use the send_email tool to send the email.
    5. Never claim an email was sent unless the tool confirms success.
    6. If sending is cancelled or fails, clearly explain that.
    7. Never invent a recipient's email address.
    """
)


# Run the agent
print("Email AI Agent is ready!")
print("Type 0 to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input == "0":
        print("Agent: Goodbye!")
        break

    if not user_input:
        continue

    try:
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": user_input}
            ]
        })

        print("\nAgent:", result["messages"][-1].content, "\n")

    except Exception as error:
        print("\nError:", error, "\n")