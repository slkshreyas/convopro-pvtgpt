# services/chat_utilities.py
from datetime import datetime
import pytz
from llm_factory.get_llm import get_gemini_llm


def get_system_context() -> str:
    """Builds a system prompt with current date, time and region info."""
    # India timezone (change to your region if needed)
    IST = pytz.timezone("Asia/Kolkata")
    now = datetime.now(IST)

    day_name = now.strftime("%A")           # e.g. Monday
    date_str = now.strftime("%d %B %Y")     # e.g. 13 March 2026
    time_str = now.strftime("%I:%M %p")     # e.g. 04:30 PM
    timezone = "IST (UTC+5:30)"
    region = "India"

    return (
        f"You are ConvoPro, a helpful AI assistant.\n"
        f"Current Date: {day_name}, {date_str}\n"
        f"Current Time: {time_str}\n"
        f"Timezone: {timezone}\n"
        f"Region: {region}\n"
        f"Always use this information when the user asks about "
        f"current date, time, day, or anything time-related. "
        f"Never say you don't know the current date or time."
    )


def get_answer(model_name: str, chat_history: list) -> str:
    """
    Sends chat history to Gemini with date/time context injected.
    """
    llm = get_gemini_llm(model_name)

    # Build Gemini history (excluding last user message)
    gemini_history = []

    # Inject system context as first user+model exchange
    gemini_history.append({
        "role": "user",
        "parts": [get_system_context()]
    })
    gemini_history.append({
        "role": "model",
        "parts": ["Understood! I'm ConvoPro. I have the current date, time and region context and will use it accurately in my responses."]
    })

    # Add rest of conversation history
    for msg in chat_history[:-1]:
        gemini_role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({
            "role": gemini_role,
            "parts": [msg["content"]]
        })

    # Start chat and send latest message
    chat = llm.start_chat(history=gemini_history)
    last_message = chat_history[-1]["content"]
    response = chat.send_message(last_message)

    return response.text


