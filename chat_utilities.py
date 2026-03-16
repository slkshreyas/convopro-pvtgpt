# services/chat_utilities.py
from datetime import datetime
import pytz
from llm_factory.get_llm import get_gemini_llm


def get_current_datetime_context() -> str:
    """Gets real current date and time in IST."""
    IST = pytz.timezone("Asia/Kolkata")
    now = datetime.now(IST)
    return (
        f"Today is {now.strftime('%A, %d %B %Y')}. "
        f"Current time is {now.strftime('%I:%M %p')} IST (India). "
        f"Current year is {now.year}. Current month is {now.strftime('%B')}."
    )


def get_answer(model_name: str, chat_history: list) -> str:
    llm = get_gemini_llm(model_name)

    # Get real datetime RIGHT when user sends message
    datetime_context = get_current_datetime_context()

    # Inject datetime directly into the user's message
    last_message = chat_history[-1]["content"]
    augmented_message = (
        f"[SYSTEM CONTEXT — USE THIS AS GROUND TRUTH: {datetime_context}]\n\n"
        f"User message: {last_message}"
    )

    # Build Gemini history (all except last message)
    gemini_history = []
    for msg in chat_history[:-1]:
        gemini_role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({
            "role": gemini_role,
            "parts": [msg["content"]]
        })

    # Start chat with history and send augmented message
    chat = llm.start_chat(history=gemini_history)
    response = chat.send_message(augmented_message)

    return response.text