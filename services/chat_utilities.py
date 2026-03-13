# services/chat_utilities.py
# Sends chat history to Gemini and returns the assistant's reply

from llm_factory.get_llm import get_gemini_llm


def get_answer(model_name: str, chat_history: list) -> str:
    """
    Sends the full chat history to Gemini and returns the response text.

    chat_history format: [{"role": "user"/"assistant", "content": "..."}]
    Gemini uses "user" and "model" as roles (not "assistant").
    """
    llm = get_gemini_llm(model_name)

    # Convert history (excluding last message) to Gemini format
    gemini_history = []
    for msg in chat_history[:-1]:
        gemini_role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({
            "role": gemini_role,
            "parts": [msg["content"]]
        })

    # Start chat session with history
    chat = llm.start_chat(history=gemini_history)

    # Send the latest user message
    last_message = chat_history[-1]["content"]
    response = chat.send_message(last_message)

    return response.text
