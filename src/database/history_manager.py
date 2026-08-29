"""
    Module đọc, ghi message vào DB.
"""
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory


def read_history(conn, conversation_id: str) -> list[BaseMessage]:
    """Lấy full list LangChain messages từ DB."""

    conversation_history = PostgresChatMessageHistory(
        "chat_history", conversation_id, sync_connection=conn
    )

    return conversation_history.get_messages()

def persist_turn(
        conn, conversation_id: str, question: str, answer: str
) -> None:
    """Append cặp messages gồm AI, Human type vào DB."""

    conversation_history = PostgresChatMessageHistory(
        "chat_history", conversation_id, sync_connection=conn 
    )

    # Define 2 message và gán type tương ứng
    list_messages = [
        HumanMessage(content=question),
        AIMessage(content=answer)
    ]

    conversation_history.add_messages(list_messages)
