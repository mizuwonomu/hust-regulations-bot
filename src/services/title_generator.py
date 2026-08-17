"""
    Định nghĩa LLM chain sinh title
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.rag.config import TITLE_GENERATOR_MODEL

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an AI model that create concise chat title for a chatbot system conversation. 
            Your task is to read the first query of user and inference LLM's response, then write a summarization title.
             
            Strict rules:
            - Output must be Vietnamese only.
            - Output must be at most 8 words (strictly below 9 words).
            - Return title text only.
            - Do not include explanation.
            - Do not wrap output in quotes, brackets, markdown, or punctuation-only wrappers.

            Examples:
                - user: Khi nào sẽ được mở lớp học phần rút gọn vậy?
                - ai: Theo điều 10, khoản 5, sinh viên sẽ được mở học phần rút gọn khi thoả mãn đồng thời các điều kiện sau...
                YOUR output: Điều kiện học phần rút gọn

                - user: Giờ t đang nợ 9 tín, thì có bị sao không?
                - ai: Theo điều 19, khoản 1, sinh viên có số tín chỉ không đạt trong học kỳ lớn hơn 8 sẽ bị nâng một mức cảnh báo học tập ...
                YOUR output: Cảnh báo học tập

                - user: Nếu học phần được 3.38 điểm, vậy theo thang 10 là bao nhiêu?
                - ai: Theo điều 12, khoản 7, dải điểm tương đương và công thức quy đổi là...
                YOUR output: Quy đổi điểm học phần
            """),
        
        ("human", 
         """First user query: {question}
            First AI answer: {full_response}

            Generate a single Vietnamese title that follows all rules.
         """), #truyền luôn response string của llm vào human để tránh missing fields
    ]
)

#1. Reuse: Cache để backend không cần tạo lại chain mỗi lần invoke sinh title 
#=> tạo chain một lần. Lần sau dùng chỉ cần trả lại chain cũ và reuse
#2. Lazy initialization: Không tạo ra LLM chain cho đến khi thực sự cần sinh title
#Note: Đây chỉ cache chain object, kgông hề cache output title. Với mỗi input khác nhau, output cũng khác nhau
#Tức các title vẫn được LLM gọi bình thường, nó không nhớ câu hỏi nào tạo ra title nào, chỉ nhớ pipeline dùng để gọi LLM
@lru_cache(maxsize=1)
def get_title_chain():
    """Chain LCEL sinh title"""

    generate_title_llm = ChatGroq(
        model=TITLE_GENERATOR_MODEL,
            max_retries=0,
            temperature= 0.3,
            reasoning_effort="none"  #qwen3: tóm tắt 8 từ, không cần thinking token
    )
    title = prompt | generate_title_llm | StrOutputParser()
    return title

def _normalize_assistant_answer(full_response: Any) -> str:
    """Chuẩn hoá AIMessage từ inference response sang dạng string
    để làm đầu vào cho title generation"""
    if isinstance(full_response, str):
        return full_response
    
    if isinstance(full_response, AIMessage):
        return full_response.content if isinstance(full_response.content, str) else str(full_response.content)
    
    if isinstance(full_response, dict):
        for key in ("answer", "content", "output", "text"):
            value = full_response.get(key)
            if value:
                return str(value)
        return str(full_response)
    
    if isinstance(full_response, list):
        return " ".join(str(item) for item in full_response if item is not None)
    
    return str(full_response)


def generate_title(question: str, full_response: Any) -> str:
    """Sync title generation for FastAPI background tasks"""

    assistant_text = _normalize_assistant_answer(full_response).strip()
    result = get_title_chain().invoke(
        {
            "question": question.strip(),
            "full_response": assistant_text,
        }
    )

    return result.strip()


async def generate_title_async(question: str, full_response: Any) -> str:
    """Async title generation for FastAPI background tasks"""

    assistant_text = _normalize_assistant_answer(full_response).strip()
    result = await get_title_chain().ainvoke(
        {
            "question": question.strip(),
            "full_response": assistant_text,
        }
    )

    return result.strip()
