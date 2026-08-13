import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random

load_dotenv()

DEFAULT_RETRIEVAL_RESULTS = "evals/v2/results/eval_20260405_063409.json"
#model sinh đáp án — để lộ ra CLI vì đây là biến duy nhất khi so sánh model
DEFAULT_INFERENCE_MODEL = "qwen/qwen3-32b"
DEFAULT_TEMPERATURE = 0.1


class E2EInputRow(BaseModel):
    id: Any
    query: str
    reference: str
    retrieved_contexts: list[str] = Field(default_factory=list)


class E2EGeneratedRow(BaseModel):
    id: Any
    query: str
    predicted_response: str
    reference: str
    retrieved_contexts: list[str]


def format_docs(docs: list[str]) -> str:
    """Mirror src/rag/qa_chain.py::format_docs

    Production ghép title vào nhãn source và làm phẳng xuống dòng trong nội
    dung. Ở đây context đã là string (title + content do retrieval eval ghép),
    nên chỉ cần làm phẳng cho khớp cách LLM nhìn thấy ở production
    """
    formatted = []
    for i, content in enumerate(docs):
        flattened = content.replace("\n", " ")
        formatted.append(f"Source [{i+1}]:\n{flattened}")
    return "\n\n".join(formatted)


def _random_offset_sleep(min_seconds: int = 5, max_seconds: int = 10) -> None:
    seconds = random.randint(min_seconds, max_seconds)
    print(f"[inference_offset] Sleeping {seconds}s...")
    time.sleep(seconds)


def _is_rate_limited_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return ("429" in message) or ("rate limit" in message) or ("too many requests" in message)


def run_generation(
    retrieval_results_path: str,
    output_path: str,
    model: str = DEFAULT_INFERENCE_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> None:
    if "GROQ_API_KEY" not in os.environ:
        raise EnvironmentError("GROQ_API_KEY is required in environment or .env")

    raw = json.loads(Path(retrieval_results_path).read_text(encoding="utf-8"))
    rows = raw.get("results", [])

    dataset = [
        E2EInputRow.model_validate(
            {
                "id": row["id"],
                "query": row.get("query") or row.get("user_input"),
                "reference": row.get("reference") or row.get("response"),
                "retrieved_contexts": row.get("retrieved_contexts", []),
            }
        )
        for row in rows
    ]

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system", """M là trợ lý AI chuyên hỗ trợ sinh viên HUST - Đại học Bách Khoa tra cứu Quy chế đào tạo (Academic Regulations)
            Nhiệm vụ của m là trả lời chính xác dựa trên Context được cung cấp.
            
            Quy tắc tuyệt đối (絶対ルール):
            1. Luôn trích dẫn rõ ràng thông tin nằm ở **Điều nào** (Dựa vào dòng đầu tiên của context). Với các con số (tín chỉ, học phí, mức cảnh báo), phải tuyệt đối chính xác.
            2. KHÔNG được sử dụng kiến thức bên ngoài (Outside knowledge) hoặc kiến thức có sẵn trong model (Pre-trained knowledge) để trả lời.
            3. Nếu thông tin không có trong Context, hãy trả lời: "Thông tin này không có trong quy chế hiện tại."
            4. KHÔNG được bịa đặt nội dung (Hallucination). 
            5. Giọng văn nghiêm túc nhưng dễ hiểu, phù hợp với sinh viên.

            Quy tắc trích dẫn (Citation Rules): 
            1. Khi đưa ra bất kỳ thông tin nào, PHẢI trích dẫn nguồn gốc theo điều, khoản nào và bằng cú pháp [index]. 
            Ví dụ:
            - "User": "Quy định về học phí"
            - Answer: "Theo quy định về học phí tại điều 9 [1], sinh viên phải..." 

            - "User": "Học phần song hành là gì?"
            - Answer: "Theo điểm c khoản 6 điều 4 [1], học phần song hành là: Học phần A là học phần song hành của học phần B thì sinh
            viên phải theo học trước hoặc học đồng thời với học phần B..."
            
            2. Nếu thông tin đến từ nhiều nguồn, hãy liệt kê đủ: [1], [2], [3],.... 
            Ví dụ:
            - "User": "Tôi có điểm CPA toàn khoá là 3,4 và tổng số tín chỉ học lại chiếm 6% tổng số tín chỉ dùng để tính điểm. Vậy tôi đang được xếp loại học lực gì và khi tốt nghiệp bằng cử nhân của tôi sẽ nhận được hạng loại gì?
            - Answer: "Với điểm CPA toàn khóa là 3,4 (nằm trong khoảng 3,2 đến 3,59), bạn sẽ được xếp loại học lực Giỏi theo khoản 6 Điều 12 [1]. Tuy nhiên, do khối lượng học phần phải học lại của bạn chiếm 6% (vượt quá mức 5% của tổng số tín chỉ được dùng tính điểm trung bình toàn khóa), hạng tốt nghiệp của bạn sẽ bị giảm đi một mức xuống thành loại Khá theo điểm a khoản 2 Điều 15 [2]."
            
            3. Cuối câu trả lời KHÔNG cần tạo danh sách tài liệu tham khảo (vì giao diện sẽ tự hiển thị).   

            Context:
            {context}
            """,
        ),
        ("human", "{question}"),
    ])

    inference_llm = ChatGroq(
        model=model,
        temperature=temperature,
        reasoning_format="parsed",
    )
    answer_chain = qa_prompt | inference_llm | StrOutputParser()

    e2e_chain = RunnablePassthrough().assign(
        context=RunnableLambda(lambda x: format_docs(x["retrieved_contexts"]))
    ).assign(
        predicted_response=(
            RunnableLambda(lambda x: {"context": x["context"], "question": x["query"]})
            | answer_chain
        )
    )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random(min=5, max=10),
        retry=retry_if_exception(_is_rate_limited_error),
        reraise=True,
    )
    def _invoke_inference(payload: dict[str, Any]) -> dict[str, Any]:
        return e2e_chain.invoke(payload)

    generated: list[dict[str, Any]] = []
    for i, row in enumerate(dataset):
        out = _invoke_inference(row.model_dump())
        generated.append(
            E2EGeneratedRow(
                id=row.id,
                query=row.query,
                predicted_response= out["predicted_response"],
                reference=row.reference,
                retrieved_contexts=row.retrieved_contexts,
            ).model_dump()
        )
        if i < len(dataset) - 1:
            _random_offset_sleep(5, 10)

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        #model/temperature là thuộc tính của CẢ lần chạy, không phải của từng row
        #-> nhờ đó file kết quả tự mô tả được nó sinh ra từ model nào
        "model": model,
        "temperature": temperature,
        "retrieval_results_path": retrieval_results_path,
        "dataset_path": raw.get("dataset_path", ""),
        "num_samples": len(generated),
        "results": generated,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved generated responses to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate answers from retrieval results with a chosen model")
    parser.add_argument("--retrieval-results", type=str, default=DEFAULT_RETRIEVAL_RESULTS)
    parser.add_argument("--model", type=str, default=DEFAULT_INFERENCE_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--output",
        type=str,
        default=f"evals/v2/results/eval_e2e_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_generation(
        retrieval_results_path=args.retrieval_results,
        output_path=args.output,
        model=args.model,
        temperature=args.temperature,
    )
