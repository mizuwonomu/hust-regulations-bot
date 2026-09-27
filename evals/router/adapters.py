"""Hai adapter định tuyến chỉ-gọi-router: arm Groq (snapshot production) và arm Jev (TypeSafe Choice).

- Arm Groq giữ nguyên prompt, model, temperature, reasoning_effort, max_retries và rule chọn nhánh
  của production; rule đó được kiểm tra parity trực tiếp trên source `src/rag/qa_chain.py` (AST,
  không import - module đó gọi load_dotenv lúc import)
- Arm Jev hỏi đúng một câu `Choice` với hai criteria `RAG`/`chat`; request và response đều bị guard
  theo type contract trước/sau network call, không thêm output parser
- Cả hai adapter chỉ nhận query làm input: `id`, `group`, `type` của corpus không bao giờ được gửi
- Mọi lỗi provider trở thành trial lỗi (executed branch null), không bao giờ bị ép thành một route
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_typesafe import (
    Choice,
    ChoiceAnswer,
    ClassifierResponse,
    TypeSafeClassifier,
)

from evals.router.corpus import ROUTE_LABELS, RouteLabel

ArmName = Literal["groq", "jev"]
ARMS: tuple[ArmName, ...] = ("groq", "jev")

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_QA_CHAIN_PATH = REPO_ROOT / "src" / "rag" / "qa_chain.py"

ROUTE_QUESTION_ID = "route"
TIMEOUT_SECONDS = 30.0
JEV_MODEL = "jev-1.13.0"
# langchain_groq tự đổi temperature 0 thành 1e-08 trên wire (chat_models.py), giữ 0 cho parity
GROQ_TEMPERATURE = 0.0
GROQ_REASONING_EFFORT = "none"
GROQ_MAX_RETRIES = 0
GROQ_REQUEST_ID_HEADER = "x-request-id"
FORMAT_VALID_TEXTS = ("chat", "RAG")

# Snapshot nguyên văn của `router_template` trong src/rag/qa_chain.py, giữ nguyên cả thụt lề và
# khoảng trắng cuối dòng vì đó là byte thật được gửi cho model
# Test parity fail nếu snapshot này lệch production
GROQ_ROUTER_PROMPT = """
    M là một chuyên gia phân loại câu hỏi. Hãy đọc câu hỏi của người dùng và quyết định câu hỏi đó thuộc loại nào"

    1. 'chat': Các câu chào hỏi xã giao, hỏi thăm sức khỏe, không liên quan đến thông tin cụ thể trong tài liệu 
    - Ví dụ: 
    + Human: Hôm nay nên ăn gì nhỉ? -> Tao nghĩ hôm nay mày nên ăn phở đó! 
    + Human: Tự nhiên buồn ghê -> Sao thế, có chuyện gì m muốn nói không?
    
    2. 'RAG': 
        - Các câu hỏi liên quan đến quy chế đào tạo, Luật, Quy định nhà trường
        - Từ khóa nhận diện: Ví dụ: "học phí", "tín chỉ", "cảnh báo học tập", "điểm học phần",...
        - Các câu hỏi về thủ tục, điều kiện, thời gian học tập tại HUST.

    Chỉ được trả kết quả theo 1 từ duy nhất: 'chat' hoặc 'RAG'. KHÔNG trả lời thêm bất cứ điều gì khác.

    Câu hỏi: {question}
    Phân loại:
    """

# Câu hỏi Choice dùng đúng câu hỏi phân loại của production
JEV_ROUTE_INSTRUCTION = (
    "Hãy đọc câu hỏi của người dùng và quyết định câu hỏi đó thuộc loại nào"
)

# Hai criteria giữ nguyên định nghĩa hai nhãn trong prompt Groq, không thêm guidance và không lấy
# ví dụ nào từ corpus heldout
ROUTE_CRITERIA: dict[str, str] = {
    "RAG": (
        "Các câu hỏi liên quan đến quy chế đào tạo, Luật, Quy định nhà trường. "
        "Từ khóa nhận diện ví dụ: \"học phí\", \"tín chỉ\", \"cảnh báo học tập\", \"điểm học phần\". "
        "Các câu hỏi về thủ tục, điều kiện, thời gian học tập tại HUST"
    ),
    "chat": (
        "Các câu chào hỏi xã giao, hỏi thăm sức khỏe, không liên quan đến thông tin cụ thể trong tài liệu"
    ),
}


class PromptParityError(RuntimeError):
    """Prompt hoặc rule chọn nhánh của production đã trôi khỏi snapshot của harness"""


class RouteContractError(RuntimeError):
    """Request hoặc response của TypeSafe vi phạm type contract"""


class ChatInvoker(Protocol):
    """Tối thiểu mà arm Groq cần từ một chat model"""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage: ...


class ClassifierInvoker(Protocol):
    """Tối thiểu mà arm Jev cần từ TypeSafe classifier"""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> ClassifierResponse: ...


@dataclass(frozen=True)
class TokenUsage:
    """Usage nullable: None nghĩa provider không báo, khác hẳn 0"""

    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class RouteTrial:
    """Kết quả một lượt gọi định tuyến của một arm (chưa gắn case index/gold label)"""

    arm: ArmName
    raw_answer: str | None
    executed_branch: RouteLabel | None
    format_valid: bool | None
    usage: TokenUsage | None
    latency_ms: float
    request_id: str | None
    model: str | None
    attempts: int
    error_type: str | None = None
    error_message: str | None = None
    detail: dict | None = None

    def as_dict(self) -> dict:
        """Bản ghi JSON-safe của trial, đủ field cho artifacts và scoring"""
        return {
            "arm": self.arm,
            "raw_answer": self.raw_answer,
            "executed_branch": self.executed_branch,
            "format_valid": self.format_valid,
            "usage": None
            if self.usage is None
            else {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
            "latency_ms": self.latency_ms,
            "request_id": self.request_id,
            "model": self.model,
            "attempts": self.attempts,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "detail": self.detail,
        }


def executed_branch_from_text(raw_text: str) -> RouteLabel:
    """Rule chọn nhánh của production: trim, lower, chứa 'chat' thì chat, còn lại RAG

    raw_text: text thô model trả về, chưa qua bất kỳ parser nào
    """
    return "chat" if "chat" in raw_text.strip().lower() else "RAG"


def format_is_valid(raw_text: str) -> bool:
    """Format hợp lệ chỉ khi text đã trim đúng bằng 'chat' hoặc 'RAG'

    raw_text: text thô model trả về
    """
    return raw_text.strip() in FORMAT_VALID_TEXTS


def raw_text_from_message(message: BaseMessage) -> str:
    """Đổi message thành text bằng đúng StrOutputParser mà production dùng sau router_llm

    message: AIMessage đã được giữ lại cùng usage trước khi đổi content
    """
    return StrOutputParser().invoke(message)


def usage_from_message(message: BaseMessage) -> TokenUsage | None:
    """Lấy usage_metadata của message; thiếu dữ liệu thì None chứ không phải 0

    message: AIMessage nhận từ provider
    """
    metadata = message.usage_metadata
    if not metadata:
        return None
    return usage_from_tokens(metadata.get("input_tokens"), metadata.get("output_tokens"))


def usage_from_tokens(
    input_tokens: int | None,
    output_tokens: int | None,
) -> TokenUsage | None:
    """Chuẩn hoá usage của cả hai arm: không có số nào thì trả None, khác hẳn dict token null

    input_tokens: số token vào provider báo, None nếu không báo
    output_tokens: số token ra provider báo, None nếu không báo
    """
    if input_tokens is None and output_tokens is None:
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def groq_prompt_sha256() -> str:
    """SHA256 của đúng prompt text gửi cho Groq, ghi vào manifest"""
    return hashlib.sha256(GROQ_ROUTER_PROMPT.encode("utf-8")).hexdigest()


def jev_criteria_sha256() -> str:
    """SHA256 của criteria Jev theo canonical JSON, ghi vào manifest"""
    payload = json.dumps(
        {"criteria": ROUTE_CRITERIA, "instructions": JEV_ROUTE_INSTRUCTION},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_route_questions() -> dict[str, Choice]:
    """Tạo đúng một câu hỏi Choice id 'route' với hai criteria RAG/chat"""
    return {
        ROUTE_QUESTION_ID: Choice(
            instructions=JEV_ROUTE_INSTRUCTION,
            criteria=dict(ROUTE_CRITERIA),
        )
    }


def validate_route_questions(questions: Mapping[str, object]) -> None:
    """Chặn request sai contract trước mọi network call

    questions: mapping id -> question sắp gửi cho TypeSafe
    So cả tên và nội dung criteria/instruction, vì fingerprint của manifest chỉ hash constants
    gốc nên một question bị sửa nội dung vẫn sẽ vượt qua nếu chỉ kiểm tra tên
    """
    if set(questions) != {ROUTE_QUESTION_ID}:
        raise RouteContractError(
            f"chỉ được có đúng một câu hỏi id {ROUTE_QUESTION_ID!r}, thấy {sorted(questions)}"
        )
    question = questions[ROUTE_QUESTION_ID]
    # Exact type: Noul, Score và mọi subclass của Choice đều bị loại
    if type(question) is not Choice:
        raise RouteContractError(
            f"question phải đúng type Choice, thấy {type(question).__name__}"
        )
    if question.criteria != ROUTE_CRITERIA:
        raise RouteContractError(
            f"criteria phải đúng {sorted(ROUTE_CRITERIA)} với đúng nội dung đã freeze, "
            f"thấy {sorted(question.criteria)}"
        )
    if question.instructions != JEV_ROUTE_INSTRUCTION:
        raise RouteContractError(
            f"instructions phải đúng câu hỏi đã freeze, thấy {question.instructions!r}"
        )


def extract_route_choice(response: object) -> ChoiceAnswer:
    """Validate response TypeSafe và trả ChoiceAnswer ở 'route'

    response: giá trị thô mà classifier trả về
    """
    if not isinstance(response, ClassifierResponse):
        raise RouteContractError(
            f"response phải là ClassifierResponse, thấy {type(response).__name__}"
        )
    if response.nouls or response.scores:
        raise RouteContractError("response có nouls/scores, không được phép")
    if set(response.answers) != {ROUTE_QUESTION_ID}:
        raise RouteContractError(
            f"response chỉ được có answer {ROUTE_QUESTION_ID!r}, thấy {sorted(response.answers)}"
        )
    answer = response.answers[ROUTE_QUESTION_ID]
    # Exact type: subclass của ChoiceAnswer cũng bị loại
    if type(answer) is not ChoiceAnswer:
        raise RouteContractError(
            f"answer phải đúng type ChoiceAnswer, thấy {type(answer).__name__}"
        )
    if answer.choice not in ROUTE_LABELS:
        raise RouteContractError(
            f"label phải thuộc {list(ROUTE_LABELS)}, thấy {answer.choice!r}"
        )
    return answer


def production_router_template(source: str | None = None) -> str:
    """Trích `router_template` của production bằng AST, không import module

    source: nội dung qa_chain.py; None thì đọc từ PRODUCTION_QA_CHAIN_PATH
    """
    tree = ast.parse(source if source is not None else _production_source())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(getattr(target, "id", None) == "router_template" for target in node.targets):
            return ast.literal_eval(node.value)
    raise PromptParityError("không tìm thấy router_template trong qa_chain.py")


def production_route_decision_source(source: str | None = None) -> str:
    """Trích source hàm route_decision của production để kiểm tra rule fallback

    source: nội dung qa_chain.py; None thì đọc từ PRODUCTION_QA_CHAIN_PATH
    """
    text = source if source is not None else _production_source()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == "route_decision":
            return ast.get_source_segment(text, node) or ""
    raise PromptParityError("không tìm thấy route_decision trong qa_chain.py")


# Rule production phải còn nguyên cấu trúc: chuẩn hoá text, test chứa 'chat', rồi ánh xạ nhánh
# chat -> chat_chain và nhánh còn lại -> rag_chain. So bằng AST nên comment, khoảng trắng hay một
# dòng comment chứa đúng các token cũ đều không thể làm parity pass giả
EXPECTED_ROUTE_DECISION_SOURCE = '''
def route_decision(info):
    decision = router_chain.invoke({"question": info["question"]})
    decision = decision.strip().lower()
    if "chat" in decision:
        return chat_chain
    else:
        return rag_chain
'''


def check_production_prompt_parity(source: str | None = None) -> None:
    """Fail nếu prompt snapshot hoặc rule chọn nhánh của production đã trôi

    source: nội dung qa_chain.py, dùng cho test drift; None thì đọc file thật
    """
    if production_router_template(source) != GROQ_ROUTER_PROMPT:
        raise PromptParityError("router_template của production khác snapshot trong harness")
    production_rule = production_route_decision_source(source)
    expected_rule = ast.dump(ast.parse(EXPECTED_ROUTE_DECISION_SOURCE).body[0])
    actual_rule = ast.dump(ast.parse(production_rule).body[0])
    if actual_rule != expected_rule:
        raise PromptParityError(
            "rule route_decision của production khác reference: "
            f"{' '.join(production_rule.split())}"
        )


def _production_source() -> str:
    return PRODUCTION_QA_CHAIN_PATH.read_text(encoding="utf-8")


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


def _error_trial(arm: ArmName, started_at: float, error: BaseException) -> RouteTrial:
    """Biến lỗi provider thành trial lỗi, giữ nguyên type/message để đọc lại sau"""
    return RouteTrial(
        arm=arm,
        raw_answer=None,
        executed_branch=None,
        format_valid=None,
        usage=None,
        latency_ms=_elapsed_ms(started_at),
        request_id=None,
        model=None,
        attempts=1,
        error_type=type(error).__name__,
        error_message=str(error),
    )


class GroqWireCapture:
    """Bắt id và model thực phục vụ từ HTTP response thô của Groq.

    langchain_groq bỏ id completion (AIMessage.id là run id `lc_run--...` của LangChain) và echo
    lại model đã request, nên hai field này chỉ có ở tầng HTTP. Run dùng concurrency 1 nên một
    capture dùng chung là an toàn; `start()` xoá giá trị cũ trước mỗi call
    """

    def __init__(self) -> None:
        self.request_id: str | None = None
        self.served_model: str | None = None

    def start(self) -> None:
        """Xoá giá trị của call trước để không lẫn sang trial sau"""
        self.request_id = None
        self.served_model = None

    def on_response(self, response: httpx.Response) -> None:
        """Event hook của httpx: đọc id và model trong body, header chỉ là fallback

        response: response thô do httpx trả về
        """
        self.request_id = response.headers.get(GROQ_REQUEST_ID_HEADER) or self.request_id
        try:
            body = json.loads(response.read())
        except ValueError:
            return
        if not isinstance(body, dict):
            return
        self.served_model = body.get("model") or self.served_model
        self.request_id = body.get("id") or self.request_id


class GroqRouterAdapter:
    """Arm Groq: đúng prompt/model/tham số của production, không retry ở tầng harness"""

    arm: ArmName = "groq"

    def __init__(
        self,
        model: str,
        llm: ChatInvoker | None = None,
        timeout: float = TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """model: model ID gửi cho provider (production truyền ROUTER_MODEL)
        llm: chat model tiêm vào cho test; None thì dựng ChatGroq thật theo model
        timeout: timeout chung của mọi call trong experiment
        transport: transport httpx thay thế, chỉ dùng cho test offline
        """
        self.model = model
        self.timeout = timeout
        self._prompt = ChatPromptTemplate.from_template(GROQ_ROUTER_PROMPT)
        self._capture = GroqWireCapture()
        self._llm = llm if llm is not None else self._build_llm(model, timeout, transport)

    def _build_llm(
        self,
        model: str,
        timeout: float,
        transport: httpx.BaseTransport | None,
    ) -> ChatGroq:
        client = httpx.Client(
            timeout=timeout,
            transport=transport,
            event_hooks={"response": [self._capture.on_response]},
        )
        return ChatGroq(
            model=model,
            temperature=GROQ_TEMPERATURE,
            reasoning_effort=GROQ_REASONING_EFFORT,
            max_retries=GROQ_MAX_RETRIES,
            timeout=timeout,
            http_client=client,
        )

    def route(self, query: str) -> RouteTrial:
        """Gọi một lượt định tuyến, không raise ra ngoài

        query: câu hỏi người dùng, là input duy nhất của model
        """
        messages = self._prompt.format_messages(question=query)
        self._capture.start()
        started_at = time.perf_counter()
        try:
            message = self._llm.invoke(messages)
            # Giữ message và usage trước khi đổi content thành text
            usage = usage_from_message(message)
            raw_text = raw_text_from_message(message)
        except Exception as error:  # noqa: BLE001 - lỗi provider là dữ liệu, không phải crash
            return _error_trial(self.arm, started_at, error)
        return RouteTrial(
            arm=self.arm,
            raw_answer=raw_text,
            executed_branch=executed_branch_from_text(raw_text),
            format_valid=format_is_valid(raw_text),
            usage=usage,
            latency_ms=_elapsed_ms(started_at),
            request_id=self._capture.request_id,
            model=self._capture.served_model,
            attempts=1,
        )


class JevRouterAdapter:
    """Arm Jev: một câu Choice duy nhất, guard cả request lẫn response theo type contract"""

    arm: ArmName = "jev"

    def __init__(
        self,
        classifier: ClassifierInvoker | None = None,
        model: str = JEV_MODEL,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        """classifier: classifier tiêm vào cho test; None thì dựng TypeSafeClassifier thật
        model: model TypeSafe (pin theo manifest của run)
        timeout: timeout chung của mọi call trong experiment
        """
        self.model = model
        self.timeout = timeout
        self._questions = build_route_questions()
        validate_route_questions(self._questions)
        self._classifier = (
            classifier
            if classifier is not None
            else TypeSafeClassifier(model=model, timeout=timeout)
        )

    def route(self, query: str) -> RouteTrial:
        """Gọi một lượt định tuyến; mọi lỗi provider thành trial lỗi, không raise ra ngoài

        query: câu hỏi người dùng, là state duy nhất gửi cho TypeSafe
        Raise RouteContractError chỉ khi request của chính harness sai contract - đó là lỗi lập
        trình, phải fail to thay vì ghi thành một trial lỗi
        """
        # Guard trước mọi network call để request sai contract không tốn call nào
        validate_route_questions(self._questions)
        started_at = time.perf_counter()
        try:
            response = self._classifier.invoke({"state": query, "questions": self._questions})
            answer = extract_route_choice(response)
            usage = usage_from_tokens(
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
        except Exception as error:  # noqa: BLE001 - lỗi provider/contract đều thành trial lỗi
            return _error_trial(self.arm, started_at, error)
        return RouteTrial(
            arm=self.arm,
            raw_answer=answer.choice,
            executed_branch=answer.choice,
            # Response đã qua guard nên nhãn chắc chắn hợp lệ
            format_valid=True,
            usage=usage,
            latency_ms=_elapsed_ms(started_at),
            request_id=response.request_id,
            model=response.model,
            attempts=1,
            detail={
                "probabilities": answer.probabilities,
                "confidence": answer.confidence,
            },
        )
