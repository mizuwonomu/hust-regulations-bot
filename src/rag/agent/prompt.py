"""Chat prompt phân cấp role và GBNF grammar động cho citation gate."""

from __future__ import annotations

from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate

_SYSTEM_PROMPT = """
You are the retrieval gate of a Vietnamese academic regulation QA system.
Each turn the loop shows you the question, the articles collected so far
(titles plus the sentences that cite other articles), and the candidate
articles cited by them. You decide the next retrieval step.

Rules:
- Follow a candidate only when its cited article contains information still needed to answer the question.
- Stop when none of the remaining candidates would add information needed for the question - being cited does not mean being needed.
- Stopping means "no current citation edge is needed for this question", not that the collected articles already prove a complete answer.
- In each decision choose exactly one action: stop, or follow one candidate.
- After following a candidate, the loop collects that article. If another unvisited candidate remains, you may be called again with an updated observation; if none remains, the loop ends without another gate call. Decide only the next single step, never the whole path.
- You may only follow articles that appear in the candidate list of the current decision.
- Reply with exactly one JSON Decision object and nothing else, no explanation: {{"stop": true, "dieu": null}} to stop, or {{"stop": false, "dieu": <candidate number>}} to follow."""

_FINAL_HUMAN_TEMPLATE = """Question:
{question}

Observation:
{observation}

Unvisited candidates:
{candidates}

Decide to stop or follow only one candidate."""

_GATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", """Question:
Luận văn lần 2 của tôi vẫn chưa đạt, tôi có thể đăng ký đề tài mới nếu còn đủ thời gian học không?

Observation:
Các Điều đã thu thập:
[Context 33] ĐÀO TẠO THẠC SĨ - Điều 33. Bảo vệ luận văn lần thứ hai
- 2. Trong trường hợp luận văn bảo vệ lần thứ hai vẫn không đạt yêu cầu, nếu học viên có nguyện vọng và còn được phép học tại ĐHBK Hà Nội đủ 6 tháng theo quy định về thời gian học tập tối đa tại điểm a khoản 3 Điều 3 Quy chế này thì có thể đăng ký đề tài mới cho luận văn thạc sĩ để thực hiện.

Unvisited candidates:
Điều 3"""),
        ("ai", '{{"stop": false, "dieu": 3}}'),
        ("human", """Question:
Luận văn lần thứ hai không đạt thì có được tổ chức bảo vệ lần thứ ba không?

Observation:
Các Điều đã thu thập:
[Context 33] ĐÀO TẠO THẠC SĨ - Điều 33. Bảo vệ luận văn lần thứ hai
- 2. Trong trường hợp luận văn bảo vệ lần thứ hai vẫn không đạt yêu cầu, nếu học viên có nguyện vọng và còn được phép học tại ĐHBK Hà Nội đủ 6 tháng theo quy định về thời gian học tập tối đa tại điểm a khoản 3 Điều 3 Quy chế này thì có thể đăng ký đề tài mới cho luận văn thạc sĩ để thực hiện. Không tổ chức bảo vệ luận văn lần thứ ba.

Unvisited candidates:
Điều 3"""),
        ("ai", '{{"stop": true, "dieu": null}}'),
        ("human", """Question:
Việc xử lý vi phạm đối với học viên cao học được thực hiện theo quy định nào?

Observation:
Các Điều đã thu thập:
[Context 26] ĐÀO TẠO KỸ SƯ - Điều 26. Xử lý vi phạm đối với học viên
- Việc xử lý vi phạm đối với học viên thực hiện theo Điều 20 Quy chế này.

Unvisited candidates:
Điều 20"""),
        ("ai", '{{"stop": false, "dieu": 20}}'),
        ("human", """Question:
Tôi muốn biết đầy đủ điều kiện bảo vệ luận án tiến sĩ ở cả cấp cơ sở và cấp Đại học, các yêu cầu đối với luận án, và mốc nộp hồ sơ cấp cơ sở được tính theo thời gian học tối đa như thế nào?

Observation:
Các Điều đã thu thập:
[Context 42] ĐÀO TẠO TIẾN SĨ - Điều 42. Đánh giá luận án tiến sĩ
- a) NCS được bảo vệ luận án cấp cơ sở nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 1 Điều 41 Quy chế này và hoàn thành hồ sơ đăng ký bảo vệ cấp cơ sở trong thời hạn đủ 90 ngày tính đến thời điểm kết thúc thời gian học tập chương trình tiến sĩ quy định tại khoản 3 Điều 3 Quy chế này.
- a) NCS được bảo vệ luận án cấp Đại học nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 2 Điều 40 Quy chế này và hoàn thành hồ sơ đề nghị đánh giá luận án ở hội đồng đánh giá luận án cấp Đại học.

Unvisited candidates:
Điều 41, Điều 3, Điều 40"""),
        ("ai", '{{"stop": false, "dieu": 41}}'),
        ("human", """Question:
Tôi muốn biết đầy đủ điều kiện bảo vệ luận án tiến sĩ ở cả cấp cơ sở và cấp Đại học, các yêu cầu đối với luận án, và mốc nộp hồ sơ cấp cơ sở được tính theo thời gian học tối đa như thế nào?

Observation:
Các Điều đã thu thập:
[Context 42] ĐÀO TẠO TIẾN SĨ - Điều 42. Đánh giá luận án tiến sĩ
- a) NCS được bảo vệ luận án cấp cơ sở nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 1 Điều 41 Quy chế này và hoàn thành hồ sơ đăng ký bảo vệ cấp cơ sở trong thời hạn đủ 90 ngày tính đến thời điểm kết thúc thời gian học tập chương trình tiến sĩ quy định tại khoản 3 Điều 3 Quy chế này.
- a) NCS được bảo vệ luận án cấp Đại học nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 2 Điều 40 Quy chế này và hoàn thành hồ sơ đề nghị đánh giá luận án ở hội đồng đánh giá luận án cấp Đại học.
[Context 41] ĐÀO TẠO TIẾN SĨ - Điều 41. Điều kiện được bảo vệ luận án tiến sĩ

Unvisited candidates:
Điều 3, Điều 40"""),
        ("ai", '{{"stop": false, "dieu": 3}}'),
        ("human", """Question:
Tôi muốn biết đầy đủ điều kiện bảo vệ luận án tiến sĩ ở cả cấp cơ sở và cấp Đại học, các yêu cầu đối với luận án, và mốc nộp hồ sơ cấp cơ sở được tính theo thời gian học tối đa như thế nào?

Observation:
Các Điều đã thu thập:
[Context 42] ĐÀO TẠO TIẾN SĨ - Điều 42. Đánh giá luận án tiến sĩ
- a) NCS được bảo vệ luận án cấp cơ sở nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 1 Điều 41 Quy chế này và hoàn thành hồ sơ đăng ký bảo vệ cấp cơ sở trong thời hạn đủ 90 ngày tính đến thời điểm kết thúc thời gian học tập chương trình tiến sĩ quy định tại khoản 3 Điều 3 Quy chế này.
- a) NCS được bảo vệ luận án cấp Đại học nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 2 Điều 40 Quy chế này và hoàn thành hồ sơ đề nghị đánh giá luận án ở hội đồng đánh giá luận án cấp Đại học.
[Context 41] ĐÀO TẠO TIẾN SĨ - Điều 41. Điều kiện được bảo vệ luận án tiến sĩ
[Context 3] NHỮNG QUY ĐỊNH CHUNG - Điều 3. Thời gian và kế hoạch học tập

Unvisited candidates:
Điều 40"""),
        ("ai", '{{"stop": false, "dieu": 40}}'),
        ("human", _FINAL_HUMAN_TEMPLATE),
    ]
)


def render_gate_prompt(
    question: str,
    observation: str,
    candidates: list[int],
) -> ChatPromptValue:
    """Render chat prompt cho một quyết định gate.

    Params:
    - question: Câu hỏi gốc của người dùng
    - observation: Compact observation từ excerpt chứa dẫn chiếu
    - candidates: Các Điều hợp lệ chưa được thăm
    """
    return _GATE_PROMPT.invoke(
        {
            "question": question,
            "observation": observation,
            "candidates": ", ".join(f"Điều {dieu}" for dieu in candidates),
        }
    )


def _validate_candidates(candidates: list[int]) -> None:
    if not candidates:
        raise ValueError("candidates rỗng")

    for dieu in candidates:
        if isinstance(dieu, bool) or not isinstance(dieu, int) or dieu <= 0:
            raise ValueError(f"candidate không hợp lệ: {dieu!r}")

    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates có phần tử trùng lặp")


def build_decision_grammar(candidates: list[int]) -> str:
    """Build GBNF grammar theo candidates của hop hiện tại.

    Params:
    - candidates: Các số Điều hợp lệ theo đúng thứ tự đầu vào
    """
    _validate_candidates(candidates)

    alternatives = " | ".join(f'"{dieu}"' for dieu in candidates)
    return (
        'root ::= "{\\"stop\\": true, \\"dieu\\": null}" '
        '| "{\\"stop\\": false, \\"dieu\\": " dieu "}"\n'
        f"dieu ::= {alternatives}"
    )
