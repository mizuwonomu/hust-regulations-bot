"""Offline loop suite cho agent_retrieve: S-series + L-series.

Inject seed/fetch/extract/decide, dùng nguyên văn Điều dựng inline. Không có
real retrieval, Chroma, parent store, LLM hay eval. Extract dùng helper thật
vì nó pure và contract của loop dựa trên hành vi parse thật của nó.
"""

import logging

import pytest

from src.rag.agent.loop import agent_retrieve
from src.rag.agent.schema import Decision
from src.rag.agent.tools import extract_citation_mentions


def article_text(dieu_num, title, *body_lines):
    """Dựng nguyên văn một Điều: dòng đầu là heading, theo sau là body"""
    return f"Điều {dieu_num}. {title}\n" + "\n".join(body_lines)


def follow(dieu_num):
    return Decision(stop=False, dieu=dieu_num)


def stop():
    return Decision(stop=True, dieu=None)


class SeedStub:
    """Seed callable giả: ghi lại question nhận được, trả fixture cố định"""

    def __init__(self, contexts, dieu_set=()):
        self.contexts = list(contexts)
        self.dieu = set(dieu_set)
        self.questions = []

    def __call__(self, question):
        self.questions.append(question)
        return list(self.contexts), set(self.dieu)


class GateStub:
    """Gate giả: ghi lại (question, observation, candidates) rồi trả decision theo kịch bản"""

    def __init__(self, decisions):
        self.calls = []
        self._decisions = list(decisions)

    def __call__(self, question, observation, candidates):
        self.calls.append(
            {
                "question": question,
                "observation": observation,
                "candidates": list(candidates),
            }
        )
        decision = self._decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision


class FetchStub:
    """get_article_fn giả: ghi lại thứ tự fetch, số Điều lạ trả None"""

    def __init__(self, articles):
        self.articles = dict(articles)
        self.calls = []

    def __call__(self, dieu_num):
        self.calls.append(dieu_num)
        return self.articles.get(dieu_num)


def run_agent(
    seed_contexts,
    seed_dieu=(),
    decisions=(),
    articles=None,
    internal_dieu=(),
    max_follow_articles=5,
    question="câu hỏi test?",
):
    """Chạy agent_retrieve với toàn bộ stub và trả về (kết quả, các stub)"""
    seed = SeedStub(seed_contexts, seed_dieu)
    gate = GateStub(decisions)
    fetch = FetchStub(articles or {})
    result = agent_retrieve(
        question,
        retrieve_seed_fn=seed,
        get_article_fn=fetch,
        extract_citations_fn=extract_citation_mentions,
        decide_fn=gate,
        internal_dieu=set(internal_dieu),
        max_follow_articles=max_follow_articles,
    )
    return {
        "result": result,
        "seed": seed,
        "gate": gate,
        "fetch": fetch,
        "question": question,
    }


def sections_of(observation):
    return observation.splitlines()[1:] # bỏ dòng "Các Điều đã thu thập:"


# S-series: merged state và compact observation


class TestS1SeedAndFollow:
    def test_seed_then_follow_once(self):
        seed10 = article_text(10, "Tổng quan", "quy định tại Điều 20")
        art20 = article_text(20, "Điều kiện", "nội dung riêng của điều 20")
        run = run_agent(
            [seed10],
            decisions=[follow(20)],
            articles={20: art20},
            internal_dieu={10, 20},
        )

        contexts, ids = run["result"]
        assert contexts == [seed10, art20]
        assert ids == {10, 20}
        assert run["fetch"].calls == [20]
        assert len(run["gate"].calls) == 1
        assert run["gate"].calls[0]["candidates"] == [20]


class TestS2FullTextPreserved:
    def test_full_text_exact_and_observation_compact(self):
        seed10 = (
            "Điều 10. Quy định chung\n"
            "Đây là đoạn văn không chứa dẫn chiếu.\n"
            "| Cột A | Cột B |\n"
            "| --- | --- |\n"
            "thực hiện theo Điều 20 Quy chế này\n"
            "HẾT"
        )
        run = run_agent([seed10], decisions=[stop()], internal_dieu={10, 20})

        contexts, ids = run["result"]
        assert contexts == [seed10] # so sánh nguyên chuỗi, không phải substring
        assert ids == {10}

        # Qua seam synthesis: consumer downstream nhận đúng nguyên bản context
        captured = []
        captured.extend(contexts)
        assert captured == [seed10]

        observation = run["gate"].calls[0]["observation"]
        assert "thực hiện theo Điều 20 Quy chế này" in observation
        assert "Đây là đoạn văn không chứa dẫn chiếu." not in observation
        assert "| Cột A | Cột B |" not in observation
        assert "HẾT" not in observation


class TestS3SiblingRemainsEligible:
    def test_selected_sibling_excluded_unselected_remains(self):
        seed10 = article_text(10, "A", "dẫn chiếu Điều 20 và Điều 30")
        art20 = article_text(20, "B", "không dẫn chiếu")
        art30 = article_text(30, "C", "không dẫn chiếu")
        run = run_agent(
            [seed10],
            decisions=[follow(20), follow(30)],
            articles={20: art20, 30: art30},
            internal_dieu={10, 20, 30},
        )

        contexts, ids = run["result"]
        assert contexts == [seed10, art20, art30]
        assert ids == {10, 20, 30}
        assert run["fetch"].calls == [20, 30]

        # Gate call 1 thấy cả hai; call 2 chỉ còn 30, 20 đã collected
        assert run["gate"].calls[0]["candidates"] == [20, 30]
        assert run["gate"].calls[1]["candidates"] == [30]


class TestS4CrossCitations:
    def test_seeds_cite_each_other_and_shared_target(self):
        seed10 = article_text(
            10, "A",
            "phối hợp Điều 20 để xét",
            "tham khảo Điều 30",
            "xem lại Điều 20 lần nữa",
        )
        seed20 = article_text(
            20, "B",
            "theo Điều 10 quy định",
            "đối chiếu Điều 30",
        )
        art30 = article_text(30, "C", "không dẫn chiếu")
        run = run_agent(
            [seed10, seed20],
            decisions=[follow(30)],
            articles={30: art30},
            internal_dieu={10, 20, 30},
        )

        contexts, ids = run["result"]
        assert contexts == [seed10, seed20, art30]
        assert ids == {10, 20, 30}
        assert run["fetch"].calls == [30]

        observation = run["gate"].calls[0]["observation"]
        assert run["gate"].calls[0]["candidates"] == [30]
        # Excerpt dẫn tới target đã collected vẫn còn trong observation
        assert "- phối hợp Điều 20 để xét" in observation
        assert "- xem lại Điều 20 lần nữa" in observation
        assert "- theo Điều 10 quy định" in observation
        assert "- tham khảo Điều 30" in observation
        assert "- đối chiếu Điều 30" in observation


class TestS5NoRefetchOfSeed:
    def test_chain_back_to_seed_stops(self):
        seed10 = article_text(10, "A", "theo Điều 20")
        art20 = article_text(20, "B", "quay lại Điều 10")
        run = run_agent(
            [seed10],
            decisions=[follow(20)],
            articles={20: art20},
            internal_dieu={10, 20},
        )

        contexts, ids = run["result"]
        assert contexts == [seed10, art20]
        assert ids == {10, 20}
        assert run["fetch"].calls == [20]
        # Frontier rỗng sau fetch -> dừng chắc chắn, không gate call thứ hai
        assert len(run["gate"].calls) == 1


class TestS6ExcerptDedup:
    def test_distinct_lines_survive_duplicate_renders_once(self):
        seed10 = article_text(
            10, "A",
            "khoản 1 Điều 20 quy định A",
            "khoản 1 Điều 20 quy định A",
            "khoản 2 Điều 20 quy định B",
            "đối chiếu Điều 10 với chính nó",
        )
        run = run_agent([seed10], decisions=[stop()], internal_dieu={10, 20})

        observation = run["gate"].calls[0]["observation"]
        assert run["gate"].calls[0]["candidates"] == [20]
        assert observation.count("khoản 1 Điều 20 quy định A") == 1
        assert "- khoản 2 Điều 20 quy định B" in observation
        # Self header và dòng self-reference không thành excerpt/action
        assert "- đối chiếu Điều 10 với chính nó" not in observation
        assert "- Điều 10. A" not in observation


class TestS7NoEligibleCitations:
    def test_empty_seed_input(self):
        run = run_agent([], decisions=[])
        contexts, ids = run["result"]
        assert contexts == []
        assert ids == set()
        assert run["gate"].calls == []
        assert run["fetch"].calls == []

    def test_seed_without_citations(self):
        seed10 = article_text(10, "A", "nội dung thuần, không dẫn chiếu")
        run = run_agent([seed10], decisions=[])
        contexts, ids = run["result"]
        assert contexts == [seed10]
        assert ids == {10}
        assert run["gate"].calls == []
        assert run["fetch"].calls == []


class TestS8UnknownTitle:
    def test_empty_first_line_body_heading_not_promoted(self):
        seed = "\nĐiều 5. Giả heading giữa body\nthực hiện theo Điều 20 Quy chế này"
        run = run_agent([seed], decisions=[stop()], internal_dieu={20})

        contexts, ids = run["result"]
        assert contexts == [seed] # nguyên văn giữ nguyên
        assert ids == set() # dòng body không được nâng thành seed identity

        header = sections_of(run["gate"].calls[0]["observation"])[0]
        assert header == "[Context ?]" # không title, không literal None, không body text
        assert "None" not in run["gate"].calls[0]["observation"]
        assert "- thực hiện theo Điều 20 Quy chế này" in run["gate"].calls[0]["observation"]


class TestS9HeadingForms:
    def test_plain_heading(self):
        seed = article_text(42, "Đánh giá luận án tiến sĩ", "thực hiện theo Điều 3 Quy chế này")
        run = run_agent([seed], decisions=[stop()], internal_dieu={42, 3})
        _, ids = run["result"]
        assert ids == {42}
        assert sections_of(run["gate"].calls[0]["observation"])[0] == "[Context 42] Điều 42. Đánh giá luận án tiến sĩ"

    def test_markdown_heading(self):
        seed = "### Điều 42. Đánh giá luận án tiến sĩ\nthực hiện theo Điều 3 Quy chế này"
        run = run_agent([seed], decisions=[stop()], internal_dieu={42, 3})
        _, ids = run["result"]
        assert ids == {42}
        assert sections_of(run["gate"].calls[0]["observation"])[0] == "[Context 42] Điều 42. Đánh giá luận án tiến sĩ"

    def test_serialized_corpus_heading(self):
        seed = "ĐÀO TẠO TIẾN SĨ - Điều 42. Đánh giá luận án tiến sĩ\nthực hiện theo Điều 3 Quy chế này"
        run = run_agent([seed], decisions=[stop()], internal_dieu={42, 3})
        _, ids = run["result"]
        assert ids == {42}
        assert sections_of(run["gate"].calls[0]["observation"])[0] == (
            "[Context 42] ĐÀO TẠO TIẾN SĨ - Điều 42. Đánh giá luận án tiến sĩ"
        )


class TestS10MentionOnlyHeading:
    # Contract: câu body chỉ NHẮC Điều không được nâng thành title/identity
    # Regex heading yêu cầu dấu chấm sau số Điều + prefix chương IN HOA nên cả hai bị từ chối
    def test_body_sentence_with_dieu_not_promoted(self):
        seed = "Điều 20 quy định cách xử lý vi phạm\nthực hiện theo Điều 3 Quy chế này"
        run = run_agent([seed], decisions=[stop()], internal_dieu={3, 20})

        _, ids = run["result"]
        assert ids == set()
        observation = run["gate"].calls[0]["observation"]
        header = sections_of(observation)[0]
        assert header == "[Context ?]"
        assert "quy định cách xử lý vi phạm" not in header

    def test_leaded_sentence_may_appear_only_as_excerpt(self):
        seed = "Theo quy định - Điều 20 được áp dụng"
        run = run_agent([seed], decisions=[stop()], internal_dieu={20})

        _, ids = run["result"]
        assert ids == set()
        observation = run["gate"].calls[0]["observation"]
        header = sections_of(observation)[0]
        assert header == "[Context ?]"
        assert "- Theo quy định - Điều 20 được áp dụng" in observation


class TestS11MultipleUnknownSeeds:
    def test_distinct_keys_order_and_no_leak(self):
        text_a = "Hướng dẫn tổng quan\ntheo Điều 20 được áp dụng"
        text_b = "Quy trình xét tốt nghiệp\nđối chiếu Điều 30 để xét"
        run = run_agent(
            [text_a, text_b],
            seed_dieu={7},
            decisions=[stop()],
            internal_dieu={7, 20, 30},
        )

        contexts, ids = run["result"]
        assert contexts == [text_a, text_b] # giữ nguyên thứ tự seed
        assert ids == {7} # key nội bộ -1/-2 không leak vào returned IDs

        observation = run["gate"].calls[0]["observation"]
        assert run["gate"].calls[0]["candidates"] == [20, 30]
        assert "- theo Điều 20 được áp dụng" in observation
        assert "- đối chiếu Điều 30 để xét" in observation
        # Sentinel membership-only (7) không render section
        assert "[Context 7]" not in observation
        # Key nội bộ âm không leak vào header
        assert "[Context -1]" not in observation
        assert "[Context -2]" not in observation


class TestS12FetchedMalformedTitle:
    def test_target_number_stays_authoritative(self):
        seed10 = article_text(10, "A", "theo Điều 20")
        malformed20 = "Bản tin Điều 20 không có heading\ntheo Điều 30 Quy chế này"
        run = run_agent(
            [seed10],
            decisions=[follow(20), stop()],
            articles={20: malformed20},
            internal_dieu={10, 20, 30},
        )

        contexts, ids = run["result"]
        assert contexts == [seed10, malformed20] # nguyên văn giữ nguyên
        assert ids == {10, 20} # số Điều fetch là identity chính thức
        assert run["fetch"].calls == [20]

        observation = run["gate"].calls[1]["observation"]
        assert "[Context 20]" in observation # fallback về target identity
        assert "None" not in observation
        assert "[Context 20] Bản tin Điều 20" not in observation # title hỏng không render


class TestS13FetchReturnsNone:
    def test_state_unchanged_no_more_gate(self):
        seed10 = article_text(10, "A", "theo Điều 20")
        run = run_agent([seed10], decisions=[follow(20)], internal_dieu={10, 20})

        contexts, ids = run["result"]
        assert contexts == [seed10]
        assert ids == {10}
        assert run["fetch"].calls == [20]
        assert len(run["gate"].calls) == 1


class TestS14FollowLimit:
    def test_zero_forbids_gate_and_fetch(self):
        seed10 = article_text(10, "A", "theo Điều 20")
        run = run_agent(
            [seed10],
            decisions=[follow(20)],
            articles={20: article_text(20, "B")},
            internal_dieu={10, 20},
            max_follow_articles=0,
        )

        contexts, ids = run["result"]
        assert contexts == [seed10]
        assert ids == {10}
        assert run["gate"].calls == []
        assert run["fetch"].calls == []

    def test_positive_limit_counts_successful_follows_only(self):
        chain = {
            10: article_text(10, "A", "theo Điều 20"),
            20: article_text(20, "B", "theo Điều 30"),
            30: article_text(30, "C", "theo Điều 40"),
            40: article_text(40, "D"),
        }
        run = run_agent(
            [chain[10]],
            decisions=[follow(20), follow(30)],
            articles=chain,
            internal_dieu={10, 20, 30, 40},
            max_follow_articles=2,
        )

        contexts, ids = run["result"]
        assert contexts == [chain[10], chain[20], chain[30]] # seed không bị cắt
        assert ids == {10, 20, 30}
        assert run["fetch"].calls == [20, 30]


class TestS15GateStopWithCandidates:
    def test_collected_texts_unchanged_no_fetch(self):
        seed10 = article_text(10, "A", "dẫn chiếu Điều 20 và Điều 30")
        run = run_agent([seed10], decisions=[stop()], internal_dieu={10, 20, 30})

        contexts, ids = run["result"]
        assert contexts == [seed10]
        assert ids == {10}
        assert run["fetch"].calls == []
        assert len(run["gate"].calls) == 1


class TestS16MembershipOnlySentinel:
    def test_unparseable_seed_with_metadata_id(self):
        seed = "Văn bản không có heading Điều nào\ntheo Điều 20 được áp dụng"
        run = run_agent([seed], seed_dieu={10}, decisions=[stop()], internal_dieu={10, 20})

        contexts, ids = run["result"]
        assert contexts == [seed] # full text của seed giữ nguyên
        assert ids == {10} # ID metadata-derived được trả về
        assert run["fetch"].calls == [] # không refetch 10

        observation = run["gate"].calls[0]["observation"]
        assert "- theo Điều 20 được áp dụng" in observation
        # Sentinel None không sinh text lẫn observation section
        assert "[Context 10]" not in observation


class TestS17UnorderedMetadataIds:
    def test_no_positional_pairing_no_none_leak(self):
        text_a = "Tổng quan chế độ\ntheo Điều 20 được áp dụng"
        text_b = "Quy trình xét\nđối chiếu Điều 25 để xét"
        run = run_agent(
            [text_a, text_b],
            seed_dieu={30, 20}, # thứ tự metadata không đảm bảo
            decisions=[stop()],
            internal_dieu={20, 25, 30},
        )

        contexts, ids = run["result"]
        assert contexts == [text_a, text_b] # giữ nguyên cả hai chuỗi
        assert ids == {20, 30} # mọi ID dương từ seed_dieu đều còn

        observation = run["gate"].calls[0]["observation"]
        # Candidates theo thứ tự xuất hiện trong TEXT: 20 đã là membership sentinel
        # (từ seed_dieu) nên bị lọc khỏi candidates, chỉ 25 còn
        assert run["gate"].calls[0]["candidates"] == [25]
        assert "- theo Điều 20 được áp dụng" in observation
        assert "- đối chiếu Điều 25 để xét" in observation
        assert "None" not in observation
        assert "[Context 20]" not in observation
        assert "[Context 30]" not in observation
        assert "[Context 25]" not in observation


class TestS18DuplicateSeedIdentity:
    def test_both_strings_kept_number_returned_once(self):
        text_a = article_text(10, "Chế độ học bổng", "theo Điều 15 Quy chế này")
        text_b = article_text(10, "Chế độ học bổng sửa đổi", "theo Điều 25 Quy chế mới")
        run = run_agent([text_a, text_b], decisions=[stop()], internal_dieu={10, 15, 25})

        contexts, ids = run["result"]
        assert contexts == [text_a, text_b] # cả hai chuỗi đầy đủ, không ghi đè
        assert ids == {10} # số Điều trùng trả về một lần

        observation = run["gate"].calls[0]["observation"]
        assert run["gate"].calls[0]["candidates"] == [15, 25]
        # Heading của chính hai text không được tái xuất hiện làm excerpt:
        # source_dieu của key âm phải suy đúng từ title (vẫn là 10)
        assert "- Điều 10. Chế độ học bổng" not in observation
        assert "- Điều 10. Chế độ học bổng sửa đổi" not in observation
        # Self reference bị loại khỏi observation, mention thật của text_b vẫn còn
        assert "- theo Điều 25 Quy chế mới" in observation


# L-series: loop mechanics


class TestL1RecoveryChain:
    def test_seed_lacks_target_cited_by_seed_article(self):
        seed10 = article_text(10, "A", "theo Điều 20")
        art20 = article_text(20, "B", "không dẫn chiếu gì thêm")
        run = run_agent(
            [seed10],
            decisions=[follow(20)],
            articles={20: art20},
            internal_dieu={10, 20},
        )

        contexts, ids = run["result"]
        assert contexts == [seed10, art20] # full text B được fetch và nối vào đúng một lần
        assert ids == {10, 20}
        assert run["fetch"].calls == [20]
        # Sau fetch không sinh candidate mới -> KHÔNG có gate call tiếp theo
        assert len(run["gate"].calls) == 1
        assert run["gate"].calls[0]["candidates"] == [20]


class TestL2GateShutSingleHop:
    def test_loop_obeys_immediate_stop(self):
        seed10 = article_text(10, "A", "theo Điều 20")
        run = run_agent([seed10], decisions=[stop()], internal_dieu={10, 20})

        contexts, ids = run["result"]
        assert contexts == [seed10]
        assert ids == {10}
        assert run["fetch"].calls == []
        assert len(run["gate"].calls) == 1
        # Loop vâng lệnh stop - test này chứng minh obedience của loop,
        # KHÔNG chứng minh LLM biết khi nào nên dừng


class TestL3FollowLimitSeatbelt:
    def test_chain_longer_than_cap_stops_at_cap(self):
        chain = {
            10: article_text(10, "A", "theo Điều 20"),
            20: article_text(20, "B", "theo Điều 30"),
            30: article_text(30, "C", "theo Điều 40"),
            40: article_text(40, "D"),
        }
        run = run_agent(
            [chain[10]],
            decisions=[follow(20), follow(30)],
            articles=chain,
            internal_dieu={10, 20, 30, 40},
            max_follow_articles=2,
        )

        contexts, ids = run["result"]
        assert run["fetch"].calls == [20, 30] # follow thành công đúng bằng cap
        assert ids == {10, 20, 30}
        assert contexts == [chain[10], chain[20], chain[30]]


class TestL4NoGateCallAfterBudgetExhausted:
    def test_no_extra_gate_call_when_cap_reached(self):
        chain = {
            10: article_text(10, "A", "theo Điều 20"),
            20: article_text(20, "B", "theo Điều 30"),
            30: article_text(30, "C", "theo Điều 40"),
            40: article_text(40, "D"),
        }
        run = run_agent(
            [chain[10]],
            decisions=[follow(20), follow(30)],
            articles=chain,
            internal_dieu={10, 20, 30, 40},
            max_follow_articles=2,
        )

        # Hai gate call cho hai follow; sau fetch cuối làm cạn budget
        # thì loop kết thúc mà KHÔNG hỏi gate thêm lần nữa
        assert len(run["gate"].calls) == 2
        assert run["fetch"].calls == [20, 30]


class TestL5EmptyFrontierNoGate:
    def test_no_candidates_no_gate_call(self):
        seed10 = article_text(10, "A", "nội dung không dẫn chiếu")
        run = run_agent([seed10], decisions=[])

        contexts, ids = run["result"]
        assert contexts == [seed10]
        assert ids == {10}
        assert run["gate"].calls == []
        assert run["fetch"].calls == []


class TestL6GateStopWhileCandidatesRemain:
    def test_no_fetch_state_returned(self):
        seed10 = article_text(10, "A", "dẫn chiếu Điều 20 và Điều 30")
        run = run_agent([seed10], decisions=[stop()], internal_dieu={10, 20, 30})

        contexts, ids = run["result"]
        assert contexts == [seed10]
        assert ids == {10}
        assert run["fetch"].calls == []


class TestL7OutOfSetTarget:
    def test_out_of_set_decision_no_fetch_no_retry(self):
        seed10 = article_text(10, "A", "theo Điều 20")
        run = run_agent(
            [seed10],
            decisions=[follow(99)], # stub không bị GBNF ràng buộc -> chủ động feed sai
            internal_dieu={10, 20},
        )

        contexts, ids = run["result"]
        assert contexts == [seed10]
        assert ids == {10}
        assert run["fetch"].calls == [] # guard whitelist + membership của orchestrator
        assert len(run["gate"].calls) == 1 # không retry cùng observation


class TestL8ValidTargetFetchNone:
    def test_exactly_one_fetch_then_stop(self):
        seed10 = article_text(10, "A", "theo Điều 20")
        run = run_agent([seed10], decisions=[follow(20)], internal_dieu={10, 20})

        contexts, ids = run["result"]
        assert run["fetch"].calls == [20] # đúng MỘT lần fetch
        assert contexts == [seed10]
        assert ids == {10} # target không vào state
        assert len(run["gate"].calls) == 1 # không gate call tiếp theo


class TestL9GateRaises:
    def test_exception_propagates_unchanged(self):
        # Khác L7: gate raise exception thay vì trả decision well-formed sai set
        with pytest.raises(RuntimeError, match="transport down"):
            run_agent(
                [article_text(10, "A", "theo Điều 20")],
                decisions=[RuntimeError("transport down")],
                internal_dieu={10, 20},
            )


class TestL10SeamShape:
    def test_return_is_tuple_list_str_set_int(self):
        seed10 = article_text(10, "A")
        result = run_agent([seed10])["result"]
        assert isinstance(result, tuple)
        assert len(result) == 2
        contexts, ids = result
        assert isinstance(contexts, list)
        assert all(isinstance(text, str) for text in contexts)
        assert isinstance(ids, set)
        assert all(isinstance(dieu_num, int) for dieu_num in ids)


class TestL11SeedCallableContract:
    def test_question_verbatim_called_once(self):
        run = run_agent([article_text(10, "A")], question="câu hỏi gốc verbatim?")
        assert run["seed"].questions == ["câu hỏi gốc verbatim?"]
        assert len(run["seed"].questions) == 1


class TestL12InvalidFollowLimit:
    @pytest.mark.parametrize("bad_limit", [-1, True, False, 1.5, "2", None])
    def test_invalid_values_raise_before_seed(self, bad_limit):
        seed = SeedStub([article_text(10, "A")])
        with pytest.raises(ValueError):
            agent_retrieve(
                "q",
                retrieve_seed_fn=seed,
                get_article_fn=FetchStub({}),
                extract_citations_fn=extract_citation_mentions,
                decide_fn=GateStub([]),
                internal_dieu={10},
                max_follow_articles=bad_limit,
            )
        assert seed.questions == [] # seed callable chưa từng chạy


class TestL13NoReverseTraversal:
    def test_unfetched_source_never_entered(self):
        # Fixture giả định edge A -> B trong text của A (Điều 10, không thuộc seed)
        seed_b = article_text(20, "B", "không dẫn chiếu")
        run = run_agent(
            [seed_b],
            decisions=[],
            internal_dieu={10, 20},
        )

        contexts, ids = run["result"]
        assert contexts == [seed_b]
        assert ids == {20}
        assert run["gate"].calls == []
        assert run["fetch"].calls == []


class TestL14FilteringVsObservation:
    def test_collected_filtered_from_candidates_kept_in_observation(self):
        seed10 = article_text(10, "A", "phối hợp Điều 20 đã có", "theo Điều 30 mới")
        seed20 = article_text(20, "B", "quay lại Điều 10")
        run = run_agent(
            [seed10, seed20],
            decisions=[stop()],
            internal_dieu={10, 20, 30},
        )

        assert run["gate"].calls[0]["candidates"] == [30]
        observation = run["gate"].calls[0]["observation"]
        # Excerpt trỏ tới target đã collected vẫn render (chỉ self-ref bị bỏ)
        assert "- phối hợp Điều 20 đã có" in observation
        assert "- quay lại Điều 10" in observation

    def test_external_target_filtered_from_candidates(self):
        # Whitelist internal_dieu: target ngoài whitelist bị loại khỏi candidates,
        # nhưng excerpt của nó vẫn render trong observation (chỉ lọc ở bước candidate)
        seed10 = article_text(
            10, "A",
            "theo Điều 20 nội bộ",
            "Áp dụng Điều 300 của văn bản ngoài",
        )
        run = run_agent([seed10], decisions=[stop()], internal_dieu={10, 20})

        assert run["gate"].calls[0]["candidates"] == [20]
        observation = run["gate"].calls[0]["observation"]
        assert "- theo Điều 20 nội bộ" in observation
        assert "- Áp dụng Điều 300 của văn bản ngoài" in observation


class TestFollowLimitSummary:
    # Log follow_limit_summary chỉ phát ở nhánh budget cạn, chứa đúng 5 field
    # reason/gate_calls/follow_count/remaining_candidates/remaining_candidate_count
    def _summary_messages(self, caplog):
        return [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("follow_limit_summary")
        ]

    def test_cap_zero_logs_remaining_without_gate_or_fetch(self, caplog):
        caplog.set_level(logging.INFO, logger="src.rag.agent.loop")
        seed10 = article_text(10, "A", "theo Điều 20")
        run = run_agent(
            [seed10],
            internal_dieu={10, 20},
            max_follow_articles=0,
        )

        summaries = self._summary_messages(caplog)
        assert len(summaries) == 1
        message = summaries[0]
        assert "reason=follow_limit_reached" in message
        assert "gate_calls=0" in message
        assert "follow_count=0" in message
        assert "remaining_candidates=[20]" in message
        assert "remaining_candidate_count=1" in message
        # Không được dùng tiền tố gate_call= khiến người đọc tưởng vừa gọi LLM
        assert "gate_call=" not in message
        # Log không được kéo theo thêm gate hay fetch nào
        assert run["gate"].calls == []
        assert run["fetch"].calls == []

    def test_cap_reached_frontier_empty_logs_zero_remaining(self, caplog):
        caplog.set_level(logging.INFO, logger="src.rag.agent.loop")
        chain = {
            10: article_text(10, "A", "theo Điều 20"),
            20: article_text(20, "B"),
        }
        run = run_agent(
            [chain[10]],
            decisions=[follow(20)],
            articles=chain,
            internal_dieu={10, 20},
            max_follow_articles=1,
        )

        summaries = self._summary_messages(caplog)
        assert len(summaries) == 1
        message = summaries[0]
        assert "gate_calls=1" in message
        assert "follow_count=1" in message
        assert "remaining_candidates=[]" in message
        assert "remaining_candidate_count=0" in message
        # Không tăng gate/fetch nào so với trước khi log
        assert len(run["gate"].calls) == 1
        assert run["fetch"].calls == [20]

    def test_cap_reached_with_candidates_logs_remaining(self, caplog):
        caplog.set_level(logging.INFO, logger="src.rag.agent.loop")
        chain = {
            10: article_text(10, "A", "theo Điều 20"),
            20: article_text(20, "B", "theo Điều 30"),
            30: article_text(30, "C"),
        }
        run = run_agent(
            [chain[10]],
            decisions=[follow(20)],
            articles=chain,
            internal_dieu={10, 20, 30},
            max_follow_articles=1,
        )

        summaries = self._summary_messages(caplog)
        assert len(summaries) == 1
        message = summaries[0]
        assert "gate_calls=1" in message
        assert "follow_count=1" in message
        assert "remaining_candidates=[30]" in message
        assert "remaining_candidate_count=1" in message
        assert len(run["gate"].calls) == 1
        assert run["fetch"].calls == [20]

    def test_no_summary_when_not_follow_limit(self, caplog):
        # Gate stop không phải nhánh budget cạn -> không phát follow_limit_summary
        caplog.set_level(logging.INFO, logger="src.rag.agent.loop")
        run = run_agent(
            [article_text(10, "A", "theo Điều 20")],
            decisions=[stop()],
            internal_dieu={10, 20},
        )
        assert self._summary_messages(caplog) == []
        assert len(run["gate"].calls) == 1
