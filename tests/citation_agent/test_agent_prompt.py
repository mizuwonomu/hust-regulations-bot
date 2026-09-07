"""Unit test prompt và GBNF động cho gate: pure, store-free.

Không snapshot nguyên chuỗi grammar: parse cấu trúc (rule, branch, literal,
thứ tự) với whitespace ngoài literal được chuẩn hoá, giữ nguyên khoảng trắng
bên trong JSON literal. Chỉ assert bằng seam nhỏ tự viết, không parser GBNF
tổng quát.
"""

import json

import pytest

from src.rag.agent.prompt import build_decision_grammar, render_gate_prompt


def messages_of(prompt_value):
    return prompt_value.to_messages()


def _rules_of(grammar):
    """Map rule name -> RHS; dòng không chứa ::= được coi là dòng continuation"""
    rules = {}
    current = None
    for line in grammar.splitlines():
        if "::=" in line:
            name, rhs = line.split("::=", 1)
            current = name.strip()
            rules[current] = rhs.strip()
        elif current is not None:
            rules[current] = (rules[current] + " " + line.strip()).strip()
    return rules


def _split_outside_literal(text, separator):
    """Tách theo separator nằm ngoài literal, giữ nguyên nội dung trong nháy kép"""
    parts, buf, in_literal = [], "", False
    index = 0
    while index < len(text):
        char = text[index]
        if in_literal and char == "\\":
            buf += text[index:index + 2] # escape trong literal, không kết thúc literal
            index += 2
            continue
        if char == '"':
            in_literal = not in_literal
        if char == separator and not in_literal:
            parts.append(buf.strip())
            buf = ""
        else:
            buf += char
        index += 1
    parts.append(buf.strip())
    return [part for part in parts if part]


def _tokens_outside_literal(text):
    """Tách token: literal (kể cả escape) là một token, whitespace ngoài literal là delimiter"""
    tokens, buf, in_literal = [], "", False
    index = 0
    while index < len(text):
        char = text[index]
        if in_literal and char == "\\":
            buf += text[index:index + 2]
            index += 2
            continue
        if char == '"':
            in_literal = not in_literal
            buf += char
        elif char.isspace() and not in_literal:
            if buf:
                tokens.append(buf)
                buf = ""
        else:
            buf += char
        index += 1
    if buf:
        tokens.append(buf)
    return tokens


def _literal_content(token):
    """Bỏ cặp nháy bao ngoài và unescape \\" thành " để parse JSON"""
    return token[1:-1].replace('\\"', '"')


class TestRenderGatePrompt:
    def test_roles_ordered_and_live_inputs_in_final_human(self):
        prompt_value = render_gate_prompt(
            "câu hỏi live về Điều 20?", "OBSERVATION_MARKER", [20, 26]
        )
        msgs = messages_of(prompt_value)
        roles = [m.type for m in msgs]

        assert roles[0] == "system" # policy nằm ở system message
        assert roles[-1] == "human" # live input là final human message
        assert all(role in {"human", "ai"} for role in roles[1:])

        final = msgs[-1].content
        assert "câu hỏi live về Điều 20?" in final
        assert "OBSERVATION_MARKER" in final
        assert "Điều 20, Điều 26" in final

    def test_render_does_not_validate_candidates(self):
        # Validation candidates thuộc về grammar builder, render_gate_prompt phải bỏ qua
        prompt_value = render_gate_prompt("q", "obs", [])
        assert "Unvisited candidates:" in messages_of(prompt_value)[-1].content


def _decision_of(ai_message):
    """Parse decision của few-shot bằng JSON để không phụ thuộc khoảng trắng"""
    return json.loads(ai_message.content.strip())


class TestFewShotPairing:
    def _few_shot_pairs(self):
        msgs = messages_of(render_gate_prompt("q", "obs", [3]))
        fewshots = msgs[1:-1] # bỏ system đầu và final human cuối
        roles = [m.type for m in fewshots]
        # Few-shot phải ghép cặp chặt human -> ai, không lệch nhịp
        assert roles[0] == "human"
        assert all(role == "human" if i % 2 == 0 else role == "ai" for i, role in enumerate(roles))
        return [(fewshots[i], fewshots[i + 1]) for i in range(0, len(fewshots), 2)]

    def _find_pair(self, pairs, context_marker, decision):
        """Tìm cặp human/ai thỏa CẢ context lẫn decision - không phụ thuộc thứ tự few-shot"""
        return next(
            pair
            for pair in pairs
            if context_marker in pair[0].content and _decision_of(pair[1]) == decision
        )

    def test_documented_example_edges_exist(self):
        pairs = self._few_shot_pairs()

        # Context và decision phải nằm trong CÙNG một ví dụ (human -> ai);
        # Điều 33 có cả ví dụ follow lẫn stop nên phải khớp theo decision, không lấy cặp đầu
        edge_33 = self._find_pair(pairs, "[Context 33]", {"stop": False, "dieu": 3})
        assert edge_33 is not None

        edge_26 = self._find_pair(pairs, "[Context 26]", {"stop": False, "dieu": 20})
        assert edge_26 is not None

        # Một ví dụ gate-shut (stop) tồn tại trong few-shots
        assert any(
            _decision_of(pair[1]) == {"stop": True, "dieu": None} for pair in pairs
        )


class TestDecisionGrammarStructure:
    def test_root_has_exactly_stop_and_follow_branches_referencing_dieu_rule(self):
        grammar = build_decision_grammar([20, 26])
        rules = _rules_of(grammar)
        assert "root" in rules
        assert "dieu" in rules

        branches = _split_outside_literal(rules["root"], "|")
        assert len(branches) == 2 # đúng hai nhánh: stop và follow

        # Nhánh stop: đúng một literal, parse ra JSON stop hợp lệ
        stop_tokens = _tokens_outside_literal(branches[0])
        assert len(stop_tokens) == 1
        assert json.loads(_literal_content(stop_tokens[0])) == {"stop": True, "dieu": None}

        # Nhánh follow: literal + tham chiếu rule dieu + literal; thay từng
        # candidate vào phải parse ra JSON follow hợp lệ
        follow_tokens = _tokens_outside_literal(branches[1])
        assert len(follow_tokens) == 3
        assert follow_tokens[1] == "dieu" # follow thực sự dùng rule dieu
        for candidate in [20, 26]:
            follow_json = (
                _literal_content(follow_tokens[0])
                + str(candidate)
                + _literal_content(follow_tokens[2])
            )
            assert json.loads(follow_json) == {"stop": False, "dieu": candidate}

    def test_dieu_rule_literals_exact_set_and_order(self):
        grammar = build_decision_grammar([20, 26])
        rules = _rules_of(grammar)
        literals = [
            json.loads(_literal_content(literal))
            for literal in _split_outside_literal(rules["dieu"], "|")
        ]
        assert literals == [20, 26]

    def test_unrelated_dieu_excluded(self):
        grammar = build_decision_grammar([20, 26])
        rules = _rules_of(grammar)
        literals = [
            json.loads(_literal_content(literal))
            for literal in _split_outside_literal(rules["dieu"], "|")
        ]
        assert 15 not in literals
        assert 9999 not in literals

    def test_input_order_preserved(self):
        grammar = build_decision_grammar([26, 20])
        rules = _rules_of(grammar)
        literals = [
            json.loads(_literal_content(literal))
            for literal in _split_outside_literal(rules["dieu"], "|")
        ]
        assert literals == [26, 20]

    def test_empty_candidates_rejected(self):
        with pytest.raises(ValueError):
            build_decision_grammar([])

    @pytest.mark.parametrize(
        "bad_candidates",
        [[20, 20], ["20"], [True], [0], [-5], [20, 0, 26]],
    )
    def test_invalid_candidates_rejected(self, bad_candidates):
        with pytest.raises(ValueError):
            build_decision_grammar(bad_candidates)
