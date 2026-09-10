"""Offline CLI checks for import, preparation, first-only execution, and summary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from artifacts import load_run, read_snapshot, sha256_file, write_snapshot
from run_experiments import main
from seed_cases import read_cases, write_cases


def _input_files(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps([{"id": 1, "user_input": "Câu hỏi"}], ensure_ascii=False),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "config": {"agent": False},
                "results": [
                    {
                        "id": 1,
                        "query": "Câu hỏi",
                        "retrieved_contexts": ["Điều 10. A\nTheo Điều 20"],
                        "hop_scores": {"retrieved_dieu": [10]},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    whitelist = tmp_path / "whitelist.json"
    whitelist.write_text("[10, 20]", encoding="utf-8")
    return baseline, dataset, whitelist


def test_cli_offline_first_workflow(tmp_path):
    baseline, dataset, whitelist = _input_files(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"
    cases_path = tmp_path / "cases.jsonl"

    assert main(
        [
            "import-seeds",
            "--baseline",
            str(baseline),
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(snapshot_path),
        ]
    ) == 0
    assert main(
        [
            "prepare-cases",
            "--seeds",
            str(snapshot_path),
            "--output",
            str(cases_path),
        ]
    ) == 0

    snapshot = read_snapshot(snapshot_path)
    cases = read_cases(cases_path)
    case = cases[0].model_dump(mode="python")
    case.update(
        {
            "expected_action": "follow",
            "acceptable_dieu": [20],
            "label_reason": "Điều 20 cần cho câu hỏi",
            "label_status": "approved",
        }
    )
    from contracts import GateCase

    write_cases([GateCase.model_validate(case)], cases_path)
    result_code = main(
        [
            "run",
            "--experiment",
            "initial-selection",
            "--seeds",
            str(snapshot_path),
            "--cases",
            str(cases_path),
            "--policies",
            "first",
            "--repeats",
            "1",
            "--output-root",
            str(tmp_path / "results"),
        ]
    )
    assert result_code == 0
    run_dirs = list((tmp_path / "results").iterdir())
    assert len(run_dirs) == 1
    manifest, loaded_cases, results = load_run(run_dirs[0])
    assert manifest.policies == ["first"]
    assert loaded_cases[0].case_id == cases[0].case_id
    assert len(results) == 1
    assert (run_dirs[0] / "results_first.jsonl").exists()
    assert not (run_dirs[0] / "results_llm.jsonl").exists()
    assert sha256_file(snapshot_path) == manifest.snapshot_source.sha256

    assert main(["summarize", "--run-dir", str(run_dirs[0])]) == 0


def _approved_inputs(tmp_path: Path):
    baseline, dataset, whitelist = _input_files(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"
    cases_path = tmp_path / "cases.jsonl"
    assert main(
        [
            "import-seeds",
            "--baseline",
            str(baseline),
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(snapshot_path),
        ]
    ) == 0
    assert main(
        ["prepare-cases", "--seeds", str(snapshot_path), "--output", str(cases_path)]
    ) == 0

    from contracts import GateCase

    draft = read_cases(cases_path)[0]
    approved = GateCase.model_validate(
        {
            **draft.model_dump(mode="python"),
            "expected_action": "follow",
            "acceptable_dieu": [20],
            "label_reason": "Điều 20 cần cho câu hỏi",
            "label_status": "approved",
        }
    )
    write_cases([approved], cases_path)
    return snapshot_path, cases_path


def test_partial_case_roster_exits_two(tmp_path):
    baseline, dataset, whitelist = _input_files(tmp_path)
    baseline_data = json.loads(baseline.read_text(encoding="utf-8"))
    baseline_data["results"].append(
        {
            "id": 2,
            "query": "Câu hỏi hai",
            "retrieved_contexts": ["Điều 10. B\nTheo Điều 20"],
            "hop_scores": {"retrieved_dieu": [10]},
        }
    )
    baseline.write_text(json.dumps(baseline_data, ensure_ascii=False), encoding="utf-8")
    dataset_data = json.loads(dataset.read_text(encoding="utf-8"))
    dataset_data.append({"id": 2, "user_input": "Câu hỏi hai"})
    dataset.write_text(json.dumps(dataset_data, ensure_ascii=False), encoding="utf-8")

    snapshot_path = tmp_path / "snapshot.json"
    cases_path = tmp_path / "cases.jsonl"
    assert main(
        [
            "import-seeds",
            "--baseline",
            str(baseline),
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(snapshot_path),
        ]
    ) == 0
    assert main(
        ["prepare-cases", "--seeds", str(snapshot_path), "--output", str(cases_path)]
    ) == 0
    assert len(read_cases(cases_path)) == 2

    from contracts import GateCase

    draft = read_cases(cases_path)[0]
    approved = GateCase.model_validate(
        {
            **draft.model_dump(mode="python"),
            "expected_action": "follow",
            "acceptable_dieu": [20],
            "label_reason": "Điều 20 cần cho câu hỏi",
            "label_status": "approved",
        }
    )
    write_cases([approved], cases_path)

    assert main(
        [
            "run",
            "--experiment",
            "initial-selection",
            "--seeds",
            str(snapshot_path),
            "--cases",
            str(cases_path),
            "--policies",
            "first",
            "--output-root",
            str(tmp_path / "results"),
        ]
    ) == 2
    assert not (tmp_path / "results").exists()


def test_runner_interrupt_persists_completed_trials(tmp_path, monkeypatch):
    import policies.first as first_policy

    real_decide = first_policy.decide
    calls = {"count": 0}

    def _interrupt_on_second(*arguments, **keywords):
        calls["count"] += 1
        if calls["count"] == 2:
            # Ngắt lần gọi thứ hai sau khi trial đầu đã được ghi xuống đĩa
            raise KeyboardInterrupt
        return real_decide(*arguments, **keywords)

    monkeypatch.setattr(first_policy, "decide", _interrupt_on_second)
    snapshot_path, cases_path = _approved_inputs(tmp_path)

    code = main(
        [
            "run",
            "--experiment",
            "initial-selection",
            "--seeds",
            str(snapshot_path),
            "--cases",
            str(cases_path),
            "--policies",
            "first",
            "--repeats",
            "2",
            "--output-root",
            str(tmp_path / "results"),
        ]
    )
    assert code == 130

    run_dir = next((tmp_path / "results").iterdir())
    manifest, _, results = load_run(run_dir)
    assert manifest.status == "incomplete"
    assert len(results) == 1
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    first_summary = summary_payload["by_policy"]["first"]
    assert first_summary["valid"] == 1
    assert first_summary["missing"] == 1


def test_runner_client_init_failure_exits_one(tmp_path, monkeypatch):
    import policies.llm as llm_policy

    def _broken_client():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm_policy, "create_client", _broken_client)
    snapshot_path, cases_path = _approved_inputs(tmp_path)
    # Client khởi tạo lỗi sau khi manifest tồn tại nên run phải giữ trạng thái incomplete

    code = main(
        [
            "run",
            "--experiment",
            "initial-selection",
            "--seeds",
            str(snapshot_path),
            "--cases",
            str(cases_path),
            "--policies",
            "llm",
            "--output-root",
            str(tmp_path / "results"),
        ]
    )
    assert code == 1

    run_dir = next((tmp_path / "results").iterdir())
    manifest, _, results = load_run(run_dir)
    assert manifest.status == "incomplete"
    assert results == []
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    first_summary = summary_payload["by_policy"]["llm"]
    assert first_summary["valid"] == 0
    assert first_summary["missing"] == 1


def test_cli_rejects_no_eligible_cases(tmp_path):
    baseline, dataset, whitelist = _input_files(tmp_path)
    snapshot = tmp_path / "snapshot.json"
    cases = tmp_path / "cases.jsonl"
    assert main(
        [
            "import-seeds",
            "--baseline",
            str(baseline),
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(snapshot),
        ]
    ) == 0
    assert main(
        [
            "prepare-cases",
            "--seeds",
            str(snapshot),
            "--output",
            str(cases),
        ]
    ) == 0
    assert main(
        [
            "run",
            "--experiment",
            "initial-selection",
            "--seeds",
            str(snapshot),
            "--cases",
            str(cases),
            "--policies",
            "first",
            "--output-root",
            str(tmp_path / "results"),
        ]
    ) == 2


def test_cli_help_and_invalid_arguments_exit_two_or_zero(tmp_path):
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0

    with pytest.raises(SystemExit) as command_exit:
        main(["invalid-command"])
    assert command_exit.value.code == 2

    with pytest.raises(SystemExit) as experiment_exit:
        main(
            [
                "run",
                "--experiment",
                "invalid",
                "--seeds",
                "snapshot.json",
                "--cases",
                "cases.jsonl",
                "--policies",
                "first",
                "--output-root",
                str(tmp_path),
            ]
        )
    assert experiment_exit.value.code == 2


def test_cli_missing_input_returns_data_error(tmp_path, capsys):
    baseline, dataset, _ = _input_files(tmp_path)
    code = main(
        [
            "import-seeds",
            "--baseline",
            str(baseline),
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "snapshot.json"),
        ]
    )
    assert code == 2
    assert "missing.json" in capsys.readouterr().err


def test_cli_help_lists_all_commands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    output = capsys.readouterr().out
    assert all(command in output for command in ("import-seeds", "prepare-cases", "run", "summarize"))


class _FakeGateClient:
    """Provide a complete invoke surface for the real gate parser."""

    def __init__(self, content: str):
        self.content = content
        self.calls = 0
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def invoke(self, prompt, extra_body=None):
        self.calls += 1
        return type("Message", (), {"content": self.content})()


def test_cli_full_run_uses_fake_client_and_shared_trial_ids(tmp_path, monkeypatch):
    baseline, dataset, whitelist = _input_files(tmp_path)
    baseline_payload = json.loads(baseline.read_text(encoding="utf-8"))
    baseline_payload["results"][0]["retrieved_contexts"] = ["Điều 10. A\nTheo Điều 20 và Điều 30"]
    baseline.write_text(json.dumps(baseline_payload, ensure_ascii=False), encoding="utf-8")
    whitelist.write_text("[10, 20, 30]", encoding="utf-8")
    snapshot_path = tmp_path / "snapshot.json"
    cases_path = tmp_path / "cases.jsonl"
    assert main(["import-seeds", "--baseline", str(baseline), "--dataset", str(dataset), "--internal-dieu", str(whitelist), "--output", str(snapshot_path)]) == 0
    assert main(["prepare-cases", "--seeds", str(snapshot_path), "--output", str(cases_path)]) == 0
    from contracts import GateCase

    draft = read_cases(cases_path)[0]
    approved = GateCase.model_validate({
        **draft.model_dump(mode="python"),
        "expected_action": "follow",
        "acceptable_dieu": [20],
        "label_reason": "Điều 20 cần cho câu hỏi",
        "label_status": "approved",
    })
    write_cases([approved], cases_path)
    import policies.llm as llm_policy

    fake = _FakeGateClient('{"stop": false, "dieu": 30}')
    monkeypatch.setattr(llm_policy, "create_client", lambda: fake)
    # Hai policy chạy cùng input và trial_id nhưng ghi vào hai file kết quả riêng
    code = main(
        [
            "run",
            "--experiment",
            "initial-selection",
            "--seeds",
            str(snapshot_path),
            "--cases",
            str(cases_path),
            "--policies",
            "first",
            "llm",
            "--repeats",
            "1",
            "--output-root",
            str(tmp_path / "results"),
        ]
    )
    assert code == 0
    run_dir = next((tmp_path / "results").iterdir())
    manifest, _, results = load_run(run_dir)
    first_ids = {record.trial.trial_id for record in results if record.policy == "first"}
    llm_ids = {record.trial.trial_id for record in results if record.policy == "llm"}
    assert first_ids == llm_ids
    assert len(first_ids) == 1
    assert fake.calls == 1
    assert (run_dir / "results_first.jsonl").exists()
    assert (run_dir / "results_llm.jsonl").exists()
    assert manifest.policy_config["llm"]["thinking"] is False
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["paired"]["first_wins"] == 1
    assert summary["paired"]["different_trial_ids"] == sorted(first_ids)


def test_cli_policy_error_records_error_and_returns_one(tmp_path, monkeypatch):
    import policies.first as first_policy

    def fail(*args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(first_policy, "decide", fail)
    snapshot_path, cases_path = _approved_inputs(tmp_path)
    # Policy lỗi vẫn lưu error record và CLI phải trả mã khác 0
    code = main(
        [
            "run",
            "--experiment",
            "initial-selection",
            "--seeds",
            str(snapshot_path),
            "--cases",
            str(cases_path),
            "--policies",
            "first",
            "--output-root",
            str(tmp_path / "results"),
        ]
    )
    assert code == 1
    run_dir = next((tmp_path / "results").iterdir())
    manifest, _, results = load_run(run_dir)
    assert manifest.status == "completed_with_errors"
    assert len(results) == 1
    assert results[0].outcome.error is not None
    assert results[0].outcome.error.category == "transport"
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["by_policy"]["first"]["errors"] == 1
    assert summary_payload["by_policy"]["first"]["valid"] == 0


def test_cli_fresh_process_preparation_does_not_initialize_llm(tmp_path):
    baseline, dataset, whitelist = _input_files(tmp_path)
    output = tmp_path / "snapshot.json"
    code = (
        "import json, sys;"
        "sys.path.insert(0, 'evals/agent-exp/scripts'); sys.path.insert(0, '.');"
        "import run_experiments as cli;"
        "rc=cli.main(['import-seeds','--baseline',sys.argv[1],'--dataset',sys.argv[2],"
        "'--internal-dieu',sys.argv[3],'--output',sys.argv[4]]);"
        "print(json.dumps({'rc':rc,'llm':'policies.llm' in sys.modules,'langchain_openai':'langchain_openai' in sys.modules}))"
    )
    # Chặn model và tracing trong process mới để kiểm tra đường CLI offline độc lập
    completed = subprocess.run(
        [str(Path(sys.executable)), "-c", code, str(baseline), str(dataset), str(whitelist), str(output)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report == {"rc": 0, "llm": False, "langchain_openai": False}
