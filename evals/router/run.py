"""CLI cho experiment router: dry-run/ live, schedule xen kẽ arm, manifest bất biến và trials.jsonl ghi dần.

- Dry-run (mặc định): validate corpus, kiểm tra prompt parity với production, kiểm tra type contract
  của TypeSafe, in schedule và tổng số call. Không tạo run, không gọi network
- Live: preflight API key, tạo run độc quyền (không bao giờ ghi đè), ghi manifest.json bất biến, rồi
  append + flush từng trial vào trials.jsonl ngay sau mỗi lần thử
- Resume chỉ khi corpus hash, request fingerprint và schedule khớp manifest cũ; trial trùng key bị chặn
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

from evals.router.adapters import (
    ARMS,
    GROQ_MAX_RETRIES,
    GROQ_REASONING_EFFORT,
    GROQ_TEMPERATURE,
    JEV_MODEL,
    REPO_ROOT,
    ROUTE_CRITERIA,
    ROUTE_QUESTION_ID,
    TIMEOUT_SECONDS,
    ArmName,
    GroqRouterAdapter,
    JevRouterAdapter,
    PromptParityError,
    RouteContractError,
    RouteTrial,
    build_route_questions,
    check_production_prompt_parity,
    groq_prompt_sha256,
    jev_criteria_sha256,
    validate_route_questions,
)
from evals.router.corpus import (
    DEFAULT_CORPUS_PATH,
    ClassificationCase,
    CorpusError,
    corpus_sha256,
    gold_counts,
    load_corpus,
)
from src.rag.config import ROUTER_MODEL

MANIFEST_NAME = "manifest.json"
TRIALS_NAME = "trials.jsonl"
DEFAULT_REPEATS = 3

# Giá niêm yết tại ngày quan sát, đơn vị USD trên một triệu token, key theo đúng model ID
# Owner đã xác nhận giá đúng và baseline Groq không bị deprecate (2026-09-25)
PRICING_OBSERVED_AT = "2026-09-25"
PRICING_BY_MODEL: dict[str, dict[str, Any]] = {
    ROUTER_MODEL: {
        "usd_per_million_input": 0.60,
        "usd_per_million_output": 3.00,
        "source_url": "https://console.groq.com/docs/models",
        "observed_at": PRICING_OBSERVED_AT,
    },
    JEV_MODEL: {
        "usd_per_million_input": 0.042,
        "usd_per_million_output": 0.0,
        "source_url": "https://www.layer3labs.io/guides/jev-pricing",
        "observed_at": PRICING_OBSERVED_AT,
    },
}

REQUIRED_KEYS = {"GROQ_API_KEY": "Groq", "TYPESAFE_API_KEY": "TypeSafe"}


class RunError(RuntimeError):
    """Run không hợp lệ: thiếu key, run đã tồn tại, hoặc resume không khớp"""


class RouteAdapter(Protocol):
    """Tối thiểu mà runner cần ở một arm"""

    arm: ArmName

    def route(self, query: str) -> RouteTrial: ...


@dataclass(frozen=True)
class ScheduleSlot:
    """Một call trong schedule: arm nào, case nào, repeat thứ mấy"""

    case_index: int
    arm: ArmName
    repeat_index: int

    @property
    def key(self) -> tuple[int, ArmName, int]:
        """Trial identity: case index, arm, repeat index - không dùng id/group debug"""
        return (self.case_index, self.arm, self.repeat_index)


def build_schedule(case_count: int, repeats: int) -> list[ScheduleSlot]:
    """Sinh schedule tất định, xen kẽ thứ tự arm theo (case index + repeat index)

    case_count: số case trong corpus
    repeats: số lần lặp mỗi case mỗi arm
    """
    slots: list[ScheduleSlot] = []
    for case_index in range(case_count):
        for repeat_index in range(repeats):
            arms = ARMS if (case_index + repeat_index) % 2 == 0 else tuple(reversed(ARMS))
            for arm in arms:
                slots.append(
                    ScheduleSlot(
                        case_index=case_index,
                        arm=arm,
                        repeat_index=repeat_index,
                    )
                )
    return slots


def schedule_sha256(slots: Sequence[ScheduleSlot]) -> str:
    """SHA256 của thứ tự call, để resume không thể trộn hai schedule khác nhau

    slots: schedule cần hash
    """
    payload = [[slot.case_index, slot.arm, slot.repeat_index] for slot in slots]
    return _sha256_json(payload)


def request_fingerprint(
    *,
    corpus_path: Path | str,
    slots: Sequence[ScheduleSlot],
    repeats: int,
) -> str:
    """Vân tay của toàn bộ cấu hình ảnh hưởng tới request gửi model

    corpus_path: file corpus đã dùng
    slots: schedule đã sinh
    repeats: số repeat mỗi case mỗi arm
    """
    payload = {
        "corpus_sha256": corpus_sha256(corpus_path),
        "schedule_sha256": schedule_sha256(slots),
        "repeats": repeats,
        "timeout_seconds": TIMEOUT_SECONDS,
        "groq": {
            "model": ROUTER_MODEL,
            "temperature": GROQ_TEMPERATURE,
            "reasoning_effort": GROQ_REASONING_EFFORT,
            "max_retries": GROQ_MAX_RETRIES,
            "prompt_sha256": groq_prompt_sha256(),
        },
        "jev": {
            "model": JEV_MODEL,
            "question_id": ROUTE_QUESTION_ID,
            "criteria_sha256": jev_criteria_sha256(),
        },
    }
    return _sha256_json(payload)


def pricing_snapshot(
    *,
    groq_model: str = ROUTER_MODEL,
    jev_model: str = JEV_MODEL,
) -> dict[str, dict[str, Any]]:
    """Lấy snapshot giá theo đúng model ID đang chạy; fail nếu model chưa có giá

    groq_model: model ID của arm Groq
    jev_model: model ID của arm Jev
    """
    snapshot: dict[str, dict[str, Any]] = {}
    for arm, model in (("groq", groq_model), ("jev", jev_model)):
        entry = PRICING_BY_MODEL.get(model)
        if entry is None:
            raise RunError(
                f"chưa có giá cho model {model!r} của arm {arm}, phải bổ sung trước khi chạy"
            )
        if not isinstance(entry["usd_per_million_input"], (int, float)) or entry[
            "usd_per_million_input"
        ] <= 0:
            raise RunError(f"giá input của {model!r} phải là số dương")
        if not isinstance(entry["usd_per_million_output"], (int, float)) or entry[
            "usd_per_million_output"
        ] < 0:
            raise RunError(f"giá output của {model!r} phải là số không âm")
        snapshot[arm] = {"model_id": model, **entry}
    return snapshot


def build_manifest(
    *,
    corpus_path: Path | str,
    cases: Sequence[ClassificationCase],
    repeats: int,
    slots: Sequence[ScheduleSlot],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Manifest bất biến: bind corpus, request, model, prompt, schedule, giá và môi trường

    corpus_path: file corpus đã validate
    cases: case đã load
    repeats: số repeat mỗi case mỗi arm
    slots: schedule đã sinh
    created_at: thời điểm tạo, None thì lấy UTC hiện tại
    """
    return {
        "created_at": created_at or _utc_now(),
        "corpus": {
            "path": str(Path(corpus_path).resolve()),
            "sha256": corpus_sha256(corpus_path),
            "rows": len(cases),
            "gold_counts": gold_counts(list(cases)),
        },
        "git": _git_state(),
        "schedule": {
            "repeats": repeats,
            "arms": list(ARMS),
            "calls": len(slots),
            "sha256": schedule_sha256(slots),
            "rule": "arm order alternates by (case_index + repeat_index) % 2",
        },
        "request_fingerprint": request_fingerprint(
            corpus_path=corpus_path,
            slots=slots,
            repeats=repeats,
        ),
        "timeout_seconds": TIMEOUT_SECONDS,
        "groq": {
            "model": ROUTER_MODEL,
            "temperature": GROQ_TEMPERATURE,
            "reasoning_effort": GROQ_REASONING_EFFORT,
            "max_retries": GROQ_MAX_RETRIES,
            "prompt_sha256": groq_prompt_sha256(),
        },
        "jev": {
            "model": JEV_MODEL,
            "question_id": ROUTE_QUESTION_ID,
            "criteria": ROUTE_CRITERIA,
            "criteria_sha256": jev_criteria_sha256(),
        },
        "packages": _package_versions(),
        "pricing": pricing_snapshot(),
    }


def validate_offline_inputs(corpus_path: Path | str) -> list[ClassificationCase]:
    """Bước offline bắt buộc: validate corpus, prompt parity và type contract của TypeSafe

    corpus_path: file corpus cần load
    """
    cases = load_corpus(corpus_path)
    check_production_prompt_parity()
    validate_route_questions(build_route_questions())
    return cases


def trial_record(
    *,
    case: ClassificationCase,
    case_index: int,
    repeat_index: int,
    trial: RouteTrial,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Ghép kết quả một lần thử với identity và gold label thành một dòng JSONL

    case: case nguồn, chỉ dùng id/group làm chú thích debug
    case_index: vị trí dòng trong corpus, là một phần của trial identity
    repeat_index: lần lặp thứ mấy
    trial: kết quả adapter trả về
    captured_at: thời điểm ghi, None thì lấy UTC hiện tại
    """
    return {
        "case_index": case_index,
        "id": case.id,
        "group": case.group,
        "repeat": repeat_index,
        "gold": case.type,
        "captured_at": captured_at or _utc_now(),
        **trial.as_dict(),
    }


def build_adapters() -> dict[ArmName, RouteAdapter]:
    """Dựng hai adapter thật; phải gọi check_api_keys trước đó để fail sớm khi thiếu key"""
    return {
        "groq": GroqRouterAdapter(model=ROUTER_MODEL, timeout=TIMEOUT_SECONDS),
        "jev": JevRouterAdapter(model=JEV_MODEL, timeout=TIMEOUT_SECONDS),
    }


def check_api_keys(environment: Mapping[str, str] | None = None) -> None:
    """Fail sớm nếu thiếu API key, để không tạo run rỗng - không đọc file .env

    environment: mapping thay thế cho os.environ, dùng cho test
    """
    source = os.environ if environment is None else environment
    missing = [
        name for name, provider in REQUIRED_KEYS.items() if not (source.get(name) or "").strip()
    ]
    if missing:
        providers = ", ".join(REQUIRED_KEYS[name] for name in missing)
        raise RunError(f"thiếu API key trong environment cho {providers}: {missing}")


def run_dry(corpus_path: Path | str = DEFAULT_CORPUS_PATH, repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    """In schedule và tổng số call cho một run dự kiến, không ghi artifact và không gọi model

    corpus_path: file corpus dùng để lên schedule
    repeats: số repeat mỗi case mỗi arm
    """
    cases = validate_offline_inputs(corpus_path)
    slots = build_schedule(len(cases), repeats)
    counts = gold_counts(cases)
    manifest = build_manifest(
        corpus_path=corpus_path,
        cases=cases,
        repeats=repeats,
        slots=slots,
    )
    lines = [
        f"Corpus: {corpus_path} ({len(cases)} case, RAG={counts['RAG']}, chat={counts['chat']})",
        f"Corpus sha256: {manifest['corpus']['sha256']}",
        "Prompt parity với production: OK",
        "TypeSafe type contract: OK",
        f"Schedule: {len(ARMS)} arm x {repeats} repeat x {len(cases)} case = {len(slots)} call",
        f"Schedule sha256: {manifest['schedule']['sha256']}",
        f"Request fingerprint: {manifest['request_fingerprint']}",
    ]
    for case_index, case in enumerate(cases):
        per_repeat = [
            f"r{repeat_index} " + " -> ".join(slot.arm for slot in _case_repeat_slots(slots, case_index, repeat_index))
            for repeat_index in range(repeats)
        ]
        lines.append(
            f"case {case_index:02d} (id {case.id}, {case.group}, gold {case.type}): "
            + " | ".join(per_repeat)
        )
    print("\n".join(lines))
    return {"cases": len(cases), "slots": slots, "manifest": manifest}


def _case_repeat_slots(
    slots: Sequence[ScheduleSlot],
    case_index: int,
    repeat_index: int,
) -> list[ScheduleSlot]:
    return [
        slot
        for slot in slots
        if slot.case_index == case_index and slot.repeat_index == repeat_index
    ]


def run_live(
    *,
    corpus_path: Path | str = DEFAULT_CORPUS_PATH,
    output_dir: Path | str,
    repeats: int = DEFAULT_REPEATS,
    resume: bool = False,
    adapters: Mapping[ArmName, RouteAdapter] | None = None,
) -> dict[str, Any]:
    """Ghi run thật: tạo hoặc resume, append từng trial, trả tóm tắt cuối run

    corpus_path: file corpus đã freeze
    output_dir: thư mục run, chứa manifest.json và trials.jsonl
    repeats: số repeat mỗi case mỗi arm
    resume: True thì nối vào run đã có thay vì từ chối ghi đè
    adapters: hai adapter tiêm vào cho test; None thì dựng adapter thật
    """
    cases = validate_offline_inputs(corpus_path)
    slots = build_schedule(len(cases), repeats)
    manifest = build_manifest(
        corpus_path=corpus_path,
        cases=cases,
        repeats=repeats,
        slots=slots,
    )
    # Preflight trước khi tạo bất cứ artifact nào
    if adapters is None:
        check_api_keys()
    run_dir = Path(output_dir)
    trials_path = run_dir / TRIALS_NAME
    # Lock ngoài run dir để hai process không bao giờ cùng ghi một run
    with _run_lock(run_dir.parent / f".{run_dir.name}.lock"):
        rows = _open_or_resume_run(run_dir, manifest, cases, slots, resume=resume)
        active = _active_adapters(adapters)
        done = {(row["case_index"], row["arm"], row["repeat"]) for row in rows}
        pending = [slot for slot in slots if slot.key not in done]
        print(f"Run: {run_dir} | đã có {len(rows)} trial, còn {len(pending)} call")
        with trials_path.open("a", encoding="utf-8") as handle:
            for slot in pending:
                case = cases[slot.case_index]
                trial = active[slot.arm].route(case.query)
                row = trial_record(
                    case=case,
                    case_index=slot.case_index,
                    repeat_index=slot.repeat_index,
                    trial=trial,
                )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                # Flush từng dòng để run chết giữa đường vẫn còn artifact dùng được
                handle.flush()
                os.fsync(handle.fileno())
                rows.append(row)
                print(
                    f"[{len(rows)}/{len(slots)}] case {slot.case_index} {slot.arm} "
                    f"r{slot.repeat_index} -> {row['executed_branch'] or row['error_type']}"
                )
    models = summarize_models(rows)
    print(f"Model thực tế theo arm: {json.dumps(models, ensure_ascii=False)}")
    return {"run_dir": run_dir, "rows": rows, "manifest": manifest, "models": models}


def _open_or_resume_run(
    run_dir: Path,
    manifest: Mapping[str, Any],
    cases: Sequence[ClassificationCase],
    slots: Sequence[ScheduleSlot],
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    """Mở run mới hoặc đọc lại trial của run cũ sau khi kiểm tra toàn vẹn"""
    if not resume:
        _publish_run(run_dir, manifest)
        return []
    _check_resume_integrity(run_dir / MANIFEST_NAME, manifest)
    rows = _read_trials(run_dir / TRIALS_NAME)
    _check_trial_rows(rows, cases, slots)
    return rows


def _active_adapters(
    adapters: Mapping[ArmName, RouteAdapter] | None,
) -> dict[ArmName, RouteAdapter]:
    """Dùng adapter tiêm vào, còn không thì dựng adapter thật và báo lỗi cấu hình cho gọn"""
    if adapters is not None:
        return dict(adapters)
    try:
        return build_adapters()
    except Exception as error:  # noqa: BLE001 - lỗi SDK/khởi tạo client phải thành RunError
        raise RunError(
            f"không dựng được adapter ({type(error).__name__}): {error}"
        ) from error


def summarize_models(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Liệt kê model ID thực tế đã trả về theo từng arm

    rows: các dòng trial đã ghi
    """
    models: dict[str, list[str]] = {arm: [] for arm in ARMS}
    for row in rows:
        model = row.get("model")
        if model and model not in models[row["arm"]]:
            models[row["arm"]].append(model)
    return models


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point CLI; trả exit code 0 khi thành công, 1 khi input/harness sai

    argv: tham số dòng lệnh, None thì lấy từ sys.argv
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.resume and not args.live:
        parser.error("--resume chỉ dùng được với --live")
    if args.live and not args.output_dir:
        parser.error("--live cần --output-dir")
    try:
        if args.live:
            run_live(
                corpus_path=args.corpus,
                output_dir=args.output_dir,
                repeats=args.repeats,
                resume=args.resume,
            )
        else:
            run_dry(corpus_path=args.corpus, repeats=args.repeats)
    except (CorpusError, PromptParityError, RouteContractError, RunError) as error:
        print(f"lỗi: {error}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.router.run",
        description="Harness so sánh router Groq với Jev trên corpus định tuyến đã freeze",
    )
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_PATH), help="file corpus JSON")
    parser.add_argument("--output-dir", default=None, help="thư mục run cho --live")
    parser.add_argument("--repeats", type=_positive_int, default=DEFAULT_REPEATS, help="số repeat mỗi case mỗi arm")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="chỉ in schedule, không gọi model (mặc định)")
    mode.add_argument("--live", action="store_true", help="gọi model thật và ghi artifact")
    parser.add_argument("--resume", action="store_true", help="nối tiếp run đã có trong --output-dir")
    return parser


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("--repeats phải >= 1")
    return number


def _publish_run(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Publish run nguyên tử: dựng đủ artifact trong thư mục staging rồi rename vào chỗ thật

    Nhờ vậy không bao giờ quan sát được run dir nửa vời (có manifest mà thiếu trials)
    """
    if run_dir.exists():
        raise RunError(f"{run_dir} đã tồn tại, không ghi đè run cũ")
    if run_dir.parent.exists() and not run_dir.parent.is_dir():
        raise RunError(f"{run_dir.parent} không phải thư mục")
    staging = run_dir.parent / f".{run_dir.name}.staging"
    if staging.exists():
        raise RunError(f"{staging} còn sót từ lần chạy trước, dọn tay rồi chạy lại")
    staging.mkdir(parents=True)
    _write_json_exclusive(staging / MANIFEST_NAME, manifest)
    (staging / TRIALS_NAME).touch(exist_ok=False)
    try:
        os.rename(staging, run_dir)
    except OSError as error:
        raise RunError(f"không publish được run {run_dir}: {error}") from error


@contextmanager
def _run_lock(lock_path: Path) -> Iterator[None]:
    """Giữ lock độc quyền cho một run dir để hai process không cùng ghi một run

    lock_path: file lock nằm cạnh run dir, giữ nguyên sau khi chạy xong
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RunError(
                f"run ứng với {lock_path.name} đang được process khác ghi"
            ) from error
        yield


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RunError(f"{path} đã tồn tại, manifest là bất biến")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _check_resume_integrity(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    if not manifest_path.exists():
        raise RunError(f"không resume được vì thiếu {manifest_path}")
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(stored, dict) or not {"corpus", "schedule", "request_fingerprint"} <= set(stored):
        raise RunError(f"{manifest_path} không phải manifest của harness")
    mismatches = [
        f"{field}: manifest {expected!r} vs hiện tại {actual!r}"
        for field, expected, actual in (
            ("schedule.repeats", stored["schedule"]["repeats"], manifest["schedule"]["repeats"]),
            ("schedule.sha256", stored["schedule"]["sha256"], manifest["schedule"]["sha256"]),
            ("corpus.sha256", stored["corpus"]["sha256"], manifest["corpus"]["sha256"]),
            (
                "request_fingerprint",
                stored["request_fingerprint"],
                manifest["request_fingerprint"],
            ),
        )
        if expected != actual
    ]
    if mismatches:
        raise RunError("resume không khớp " + " | ".join(mismatches))


def _read_trials(trials_path: Path) -> list[dict[str, Any]]:
    """Đọc trial đã ghi; thiếu file là artifact hỏng, dòng cuối đứt thì bị cắt bỏ

    trials_path: file JSONL của run, luôn tồn tại từ lúc publish run
    Chỉ dòng cuối KHÔNG kết thúc bằng newline mới được coi là torn write; một dòng hoàn chỉnh
    nhưng JSON hỏng là artifact hỏng và phải fail, nếu không attempt đó sẽ bị gọi lại
    """
    if not trials_path.exists():
        # Publish luôn tạo file này, nên mất file nghĩa là artifact đã bị hỏng hoặc bị xoá
        raise RunError(
            f"thiếu {trials_path}: run artifact hỏng, không resume để tránh gọi lại toàn bộ schedule"
        )
    text = trials_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    complete_length = 0
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            complete_length += len(raw_line)
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            if line_number == len(lines) and not raw_line.endswith("\n"):
                print(f"cảnh báo: bỏ dòng cuối bị đứt trong {trials_path}:{line_number} ({error})")
                break
            raise RunError(f"{trials_path}:{line_number} không parse được ({error})") from error
        if not isinstance(row, dict):
            raise RunError(f"{trials_path}:{line_number} không phải JSON object")
        rows.append(row)
        complete_length += len(raw_line)
    if complete_length != len(text):
        # Cắt phần đứt để lần append sau không nối vào dòng hỏng
        trials_path.write_text(text[:complete_length], encoding="utf-8")
    return rows


def _check_trial_rows(
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[ClassificationCase],
    slots: Sequence[ScheduleSlot],
) -> None:
    allowed = {slot.key for slot in slots}
    seen: set[tuple[int, ArmName, int]] = set()
    for row in rows:
        case_index = row.get("case_index")
        arm = row.get("arm")
        repeat = row.get("repeat")
        if isinstance(case_index, bool) or not isinstance(case_index, int):
            raise RunError(f"trial có case_index không phải số nguyên: {case_index!r}")
        if not isinstance(arm, str):
            raise RunError(f"trial có arm không phải string: {arm!r}")
        if isinstance(repeat, bool) or not isinstance(repeat, int):
            raise RunError(f"trial có repeat không phải số nguyên: {repeat!r}")
        if not 0 <= case_index < len(cases):
            raise RunError(f"trial có case index ngoài phạm vi corpus: {case_index}")
        key = (case_index, arm, repeat)
        if key not in allowed:
            raise RunError(f"trial {key} không thuộc schedule hiện tại")
        if key in seen:
            raise RunError(f"trial trùng key {key}")
        seen.add(key)
        expected_gold = cases[case_index].type
        if row.get("gold") != expected_gold:
            raise RunError(
                f"trial {key} có gold {row.get('gold')!r} khác corpus {expected_gold!r}"
            )


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "revision": revision,
        "dirty": None if status is None else bool(status),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("langchain-core", "langchain-groq", "langchain-typesafe", "httpx2", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
