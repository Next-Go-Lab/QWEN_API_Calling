#!/usr/bin/env python3
"""Unified tool: generate prompt, fill one, fill batch."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


API_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
USER_API_KEY = "sk-cd586792f98c4f5b826164a721c0a479"
REGION_IDS = [
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_key(cli_key: str | None) -> str:
    key = cli_key or USER_API_KEY or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("Missing API key. Use --api-key or set USER_API_KEY / DASHSCOPE_API_KEY.")
    return key


def build_system_prompt() -> str:
    return (
        "你是职业围棋内容标注助手。"
        "你会基于给定棋盘信息产出结构化中文描述字段。"
        "输出严格 JSON，禁止输出 markdown。"
    )


def build_user_prompt(sample: Dict[str, Any]) -> str:
    ctx = {
        "meta": sample.get("meta", {}),
        "rules": sample.get("rules", {}),
        "state": sample.get("state", {}),
        "targets": sample.get("targets", {}),
    }
    schema = {
        "semantic_description": {
            "global_tags": [],
            "strategic_focus": [],
            "global_summary_cn": "",
            "phase_explanation": "",
        },
        "regions": [
            {
                "region_id": rid,
                "summary": "",
                "shapes": [],
                "local_tags": [],
                "key_points": [],
                "group_status": [],
            }
            for rid in REGION_IDS
        ],
    }
    return f"""
请基于输入棋盘信息，生成“语义补全字段”。

【核心要求】
1) 仅输出这两个顶层字段：`semantic_description`、`regions`。
2) 不要输出 `meta/rules/state/targets`，也不要输出任何解释文本。
3) 输出必须是合法 JSON。
4) 坐标统一使用 GTP（如 D4、Q16、PASS）。
5) 标签/术语不做限制，按局面自由生成即可；没有合适词汇可留空。

【结构约束】
- `regions` 必须包含九宫格 9 个分区，且每个 `region_id` 恰好出现一次。
- `regions[].region_id` 只能使用：{", ".join(REGION_IDS)}

【输入：棋盘上下文】
{json.dumps(ctx, ensure_ascii=False, indent=2)}

【输出结构示例（仅结构示意）】
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


def build_payload(model: str, temp: float, user_prompt: str) -> Dict[str, Any]:
    return {
        "model": model,
        "temperature": temp,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
    }


def parse_json_text(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        s = t.find("{")
        e = t.rfind("}")
        if s >= 0 and e > s:
            return json.loads(t[s : e + 1])
        raise


def _validate_generated(sample: Dict[str, Any], data: Dict[str, Any]) -> None:
    for k in ("semantic_description", "regions"):
        if k not in data:
            raise ValueError(f"Model output missing field: {k}")

    regions = data["regions"]
    if not isinstance(regions, list) or len(regions) != len(REGION_IDS):
        raise ValueError("regions must contain exactly 9 items.")
    ids = [item.get("region_id") for item in regions if isinstance(item, dict)]
    if sorted(ids) != sorted(REGION_IDS):
        raise ValueError("regions must include all required region_id exactly once.")

    semantic = data.get("semantic_description", {})
    if not isinstance(semantic, dict):
        raise ValueError("semantic_description must be an object.")
    semantic["global_tags"] = [str(x).strip() for x in semantic.get("global_tags", []) if str(x).strip()]
    semantic["strategic_focus"] = [str(x).strip() for x in semantic.get("strategic_focus", []) if str(x).strip()]
    if not isinstance(semantic.get("global_summary_cn", ""), str):
        semantic["global_summary_cn"] = ""
    if not isinstance(semantic.get("phase_explanation", ""), str):
        semantic["phase_explanation"] = ""
    data["semantic_description"] = semantic

    for i, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError(f"regions[{i}] must be an object.")
        region["shapes"] = [str(x).strip() for x in region.get("shapes", []) if str(x).strip()]
        if not isinstance(region.get("summary", ""), str):
            region["summary"] = ""
        region["local_tags"] = [str(x).strip() for x in region.get("local_tags", []) if str(x).strip()]
        region["key_points"] = [str(x).strip().upper() for x in region.get("key_points", []) if str(x).strip()]
        if not isinstance(region.get("group_status", []), list):
            region["group_status"] = []
        regions[i] = region

def call_api(
    api_key: str,
    model: str,
    temp: float,
    sample: Dict[str, Any],
    user_prompt: str,
    timeout: int,
    url: str,
) -> Dict[str, Any]:
    payload = build_payload(model, temp, user_prompt)
    res = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    res.raise_for_status()
    data = parse_json_text(res.json()["choices"][0]["message"]["content"])
    _validate_generated(sample, data)
    return data


def merge_fields(base: Dict[str, Any], fields: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("meta", "rules", "state", "targets"):
        if key in base:
            out[key] = base[key]
    out["semantic_description"] = fields["semantic_description"]
    out["regions"] = fields["regions"]
    return out


def run_prompt(args: argparse.Namespace) -> None:
    sample = load_json(args.input_json)
    prompt = build_user_prompt(sample)
    payload = build_payload(args.model, args.temperature, prompt)
    Path(args.output_prompt).write_text(prompt, encoding="utf-8")
    Path(args.output_payload).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prompt written: {Path(args.output_prompt).resolve()}")
    print(f"Payload written: {Path(args.output_payload).resolve()}")


def run_fill_one(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    sample = load_json(args.input_json)
    key = get_key(args.api_key)
    fields = call_api(key, args.model, args.temperature, sample, build_user_prompt(sample), args.timeout, args.base_url)
    out = args.output_json or args.input_json
    save_json(Path(out), merge_fields(sample, fields))
    elapsed = time.perf_counter() - t0
    print(f"Merged file written: {Path(out).resolve()}")
    print(f"Elapsed (one sample): {elapsed:.2f}s")


def _copy_non_step_jsons(src_root: Path, dst_root: Path) -> None:
    for p in src_root.rglob("*.json"):
        if p.name.startswith("step_"):
            continue
        d = dst_root / p.relative_to(src_root)
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")


def _progress(done: int, total: int, ok: int, failed: int) -> None:
    if total <= 0:
        return
    w = 30
    r = min(max(done / total, 0.0), 1.0)
    f = int(r * w)
    bar = "#" * f + "-" * (w - f)
    print(f"\r[{bar}] {done}/{total} {r*100:6.2f}% | ok={ok} | failed={failed}", end="", flush=True)


def run_fill_batch(args: argparse.Namespace) -> None:
    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    if not in_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {in_root}")

    key = ""
    if not args.dry_run:
        key = get_key(args.api_key)

    step_files = sorted(in_root.rglob("step_*.json"))
    if args.limit_files is not None:
        step_files = step_files[: args.limit_files]

    _copy_non_step_jsons(in_root, out_root)

    total, ok, failed = len(step_files), 0, 0
    fails: List[str] = []
    total_elapsed = 0.0
    success_elapsed = 0.0
    if total:
        _progress(0, total, ok, failed)

    for i, src in enumerate(step_files, start=1):
        t0 = time.perf_counter()
        dst = out_root / src.relative_to(in_root)
        try:
            sample = load_json(src)
            if args.dry_run:
                save_json(dst, sample)
            else:
                prompt = build_user_prompt(sample)
                last_err: Exception | None = None
                for _ in range(args.retries + 1):
                    try:
                        fields = call_api(key, args.model, args.temperature, sample, prompt, args.timeout, args.base_url)
                        save_json(dst, merge_fields(sample, fields))
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(args.retry_wait)
                if last_err is not None:
                    raise last_err
            ok += 1
            success_elapsed += time.perf_counter() - t0
        except Exception as e:
            failed += 1
            fails.append(f"{src}: {e}")
        finally:
            total_elapsed += time.perf_counter() - t0
            if total:
                _progress(i, total, ok, failed)

    if total:
        print()
    print(f"Input root: {in_root.resolve()}")
    print(f"Output root: {out_root.resolve()}")
    print(f"Total step files: {total}")
    print(f"Succeeded: {ok}")
    print(f"Failed: {failed}")
    if total > 0:
        print(f"Avg elapsed per sample (all): {total_elapsed / total:.2f}s")
    if ok > 0:
        print(f"Avg elapsed per sample (success): {success_elapsed / ok:.2f}s")
    if fails:
        log = out_root / "_failures.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(fails), encoding="utf-8")
        print(f"Failure log: {log.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified tool for simpleproto + Qwen.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prompt = sub.add_parser("prompt", help="Generate prompt/payload json only.")
    p_prompt.add_argument("--input-json", type=Path, required=True)
    p_prompt.add_argument("--output-prompt", type=Path, default=Path("./api_prompt.txt"))
    p_prompt.add_argument("--output-payload", type=Path, default=Path("./api_payload.json"))
    p_prompt.add_argument("--model", type=str, default="qwen-plus")
    p_prompt.add_argument("--temperature", type=float, default=0.2)
    p_prompt.set_defaults(func=run_prompt)

    p_one = sub.add_parser("fill-one", help="Fill one step json from API.")
    p_one.add_argument("--input-json", type=Path, required=True)
    p_one.add_argument("--output-json", type=Path, default=None)
    p_one.add_argument("--model", type=str, default="qwen-plus")
    p_one.add_argument("--temperature", type=float, default=0.2)
    p_one.add_argument("--timeout", type=int, default=120)
    p_one.add_argument("--api-key", type=str, default=None)
    p_one.add_argument("--base-url", type=str, default=API_URL)
    p_one.set_defaults(func=run_fill_one)

    p_batch = sub.add_parser("fill-batch", help="Fill all step jsons and mirror folder.")
    p_batch.add_argument("--input-root", type=Path, default=Path("./simpleproto_from_sgf"))
    p_batch.add_argument("--output-root", type=Path, default=Path("./prompted_json"))
    p_batch.add_argument("--model", type=str, default="qwen-plus")
    p_batch.add_argument("--temperature", type=float, default=0.2)
    p_batch.add_argument("--timeout", type=int, default=120)
    p_batch.add_argument("--api-key", type=str, default=None)
    p_batch.add_argument("--base-url", type=str, default=API_URL)
    p_batch.add_argument("--retries", type=int, default=1)
    p_batch.add_argument("--retry-wait", type=float, default=1.5)
    p_batch.add_argument("--limit-files", type=int, default=None)
    p_batch.add_argument("--dry-run", action="store_true")
    p_batch.set_defaults(func=run_fill_batch)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
