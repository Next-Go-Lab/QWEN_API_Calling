#!/usr/bin/env python3
"""Fill semantic_context for step_*.json via Qwen-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI

# ----- 运行前按需修改 -----
# 百炼 OpenAI 兼容模式：只配到 /v1，不要带 /chat/completions（SDK 会请求该路径）
OPENAI_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-50d636b4031d4f2e86cb25670f9dc5ab"  # 可填 sk-xxx；不设则用环境变量 DASHSCOPE_API_KEY
MODEL = "qwen-plus"
TEMPERATURE = 0.2 # 温度，0.0-1.0，越小越确定，越大越随机
REQUEST_TIMEOUT = 120 # 请求超时时间
DRY_RUN = False # True：只复制 json，不调 API
MAX_STEP_FILES = 50  # 填正整数则只处理前 N 个 step_*.json（全树排序后）；None 表示全部
# 首次请求后，若仍复述 state.last_move.point，则最多再请求此次数；耗尽后整批退出
MAX_LEAK_RETRIES = 10

DEFAULT_LABEL_SPACE: Dict[str, List[str]] = {
    "move_style_candidates": [
        "强攻", "先手压迫", "补强", "收束", "腾挪", "治孤", "围空", "扩张", "压迫",
        "限制", "断联", "劫争", "转换", "试探",
    ],
    "move_priority_candidates": [
        "胜负手", "先手官子", "普通收束", "布局要点", "中盘要点", "收官要点",
    ],
    "move_risk_candidates": ["激烈", "稳健", "折中"],
}

STR_KEYS = (
    "global_summary", "move_goal", "move_location_hint", "move_effect", "move_exclusivity",
)
LIST_KEYS = ("move_style", "move_priority", "move_risk")
ALL_KEYS = STR_KEYS + LIST_KEYS

# 描述棋形与本手时优先使用的规范中文围棋术语（与常见棋书/解说用语一致）
GO_TERMINOLOGY_GUIDE = """
【专业术语】概括局面、棋形与「本手」意图时，**优先选用下列范畴中的规范用语**，贴近职业棋语；按局面择词、自然连贯，避免生造口号或与局面不符的堆砌。
- 基本概念：目、气、眼、布局、收官、死活、双活、打劫
- 棋盘位置：天元、星位、小目、高目、目外、三三
- 一字着法：长、立、飞、尖、虎、拆、提、关、冲、跳、曲（拐）、镇、夹、断、跨、刺（觑）、托、退、碰、压、爬、接、顶、并、扳、挡、双、挤、逼、封、点、渡、扑
- 复合着法或手段：叫吃、占角、挂角、缔角、征子、缓征、紧气、脱先、投子、割分、整地、倒脱靴、滚打包收
- 常见棋形：直三、直四、弯三、弯四、方四、板六、笠帽四（丁四）、刀把五、梅花五、拳头六、大猪嘴、小猪嘴、长生劫、金鸡独立、盘角曲四
- 布局/流派：小林流、星无忧角、中国流、迷你中国流、错小目、三连星、宇宙流

字段侧重：global_summary 偏全局大势与攻防方向；move_goal / move_location_hint / move_effect / move_exclusivity 偏本手与局部时，**多用着法名、棋形名、战术名**（如「镇」「靠断」「模样」「先手」等），仍须遵守禁坐标与防泄露规则。
""".strip()


class LastMoveSemanticLeakError(ValueError):
    """semantic_context 五个字符串字段中出现了与 state.last_move.point 相同的 GTP。"""


class LastMoveLeakExhaustedError(RuntimeError):
    """首请求 + MAX_LEAK_RETRIES 次重试后仍泄露，须停止整批生成。"""


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_key(cli: str | None) -> str:
    k = cli or API_KEY or os.getenv("DASHSCOPE_API_KEY", "")
    if not k:
        raise ValueError("在文件顶部填写 API_KEY，或设环境变量 DASHSCOPE_API_KEY，或传 --api-key")
    return k


def make_client(api_key: str) -> OpenAI:
    base = OPENAI_BASE_URL.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")].rstrip("/")
    return OpenAI(api_key=api_key, base_url=base, timeout=REQUEST_TIMEOUT)


def label_space(sample: Dict[str, Any]) -> Dict[str, List[str]]:
    ls = sample.get("label_space")
    if isinstance(ls, dict) and ls.get("move_style_candidates"):
        return {a: list(b) if isinstance(b, list) else [] for a, b in ls.items()}
    return dict(DEFAULT_LABEL_SPACE)


def _normalize_side_color(v: Any) -> Optional[str]:
    s = str(v).strip().lower()
    if s in ("black", "b", "1"):
        return "black"
    if s in ("white", "w", "0"):
        return "white"
    return None


def _stance_instruction(state: Dict[str, Any]) -> str:
    """语义描述须与 state.last_move.color 一致：该方=我方，另一方=对方。"""
    lm = state.get("last_move")
    side: Optional[str] = None
    from_last = False
    if isinstance(lm, dict):
        side = _normalize_side_color(lm.get("color"))
        from_last = side is not None
    fallback_note = ""
    if side is None:
        side = _normalize_side_color(state.get("to_play"))
        if side is not None:
            fallback_note = (
                "【说明】当前无 state.last_move，以下「我方」暂按 state.to_play（即将行棋方）理解。 "
            )
    recent = (
        "recent_moves 每项含 color/point/is_pass，仅供理解进程，**勿在输出里逐条复述**。 "
    )
    if side is None:
        return (
            "【立场】无法从 state.last_move / state.to_play 判定黑白时请据棋盘自洽使用「我方/对方」。"
            + recent
        )
    if not from_last:
        if side == "black":
            core = (
                "【立场】**黑=我方、白=对方**（尚无上一手，按即将落子的黑方语境写）。"
                "勿改作白方主语或替白方制定计划。"
            )
        else:
            core = (
                "【立场】**白=我方、黑=对方**（尚无上一手，按即将落子的白方语境写）。"
                "勿改作黑方主语或替黑方制定计划。"
            )
        return fallback_note + core + recent
    if side == "black":
        core = (
            "【立场】**与 state.last_move.color=black 一致**：**黑=我方、白=对方**。"
            "叙述立足于**黑方刚下完上一手**后的局面语境，勿改作白方主语或替白方制定计划。"
            " state.to_play 表示**下一手轮到谁**（常与 last_move 异色），"
            "但全文「我方/对方」**只以 last_move.color 为准**。"
        )
        return fallback_note + core + recent
    core = (
        "【立场】**与 state.last_move.color=white 一致**：**白=我方、黑=对方**。"
        "叙述立足于**白方刚下完上一手**后的局面语境，勿改作黑方主语或替黑方制定计划。"
        " state.to_play 表示**下一手轮到谁**（常与 last_move 异色），"
        "但全文「我方/对方」**只以 last_move.color 为准**。"
    )
    return fallback_note + core + recent


def build_system_prompt() -> str:
    return (
        "你是围棋训练数据标注员，只输出合法 JSON，禁止 markdown。\n"
        "用语须**专业**：用用户消息中的围棋术语表描述棋形、着手性质与本手意图，避免空泛口语。\n"
        "硬性约束：若输入含 state.last_move 且为具体落点，则 global_summary、move_goal、"
        "move_location_hint、move_effect、move_exclusivity 五个字段中**禁止出现与该点相同的 GTP**"
        "（大小写都不行），也不得用拆字、空格等方式变相写出同一点；"
        "可用「对方上一手所在块」「刚被碰/托的那条线」等描述代替，避免与结构化 last_move 重复造成训练数据泄露。\n"
        "state.recent_moves 为若干条对象（color/point/is_pass），仅供理解进程，**禁止**在五个字符串中按条抄录该序列（等同复述手顺）。\n"
        "其它须遵守用户消息中的立场与禁则。"
    )


def last_move_point_to_avoid_in_semantics(state: Dict[str, Any]) -> Optional[str]:
    """
    若存在上一手具体点，则五个字符串字段不得再写出该 GTP，否则与结构化 last_move 重复、易造成标签泄露。
    """
    lm = state.get("last_move")
    if not isinstance(lm, dict) or lm.get("is_pass"):
        return None
    pt = str(lm.get("point", "")).strip()
    if not pt or pt.upper() == "PASS":
        return None
    return pt.upper()


def _last_move_leak_pattern(forbidden_pt: str) -> re.Pattern[str]:
    """在中文邻接下也能识别独立的 GTP 点（如 「P3」「在P3」）。"""
    esc = re.escape(forbidden_pt)
    return re.compile(rf"(?<![A-HJ-Ta-hj-t]){esc}(?![0-9])", re.IGNORECASE)


def build_prompt(sample: Dict[str, Any]) -> str:
    state = sample.get("state") or {}
    stance = _stance_instruction(state)
    ctx: Dict[str, Any] = {
        "meta": sample.get("meta", {}),
        "rules": sample.get("rules", {}),
        "state": sample.get("state", {}),
        "label_space": label_space(sample),
    }
    if sample.get("targets"):
        ctx["targets"] = sample["targets"]

    leak_rule = ""
    lmpt = last_move_point_to_avoid_in_semantics(state)
    if lmpt:
        leak_rule = (
            f"\n【防泄露】state.last_move.point = {lmpt} 已在结构化字段中给出，"
            f"以下五个字符串**禁止再次出现**「{lmpt}」及其大小写变体；请用「对方上一手所在块」「角上刚落地一子」等描述，避免训练时从语文复述抄点位。"
        )

    ex = {
        "global_summary": "中腹互缠，我方厚势对对方欠棋施压，胜负在迂回占先。",
        "move_goal": "保持对对方弱形压力，争先不求一击必杀。",
        "move_location_hint": "对方刚动手所在块的**外侧封锁带**，偏厚一侧",
        "move_effect": "收紧对方出路、保留后续搜根手段",
        "move_exclusivity": "脱先则对方两块易腾挪，攻势断档",
        "move_style": ["限制整形", "先手压迫"],
        "move_priority": ["中盘要点"],
        "move_risk": ["折中"],
    }

    rules = f"""
{stance}
{leak_rule}

{GO_TERMINOLOGY_GUIDE}

【输出】仅一个顶层键 semantic_context，字段必含：{", ".join(ALL_KEYS)}。
前五项为**短字符串**（各建议≤80字，尽量**名词短语/标签化短句**，一整句亦可但要极练）；
move_style / move_priority / move_risk 为数组，元素**只能**来自输入 label_space 对应 candidates。

【禁止】
- 在五个字符串字段中写出与 **state.last_move.point** 相同的 GTP（若存在上一手坐标）；其它坐标若有助于说明局面**可以**谨慎使用。
- 直接点明「本手（待下的一手）应下在某某具体点」式的标准答案表述。
- 按 state.recent_moves **逐条复述**手顺（其为 {{color, point, is_pass}} 对象列表，与结构化字段重复）；其它非 recent 的坐标在必要时可谨慎用于说明。
- 立场错误：「我方/对方」与 **state.last_move.color**（无 last_move 时按说明用 state.to_play）不一致。

【应写】
- 区域与棋形关系、相对位置；可点名与 last_move **不同**的参考点，但勿复述 last_move 点位。
- move_location_hint 以方位/带状区域/棋块关系为主，并尽量嵌入**着法/棋形术语**（尖、飞、镇、模样、厚势等）写清「像什么、在做什么」。
- 让读者结合棋盘 + 本描述理解攻防，同时避免与 last_move 字段信息重复泄密。

【输入棋盘上下文】
{json.dumps(ctx, ensure_ascii=False, indent=2)}

【结构示例】
{json.dumps({"semantic_context": ex}, ensure_ascii=False, indent=2)}
""".strip()
    return rules


def parse_json_text(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}")
        if s >= 0 and e > s:
            return json.loads(t[s : e + 1])
        raise


def norm_root(d: Dict[str, Any]) -> Dict[str, Any]:
    if "semantic_context" in d:
        return d
    if "global_summary" in d:
        return {"semantic_context": d}
    return d


def as_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


def validate(sample: Dict[str, Any], data: Dict[str, Any]) -> None:
    data = norm_root(data)
    sc = data.get("semantic_context")
    if not isinstance(sc, dict):
        raise ValueError("need semantic_context object")
    for k in ALL_KEYS:
        if k not in sc:
            raise ValueError(f"missing {k}")
    state = sample.get("state") or {}
    forbid_pt = last_move_point_to_avoid_in_semantics(state)
    pat = _last_move_leak_pattern(forbid_pt) if forbid_pt else None

    for k in STR_KEYS:
        sc[k] = "" if sc[k] is None else str(sc[k]).strip()
        if pat is not None and pat.search(sc[k]):
            raise LastMoveSemanticLeakError(
                f"字段 {k} 不得包含上一手坐标 {forbid_pt!r}（与 state.last_move 重复）。原文：{sc[k]!r}"
            )
    ls = label_space(sample)
    ok_s, ok_p, ok_r = (
        set(ls.get("move_style_candidates", [])),
        set(ls.get("move_priority_candidates", [])),
        set(ls.get("move_risk_candidates", [])),
    )

    def filt(xs: List[str], ok: Set[str]) -> List[str]:
        return xs if not ok else [x for x in xs if x in ok]

    sc["move_style"] = filt(as_str_list(sc["move_style"]), ok_s)
    sc["move_priority"] = filt(as_str_list(sc["move_priority"]), ok_p)
    sc["move_risk"] = filt(as_str_list(sc["move_risk"]), ok_r)
    data["semantic_context"] = sc


def _leak_correction_prefix(forbid_pt: str | None, last_err: LastMoveSemanticLeakError) -> str:
    pt_hint = f"（禁止点位：{forbid_pt}）" if forbid_pt else ""
    return (
        f"【校验未通过】{last_err}{pt_hint}\n"
        "你必须重新输出**一份完整**的 JSON：仅顶层键 semantic_context，且五个短字符串字段中"
        "**不得再出现**与 state.last_move.point 相同的 GTP，任何变体都不行；"
        "用方位与棋形关系描述代替点坐标。下面是原任务与上下文（请当作新回答覆盖上一轮）：\n\n"
    )


def call_api(
    client: OpenAI,
    sample: Dict[str, Any],
    *,
    source_label: str = "",
) -> Dict[str, Any]:
    state = sample.get("state") or {}
    forbid_pt = last_move_point_to_avoid_in_semantics(state)
    last_leak: LastMoveSemanticLeakError | None = None
    correction = ""

    total_attempts = 1 + MAX_LEAK_RETRIES  # 首次 + 重试 MAX_LEAK_RETRIES 次
    for attempt in range(total_attempts):
        base_prompt = build_prompt(sample)
        user_content = correction + base_prompt if correction else base_prompt
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as e:
            raise RuntimeError(
                f"{e}\n说明与错误码见: https://www.alibabacloud.com/help/model-studio/developer-reference/error-code"
            ) from e

        msg = completion.choices[0].message
        raw = msg.content if msg else None
        if not raw:
            raise ValueError("模型返回空内容")
        data = parse_json_text(raw)
        data = norm_root(data)
        try:
            validate(sample, data)
            return data
        except LastMoveSemanticLeakError as e:
            last_leak = e
            if attempt >= total_attempts - 1:
                break
            correction = _leak_correction_prefix(forbid_pt, e)
            loc = f" {source_label}" if source_label else ""
            print(
                f"  [防泄露重试 {attempt + 2}/{total_attempts}]{loc}：{e}",
                file=sys.stderr,
            )

    assert last_leak is not None
    loc = f" 文件：{source_label}" if source_label else ""
    raise LastMoveLeakExhaustedError(
        f"state.last_move 坐标防泄露：共请求 {total_attempts} 次（含首次）仍不合格。{loc}\n最后错误：{last_leak}"
    ) from last_leak


def merge(base: Dict[str, Any], gen: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: base[k] for k in ("meta", "rules", "state", "targets", "label_space") if k in base}
    out["semantic_context"] = norm_root(gen)["semantic_context"]
    return out


def copy_aux_json(src: Path, dst: Path) -> None:
    for p in src.rglob("*.json"):
        if p.name.startswith("step_"):
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill semantic_context in step_*.json")
    ap.add_argument("--input-root", type=Path, default=Path("./simpleproto_from_sgf"))
    ap.add_argument("--output-root", type=Path, default=Path("./prompted_json"))
    ap.add_argument("--api-key", type=str, default=None, help="覆盖文件顶部的 API_KEY")
    args = ap.parse_args()

    root = args.input_root
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    steps = sorted(root.rglob("step_*.json"))
    if MAX_STEP_FILES is not None:
        steps = steps[: max(0, int(MAX_STEP_FILES))]

    copy_aux_json(root, args.output_root)
    client: OpenAI | None = None
    if not DRY_RUN:
        client = make_client(get_key(args.api_key))

    ok = fail = 0
    for i, src in enumerate(steps, 1):
        dst = args.output_root / src.relative_to(root)
        try:
            sample = load_json(src)
            if DRY_RUN:
                save_json(dst, sample)
            else:
                assert client is not None
                gen = call_api(client, sample, source_label=str(src))
                save_json(dst, merge(sample, gen))
            ok += 1
            print(f"[{i}/{len(steps)}] ok {src.name}")
        except LastMoveLeakExhaustedError as e:
            print(f"[{i}/{len(steps)}] FATAL {src}:\n{e}", file=sys.stderr)
            raise SystemExit(1) from e
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(steps)}] FAIL {src}: {e}")

    print(f"done: {ok} ok, {fail} failed, total {len(steps)}")


if __name__ == "__main__":
    main()
