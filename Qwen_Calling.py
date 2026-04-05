#!/usr/bin/env python3
"""Fill semantic_context for step_*.json via Qwen-compatible API.

默认从 ``outputs/`` 读取 step JSON，写入 ``prompted_json/``（保持相同子目录结构）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from label import COL_LABELS, parse_coord

# ----- 运行前按需修改 -----
# 百炼 OpenAI 兼容模式：只配到 /v1，不要带 /chat/completions（SDK 会请求该路径）
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-546774f87cc4466eb1bc467445398242"  # 可填 sk-xxx；不设则用环境变量 DASHSCOPE_API_KEY
MODEL = "qwen-plus"
TEMPERATURE = 0.1  # 温度，0.0-1.0，越小越确定，越大越随机
REQUEST_TIMEOUT = 120  # 请求超时时间
DRY_RUN = False  # True：只复制 json，不调 API
MAX_STEP_FILES = 284  # 填正整数则只处理前 N 个 step_*.json（全树排序后）；None 表示全部
# 首次请求后，若输出仍不合规（含坐标泄露、格式校验失败），则最多再请求此次数；耗尽后退出
MAX_LEAK_RETRIES = 10

# semantic_context 四字段（均为字符串）
SEMANTIC_KEYS = (
    "global_summary",
    "move_goal",
    "move_location_hint",
    "move_style",
)

# 各字段硬性长度上限：仅统计汉字（CJK 统一表意文字 U+4E00–U+9FFF 个数）
SEMANTIC_FIELD_MAX_HAN = 60

# global_summary 必须以其中之一开头（长短语序：先匹配长的）
GLOBAL_ADVANTAGE_LABELS: tuple[str, ...] = (
    "黑大优",
    "黑小优",
    "势均力敌",
    "白小优",
    "白大优",
)

MOVE_STYLE_LEVELS: tuple[str, ...] = (
    "非常激进",
    "有点激进",
    "中性",
    "有点保守",
    "过于保守",
)

# 布局期：手数 ≤ 此值视为「极早」，global_summary 只要求极短补充，不写长段大势分析
EARLY_OPENING_MAX_MOVE = 10

# 描述棋形与本手时优先使用的规范中文围棋术语（与常见棋书/解说用语一致）
GO_TERMINOLOGY_GUIDE = """
【专业术语】概括局面、棋形与「本手」意图时，**优先选用下列范畴中的规范用语**，贴近职业棋语；按局面择词、自然连贯，避免生造口号或与局面不符的堆砌。
- 基本概念：目、气、眼、布局、收官、死活、双活、打劫
- 棋盘位置：天元、星位、小目、高目、目外、三三
- 一字着法：长、立、飞、尖、虎、拆、提、关、冲、跳、曲（拐）、镇、夹、断、跨、刺（觑）、托、退、碰、压、爬、接、顶、并、扳、挡、双、挤、逼、封、点、渡、扑
- 复合着法或手段：叫吃、占角、挂角、缔角、征子、缓征、紧气、脱先、投子、割分、整地、倒脱靴、滚打包收
- 常见棋形：直三、直四、弯三、弯四、方四、板六、笠帽四（丁四）、刀把五、梅花五、拳头六、大猪嘴、小猪嘴、长生劫、金鸡独立、盘角曲四
- 布局/流派：小林流、星无忧角、中国流、迷你中国流、错小目、三连星、宇宙流

字段侧重：global_summary 偏全局大势（仅在极早布局期可极短）；move_goal 偏**上一手**意图与双方处境；move_location_hint 是供**你自己的下游模型**阅读的**强指示性**自然语言 prompt：在**不落子前盘面**参照下，把 **last_move** 的落点收窄到**唯一或极少数**候选交叉点，使读者在脑中可走成与 **state.last_move** **一致或极相近**的一手（仍不得以明文复述 last_move 的 GTP）；**不是**对 to_play 的下一手预测。move_style 为五档风格定性，须遵守禁 GTP 复述规则。**四字段汉字个数硬性上限见用户消息【输出】。**
""".strip()

# move_location_hint 输入锚点常见单字/短语，释义概括自维基百科《围棋术语》：
# https://zh.wikipedia.org/wiki/%E5%9B%B4%E6%A3%8B%E6%9C%AF%E8%AF%AD
MOVE_LOCATION_HINT_TAG_MEANINGS = """
【move_location_hint 短标签释义】便于理解 **semantic_context.move_location_hint** 或 **state.last_move** 标签中的简称；义项与《围棋术语》（维基百科）一致或由其概括，扩写时须用准棋语、勿与邻近概念混淆。
- **挂 / 挂角**：对对方已在角上占据的**星位、小目、高目、三三**等处的**独子**迫近，意在分角、打破对方干净占角；落在**四线**为**高挂**，**三线**为**低挂**（参见该条目「挂」「掛角」）。**不是**己方先着占空角的「占角」。
- **占角**：在**空角**先行占据要点（星、小目等），建立角上阵地。
- **缔角 / 締角**：在角上补强、守角、立根据（如无忧角类），偏守势整形。
- **星位**：棋盘上黑点位置（十九路有九星），中央为**天元**。
- **三三**：角上**三线与三线**交点。
- **小目**：**三线与四线**交点。
- **目外**：**三线与五线**交点。
- **高目**：**四线与五线**交点。
- **飞 / 小飞 / 大飞**：沿斜向隔路类着法（如「马步」形）；作标签时需结合与参照子的相对方位说明。
- **尖、长、跳、拆、夹、托、扳、镇、碰、点……**：着法类别，含义见本条目前文【专业术语】；扩写时要写出**相对哪块棋、哪一侧**。
- 若锚点仅为单字如「**挂**」，须按**挂角**理解，扩写中写明：**哪一角**、对方占的是**星/小目/三三**等、**高挂/低挂**、相对**哪一子**，不得解成与局面不符的别义。

完整分类与更多术语见：https://zh.wikipedia.org/wiki/%E5%9B%B4%E6%A3%8B%E6%9C%AF%E8%AF%AD
""".strip()

# 十九路 GTP 与 label.parse_coord 一致：列 A–T（无 I），行数字越小越近棋盘底边；
# 画面上列字母增大≈向右，行号增大≈向棋盘顶边（与「下一格」时左右看列、上下看行号，勿混）。
# 「第 n 路」：自该角**沿路棱**向里数，**贴角第一格交叉那条线为第 1 路**（即右下以 T 列为第 1 路边、第 1 横线为最下一路）。
# 在此约定下：右下 **Q3** 为「第 4 路边 × 第 3 路边」= **小目**；**R3** 为 **三三**；右下 **星** 为 **Q4**（不得误用 P4）。
GTP_TYPICAL_CORNER_POINTS = """
【典型点 GTP 坐标参考】须与下表及输入中的 **computed_last_move_geometry**（若有）一致；**勿**用「从角数跨了几格」的含糊说法与下表抵触，**勿**把右下语境写成 A1、D1 等左下参照。
- **左下顶点 A1**：星位 **D4**；三三 **C3**；小目 **C4**、**D3**。
- **右下顶点 T1**：星位 **Q4**；三三 **R3**；小目 **Q3**、**R4**。
- **左上顶点 A19**：星位 **D16**；三三 **C17**；小目 **C16**、**D17**。
- **右上顶点 T19**：星位 **Q16**；三三 **R17**；小目 **Q17**、**R16**。
- **天元**：**K10**。

用法：**四字符串仍禁止写出与 state.last_move.point 相同的 GTP**；可用表中**非上一手**的点作锚。典型名宜与程序核算一致；全局总结可提及其他角点位，勿与本手坐标类型矛盾。
""".strip()


class LastMoveSemanticLeakError(ValueError):
    """semantic_context 四个字符串字段中出现了与 state.last_move.point 相同的 GTP。"""


class LastMoveLeakExhaustedError(RuntimeError):
    """首请求 + MAX_LEAK_RETRIES 次重试后防泄露仍不合格，须停止整批生成。"""


class SemanticFormatError(ValueError):
    """semantic_context 格式或枚举不符合约定。"""


@dataclass
class ApiCallMetrics:
    """单次 call_api（可含多轮重试）的耗时与 token 累计。"""

    wall_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_requests: int = 0

    def __iadd__(self, other: ApiCallMetrics) -> ApiCallMetrics:
        self.wall_seconds += other.wall_seconds
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.num_requests += other.num_requests
        return self


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds + 0.5), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _usage_from_completion(completion: Any) -> Tuple[int, int, int]:
    u = getattr(completion, "usage", None)
    if u is None:
        return 0, 0, 0
    pt = int(getattr(u, "prompt_tokens", None) or 0)
    ct = int(getattr(u, "completion_tokens", None) or 0)
    tt = int(getattr(u, "total_tokens", None) or 0)
    if tt <= 0:
        tt = pt + ct
    return pt, ct, tt


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


def _normalize_side_color(v: Any) -> Optional[str]:
    s = str(v).strip().lower()
    if s in ("black", "b", "1"):
        return "black"
    if s in ("white", "w", "0"):
        return "white"
    return None


def _has_concrete_last_move(state: Dict[str, Any]) -> bool:
    lm = state.get("last_move")
    if not isinstance(lm, dict):
        return False
    if lm.get("is_pass"):
        return False
    pt = str(lm.get("point", "")).strip()
    return bool(pt) and pt.upper() != "PASS"


def _seed_contradicts_geometry(seed: str, geom: Optional[Dict[str, Any]]) -> bool:
    """输入短锚与程序核算典型点名冲突时，以几何为准，不强制保留错误锚点子串。"""
    if not seed or not geom:
        return False
    t_raw = geom.get("typical_point_name_zh")
    if not t_raw:
        return False
    t = str(t_raw).strip()
    if t == "小目" and "三三" in seed:
        return True
    if t == "三三" and "小目" in seed:
        return True
    if t == "星位" and ("三三" in seed or "小目" in seed):
        return True
    return False


def input_move_location_hint_seed(sample: Dict[str, Any]) -> Optional[str]:
    """
    来自输入 JSON（如 outputs）的 semantic_context.move_location_hint 短锚点；
    若非空则生成时须据此扩写，且校验时须在输出中可检索地保留（子串匹配）。
    """
    sc = sample.get("semantic_context")
    if not isinstance(sc, dict):
        return None
    s = _coerce_semantic_string("move_location_hint", sc.get("move_location_hint")).strip()
    return s or None


def tags_consonant_with_geometry(
    tags: List[str], geom: Optional[Dict[str, Any]]
) -> List[str]:
    """
    上游 labels 可能与 GTP 坐标矛盾（如实为小目标成「三三」）。以 last_move_typical_point_geometry 为准，丢弃明显冲突的标签，
    避免「hint 须含标签」与「hint 不得写三三」死锁。
    """
    if not geom or not tags:
        return list(tags)
    t_raw = geom.get("typical_point_name_zh")
    t = str(t_raw).strip() if t_raw is not None else ""
    out: List[str] = []
    for x in tags:
        s = str(x).strip()
        if not s:
            continue
        if t == "小目" and "三三" in s and "小目" not in s:
            continue
        if t == "三三" and "小目" in s and "三三" not in s:
            continue
        if t == "星位" and (
            ("三三" in s and "星" not in s)
            or ("小目" in s and "星位" not in s and "星" not in s)
        ):
            continue
        out.append(x)
    return out


def last_move_tag_strings(state: Dict[str, Any]) -> List[str]:
    """兼容 labels / tags / label。"""
    lm = state.get("last_move")
    if not isinstance(lm, dict):
        return []
    out: List[str] = []
    for key in ("labels", "tags", "label"):
        v = lm.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
        elif isinstance(v, list):
            for x in v:
                s = str(x).strip()
                if s:
                    out.append(s)
    return out


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
        "硬性约束：若输入含 state.last_move 且为具体落点（非 pass），则 global_summary、move_goal、"
        "move_location_hint、move_style 四个字段中**禁止出现与该点相同的 GTP坐标**"
        "（大小写都不行），也不得用拆字、空格等方式变相写出同一点；"
        "布局期可用「左上角星位」「右下角三三」等语义化点位，但不得写出与 last_move.point **同一** GTP坐标。\n"
        "move_location_hint：以**上一手落下之前**的盘面为参照，用**高鉴别力**语言描述 **last_move** 落点，使读者能唯一或近乎唯一地还原该手（不得明文写该点 GTP）；"
        "禁止写成 state.to_play 的下一手或应手预测。\n"
        "state.recent_moves 仅供理解进程，**禁止**在四个字段中按条抄录该序列。\n"
        f"四个字符串字段每个硬性仅计汉字：汉字个数≤{SEMANTIC_FIELD_MAX_HAN}（计数规则见用户消息）。\n"
        "若用户消息含「落点锚点·扩写」，输出的 move_location_hint 必须包含所给锚点的**原文字符串**（子串命中即可）并据此扩写；锚点棋语含义以用户消息【move_location_hint 短标签释义】及维基《围棋术语》为准。\n"
        "优先依照用户消息中的 **computed_last_move_geometry** 与【典型点 GTP 坐标参考】描述星位/三三/小目；其它角可另作讨论，勿与上一手坐标矛盾。\n"
        "其它须遵守用户消息中的立场与字段格式。"
    )


def last_move_point_to_avoid_in_semantics(state: Dict[str, Any]) -> Optional[str]:
    """
    若存在上一手具体点，则四个字符串字段不得再写出该 GTP。
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


def _global_summary_starts_with_tier(s: str) -> bool:
    t = s.strip()
    return any(t.startswith(label) for label in GLOBAL_ADVANTAGE_LABELS)


def semantic_han_char_count(text: str) -> int:
    """
    硬性长度：仅统计汉字个数（CJK 统一表意文字 Unicode U+4E00–U+9FFF）。
    标点、空格、英文字母与数字等一律不计。
    """
    return len(re.findall(r"[\u4e00-\u9fff]", str(text)))


def _coerce_semantic_string(key: str, v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        if not v:
            return ""
        # move_style 必须为五档之一；列表时只取首项参与枚举校验
        if key == "move_style":
            return str(v[0]).strip()
        if len(v) == 1:
            return str(v[0]).strip()
        return "；".join(str(x).strip() for x in v if str(x).strip())
    return str(v).strip()


def _is_very_early_opening(state: Dict[str, Any]) -> bool:
    if str(state.get("game_phase", "")).strip() != "opening":
        return False
    try:
        mn = int(state.get("move_number", 0))
    except (TypeError, ValueError):
        return False
    return 1 <= mn <= EARLY_OPENING_MAX_MOVE


def _board_size_from_state(state: Dict[str, Any]) -> int:
    rules = state.get("rules") or {}
    try:
        n = int(rules.get("board_size", 19))
    except (TypeError, ValueError):
        return 19
    return n if n >= 2 else 19


def _corner_star_row_col(nearest_corner_zh: str, n: int) -> Tuple[int, int]:
    """最近一角的标准四四星位，(row_top, col_idx) 与 label.parse_coord 一致。"""
    t = 3
    b = n - 4
    if nearest_corner_zh == "左上":
        return t, t
    if nearest_corner_zh == "右上":
        return t, b
    if nearest_corner_zh == "左下":
        return b, t
    if nearest_corner_zh == "右下":
        return b, b
    return t, t


def _gtp_from_row_col(row_top: int, col_idx: int, n: int) -> str:
    return f"{COL_LABELS[col_idx]}{n - row_top}"


def _describe_move_vs_corner_star_zh(dr: int, dc: int, star_gtp: str, max_manhattan: int = 4) -> str:
    """
    dr = row_last - row_star（画面上自上而下为 row_top 增大，故 dr>0 为朝棋盘底边）
    dc = col_last - col_star（dc>0 为列字母增大、画面向右）
    """
    if abs(dr) + abs(dc) > max_manhattan:
        return (
            f"本手与【该角】星位 {star_gtp} 相距已远，勿用「正上/正下/正左/正右一格」类句式硬套；"
            "请据落子前子力、典型点名与锚点描述。"
        )
    if dr == 0 and dc == 0:
        return f"本手即【该角】标准四四星位（与 {star_gtp} 同点）。"
    parts: List[str] = []
    if dr < 0:
        parts.append(f"向棋盘顶边 {-dr} 格（行号增大）")
    elif dr > 0:
        parts.append(f"向棋盘底边 {dr} 格（行号减小）")
    if dc < 0:
        parts.append(f"向左 {-dc} 格（列字母减小）")
    elif dc > 0:
        parts.append(f"向右 {dc} 格（列字母增大）")
    joint = "且".join(parts) if len(parts) == 2 else parts[0]
    return (
        f"相对【该角】星位 {star_gtp}：{joint}。"
        "写 move_location_hint 时「上下」随**行号**变、「左右」随**列字母**变，勿混"
        "（例：**同行、列字母大一位**是邻格在**右**，不是「正下」；正下为**同列、行号小一**）。"
    )


def last_move_typical_point_geometry(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    相对「最近一角」两边各为第几路（自该角沿路棱向里数，靠角第一条线为第 1 路），
    据此判定星位(4-4)/三三(3-3)/小目(3-4)。与职业盘面「下小目」等一致（如右下 Q3 为小目）。
    """
    if not _has_concrete_last_move(state):
        return None
    lm = state.get("last_move")
    if not isinstance(lm, dict):
        return None
    pt = str(lm.get("point", "")).strip().upper()
    if not pt:
        return None
    n = _board_size_from_state(state)
    try:
        row_top, col_idx = parse_coord(pt, n)
    except ValueError:
        return None
    r_bottom = n - row_top  # GTP 行号（距离底边格数，1 为最下一路）
    d_left = col_idx + 1
    d_right = n - col_idx
    d_bottom = r_bottom
    d_top = n + 1 - r_bottom

    corners: List[tuple[str, str, int, int]] = [
        ("左下", "A1", d_left, d_bottom),
        ("右下", "T1", d_right, d_bottom),
        ("左上", "A19", d_left, d_top),
        ("右上", "T19", d_right, d_top),
    ]
    best_name, best_vtx, a, b = min(corners, key=lambda t: max(t[2], t[3]))
    pair = tuple(sorted((a, b)))
    if pair == (3, 3):
        name_zh = "三三"
    elif pair == (4, 4):
        name_zh = "星位"
    elif pair == (3, 4):
        name_zh = "小目"
    else:
        name_zh = None

    sr, sc = _corner_star_row_col(best_name, n)
    star_gtp = _gtp_from_row_col(sr, sc, n)
    dr = row_top - sr
    dc = col_idx - sc
    vs_star = _describe_move_vs_corner_star_zh(dr, dc, star_gtp)

    out: Dict[str, Any] = {
        "last_move_gtp": pt,
        "nearest_corner_zh": best_name,
        "corner_vertex_gtp": best_vtx,
        "edge_distances_from_corner": [a, b],
        "nearest_corner_star_gtp": star_gtp,
        "last_move_vs_corner_star_zh": vs_star,
        "note": (
            "若 typical_point_name_zh 存在，描述星位/三三/小目应与之及【典型点 GTP 坐标参考】一致；"
            "否则本手为非三三·星·小目的肩冲/挂角等，仍须以 last_move_vs_corner_star_zh 为\"上下左右\"依据。"
        ),
    }
    if name_zh is not None:
        out["typical_point_name_zh"] = name_zh
    return out


def build_prompt(sample: Dict[str, Any]) -> str:
    state = sample.get("state") or {}
    stance = _stance_instruction(state)
    ctx: Dict[str, Any] = {
        "meta": sample.get("meta", {}),
        "rules": sample.get("rules", {}),
        "state": sample.get("state", {}),
    }
    if sample.get("targets"):
        ctx["targets"] = sample["targets"]

    geometry_rule = ""
    geom = last_move_typical_point_geometry(state)
    if geom:
        ctx["computed_last_move_geometry"] = geom
        geometry_rule = (
            "\n【程序核算·典型点】输入 JSON 内 **computed_last_move_geometry** 由坐标推算，描述**本手所在角**的星位/三三/小目时应与此一致；"
            "**last_move_vs_corner_star_zh** 给出本手相对该角**四四星位**的上下左右，move_location_hint 须与之相容。"
            "GTP：列字母 A→T 为从左到右，行号 1→19 为从底到顶；故**同行邻格在左右看列**，**同列邻格在上下看行号**。"
            "global_summary 若提到其它角/未落子处，勿与本手类型混谈。"
        )

    leak_rule = ""
    lmpt = last_move_point_to_avoid_in_semantics(state)
    if lmpt:
        leak_rule = (
            f"\n【防泄露】state.last_move.point = {lmpt} 已在结构化字段中给出，"
            f"四个字段**禁止再次出现**「{lmpt}」及其大小写变体；"
            "但为**锁定落点**，**可**写出落子前盘面上**其它**已有子的 GTP 作锚点（不得复述上一手该点本身）。"
            "须以方位、典型点名、与锚点的相对关系等高鉴别力描述为主。"
        )

    raw_tags = last_move_tag_strings(state)
    tags = tags_consonant_with_geometry(raw_tags, geom)
    tag_calibration = ""
    if raw_tags and tags != raw_tags:
        dropped = [x for x in raw_tags if x not in tags]
        tag_calibration = (
            "\n【标签校准】下列 last_move 标签与 GTP 坐标核算的典型点名抵触，已从要求中移除："
            + "、".join(str(x) for x in dropped if str(x).strip())
            + "。请以 computed_last_move_geometry 为准，勿强行写入矛盾用语。"
        )
    tag_rule = ""
    if tags:
        tag_rule = (
            "\n【本手标签】state.last_move 携带标签（已按坐标校准），move_location_hint **必须**显式纳入下列至少一项语义描述："
            + "、".join(tags)
        )

    seed = input_move_location_hint_seed(sample)
    seed_rule = ""
    if seed:
        seed_json = json.dumps(seed, ensure_ascii=False)
        if geom and _seed_contradicts_geometry(seed, geom):
            seed_rule = (
                f"\n【落点锚点】输入曾给出 {seed_json}，但与 **computed_last_move_geometry** 中的点位类型矛盾，"
                "**以程序核算为准**，勿再按错误锚点写类型（如不得写三三）；可保留单字着法类锚点（如「挂」）若与局面相容。"
            )
        else:
            seed_rule = (
                f"\n【落点锚点·扩写】输入里 semantic_context.move_location_hint 已给出短提示 {seed_json}："
                "请先对照下文【move_location_hint 短标签释义】理解其围棋含义，再动笔。"
                "生成的 **move_location_hint** 必须以该用语为语义核心，全文须**可检索地包含**与之完全相同的子串"
                "（允许嵌入更长词中，如「挂」须出现在「挂角」「小飞挂」等）；"
                "在此基础上**扩写**为高鉴别力完整的落点说明，**禁止**丢弃该锚点改写无关位置；仍遵守汉字上限与禁复述 last_move 的 GTP。"
            )

    phase = str(state.get("game_phase", "")).strip()
    opening_hint = ""
    if phase == "opening":
        opening_hint = (
            "\n【布局期】game_phase=opening：若有上一手，move_location_hint **务必写满「角别+点位类型」或等价唯一描述**"
            "（如「左上角星位/小目/三三/高目/目外」），避免笼统「占角」导致多解；以落子前空枰为参照，"
            "使下游模型读hint即能还原该角上该典型点的一手；禁止写 last_move 的 GTP，禁止写成对下一手的选点建议。"
        )
    early_gs = _is_very_early_opening(state)
    global_summary_field_guide = ""
    if early_gs:
        global_summary_field_guide = (
            "\n【global_summary·布局极早】"
            f"当前为开局早期（game_phase=opening 且 move_number≤{EARLY_OPENING_MAX_MOVE}）："
            "在五档大势标签后，**只用极短语句收束**（一两小句即可），例如空枰、占角/挂角争夺尚浅、双方尚未短兵相接等；"
            "**不要**用多句并列铺陈模样、不要详写「关键局部」的长篇分析——过早局面可判信息少，宁短勿繁。"
        )

    no_last = not _has_concrete_last_move(state)
    first_move_rule = ""
    if no_last:
        first_move_rule = (
            "\n【首手/无上一手】当前无具体上一手落点："
            "global_summary 仍须以五档大势之一**开头**，可写空枰/即将开局语境；"
            "move_goal 写局面意图与下一手方（见 state.to_play）的常见构思，勿冒充「上一手」；"
            "move_location_hint 在此情形下才允许泛指**即将行棋方**可能考虑的区域（角部星位/三三等或「棋盘尚空」）；"
            "与「有上一手」时**只描述上一手落点**的规则不同。"
            "move_style 通常为「中性」。"
        )

    ex = {
        "global_summary": "势均力敌，模样与实地尚散，争夺在左上与右下两角。",
        "move_goal": "上一手占角争先：黑得要点次序紧，白宜挂角或分投破黑阵势。",
        "move_location_hint": "空枰：右上星位四四占角，距角顶标准星位；非小目非三三。",
        "move_style": "中性",
    }

    move_goal_detail = ""
    move_location_framing = ""
    if _has_concrete_last_move(state):
        move_goal_detail = (
            "紧扣 **state.last_move（上一手）**：说明为何下在此处、为何重要；"
            "并**分别**写清对**黑棋**与**对白棋**的影响（可用短分句）。"
        )
        move_location_framing = (
            "- **时间参照**：以 **上一手落下之前** 的盘面为准（该交叉点当时为空）；不要用「落子后」才出现的接触战形状来写落点，除非只为说明方位。"
            "\n- **强指示性（核心）**：本句是下游模型的**落点检索 prompt**。描述须使熟悉 GTP 的读者在落子前子力分布下，把候选交叉点压到**唯一**或**极少数相邻等价点**；"
            "优先写：**哪一角/哪条边侧、第几路感（相对角或边的典型点名）、与落子前最近棋块的相对方位与距离**（贴、托、断点、气的位置等职业用语）。"
            "\n- **布局**：必须给出**角别 + 星/小目/三三/高目/目外**等之一，或与之**等价且单义**的表述；禁止仅写「占角」「挂角」而不说清点位。"
            "\n- **中盘/接触**：相对**具体棋块/关键断点/眼形缺陷**等写清「落在某块之某侧/某条延伸线上」，必要时用**非 last_move** 的现有子 GTP 作锚（见【防泄露】）。"
            "\n- **禁止任务漂移**：不得写 **to_play** 应下何处、「下一手常见」与**尚未落下**之着的引导；不写对手应手预测。"
            f"\n- **长度**：与其它字段相同，**汉字数硬性≤{SEMANTIC_FIELD_MAX_HAN}**（计数同【输出】）；"
            "须高密度、去赘语，优先鉴别信息。"
        )

    global_summary_bullets = (
        "- 接一句：简要说明为何作此大势判断。\n"
        "- 再接一句：全局中**极为重要的一处局部**，用极精炼语言点出（可写区域+性质）。"
        if not early_gs
        else "- 开局尚早时大势往往接近「势均力敌」或轻微倾向即可；标签后**一两句短说明**即够，勿再多段展开。"
    )

    rules = f"""
{stance}
{leak_rule}
{tag_rule}
{tag_calibration}
{seed_rule}
{opening_hint}
{first_move_rule}
{global_summary_field_guide}
{geometry_rule}

{GO_TERMINOLOGY_GUIDE}

{MOVE_LOCATION_HINT_TAG_MEANINGS}

{GTP_TYPICAL_CORNER_POINTS}

【输出】仅一个顶层键 semantic_context，且**仅含**下列四键（均为字符串）：{", ".join(SEMANTIC_KEYS)}。
**硬性**：四字段每个字符串**仅计汉字个数**（U+4E00–U+9FFF），每字段≤{SEMANTIC_FIELD_MAX_HAN}；标点、英文、数字不计入。超出则校验失败，须删繁就简。

【global_summary】
- **第一句**必须以以下之一**开头**（其后可用逗号、句号隔开）：{"、".join(GLOBAL_ADVANTAGE_LABELS)}。
- 全文（含标签后补充）汉字总数≤{SEMANTIC_FIELD_MAX_HAN}。
{global_summary_bullets}

【move_goal】
- 若有具体上一手，则说明**上一手**的战略/战术意图与对局面的影响（见下条）；勿写成仅指导下一手方下棋。
{move_goal_detail}

【move_location_hint】
- **禁止**写出与 state.last_move.point **相同**的 GTP（若有上一手具体点）；**目标**仍是使读者能还原**同一手或极相近点**。
- 若有上一手具体点：本段是**高鉴别力落点说明**（落子前盘面视角），供下游模型复现 last_move，**不是**下一手预测。
{move_location_framing}
- 若有本手标签（见上文），**必须**使用。
- 若输入已给「落点锚点」（见【落点锚点·扩写】），**必须**含该子串并据此扩写。
- 无上一手时：见上文【首手/无上一手】（可泛指即将行棋方区域）。
- 有上一手且非空枰：须**尽量利用**落子前已有子的相对位置（必要时用其它点的 GTP 锚定，见【防泄露】），避免单凭「中腹一带」类弱描述。

【move_style】
- **必须且仅能**为以下之一（整段字符串完全一致）：{"、".join(MOVE_STYLE_LEVELS)}。

【禁止】
- 任一 semantic 字段汉字个数超过 {SEMANTIC_FIELD_MAX_HAN}。
- 在任一字段写出与 **state.last_move.point** 相同的 GTP（若有上一手坐标）；其它点的 GTP 仅在有助于说明时**可**谨慎使用（但仍勿复述 last_move）。
- **move_location_hint** 在有上一手时写成对 **to_play** 的「下一手应对/选点」预测（须只写 last_move 落点何在，以落子前空枰为参照）。
- 按 state.recent_moves **逐条复述**手顺。
- 立场错误：「我方/对方」与 **state.last_move.color**（无 last_move 时按 state.to_play）不一致。

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


def validate(sample: Dict[str, Any], data: Dict[str, Any]) -> None:
    data = norm_root(data)
    sc = data.get("semantic_context")
    if not isinstance(sc, dict):
        raise SemanticFormatError("need semantic_context object")

    extra = set(sc.keys()) - set(SEMANTIC_KEYS)
    if extra:
        raise SemanticFormatError(f"unexpected keys in semantic_context: {sorted(extra)}")

    for k in SEMANTIC_KEYS:
        if k not in sc:
            raise SemanticFormatError(f"missing {k}")

    state = sample.get("state") or {}
    forbid_pt = last_move_point_to_avoid_in_semantics(state)
    pat = _last_move_leak_pattern(forbid_pt) if forbid_pt else None

    for k in SEMANTIC_KEYS:
        sc[k] = _coerce_semantic_string(k, sc[k])
        if not sc[k]:
            raise SemanticFormatError(f"字段 {k} 不能为空")
        han = semantic_han_char_count(sc[k])
        if han > SEMANTIC_FIELD_MAX_HAN:
            raise SemanticFormatError(
                f"字段 {k} 超过硬性上限 {SEMANTIC_FIELD_MAX_HAN} 个汉字（当前 {han}），须压缩。"
                f"原文：{sc[k][:120]!r}{'…' if len(sc[k]) > 120 else ''}"
            )
        if pat is not None and pat.search(sc[k]):
            raise LastMoveSemanticLeakError(
                f"字段 {k} 不得包含上一手坐标 {forbid_pt!r}（与 state.last_move 重复）。原文：{sc[k]!r}"
            )

    gs = sc["global_summary"]
    if not _global_summary_starts_with_tier(gs):
        raise SemanticFormatError(
            f"global_summary 必须以以下之一开头：{GLOBAL_ADVANTAGE_LABELS}。当前：{gs[:80]!r}…"
        )

    ms = sc["move_style"]
    if ms not in MOVE_STYLE_LEVELS:
        raise SemanticFormatError(
            f"move_style 必须是以下之一：{MOVE_STYLE_LEVELS}。当前：{ms!r}"
        )

    geom = last_move_typical_point_geometry(state)
    tags = tags_consonant_with_geometry(last_move_tag_strings(state), geom)
    if tags:
        hint = sc["move_location_hint"]
        if not any(t in hint for t in tags):
            raise SemanticFormatError(
                f"move_location_hint 须显式包含 last_move 标签之一：{tags}。当前：{hint!r}"
            )

    seed = input_move_location_hint_seed(sample)
    if seed and not _seed_contradicts_geometry(seed, geom):
        hint = sc["move_location_hint"]
        if seed not in hint:
            raise SemanticFormatError(
                f"move_location_hint 须基于输入 semantic_context.move_location_hint 锚点扩写，"
                f"成文须包含子串 {seed!r}。当前：{hint!r}"
            )

    data["semantic_context"] = sc


def _leak_correction_prefix(forbid_pt: str | None, last_err: LastMoveSemanticLeakError) -> str:
    pt_hint = f"（禁止点位：{forbid_pt}）" if forbid_pt else ""
    return (
        f"【校验未通过】{last_err}{pt_hint}\n"
        "你必须重新输出**一份完整**的 JSON：仅顶层键 semantic_context，且仅含四个字符串字段；"
        "**不得再出现**与 state.last_move.point 相同的 GTP。用语义方位与棋形描述。原任务如下（覆盖上一轮）：\n\n"
    )


def _format_correction_prefix(err: Exception) -> str:
    return (
        f"【校验未通过】{err}\n"
        "你必须重新输出**一份完整**的 JSON：仅顶层键 semantic_context，"
        f"且字段为 {list(SEMANTIC_KEYS)}。"
        f"global_summary 须以 {GLOBAL_ADVANTAGE_LABELS} 之一开头；"
        f"move_style 须为 {MOVE_STYLE_LEVELS} 之一；"
        f"每字段汉字≤{SEMANTIC_FIELD_MAX_HAN}字。原任务如下（覆盖上一轮）：\n\n"
    )


def call_api(
    client: OpenAI,
    sample: Dict[str, Any],
    *,
    source_label: str = "",
    metrics: Optional[ApiCallMetrics] = None,
) -> Dict[str, Any]:
    state = sample.get("state") or {}
    forbid_pt = last_move_point_to_avoid_in_semantics(state)
    last_leak: LastMoveSemanticLeakError | None = None
    correction = ""
    loc = f" {source_label}" if source_label else ""
    m = metrics if metrics is not None else ApiCallMetrics()

    total_attempts = 1 + MAX_LEAK_RETRIES
    for attempt in range(total_attempts):
        base_prompt = build_prompt(sample)
        user_content = correction + base_prompt if correction else base_prompt
        try:
            t0 = time.perf_counter()
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": user_content},
                ],
            )
            m.wall_seconds += time.perf_counter() - t0
            m.num_requests += 1
            pt, ct, tt = _usage_from_completion(completion)
            m.prompt_tokens += pt
            m.completion_tokens += ct
            m.total_tokens += tt
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
            print(
                f"  [防泄露重试 {attempt + 2}/{total_attempts}]{loc}：{e}",
                file=sys.stderr,
            )
        except SemanticFormatError as e:
            if attempt >= total_attempts - 1:
                locf = f" 文件：{source_label}" if source_label else ""
                print(
                    f"  [本文件 API] 请求 {m.num_requests} 次, tok 累计 {m.total_tokens} "
                    f"(prompt {m.prompt_tokens}, completion {m.completion_tokens}), 耗时 {_fmt_duration(m.wall_seconds)}",
                    file=sys.stderr,
                )
                raise SemanticFormatError(
                    f"输出格式校验失败（已重试 {total_attempts} 次）。{locf}\n最后错误：{e}"
                ) from e
            correction = _format_correction_prefix(e)
            print(
                f"  [格式重试 {attempt + 2}/{total_attempts}]{loc}：{e}",
                file=sys.stderr,
            )

    assert last_leak is not None
    locf = f" 文件：{source_label}" if source_label else ""
    print(
        f"  [本文件 API] 请求 {m.num_requests} 次, tok 累计 {m.total_tokens} "
        f"(prompt {m.prompt_tokens}, completion {m.completion_tokens}), 耗时 {_fmt_duration(m.wall_seconds)}",
        file=sys.stderr,
    )
    raise LastMoveLeakExhaustedError(
        f"state.last_move 坐标防泄露：共请求 {total_attempts} 次（含首次）仍不合格。{locf}\n最后错误：{last_leak}"
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
    ap = argparse.ArgumentParser(
        description="从 outputs 的 step_*.json 生成语义字段，写入 prompted_json（默认路径可改）"
    )
    ap.add_argument(
        "--input-root",
        type=Path,
        default=Path("./outputs"),
        help="含各对局子目录与 step_*.json 的根目录（默认 ./outputs）",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("./prompted_json"),
        help="写入合并 semantic_context 后的 JSON 根目录（默认 ./prompted_json）",
    )
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
    total_api = ApiCallMetrics()
    t_batch = time.perf_counter()
    n = len(steps)

    for i, src in enumerate(steps, 1):
        dst = args.output_root / src.relative_to(root)
        t_step = time.perf_counter()
        step_m = ApiCallMetrics()
        try:
            sample = load_json(src)
            if DRY_RUN:
                save_json(dst, sample)
            else:
                assert client is not None
                gen = call_api(client, sample, source_label=str(src), metrics=step_m)
                save_json(dst, merge(sample, gen))
                total_api += step_m
            ok += 1
            step_elapsed = time.perf_counter() - t_step
            elapsed_total = time.perf_counter() - t_batch
            rem_files = n - i
            if ok > 0 and rem_files > 0:
                eta_sec = (elapsed_total / i) * rem_files
            else:
                eta_sec = 0.0

            tok_part = ""
            if DRY_RUN:
                tok_part = "tok —"
            else:
                tok_part = (
                    f"tok 本步 {step_m.total_tokens} (累计 {total_api.total_tokens}, "
                    f"{step_m.num_requests} 次请求)"
                )

            print(
                f"[{i}/{n}] ok {src.name} | "
                f"本步 {_fmt_duration(step_elapsed)} | "
                f"{tok_part} | "
                f"已用 {_fmt_duration(elapsed_total)} | "
                f"预计剩余 {_fmt_duration(eta_sec) if rem_files > 0 else '0s'}"
            )
        except LastMoveLeakExhaustedError as e:
            print(f"[{i}/{len(steps)}] FATAL {src}:\n{e}", file=sys.stderr)
            raise SystemExit(1) from e
        except Exception as e:
            fail += 1
            step_elapsed = time.perf_counter() - t_step
            print(
                f"[{i}/{len(steps)}] FAIL {src}: {e} | "
                f"本步 {_fmt_duration(step_elapsed)}",
                file=sys.stderr,
            )

    wall_total = time.perf_counter() - t_batch
    if DRY_RUN:
        print(f"done: {ok} ok, {fail} failed, total {len(steps)} | 总耗时 {_fmt_duration(wall_total)} (DRY_RUN 无 API)")
    else:
        print(
            f"done: {ok} ok, {fail} failed, total {len(steps)} | "
            f"总耗时 {_fmt_duration(wall_total)} | "
            f"API 请求共 {total_api.num_requests} 次 | "
            f"token 累计 prompt {total_api.prompt_tokens}, completion {total_api.completion_tokens}, "
            f"total {total_api.total_tokens}"
        )


if __name__ == "__main__":
    main()
