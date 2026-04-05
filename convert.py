#!/usr/bin/env python3
"""
Convert SGF games to simpleproto-style training samples.

Output format: meta / rules / state / semantic_context.
semantic_context 各字段均为空字符串（可由 API 等后续填充）。`label_space` 不写入。

Coordinate conversion:
- SGF uses a..s (includes i)
- Output uses GTP columns A..T without I
By default, each game is written under its own folder:
    <output-dir>/<sgf_stem>/step_0001.json

Batch limits and paths are controlled by module-level LIMIT_GAMES, INPUT_DIR, OUTPUT_DIR, etc.
(near the imports). Command-line flags override those defaults.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# =============================================================================
# 转换配置：直接改这里的变量即可（命令行参数仍可覆盖同名项）
# =============================================================================
LIMIT_GAMES: Optional[int] = 5  # 只处理前 N 局；None = 目录内全部 SGF（排序后截取）
INPUT_DIR = Path("./sgf")
OUTPUT_DIR = Path("./simpleproto_from_sgf")
RECENT_K = 6
MIN_MOVE_NUMBER = 0

BLACK = 1
WHITE = 0
SGF_LETTERS = "abcdefghijklmnopqrstuvwxyz"
GTP_COLS_NO_I = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def empty_semantic_context() -> dict:
    """Same keys as simpleproto.json."""
    return {
        "global_summary": "",
        "move_goal": "",
        "move_location_hint": "",
        "move_style": "",
    }


def color_to_side(color: int) -> str:
    return "black" if color == BLACK else "white"


def game_phase_for(move_idx: int) -> str:
    if move_idx < 40:
        return "opening"
    if move_idx < 160:
        return "middle_game"
    return "end_game"


def render_progress(
    done: int,
    total: int,
    converted_games: int,
    total_samples: int,
    width: int = 30,
) -> None:
    if total <= 0:
        return
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100
    print(
        f"\r[{bar}] {done}/{total} {percent:6.2f}% | converted={converted_games} | samples={total_samples}",
        end="",
        flush=True,
    )


@dataclass
class Move:
    color: int
    point: Optional[Tuple[int, int]]  # None = pass


def parse_props(node_text: str) -> Dict[str, List[str]]:
    """
    Parse SGF properties from one node text.
    Returns: {prop_ident: [value1, value2, ...]}
    """
    props: Dict[str, List[str]] = {}
    i = 0
    n = len(node_text)

    while i < n:
        while i < n and node_text[i].isspace():
            i += 1
        if i >= n or not node_text[i].isalpha():
            i += 1
            continue

        start = i
        while i < n and node_text[i].isalpha():
            i += 1
        ident = node_text[start:i]
        values: List[str] = []

        while i < n and node_text[i] == "[":
            i += 1
            value_chars: List[str] = []
            while i < n:
                ch = node_text[i]
                if ch == "\\" and i + 1 < n:
                    # SGF escaping: next char literal
                    value_chars.append(node_text[i + 1])
                    i += 2
                    continue
                if ch == "]":
                    i += 1
                    break
                value_chars.append(ch)
                i += 1
            values.append("".join(value_chars))

        if values:
            props.setdefault(ident, []).extend(values)

    return props


def parse_sgf(
    sgf_text: str,
) -> Tuple[Dict[str, List[str]], List[Move], Dict[Tuple[int, int], int]]:
    """
    Lightweight SGF parser:
    - root properties from the first node
    - moves from B[]/W[] on the main sequence (all non-root nodes in file order)
    For waltheri-like linear games, this is enough.
    """
    content = sgf_text.strip()
    if not content.startswith("("):
        raise ValueError("Invalid SGF: missing '('")

    # Split node texts by ';' that are outside property brackets
    nodes: List[str] = []
    in_bracket = False
    escaped = False
    cur: List[str] = []

    for ch in content:
        if escaped:
            cur.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_bracket:
            cur.append(ch)
            escaped = True
            continue
        if ch == "[" and not escaped:
            in_bracket = True
            cur.append(ch)
            continue
        if ch == "]" and not escaped:
            in_bracket = False
            cur.append(ch)
            continue
        if ch == ";" and not in_bracket:
            if cur:
                node_text = "".join(cur).strip()
                if node_text and node_text not in {"(", ")"}:
                    nodes.append(node_text)
            cur = []
            continue
        cur.append(ch)

    if cur:
        node_text = "".join(cur).strip()
        if node_text and node_text not in {"(", ")"}:
            nodes.append(node_text)

    if not nodes:
        raise ValueError("Invalid SGF: no nodes")

    root_props = parse_props(nodes[0])
    size = int(root_props.get("SZ", ["19"])[0])

    init_board: Dict[Tuple[int, int], int] = {}
    for p in root_props.get("AB", []):
        xy = sgf_to_xy(p, size)
        if xy is not None:
            init_board[xy] = BLACK
    for p in root_props.get("AW", []):
        xy = sgf_to_xy(p, size)
        if xy is not None:
            init_board[xy] = WHITE
    for p in root_props.get("AE", []):
        xy = sgf_to_xy(p, size)
        if xy is not None:
            init_board.pop(xy, None)

    moves: List[Move] = []

    for node in nodes[1:]:
        props = parse_props(node)
        if "B" in props:
            point = sgf_to_xy(props["B"][0], size)
            moves.append(Move(color=BLACK, point=point))
        elif "W" in props:
            point = sgf_to_xy(props["W"][0], size)
            moves.append(Move(color=WHITE, point=point))

    return root_props, moves, init_board


def sgf_to_xy(s: str, size: int) -> Optional[Tuple[int, int]]:
    """
    SGF point to board coords (x, y):
    - SGF aa = top-left
    - internal y=0 means bottom row
    """
    if not s or s == "tt":
        return None
    if len(s) != 2:
        return None

    sx, sy = s[0], s[1]
    if sx not in SGF_LETTERS or sy not in SGF_LETTERS:
        return None

    x_top = ord(sx) - ord("a")
    y_top = ord(sy) - ord("a")
    if not (0 <= x_top < size and 0 <= y_top < size):
        return None
    y_bottom = size - 1 - y_top
    return (x_top, y_bottom)


def to_gtp(x: int, y: int, size: int) -> str:
    col = GTP_COLS_NO_I[x]
    row = y + 1
    return f"{col}{row}"


def nbs(x: int, y: int, size: int) -> List[Tuple[int, int]]:
    pts = []
    if x > 0:
        pts.append((x - 1, y))
    if x < size - 1:
        pts.append((x + 1, y))
    if y > 0:
        pts.append((x, y - 1))
    if y < size - 1:
        pts.append((x, y + 1))
    return pts


def group_libs(
    board: Dict[Tuple[int, int], int], start: Tuple[int, int], size: int
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    color = board[start]
    stack = [start]
    group: Set[Tuple[int, int]] = set()
    libs: Set[Tuple[int, int]] = set()

    while stack:
        p = stack.pop()
        if p in group:
            continue
        group.add(p)
        for nb in nbs(p[0], p[1], size):
            if nb not in board:
                libs.add(nb)
            elif board[nb] == color and nb not in group:
                stack.append(nb)
    return group, libs


def apply_move(board: Dict[Tuple[int, int], int], move: Move, size: int) -> None:
    if move.point is None:
        return

    x, y = move.point
    board[(x, y)] = move.color
    opp = WHITE if move.color == BLACK else BLACK

    # Capture adjacent opponent groups without liberties
    for nb in nbs(x, y, size):
        if nb in board and board[nb] == opp:
            grp, libs = group_libs(board, nb, size)
            if not libs:
                for p in grp:
                    board.pop(p, None)

    # Handle suicide if present in source data/rules
    if (x, y) in board and board[(x, y)] == move.color:
        own_grp, own_libs = group_libs(board, (x, y), size)
        if not own_libs:
            for p in own_grp:
                board.pop(p, None)


def norm_date(raw: str) -> str:
    # Keep YYYY-MM-DD when available, fallback raw string
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", raw or "")
    if not m:
        return raw or ""
    y, mm, dd = m.group(1), int(m.group(2)), int(m.group(3))
    return f"{y}-{mm:02d}-{dd:02d}"


def safe_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", name).strip()
    return cleaned[:180] if cleaned else "unknown_game"


def short_name(stem: str, idx: int) -> str:
    m = re.match(r"^(\d+)", stem)
    if m:
        return f"g{int(m.group(1)):05d}"
    return f"g{idx:05d}"


def unique_folder_name(base: str, used: Set[str], idx: int) -> str:
    name = base or f"game_{idx:05d}"
    if name not in used:
        used.add(name)
        return name
    alt = f"{name}__{idx:05d}"
    used.add(alt)
    return alt


def point_to_gtp(point: Optional[Tuple[int, int]], size: int) -> str:
    if point is None:
        return "PASS"
    return to_gtp(point[0], point[1], size)


def move_to_state_record(mv: Move, size: int) -> dict:
    """Structured move: GTP point + side + pass flag (same shape as last_move entries)."""
    pt = point_to_gtp(mv.point, size)
    return {
        "color": color_to_side(mv.color),
        "point": pt,
        "is_pass": bool(mv.point is None or pt == "PASS"),
    }


def build_sample(
    root: Dict[str, List[str]],
    board: Dict[Tuple[int, int], int],
    moves: List[Move],
    move_idx: int,
    game_stem: str,
    recent_k: int,
) -> Optional[dict]:
    """
    Build sample at position after moves[0..move_idx-1], target is moves[move_idx].
    """
    prev = moves[move_idx - 1] if move_idx > 0 else None
    target = moves[move_idx]

    size = int(root.get("SZ", ["19"])[0])
    if size > len(GTP_COLS_NO_I):
        raise ValueError(f"Board size {size} not supported by GTP no-I columns")

    black_stones = sorted(
        (to_gtp(x, y, size) for (x, y), c in board.items() if c == BLACK),
        key=lambda p: (int(p[1:]), p[0]),
    )
    white_stones = sorted(
        (to_gtp(x, y, size) for (x, y), c in board.items() if c == WHITE),
        key=lambda p: (int(p[1:]), p[0]),
    )

    recent_start = max(0, move_idx - recent_k)
    recent_moves: List[dict] = [
        move_to_state_record(mv, size) for mv in moves[recent_start:move_idx]
    ]

    last_move = move_to_state_record(prev, size) if prev is not None else None

    state_block: Dict[str, Any] = {
        "move_number": move_idx,
        "to_play": color_to_side(target.color),
        "game_phase": game_phase_for(move_idx),
        "last_move": last_move,
        "recent_moves": recent_moves,
        "black_stones": black_stones,
        "white_stones": white_stones,
    }

    sample = {
        "meta": {
            "sample_id": f"game_{game_stem}_move_{move_idx:03d}",
            "game_name": root.get("GN", [game_stem])[0],
            "event": root.get("EV", [""])[0],
            "date": norm_date(root.get("DT", [""])[0]),
            "black_player": root.get("PB", [""])[0],
            "white_player": root.get("PW", [""])[0],
        },
        "rules": {
            "board_size": size,
            "komi": float(root.get("KM", ["0"])[0] or 0),
            "result": root.get("RE", [""])[0],
        },
        "state": state_block,
        "semantic_context": empty_semantic_context(),
    }
    return sample


def convert_game(
    sgf_path: Path,
    recent_k: int,
    min_move_number: int,
) -> List[dict]:
    text = sgf_path.read_text(encoding="utf-8", errors="ignore")
    root, moves, init_board = parse_sgf(text)
    size = int(root.get("SZ", ["19"])[0])
    if size < 1 or size > len(GTP_COLS_NO_I):
        return []
    if len(moves) < 2:
        return []

    board: Dict[Tuple[int, int], int] = dict(init_board)
    samples: List[dict] = []

    # move_idx means "next move index", current state is after move_idx moves played
    # so we apply moves[0..move_idx-1] incrementally.
    for move_idx in range(0, len(moves)):
        if move_idx > 0:
            apply_move(board, moves[move_idx - 1], size)
        if move_idx < min_move_number:
            continue
        sample = build_sample(root, board, moves, move_idx, sgf_path.stem, recent_k)
        if sample is not None:
            samples.append(sample)

    return samples


def list_sgfs(input_dir: Path) -> List[Path]:
    return sorted(input_dir.rglob("*.sgf"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SGF games to simpleproto samples")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing SGF files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root output directory, one subfolder per game",
    )
    parser.add_argument(
        "--limit-games",
        type=int,
        default=LIMIT_GAMES,
        help="Limit number of SGF files; defaults to module LIMIT_GAMES if omitted",
    )
    parser.add_argument(
        "--recent-k",
        type=int,
        default=RECENT_K,
        help="Number of recent moves to keep in state.recent_moves",
    )
    parser.add_argument(
        "--min-move-number",
        type=int,
        default=MIN_MOVE_NUMBER,
        help="Only emit samples with move_number >= this value",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar output",
    )
    args = parser.parse_args()

    sgf_files = list_sgfs(args.input_dir)
    if args.limit_games is not None:
        sgf_files = sgf_files[: args.limit_games]

    total_samples = 0
    converted_games = 0

    name_map: Dict[str, str] = {}
    used_folder_names: Set[str] = set()

    total_games = len(sgf_files)
    if total_games and not args.no_progress:
        render_progress(0, total_games, converted_games, total_samples)

    for i, sgf_path in enumerate(sgf_files, start=1):
        try:
            samples = convert_game(
                sgf_path=sgf_path,
                recent_k=args.recent_k,
                min_move_number=args.min_move_number,
            )
            if samples:
                folder_name = unique_folder_name(
                    safe_name(sgf_path.stem),
                    used_folder_names,
                    i,
                )
                game_folder = args.output_dir / folder_name
                game_folder.mkdir(parents=True, exist_ok=True)

                # Clean previous outputs for this game folder.
                for old in game_folder.glob("step_*.json"):
                    old.unlink(missing_ok=True)
                (game_folder / "index.json").unlink(missing_ok=True)

                sample_files: List[str] = []
                for j, item in enumerate(samples, start=1):
                    file_name = f"step_{j:04d}.json"
                    sample_files.append(file_name)
                    (game_folder / file_name).write_text(
                        json.dumps(item, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                (game_folder / "index.json").write_text(
                    json.dumps(
                        {
                            "game_folder": folder_name,
                            "source_sgf": sgf_path.name,
                            "samples": len(sample_files),
                            "files": sample_files,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                name_map[folder_name] = safe_name(sgf_path.stem)
                converted_games += 1
                total_samples += len(samples)
        except Exception:
            # Keep conversion robust for large datasets with rare malformed SGF.
            continue
        finally:
            if total_games and not args.no_progress:
                render_progress(i, total_games, converted_games, total_samples)

    if name_map:
        map_file = args.output_dir / "games_map.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        map_file.write_text(
            json.dumps(name_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if total_games and not args.no_progress:
        print()

    print(f"Games scanned: {len(sgf_files)}")
    print(f"Games converted: {converted_games}")
    print(f"Samples written: {total_samples}")
    print(f"Output root: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
