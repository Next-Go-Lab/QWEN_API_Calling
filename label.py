
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

# 19路围棋列坐标（跳过 I）
COL_LABELS = "ABCDEFGHJKLMNOPQRST"
COL_TO_IDX = {ch: i for i, ch in enumerate(COL_LABELS)}

# 棋盘编码
EMPTY = 0
BLACK = 1
WHITE = 2
PADDING = -1
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "simpleproto_from_sgf"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def parse_coord(coord: str, board_size: int = 19) -> Tuple[int, int]:
    """
    解析围棋坐标（如 M2）为棋盘索引 (row, col)。
    row=0 表示最上方（19线），row=18 表示最下方（1线）。
    """
    if not isinstance(coord, str):
        raise ValueError(f"Invalid coord type: {coord!r}")

    text = coord.strip().upper()
    m = re.fullmatch(r"([A-T])([1-9]|1[0-9])", text)
    if not m:
        raise ValueError(f"Invalid coord format: {coord!r}")

    col_char, row_num_str = m.groups()
    if col_char == "I":
        raise ValueError(f"Invalid coord: column 'I' is skipped in Go: {coord!r}")
    if col_char not in COL_TO_IDX:
        raise ValueError(f"Column out of range for {board_size}x{board_size}: {coord!r}")

    row_num = int(row_num_str)
    if not (1 <= row_num <= board_size):
        raise ValueError(f"Row out of range for {board_size}x{board_size}: {coord!r}")

    col = COL_TO_IDX[col_char]
    row = board_size - row_num
    return row, col


def build_board(state: Dict, board_size: int = 19) -> List[List[int]]:
    """
    根据 state.black_stones / state.white_stones 构建棋盘：
    0=空, 1=黑, 2=白
    """
    board = [[EMPTY for _ in range(board_size)] for _ in range(board_size)]

    for pt in state.get("black_stones", []):
        r, c = parse_coord(pt, board_size)
        board[r][c] = BLACK

    for pt in state.get("white_stones", []):
        r, c = parse_coord(pt, board_size)
        board[r][c] = WHITE

    return board


def _collect_group_and_liberties_on_board(
    board: List[List[int]],
    start_r: int,
    start_c: int,
    target_stone: int,
) -> Tuple[set, set]:
    """
    在整盘 board 上收集 target_stone 连通块及其气。
    连通按正交四邻，气为 EMPTY 点。
    """
    n = len(board)
    if not (0 <= start_r < n and 0 <= start_c < n):
        return set(), set()
    if board[start_r][start_c] != target_stone:
        return set(), set()

    group = set()
    liberties = set()
    stack = [(start_r, start_c)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        r, c = stack.pop()
        if (r, c) in group:
            continue
        group.add((r, c))
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            cell = board[nr][nc]
            if cell == target_stone and (nr, nc) not in group:
                stack.append((nr, nc))
            elif cell == EMPTY:
                liberties.add((nr, nc))

    return group, liberties


def match_jiaochi_on_board(
    board: List[List[int]],
    move_r: int,
    move_c: int,
    my_color: str,
) -> bool:
    """
    整盘“叫吃”判定：
    与本手直接相邻的敌方连通块，若存在仅剩一口气，则判为叫吃。
    """
    my_stone = BLACK if my_color.lower() == "black" else WHITE
    enemy_stone = WHITE if my_stone == BLACK else BLACK
    n = len(board)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    seen_enemy = set()

    for dr, dc in dirs:
        er, ec = move_r + dr, move_c + dc
        if not (0 <= er < n and 0 <= ec < n):
            continue
        if board[er][ec] != enemy_stone or (er, ec) in seen_enemy:
            continue

        group, liberties = _collect_group_and_liberties_on_board(board, er, ec, enemy_stone)
        seen_enemy |= group
        if group and len(liberties) == 1:
            return True

    return False


def _previous_step_path(current_path: Path) -> Optional[Path]:
    """
    根据 step_XXXX.json 推导上一步文件路径。
    """
    m = re.fullmatch(r"step_(\d+)\.json", current_path.name)
    if not m:
        return None
    idx = int(m.group(1))
    if idx <= 1:
        return None
    prev_name = f"step_{idx - 1:04d}.json"
    prev_path = current_path.with_name(prev_name)
    return prev_path if prev_path.exists() else None


def match_ti_on_board(
    prev_board: List[List[int]],
    curr_board: List[List[int]],
    move_r: int,
    move_c: int,
    my_color: str,
) -> bool:
    """
    整盘“提/提子”判定：
    本手落在某直接相邻敌方连通块的最后一口气上，使该敌块被移除。
    """
    my_stone = BLACK if my_color.lower() == "black" else WHITE
    enemy_stone = WHITE if my_stone == BLACK else BLACK
    n = len(curr_board)
    if not (0 <= move_r < n and 0 <= move_c < n):
        return False

    # 当前盘面本手位置应为己子
    if curr_board[move_r][move_c] != my_stone:
        return False

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in dirs:
        er, ec = move_r + dr, move_c + dc
        if not (0 <= er < n and 0 <= ec < n):
            continue
        if prev_board[er][ec] != enemy_stone:
            continue

        group, liberties = _collect_group_and_liberties_on_board(prev_board, er, ec, enemy_stone)
        if not group or len(liberties) != 1:
            continue
        if (move_r, move_c) not in liberties:
            continue

        # 该敌方连通块在当前盘面应被整体提走
        if all(curr_board[gr][gc] == EMPTY for gr, gc in group):
            return True

    return False


def _enemy_component_count_around_point_on_board(
    board: List[List[int]],
    point_r: int,
    point_c: int,
    enemy_stone: int,
) -> int:
    """
    统计某空点四邻敌子在“忽略该空点”时分成的敌方连通分量数。
    """
    n = len(board)
    if not (0 <= point_r < n and 0 <= point_c < n):
        return 0
    if board[point_r][point_c] != EMPTY:
        return 0

    orth_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    seeds = []
    for dr, dc in orth_dirs:
        nr, nc = point_r + dr, point_c + dc
        if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == enemy_stone:
            seeds.append((nr, nc))
    if len(seeds) < 2:
        return 0

    enemy_cells = set()
    for r in range(n):
        for c in range(n):
            if board[r][c] == enemy_stone and (r, c) != (point_r, point_c):
                enemy_cells.add((r, c))

    comp_id: Dict[Tuple[int, int], int] = {}
    comp = 0
    for start in enemy_cells:
        if start in comp_id:
            continue
        comp += 1
        stack = [start]
        comp_id[start] = comp
        while stack:
            ur, uc = stack.pop()
            for dr, dc in orth_dirs:
                vr, vc = ur + dr, uc + dc
                if (vr, vc) in enemy_cells and (vr, vc) not in comp_id:
                    comp_id[(vr, vc)] = comp
                    stack.append((vr, vc))

    touching = {comp_id[pos] for pos in seeds if pos in comp_id}
    return len(touching)


def match_ji_on_board(
    prev_board: List[List[int]],
    curr_board: List[List[int]],
    move_r: int,
    move_c: int,
    my_color: str,
) -> bool:
    """
    挤：
    采用局部四宫格构型判定（与“断”区分）：
    - 在包含本手的某个 2x2 四宫格中，本手两条正交邻边均为敌子；
    - 但该四宫格对角点不是友子（即“断”里那枚对角友军缺失）。
    """
    my_stone = BLACK if my_color.lower() == "black" else WHITE
    enemy_stone = WHITE if my_stone == BLACK else BLACK
    n = len(curr_board)
    if not (0 <= move_r < n and 0 <= move_c < n):
        return False
    if curr_board[move_r][move_c] != my_stone:
        return False

    # 以本手为一个角，枚举四个可能 2x2 方向
    # (角对角方向) -> (两条邻边方向)
    patterns = [
        ((-1, -1), (-1, 0), (0, -1)),
        ((-1, 1), (-1, 0), (0, 1)),
        ((1, -1), (1, 0), (0, -1)),
        ((1, 1), (1, 0), (0, 1)),
    ]
    for (ddi, ddj), (e1i, e1j), (e2i, e2j) in patterns:
        r1, c1 = move_r + e1i, move_c + e1j
        r2, c2 = move_r + e2i, move_c + e2j
        rd, cd = move_r + ddi, move_c + ddj
        if not (0 <= r1 < n and 0 <= c1 < n and 0 <= r2 < n and 0 <= c2 < n and 0 <= rd < n and 0 <= cd < n):
            continue
        if curr_board[r1][c1] != enemy_stone or curr_board[r2][c2] != enemy_stone:
            continue
        if curr_board[rd][cd] != my_stone:
            return True
    return False


def extract_local_patch(board: List[List[int]], r: int, c: int, size: int = 7) -> List[List[int]]:
    """
    以 (r,c) 为中心截取 size x size 局部棋盘，越界填充 -1。
    """
    n = len(board)
    half = size // 2
    patch = []

    for dr in range(-half, half + 1):
        row_vals = []
        rr = r + dr
        for dc in range(-half, half + 1):
            cc = c + dc
            if 0 <= rr < n and 0 <= cc < n:
                row_vals.append(board[rr][cc])
            else:
                row_vals.append(PADDING)
        patch.append(row_vals)

    return patch


def encode_patch(patch: List[List[int]], my_color: str) -> List[List[str]]:
    """
    将局部 patch 编码为：
    S=本手, A=我方, E=敌方, .=空点, B=边界
    """
    size = len(patch)
    center = size // 2
    my_stone = BLACK if my_color.lower() == "black" else WHITE
    enemy_stone = WHITE if my_stone == BLACK else BLACK

    encoded = []
    for i in range(size):
        row_chars = []
        for j in range(size):
            val = patch[i][j]
            if i == center and j == center:
                row_chars.append("S")
            elif val == PADDING:
                row_chars.append("B")
            elif val == EMPTY:
                row_chars.append(".")
            elif val == my_stone:
                row_chars.append("A")
            elif val == enemy_stone:
                row_chars.append("E")
            else:
                row_chars.append("?")
        encoded.append(row_chars)
    return encoded


def _in_bounds(patch: List[List[str]], i: int, j: int) -> bool:
    n = len(patch)
    return 0 <= i < n and 0 <= j < n


def _cell(patch: List[List[str]], i: int, j: int) -> Optional[str]:
    if not _in_bounds(patch, i, j):
        return None
    return patch[i][j]


def _center(patch: List[List[str]]) -> Tuple[int, int]:
    n = len(patch)
    return n // 2, n // 2


def _collect_group_and_liberties(
    patch: List[List[str]],
    start_i: int,
    start_j: int,
    target: str,
) -> Tuple[set, set]:
    """
    从起点收集 target 连通块及其气（仅统计 '.' 为空点的气）。
    连通按正交四邻。
    """
    if _cell(patch, start_i, start_j) != target:
        return set(), set()

    group = set()
    liberties = set()
    stack = [(start_i, start_j)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        i, j = stack.pop()
        if (i, j) in group:
            continue
        group.add((i, j))
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            cell = _cell(patch, ni, nj)
            if cell == target and (ni, nj) not in group:
                stack.append((ni, nj))
            elif cell == ".":
                liberties.add((ni, nj))

    return group, liberties


def match_kao(patch: List[List[str]]) -> bool:
    """
    靠：S 与 E 正交相邻
    """
    ci, cj = _center(patch)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if _cell(patch, ci + di, cj + dj) == "E":
            return True
    return False


def match_jiaochi(patch: List[List[str]]) -> bool:
    """
    叫吃：与 S 直接相邻的敌方棋子所在连通整体，仅剩一口气。
    """
    ci, cj = _center(patch)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    seen_enemy = set()

    for di, dj in dirs:
        ei, ej = ci + di, cj + dj
        if _cell(patch, ei, ej) != "E" or (ei, ej) in seen_enemy:
            continue

        group, liberties = _collect_group_and_liberties(patch, ei, ej, target="E")
        seen_enemy |= group
        if group and len(liberties) == 1:
            return True

    return False


def match_ban(patch: List[List[str]]) -> bool:
    """
    扳：黑白子互相贴近时，一方从斜角向对方兜头下一子。
    判定为：
    - S 正交四邻中恰有一侧相邻敌子 E
    - S 的四邻不存在友子 A
    - 以 S 为中心九宫格内，四个斜角上恰好只有一个友子 A
    - 且该唯一斜角友子需与该相邻敌子也正交相邻（同处该侧两邻角之一）
    """
    ci, cj = _center(patch)
    orth_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    enemy_dirs: List[Tuple[int, int]] = []
    for di, dj in orth_dirs:
        cell = _cell(patch, ci + di, cj + dj)
        if cell == "E":
            enemy_dirs.append((di, dj))
        elif cell == "A":
            return False

    if len(enemy_dirs) != 1:
        return False

    corner_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    ally_corners = [(di, dj) for di, dj in corner_dirs if _cell(patch, ci + di, cj + dj) == "A"]
    if len(ally_corners) != 1:
        return False
    only_ally_corner = ally_corners[0]

    # 唯一相邻敌子所在一侧，且与唯一对角友子也相邻
    edi, edj = enemy_dirs[0]
    if edi != 0:  # 敌子在上下，邻角是同一行的左右角
        diag_candidates = [(edi, -1), (edi, 1)]
    else:  # 敌子在左右，邻角是同一列的上下角
        diag_candidates = [(-1, edj), (1, edj)]
    return only_ally_corner in diag_candidates


def match_peng(patch: List[List[str]]) -> bool:
    """
    碰：敌我都是单独一子，且与本手形成正交直接两连。
    规则细化：
    - 本手四邻没有友子 A（本手为单子）
    - 本手四邻恰好有一个敌子 E，且该敌子连通块大小为 1（敌子为单子）
    """
    ci, cj = _center(patch)
    orth_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    enemy_positions_orth: List[Tuple[int, int]] = []

    # 本手不能与友军正交相连（本手单子）
    for di, dj in orth_dirs:
        if _cell(patch, ci + di, cj + dj) == "A":
            return False

    for di, dj in orth_dirs:
        if _cell(patch, ci + di, cj + dj) == "E":
            enemy_positions_orth.append((ci + di, cj + dj))

    if len(enemy_positions_orth) != 1:
        return False

    ei, ej = enemy_positions_orth[0]
    enemy_group, _ = _collect_group_and_liberties(patch, ei, ej, target="E")
    return len(enemy_group) == 1


def match_jian(patch: List[List[str]]) -> bool:
    """
    尖：在以 S 为中心九宫格内，只有一个角位有友军；
    且该友军与 S 形成的 2x2 四宫格中，另外两点必须为空。
    """
    ci, cj = _center(patch)
    corner_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    ally_corners: List[Tuple[int, int]] = []
    for di, dj in corner_dirs:
        if _cell(patch, ci + di, cj + dj) == "A":
            ally_corners.append((di, dj))

    # 九宫格四角里只能有一个友军
    if len(ally_corners) != 1:
        return False

    di, dj = ally_corners[0]
    # 与该角位友军构成的 2x2 中，除 S 和友军外的两个点
    side_1 = _cell(patch, ci + di, cj)
    side_2 = _cell(patch, ci, cj + dj)
    return side_1 == "." and side_2 == "."


def match_hu(patch: List[List[str]]) -> bool:
    """
    虎：S 与两枚友军形成“品字型”三子结构。
    这里采用两类可旋转模板，按“方向独立”判定：
    - 两枚友军位于同侧两个斜角位
    - 或一枚友军在两格正交位，另一枚在其中点斜侧位（远近虎口）
    并要求：
    - 对应方向的虎口点必须为空点 '.'
    - 远近虎口还要求远端中点为空（避免把贴身战斗形误判成虎）
    说明：
    - 虎口内有任意棋子（友/敌）都不判“虎”。
    - 若某一方向虎口被堵，但另一方向模板成立，仍判“虎”。
    """
    ci, cj = _center(patch)
    # 模板一：标准虎口（两枚斜侧友军分列同侧）
    side_diagonal_patterns = [
        ((-1, 1), (1, 1), (0, 1)),     # 右侧虎口
        ((-1, -1), (1, -1), (0, -1)),  # 左侧虎口
        ((-1, -1), (-1, 1), (-1, 0)),  # 上侧虎口
        ((1, -1), (1, 1), (1, 0)),     # 下侧虎口
    ]

    long_short_patterns = [
        ((-2, 0), (-1, -1), (0, 1)),
        ((-2, 0), (-1, 1), (0, -1)),
        ((2, 0), (1, -1), (0, 1)),
        ((2, 0), (1, 1), (0, -1)),
        ((0, -2), (-1, -1), (1, 0)),
        ((0, -2), (1, -1), (-1, 0)),
        ((0, 2), (-1, 1), (1, 0)),
        ((0, 2), (1, 1), (-1, 0)),
    ]

    for (d1i, d1j), (d2i, d2j), (mdi, mdj) in side_diagonal_patterns:
        if not (_cell(patch, ci + d1i, cj + d1j) == "A" and _cell(patch, ci + d2i, cj + d2j) == "A"):
            continue

        mouth_i, mouth_j = ci + mdi, cj + mdj
        if _cell(patch, mouth_i, mouth_j) != ".":
            continue

        return True

    for (d1i, d1j), (d2i, d2j), (mdi, mdj) in long_short_patterns:
        if not (_cell(patch, ci + d1i, cj + d1j) == "A" and _cell(patch, ci + d2i, cj + d2j) == "A"):
            continue

        # 远端与 S 之间中点要空，避免“夹着敌子”的贴身战斗形误判为虎
        mid_i, mid_j = ci + d1i // 2, cj + d1j // 2
        if _cell(patch, mid_i, mid_j) != ".":
            continue

        mouth_i, mouth_j = ci + mdi, cj + mdj
        if _cell(patch, mouth_i, mouth_j) != ".":
            continue

        return True
    return False


def match_chang(patch: List[List[str]]) -> bool:
    """
    长：S 与 A 正交相邻
    """
    ci, cj = _center(patch)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if _cell(patch, ci + di, cj + dj) == "A":
            return True
    return False


def match_nian(patch: List[List[str]]) -> bool:
    """
    粘：本手四邻至少有两个友军，且这些友军在九宫格内彼此不连通。
    连通性按围棋常规“正交相连”判断（不把中心 S 作为已有友军参与连通）。
    """
    ci, cj = _center(patch)
    orth_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    orth_allies: List[Tuple[int, int]] = []
    for di, dj in orth_dirs:
        pi, pj = ci + di, cj + dj
        if _cell(patch, pi, pj) == "A":
            orth_allies.append((pi, pj))

    # 四邻至少两个友军
    if len(orth_allies) < 2:
        return False

    # 仅在以 S 为中心九宫格内检查已有友军连通性（不含中心 S）
    allies_in_3x3 = set()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            pi, pj = ci + di, cj + dj
            if _cell(patch, pi, pj) == "A":
                allies_in_3x3.add((pi, pj))

    # 对九宫格内友军做正交连通分量划分
    comp_id: Dict[Tuple[int, int], int] = {}
    current_comp = 0
    for start in allies_in_3x3:
        if start in comp_id:
            continue
        current_comp += 1
        stack = [start]
        comp_id[start] = current_comp
        while stack:
            ui, uj = stack.pop()
            for di, dj in orth_dirs:
                vi, vj = ui + di, uj + dj
                if (vi, vj) in allies_in_3x3 and (vi, vj) not in comp_id:
                    comp_id[(vi, vj)] = current_comp
                    stack.append((vi, vj))

    # 若四邻友军分属不同分量，则“粘”成立
    touching_comp_ids = {comp_id[pos] for pos in orth_allies if pos in comp_id}
    return len(touching_comp_ids) >= 2


def match_duan(patch: List[List[str]]) -> bool:
    """
    断（四宫格切断）：
    - 在以 S 为中心的九宫格内，存在一个包含 S 的 2x2 四宫格；
    - 该四宫格里：S 的对角为友子 A，且与 S 同行/同列的两点均为敌子 E；
    - 这两枚敌子在九宫格内（不经过中心 S）彼此不连通。
    """
    ci, cj = _center(patch)
    # (对角友子方向) -> (同四宫格内与 S 相邻的两敌子方向)
    corner_to_enemy_pairs = [
        ((-1, -1), [(-1, 0), (0, -1)]),
        ((-1, 1), [(-1, 0), (0, 1)]),
        ((1, -1), [(1, 0), (0, -1)]),
        ((1, 1), [(1, 0), (0, 1)]),
    ]
    for (cdi, cdj), [(e1i, e1j), (e2i, e2j)] in corner_to_enemy_pairs:
        corner = (ci + cdi, cj + cdj)
        e1 = (ci + e1i, cj + e1j)
        e2 = (ci + e2i, cj + e2j)
        if _cell(patch, corner[0], corner[1]) != "A":
            continue
        if _cell(patch, e1[0], e1[1]) != "E" or _cell(patch, e2[0], e2[1]) != "E":
            continue
        if not _connected_in_3x3_excluding_center(patch, e1, e2, target="E"):
            return True

    return False


def _connected_in_3x3_excluding_center(
    patch: List[List[str]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    target: str,
) -> bool:
    """
    在以 S 为中心九宫格内，判断两个点是否可通过 target 正交连通，
    且路径不允许经过中心 S。
    """
    ci, cj = _center(patch)
    if _cell(patch, start[0], start[1]) != target or _cell(patch, end[0], end[1]) != target:
        return False

    allowed = set()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            pi, pj = ci + di, cj + dj
            if (pi, pj) == (ci, cj):
                continue
            if _cell(patch, pi, pj) == target:
                allowed.add((pi, pj))

    if start not in allowed or end not in allowed:
        return False

    stack = [start]
    visited = set()
    orth_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while stack:
        i, j = stack.pop()
        if (i, j) in visited:
            continue
        visited.add((i, j))
        if (i, j) == end:
            return True
        for di, dj in orth_dirs:
            ni, nj = i + di, j + dj
            if (ni, nj) in allowed and (ni, nj) not in visited:
                stack.append((ni, nj))
    return False


def match_wa(patch: List[List[str]]) -> bool:
    """
    挖：
    - 本手与友方不正交直接相连（四邻无 A）
    - 两对侧之一（上-下 或 左-右）均为敌方 E
    - 该对侧敌子在九宫格内彼此不连通（表示被本手“挖断”其跳型联系）
    """
    ci, cj = _center(patch)
    orth_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if any(_cell(patch, ci + di, cj + dj) == "A" for di, dj in orth_dirs):
        return False

    opposite_pairs = [((ci - 1, cj), (ci + 1, cj)), ((ci, cj - 1), (ci, cj + 1))]
    for p1, p2 in opposite_pairs:
        if _cell(patch, p1[0], p1[1]) == "E" and _cell(patch, p2[0], p2[1]) == "E":
            if not _connected_in_3x3_excluding_center(patch, p1, p2, target="E"):
                return True
    return False


def match_fei(patch: List[List[str]]) -> bool:
    """
    飞：S 与 A 构成飞形，且两子中间区域（除两子所在短边外）无其它棋子
    """
    return _match_fei_like(patch, target="A")


def _fei_middle_cells(si: int, sj: int, ti: int, tj: int) -> Optional[List[Tuple[int, int]]]:
    """
    计算“飞形”两子之间需要为空的两点。
    若两子不构成 2x3 飞形（差值不是 2-1 或 1-2），返回 None。
    """
    di = ti - si
    dj = tj - sj
    adi, adj = abs(di), abs(dj)
    if not ((adi == 2 and adj == 1) or (adi == 1 and adj == 2)):
        return None

    # 纵向长边（2）+ 横向短边（1）
    if adi == 2 and adj == 1:
        mid_i = si + (1 if di > 0 else -1)
        return [(mid_i, sj), (mid_i, tj)]

    # 横向长边（2）+ 纵向短边（1）
    mid_j = sj + (1 if dj > 0 else -1)
    return [(si, mid_j), (ti, mid_j)]


def _middle_region_is_clear(
    patch: List[List[str]],
    si: int,
    sj: int,
    ti: int,
    tj: int,
) -> bool:
    """
    检查两子中间区域是否无棋子：
    - 仅允许空点 '.'
    - 若出现 A/E/S/B/越界等，视为不满足
    当前用于飞/挂（两子构成 2x3 矩形）：
    - 先判断短边：横坐标差 1 或纵坐标差 1
    - 再在长边中点对应的两点上判空（两点都必须是 '.')
    """
    di = ti - si
    dj = tj - sj
    adi, adj = abs(di), abs(dj)

    # 两子需形成 2x3 矩形：一边差 1（短边），另一边差 2（长边）
    if not ((adi == 1 and adj == 2) or (adi == 2 and adj == 1)):
        return False

    cells_to_check = _fei_middle_cells(si, sj, ti, tj)
    if cells_to_check is None:
        return False

    for mi, mj in cells_to_check:
        if _cell(patch, mi, mj) != ".":
            return False
    return True


def match_chong(patch: List[List[str]]) -> bool:
    """
    冲：
    - 本手 S 与友军 A 正交直接相邻
    - 且本手位置破坏了敌方“飞”或“跳”形

    这里将“破坏”规则化为：
    1) 破坏敌方飞：存在两枚敌子可构成飞形，且飞形两处中间点之一为 S，
       另一处中间点仍为空；
    2) 破坏敌方跳：S 两侧正交相对位置为敌子（即 S 落在敌方跳的中点）。
    """
    ci, cj = _center(patch)
    orth_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    ally_dirs: List[Tuple[int, int]] = []
    for di, dj in orth_dirs:
        if _cell(patch, ci + di, cj + dj) == "A":
            ally_dirs.append((di, dj))
    if not ally_dirs:
        return False

    n = len(patch)
    all_enemies: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if _cell(patch, i, j) == "E":
                all_enemies.append((i, j))

    # 1) 破坏敌方飞
    for idx, (ei, ej) in enumerate(all_enemies):
        for fi, fj in all_enemies[idx + 1 :]:
            mids = _fei_middle_cells(ei, ej, fi, fj)
            if mids is None or (ci, cj) not in mids:
                continue

            other_mids = [pos for pos in mids if pos != (ci, cj)]
            if len(other_mids) != 1:
                continue
            oi, oj = other_mids[0]
            if _cell(patch, oi, oj) != ".":
                continue

            return True

    # 2) 破坏敌方跳（S 位于敌方两子中点）
    if _cell(patch, ci - 1, cj) == "E" and _cell(patch, ci + 1, cj) == "E":
        return True
    if _cell(patch, ci, cj - 1) == "E" and _cell(patch, ci, cj + 1) == "E":
        return True

    return False


def match_dang(patch: List[List[str]]) -> bool:
    """
    挡：
    本手直接相邻有敌子压住“飞/跳”路线，且本手四邻有友军。
    规则化为：
    - 当前局面：本手四邻至少有一枚友军 A，且至少有一枚相邻敌子 E；
    - 当前局面不直接满足我方“飞/跳”；
    - 设想把本手四邻的敌子 E 临时移除为空（视作不被贴住），
      若此时局部可判为“飞”或“跳”，则记为“挡”。
    """
    ci, cj = _center(patch)
    orth_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    has_adjacent_ally = any(_cell(patch, ci + di, cj + dj) == "A" for di, dj in orth_dirs)
    if not has_adjacent_ally:
        return False
    has_adjacent_enemy = any(_cell(patch, ci + di, cj + dj) == "E" for di, dj in orth_dirs)
    if not has_adjacent_enemy:
        return False

    # 若当前已是飞/跳，则不标“挡”
    if match_tiao(patch) or match_fei(patch):
        return False

    relaxed = [row[:] for row in patch]
    for di, dj in orth_dirs:
        pi, pj = ci + di, cj + dj
        if _cell(relaxed, pi, pj) == "E":
            relaxed[pi][pj] = "."

    return match_tiao(relaxed) or match_fei(relaxed)


def _middle_region_is_clear_by_span(
    patch: List[List[str]],
    si: int,
    sj: int,
    ti: int,
    tj: int,
    long_span: int,
) -> bool:
    """
    飞形中间区域通用判空：
    - 两子构成 (long_span x 1) 的差值（即 |di|,|dj| 为 {long_span,1}）
    - 检查除两子所在短边外的中间条带是否全为空
    """
    di = ti - si
    dj = tj - sj
    adi, adj = abs(di), abs(dj)
    if not ((adi == long_span and adj == 1) or (adi == 1 and adj == long_span)):
        return False

    if adi == long_span and adj == 1:
        step_i = 1 if di > 0 else -1
        for k in range(1, long_span):
            row = si + step_i * k
            if _cell(patch, row, sj) != "." or _cell(patch, row, tj) != ".":
                return False
        return True

    # adi == 1 and adj == long_span
    step_j = 1 if dj > 0 else -1
    for k in range(1, long_span):
        col = sj + step_j * k
        if _cell(patch, si, col) != "." or _cell(patch, ti, col) != ".":
            return False
    return True


def _match_fei_like(patch: List[List[str]], target: str) -> bool:
    """
    飞形公共判定：
    - 与 target（A 或 E）隔一格，且两子不共线（仅保留斜向飞形）
    - 两子中间区域不能为空
    """
    ci, cj = _center(patch)
    vectors = [
        (-2, -1), 
        (-2, 1), 
        (2, -1), 
        (2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
    ]
    for di, dj in vectors:
        ti = ci + di
        tj = cj + dj
        if _cell(patch, ti, tj) == target and _middle_region_is_clear(patch, ci, cj, ti, tj):
            return True
    return False


def _match_big_fei_like(patch: List[List[str]], target: str) -> bool:
    """
    大飞形公共判定（与飞同形，间距更大）：
    - 与 target（A 或 E）构成 3-1 或 1-3
    - 中间两行（或两列）条带均为空
    """
    ci, cj = _center(patch)
    vectors = [
        (-3, -1), (-3, 1),
        (3, -1), (3, 1),
        (-1, -3), (-1, 3),
        (1, -3), (1, 3),
    ]
    for di, dj in vectors:
        ti = ci + di
        tj = cj + dj
        if _cell(patch, ti, tj) == target and _middle_region_is_clear_by_span(patch, ci, cj, ti, tj, long_span=3):
            return True
    return False


def _nine_grid_is_clean_except(
    patch: List[List[str]],
    center_i: int,
    center_j: int,
    allowed_positions: set,
) -> bool:
    """
    检查某点九宫格内是否无其它棋子：
    - 除 allowed_positions 中位置外，不允许出现 A/E/S
    - '.' 和边界 'B' 允许
    """
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            pi, pj = center_i + di, center_j + dj
            if not _in_bounds(patch, pi, pj):
                continue
            if (pi, pj) in allowed_positions:
                continue
            if _cell(patch, pi, pj) in {"A", "E", "S"}:
                return False
    return True


def match_gua(patch: List[List[str]]) -> bool:
    """
    挂：形状与飞一致，但 S 与 E 构成飞形。
    附加约束：
    - S 的九宫格内除 S 和被挂 E 外无其它棋子
    - 被挂 E 的九宫格内除 E 和 S 外无其它棋子
    """
    ci, cj = _center(patch)
    vectors = [
        (-2, -1),
        (-2, 1),
        (2, -1),
        (2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
    ]
    for di, dj in vectors:
        ti = ci + di
        tj = cj + dj
        if _cell(patch, ti, tj) != "E":
            continue
        if not _middle_region_is_clear(patch, ci, cj, ti, tj):
            continue

        allowed = {(ci, cj), (ti, tj)}
        if _nine_grid_is_clean_except(patch, ci, cj, allowed) and _nine_grid_is_clean_except(patch, ti, tj, allowed):
            return True
    return False


def match_dafei(patch: List[List[str]]) -> bool:
    """
    大飞：与飞同形，但两子中间为两行（或两列）空间。
    """
    return _match_big_fei_like(patch, target="A")


def match_tiao(patch: List[List[str]]) -> bool:
    """
    跳：S 与 A 直线推进隔点（仅正交方向隔一点）
    """
    ci, cj = _center(patch)
    # 若已与友军正交相邻，则不视作“跳”
    for odi, odj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if _cell(patch, ci + odi, cj + odj) == "A":
            return False

    vectors = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    for di, dj in vectors:
        mi = ci + di // 2
        mj = cj + dj // 2
        ti = ci + di
        tj = cj + dj
        if _cell(patch, ti, tj) == "A" and _cell(patch, mi, mj) == ".":
            return True
    return False


def match_datao(patch: List[List[str]]) -> bool:
    """
    大跳：两子同一直线，中间空两格（正交方向）。
    """
    ci, cj = _center(patch)
    vectors = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    for di, dj in vectors:
        ti = ci + di
        tj = cj + dj
        if _cell(patch, ti, tj) != "A":
            continue

        if di != 0:
            step_i = 1 if di > 0 else -1
            if _cell(patch, ci + step_i, cj) == "." and _cell(patch, ci + 2 * step_i, cj) == ".":
                return True
        else:
            step_j = 1 if dj > 0 else -1
            if _cell(patch, ci, cj + step_j) == "." and _cell(patch, ci, cj + 2 * step_j) == ".":
                return True
    return False


def match_tuo(patch: List[List[str]]) -> bool:
    """
    托：紧挨着对方棋子的“下边”落子。
    这里“下边”定义为更贴近边界的一侧：
    - S 与 E 正交相邻
    - 从 S 朝“远离 E”方向到边界的距离，小于朝“靠近 E”方向到边界的距离
    - 若 S 在左右（或上下）两侧均有友军且高位直接贴敌，也应判为托
    """
    ci, cj = _center(patch)
    orth_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dist_to_edge(i: int, j: int, di: int, dj: int) -> int:
        # 在 7x7 patch 内最多看到 3 步；未见边界则记为很远
        for step in (1, 2, 3):
            cell = _cell(patch, i + di * step, j + dj * step)
            if cell == "B" or cell is None:
                return step - 1
        return 99

    def clear_to_edge(i: int, j: int, di: int, dj: int) -> bool:
        """
        检查从 (i,j) 朝某方向到边线之间是否全为空点。
        遇到边界(B/None)前若出现 A/E/S 则不满足。
        """
        for step in (1, 2, 3):
            cell = _cell(patch, i + di * step, j + dj * step)
            if cell in {"B", None}:
                return True
            if cell != ".":
                return False
        return False

    for di, dj in orth_dirs:
        if _cell(patch, ci + di, cj + dj) != "E":
            continue

        # S 处在 E 的“下边” => 从 S 远离 E 的方向更靠边界
        away_di, away_dj = -di, -dj
        toward_di, toward_dj = di, dj

        away_dist = dist_to_edge(ci, cj, away_di, away_dj)
        toward_dist = dist_to_edge(ci, cj, toward_di, toward_dj)

        # 新约束：本手到底线之间不能有棋子（通道需全空）
        if not clear_to_edge(ci, cj, away_di, away_dj):
            continue

        if away_dist < toward_dist:
            return True

        # 强化规则：低位托（两侧有友 + 高位邻敌）
        if toward_dist <= 2 and away_dist <= toward_dist:
            if di != 0:
                flank_allies = [_cell(patch, ci, cj - 1), _cell(patch, ci, cj + 1)]
            else:
                flank_allies = [_cell(patch, ci - 1, cj), _cell(patch, ci + 1, cj)]
            if flank_allies[0] == "A" and flank_allies[1] == "A":
                return True

    return False


def match_pa(patch: List[List[str]]) -> bool:
    """
    爬：在对方压迫下，沿边上一线/二线位置“长”。
    判定为：
    - S 距最近边界不超过 1 格（即一线或二线）
    - S 在与该边界平行方向上至少有一枚友军 A（沿边长）
    - S 朝棋盘内侧一格为敌方 E（受压）
    """
    ci, cj = _center(patch)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dist_to_edge(i: int, j: int, di: int, dj: int) -> int:
        for step in (1, 2, 3):
            cell = _cell(patch, i + di * step, j + dj * step)
            if cell == "B" or cell is None:
                return step - 1
        return 99

    dist_map: Dict[Tuple[int, int], int] = {}
    for di, dj in directions:
        dist_map[(di, dj)] = dist_to_edge(ci, cj, di, dj)

    min_dist = min(dist_map.values())
    # 仅考虑一线/二线
    if min_dist > 1:
        return False

    # 取最近边界方向；角部并列最近时不判“爬”
    near_dirs = [d for d, dist in dist_map.items() if dist == min_dist]
    if len(near_dirs) != 1:
        return False
    near_di, near_dj = near_dirs[0]

    # 内侧受压（远离边界方向为内侧）
    inner_cell = _cell(patch, ci - near_di, cj - near_dj)
    if inner_cell != "E":
        return False

    # 沿边“长”：与边界平行方向至少一侧有友军
    if near_di != 0:  # 近边为上下边，平行方向为左右
        side_cells = [_cell(patch, ci, cj - 1), _cell(patch, ci, cj + 1)]
    else:  # 近边为左右边，平行方向为上下
        side_cells = [_cell(patch, ci - 1, cj), _cell(patch, ci + 1, cj)]
    return any(cell == "A" for cell in side_cells)


def match_jia(patch: List[List[str]]) -> bool:
    """
    夹：本手 S 与友子 A 同线，中间夹住单个敌子 E（S-E-A）。
    规则：
    - 某正交方向上，S 的相邻点是敌子 E；
    - 同方向再前进一格是友子 A（敌子位于中间）；
    - 该敌子连通块大小必须为 1（单个敌军）。
    - 两侧友军（本手 S 与远端 A）应不连通（不能已通过其它己子绕连）。
    """
    ci, cj = _center(patch)
    my_group, _ = _collect_my_group_and_liberties_in_patch(patch)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for di, dj in directions:
        enemy_i, enemy_j = ci + di, cj + dj
        ally_i, ally_j = ci + 2 * di, cj + 2 * dj
        if _cell(patch, enemy_i, enemy_j) != "E":
            continue
        if _cell(patch, ally_i, ally_j) != "A":
            continue

        enemy_group, _ = _collect_group_and_liberties(patch, enemy_i, enemy_j, target="E")
        if len(enemy_group) != 1:
            continue
        if (ally_i, ally_j) in my_group:
            continue
        return True
    return False


def _collect_my_group_and_liberties_in_patch(patch: List[List[str]]) -> Tuple[set, set]:
    """
    在局部 patch 内，以中心 S 为起点收集己方连通块（S/A）及其气（'.'）。
    """
    ci, cj = _center(patch)
    if _cell(patch, ci, cj) != "S":
        return set(), set()

    group = set()
    liberties = set()
    stack = [(ci, cj)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        i, j = stack.pop()
        if (i, j) in group:
            continue
        cell = _cell(patch, i, j)
        if cell not in {"S", "A"}:
            continue
        group.add((i, j))
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            ncell = _cell(patch, ni, nj)
            if ncell in {"S", "A"} and (ni, nj) not in group:
                stack.append((ni, nj))
            elif ncell == ".":
                liberties.add((ni, nj))
    return group, liberties


def _enemy_component_count_around_point(patch: List[List[str]], pi: int, pj: int) -> int:
    """
    统计某空点四邻敌子在“忽略该空点”后分成多少敌方连通分量。
    该点是潜在“断点”时，分量数通常 >= 2。
    """
    if _cell(patch, pi, pj) != ".":
        return 0

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    seeds = []
    for di, dj in dirs:
        ni, nj = pi + di, pj + dj
        if _cell(patch, ni, nj) == "E":
            seeds.append((ni, nj))
    if len(seeds) < 2:
        return 0

    enemy_cells = set()
    n = len(patch)
    for i in range(n):
        for j in range(n):
            if (i, j) == (pi, pj):
                continue
            if _cell(patch, i, j) == "E":
                enemy_cells.add((i, j))

    comp_id: Dict[Tuple[int, int], int] = {}
    comp = 0
    for start in enemy_cells:
        if start in comp_id:
            continue
        comp += 1
        stack = [start]
        comp_id[start] = comp
        while stack:
            ui, uj = stack.pop()
            for di, dj in dirs:
                vi, vj = ui + di, uj + dj
                if (vi, vj) in enemy_cells and (vi, vj) not in comp_id:
                    comp_id[(vi, vj)] = comp
                    stack.append((vi, vj))

    touching = {comp_id[pos] for pos in seeds if pos in comp_id}
    return len(touching)


def _is_strict_breakpoint_in_patch(patch: List[List[str]], pi: int, pj: int) -> bool:
    """
    严格“断点”判定（局部）：
    - 该点为空 '.'
    - 且满足以下之一：
      1) 上下均为敌子 E，且两者在九宫格内不经该点不连通
      2) 左右均为敌子 E，且两者在九宫格内不经该点不连通
    """
    if _cell(patch, pi, pj) != ".":
        return False

    up = (pi - 1, pj)
    down = (pi + 1, pj)
    left = (pi, pj - 1)
    right = (pi, pj + 1)

    if _cell(patch, up[0], up[1]) == "E" and _cell(patch, down[0], down[1]) == "E":
        if not _connected_in_3x3_excluding_center(patch, up, down, target="E"):
            return True

    if _cell(patch, left[0], left[1]) == "E" and _cell(patch, right[0], right[1]) == "E":
        if not _connected_in_3x3_excluding_center(patch, left, right, target="E"):
            return True

    return False


def match_ci(patch: List[List[str]]) -> bool:
    """
    刺：
    本手与“断点”直接相邻，并且本手连通块气口大于一。
    这里“断点”定义为：
    - 某个空点四邻至少有两枚敌子；
    - 忽略该空点后，这些四邻敌子分属至少两个敌方连通分量。
    """
    ci, cj = _center(patch)
    _, my_liberties = _collect_my_group_and_liberties_in_patch(patch)
    if len(my_liberties) <= 1:
        return False

    # 本手四邻若存在“严格断点”空位，则记为刺
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        pi, pj = ci + di, cj + dj
        if _is_strict_breakpoint_in_patch(patch, pi, pj):
            return True

    return False


def match_shuang(patch: List[List[str]]) -> bool:
    """
    双：
    在以 S 为中心九宫格/近邻中，形成“两子对两子正对”的结构：
    - S 与一枚友子 A 正交相邻，构成第一对；
    - 在与该对子平行的方向上，隔一行（或一列）存在第二对友子 A；
    - 两对之间对应的中间两点必须为空（中间没有其它棋子）。
    """
    ci, cj = _center(patch)
    orth_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for pdi, pdj in orth_dirs:
        mate_i, mate_j = ci + pdi, cj + pdj
        if _cell(patch, mate_i, mate_j) != "A":
            continue

        # pair_dir 与 facing_dir 正交；双形是“对子”与“对向平行对子”
        if pdi == 0:  # 当前对子横向，正对方向为上下
            facing_dirs = [(-1, 0), (1, 0)]
        else:  # 当前对子纵向，正对方向为左右
            facing_dirs = [(0, -1), (0, 1)]

        for fdi, fdj in facing_dirs:
            opp1 = (ci + 2 * fdi, cj + 2 * fdj)
            opp2 = (mate_i + 2 * fdi, mate_j + 2 * fdj)
            mid1 = (ci + fdi, cj + fdj)
            mid2 = (mate_i + fdi, mate_j + fdj)

            if (
                _cell(patch, opp1[0], opp1[1]) == "A"
                and _cell(patch, opp2[0], opp2[1]) == "A"
                and _cell(patch, mid1[0], mid1[1]) == "."
                and _cell(patch, mid2[0], mid2[1]) == "."
            ):
                return True

    return False


def classify_move(
    patch: List[List[str]],
    is_jiaochi: Optional[bool] = None,
    is_ti: Optional[bool] = None,
    is_ji: Optional[bool] = None,
) -> List[str]:
    """
    按“距离优先级”分类：
    - 友军曼哈顿距离越近，优先级越高
    - 命中高优先级标签后，立即返回，不再考虑低优先级
    """
    if is_jiaochi is None:
        is_jiaochi = match_jiaochi(patch)

    # “提”和“叫吃”可并存
    if is_ti:
        if is_jiaochi:
            return ["提", "叫吃"]
        return ["提"]

    if is_jiaochi:
        # 多标签：叫吃可与冲/断并存
        labels = ["叫吃"]
        if is_ji:
            labels.append("挤")
        if match_nian(patch):
            labels.append("粘")
        is_duan_here = match_duan(patch)
        if not is_duan_here and (not match_kao(patch)) and match_ci(patch):
            labels.append("刺")
            if match_chang(patch):
                labels.append("长")
        if not is_duan_here and match_tiao(patch):
            labels.append("跳")
        if match_chong(patch):
            labels.append("冲")
        if is_duan_here:
            labels.append("断")
        return labels

    if is_ji:
        return ["挤"]

    is_hu = match_hu(patch)
    if is_hu:
        # 多标签：虎可与粘/靠并存
        labels = ["虎"]
        if match_nian(patch):
            labels.append("粘")
        if match_kao(patch):
            labels.append("靠")
        return labels

    is_nian = match_nian(patch)
    if is_nian:
        return ["粘"]

    if match_shuang(patch):
        return ["双"]

    # ---- 近距离层（高优先级）----
    # 近距离战术优先：刺 > 冲 > 挖 > 断 > 扳
    is_chong = match_chong(patch)
    is_wa = match_wa(patch)
    is_duan = match_duan(patch)
    is_jia = match_jia(patch)
    # 虎优先级高于扳：即使后续逻辑调整，也不允许同形落到“扳”。
    is_ban = False if (is_duan or is_hu) else match_ban(patch)
    is_pa = match_pa(patch)
    is_kao = (not is_duan and not is_ban and not is_pa and match_kao(patch))
    is_ci = (not is_duan and not is_kao and match_ci(patch))
    is_tuo = (not is_ban and not is_pa and match_tuo(patch))
    is_jian = match_jian(patch)
    ci, cj = _center(patch)
    has_adjacent_ally = any(_cell(patch, ci + di, cj + dj) == "A" for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)])

    is_chang = (not is_kao and match_chang(patch))

    if is_ci:
        labels = []
        if match_tiao(patch):
            labels.append("跳")
        labels.append("刺")
        if is_chang:
            labels.append("长")
        if is_jian:
            labels.append("尖")
        return labels
    # “冲/断”可同时命中时，按业务定义优先判“冲”
    if is_chong:
        return ["冲"]
    if is_wa:
        return ["挖"]
    if is_duan:
        return ["断"]
    if is_ban:
        return ["扳"]
    if is_jia:
        return ["夹"]
    # “托”是特定低位贴敌形，若与“靠”并发，优先判“托”
    if is_tuo and is_kao:
        return ["托"]
    # “尖顶”需是纯顶形；若已有正交相邻友军，则不判尖顶
    is_peng = match_peng(patch)
    if is_peng:
        return ["碰"]
    if is_kao and is_jian and not has_adjacent_ally:
        return ["尖顶"]
    if is_kao:
        return ["靠"]
    if not is_duan and not is_ban and match_dang(patch):
        return ["挡"]
    if is_pa:
        return ["爬"]
    if is_chang:
        return ["长"]

    # ---- 中距离层 ----
    # 长 > 跳（已在上层先判长）
    if is_jian:
        return ["尖"]
    if is_tuo:
        return ["托"]

    # ---- 远距离层 ----
    # 远距离内部：跳 > 飞 > 大跳 > 大飞
    is_tiao = False
    is_fei = False
    is_datao = False
    is_dafei = False
    if is_duan:
        return ["断"]
    if not is_ban:
        is_tiao = match_tiao(patch)
        suppress_fei = is_jian or is_kao or is_chang or is_duan
        is_fei = (not is_tiao and not suppress_fei and match_fei(patch))
        is_datao = (not is_tiao and not is_fei and match_datao(patch))
        is_dafei = (not is_tiao and not is_fei and not is_datao and match_dafei(patch))
    if is_tiao:
        return ["跳"]
    if is_fei:
        return ["飞"]
    if is_datao:
        return ["大跳"]
    if match_gua(patch):
        return ["挂"]
    if is_dafei:
        return ["大飞"]

    return []


def update_json(data: Dict, labels: List[str]) -> Dict:
    """
    将识别结果写入 semantic_context.move_location_hint
    多个结果用中文逗号连接；未识别写空字符串。
    """
    if "semantic_context" not in data or not isinstance(data["semantic_context"], dict):
        data["semantic_context"] = {}

    if labels:
        data["semantic_context"]["move_location_hint"] = "，".join(labels)
    else:
        data["semantic_context"]["move_location_hint"] = ""

    return data


def _resolve_output_path(
    input_path: str,
    source_root_name: str = "simpleproto_from_sgf",
    output_root_name: str = "outputs",
) -> Path:
    """
    将输入路径映射到输出路径：
    - .../simpleproto_from_sgf/<subdirs>/<file>.json
      -> .../outputs/<subdirs>/<file>.json
    若不在 source_root 下，则退化为 outputs/<basename>。
    """
    in_path = Path(input_path).resolve()
    cwd = Path.cwd().resolve()

    parts = in_path.parts
    if source_root_name in parts:
        idx = parts.index(source_root_name)
        rel_inside_source = Path(*parts[idx + 1 :])
        return cwd / output_root_name / rel_inside_source

    return cwd / output_root_name / in_path.name


def process_file(
    input_path: str,
    source_root_name: str = "simpleproto_from_sgf",
    output_root_name: str = "outputs",
) -> Path:
    """
    读取 JSON -> 识别 last_move 着法 -> 更新字段 -> 保存 JSON
    """
    in_path = Path(input_path).resolve()
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rules = data.get("rules", {})
    state = data.get("state", {})
    board_size = int(rules.get("board_size", 19))

    last_move = state.get("last_move", {}) or {}
    if last_move.get("is_pass", False):
        update_json(data, [])
    else:
        point = last_move.get("point")
        color = last_move.get("color")
        if not point or color not in ("black", "white"):
            # 无效 last_move，按未识别处理
            update_json(data, [])
        else:
            board = build_board(state, board_size=board_size)
            r, c = parse_coord(point, board_size=board_size)
            patch = extract_local_patch(board, r, c, size=7)
            encoded_patch = encode_patch(patch, my_color=color)

            is_ti = False
            is_ji = False
            prev_path = _previous_step_path(in_path)
            if prev_path is not None:
                with prev_path.open("r", encoding="utf-8") as pf:
                    prev_data = json.load(pf)
                prev_state = prev_data.get("state", {}) or {}
                prev_board = build_board(prev_state, board_size=board_size)
                is_ti = match_ti_on_board(prev_board, board, r, c, color)
                is_ji = match_ji_on_board(prev_board, board, r, c, color)

            is_jiaochi = match_jiaochi_on_board(board, r, c, color)
            labels = classify_move(encoded_patch, is_jiaochi=is_jiaochi, is_ti=is_ti, is_ji=is_ji)
            update_json(data, labels)

    out_path = _resolve_output_path(
        input_path=str(in_path),
        source_root_name=source_root_name,
        output_root_name=output_root_name,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out_path


def process_all_files() -> int:
    """
    使用固定路径批处理：
    - 输入根目录：PROJECT_ROOT/simpleproto_from_sgf
    - 输出根目录：PROJECT_ROOT/outputs
    """
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {SOURCE_ROOT}")

    count = 0
    for input_json in SOURCE_ROOT.rglob("*.json"):
        out_path = process_file(
            str(input_json),
            source_root_name=SOURCE_ROOT.name,
            output_root_name=OUTPUT_ROOT.name,
        )
        count += 1
        print(f"Saved: {out_path}")
    return count


def main() -> None:
    total = process_all_files()
    print(f"Done. Processed {total} file(s).")


if __name__ == "__main__":
    main()
