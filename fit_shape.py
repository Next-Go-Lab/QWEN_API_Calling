#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
围棋棋型识别系统 v4.2 (Go Pattern Recognition System)

功能模块：
1. 棋盘位置识别：天元、星位、小目、高目、目外、三三
2. 着法手段识别：长、立、飞、尖、压、托、靠、断、夹等
3. 常见棋形识别：直三、直四、弯三、曲四、方四、板六、丁四、刀把五等
4. 布局流派识别：三连星、中国流、迷你中国流、错小目、星无忧角、小林流
5. 厚势识别：铁壁、厚势、准厚势、薄势（含断点检测）

厚势判断标准：
- 四线以上为"势"，五线及以上为佳
- 向中腹（中央）方向无对手棋子侵扰
- 形状具有连续性和展开性
- 可能有断点但受保护（虎口）或难以被切断

作者：AI Assistant
版本：4.2
"""

import json
from typing import List, Dict, Tuple, Set, Optional


class GoPoint:
    """围棋坐标点类，处理19路棋盘坐标（跳过I列）"""
    __slots__ = ['coord', 'col', 'row']
    
    def __init__(self, coord: str):
        self.coord = coord.upper().strip()
        self.col = self._parse_col()
        self.row = self._parse_row()
    
    def _parse_col(self) -> int:
        col_char = self.coord[0]
        if col_char >= 'J':
            return ord(col_char) - ord('A') - 1
        return ord(col_char) - ord('A')
    
    def _parse_row(self) -> int:
        return int(self.coord[1:]) - 1
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.col, self.row)
    
    def __hash__(self):
        return hash(self.coord)
    
    def __eq__(self, other):
        return isinstance(other, GoPoint) and self.coord == other.coord
    
    def __repr__(self):
        return f"GoPoint({self.coord})"
    
    @staticmethod
    def from_tuple(col: int, row: int) -> 'GoPoint':
        if col >= 8:
            col_char = chr(ord('A') + col + 1)
        else:
            col_char = chr(ord('A') + col)
        return GoPoint(f"{col_char}{row+1}")


# ========== 棋盘位置定义 ==========
BOARD_POSITIONS = {
    "天元": ["K10"],
    "星位": ["D4", "D10", "D16", "K4", "K16", "Q4", "Q10", "Q16"],
    "小目": ["C4", "C16", "D3", "D17", "J3", "J17", "K3", "K17", "Q3", "Q17", "R4", "R16"],
    "高目": ["D5", "D15", "E4", "E16", "J4", "J16", "K5", "K15", "P4", "P16", "Q5", "Q15"],
    "目外": ["C5", "C15", "E3", "E17", "J5", "J15", "K3", "K17", "P3", "P17", "R5", "R15"],
    "三三": ["C3", "C17", "D3", "D17", "J3", "J17", "K3", "K17", "Q3", "Q17", "R3", "R17"],
}


class ThickShapeAnalyzer:
    """
    厚势分析器 v2.0
    
    厚势定义：
    - 四线以上为"势"，五线及以上为佳
    - 向中腹方向无对手棋子阻挡
    - 具有连续性和展开性
    - 可能有断点但受保护
    """
    
    def __init__(self, black_stones: Set[GoPoint], white_stones: Set[GoPoint]):
        self.black = black_stones
        self.white = white_stones
        self.all_stones = black_stones | white_stones
    
    def get_neighbors(self, point: GoPoint) -> List[GoPoint]:
        col, row = point.to_tuple()
        neighbors = []
        for dc, dr in [(0,1), (0,-1), (1,0), (-1,0)]:
            nc, nr = col + dc, row + dr
            if 0 <= nc < 19 and 0 <= nr < 19:
                neighbors.append(GoPoint.from_tuple(nc, nr))
        return neighbors
    
    def get_diagonals(self, point: GoPoint) -> List[GoPoint]:
        col, row = point.to_tuple()
        diagonals = []
        for dc, dr in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            nc, nr = col + dc, row + dr
            if 0 <= nc < 19 and 0 <= nr < 19:
                diagonals.append(GoPoint.from_tuple(nc, nr))
        return diagonals
    
    def _find_groups(self, stones: Set[GoPoint]) -> List[Set[GoPoint]]:
        if not stones:
            return []
        remaining = set(stones)
        groups = []
        while remaining:
            start = remaining.pop()
            group = {start}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for n in self.get_neighbors(current):
                    if n in remaining:
                        remaining.remove(n)
                        group.add(n)
                        queue.append(n)
            groups.append(group)
        return groups
    
    def _calculate_thickness_score(self, group: Set[GoPoint], color: str) -> Dict:
        opp_stones = self.white if color == "black" else self.black
        group_list = list(group)
        group_size = int(len(group_list))
        
        # 1. 高度分（四线=0.6, 五线=0.8, 六线+=1.0）
        avg_row = sum(s.row for s in group_list) / group_size
        if avg_row >= 6:
            height_score = 1.0
        elif avg_row >= 5:
            height_score = 0.8 + (avg_row - 5) * 0.2
        elif avg_row >= 4:
            height_score = 0.6 + (avg_row - 4) * 0.2
        else:
            height_score = avg_row / 4 * 0.6
        
        # 2. 中央开放度（指向天元方向是否被阻挡）
        center = (9, 9)
        center_direction_clear = 0
        
        for stone in group_list:
            col, row = stone.to_tuple()
            dc = center[0] - col
            dr = center[1] - row
            
            blocked = False
            steps = max(abs(dc), abs(dr))
            if steps > 0:
                for i in range(1, min(steps, 3) + 1):
                    check_col = col + round(dc * i / steps)
                    check_row = row + round(dr * i / steps)
                    if 0 <= check_col < 19 and 0 <= check_row < 19:
                        check_point = GoPoint.from_tuple(check_col, check_row)
                        if check_point in opp_stones:
                            blocked = True
                            break
            
            if not blocked:
                center_direction_clear += 1
        
        center_score = center_direction_clear / group_size
        
        # 3. 连续性
        connections = 0
        for stone in group_list:
            for neighbor in self.get_neighbors(stone):
                if neighbor in group:
                    connections += 1
        continuity_score = min(connections / (group_size * 2), 1.0)
        
        # 4. 发展潜力
        empty_neighbors = set()
        for stone in group_list:
            for neighbor_point in self.get_neighbors(stone):
                if neighbor_point not in self.all_stones:
                    empty_neighbors.add(neighbor_point)
        potential_score = min(len(empty_neighbors) / (group_size * 3), 1.0)
        
        # 5. 弱点分析
        weak_points = []
        dangerous_weak_points = []
        
        for stone in group_list:
            own_conn = sum(1 for n in self.get_neighbors(stone) if n in group)
            opp_adjacent = sum(1 for n in self.get_neighbors(stone) if n in opp_stones)
            diag_own = sum(1 for d in self.get_diagonals(stone) if d in group)
            
            if own_conn <= 1 and opp_adjacent >= 2:
                if diag_own == 0:
                    dangerous_weak_points.append(stone)
                else:
                    weak_points.append(stone)
        
        if dangerous_weak_points:
            weakness_score = max(0.3, 1.0 - len(dangerous_weak_points) * 0.3)
        elif weak_points:
            weakness_score = 0.8
        else:
            weakness_score = 1.0
        
        # 综合得分
        total_score = (
            height_score * 0.30 +
            center_score * 0.30 +
            continuity_score * 0.15 +
            potential_score * 0.15 +
            weakness_score * 0.10
        )
        
        return {
            "height_score": round(height_score, 2),
            "center_score": round(center_score, 2),
            "continuity_score": round(continuity_score, 2),
            "potential_score": round(potential_score, 2),
            "weakness_score": round(weakness_score, 2),
            "total_score": round(total_score, 2),
            "avg_row": round(avg_row, 1),
            "weak_points_count": len(weak_points),
            "dangerous_weak_points_count": len(dangerous_weak_points)
        }
    
    def _classify_thickness_type(self, group: Set[GoPoint], metrics: Dict) -> Optional[str]:
        score = metrics["total_score"]
        avg_row = metrics["avg_row"]
        dangerous = metrics["dangerous_weak_points_count"]
        
        # 基本条件：至少4线及以上且3子以上
        if avg_row < 3.5 or len(group) < 3:
            return None
        
        # 铁壁：5线及以上，得分≥0.85，无危险断点，中央开放度≥0.8
        if avg_row >= 5 and score >= 0.85 and dangerous == 0 and metrics["center_score"] >= 0.8:
            return "铁壁"
        
        # 厚势：4.5线及以上，得分≥0.70，无危险断点
        if avg_row >= 4.5 and score >= 0.70 and dangerous == 0:
            return "厚势"
        
        # 准厚势：4线及以上，得分≥0.55
        if avg_row >= 4 and score >= 0.55:
            return "准厚势"
        
        # 薄势：4线及以上，得分≥0.40
        if avg_row >= 4 and score >= 0.40:
            return "薄势"
        
        return None
    
    def analyze_thickness(self) -> List[Dict]:
        """分析双方厚势"""
        results = []
        
        for color, stones in [("black", self.black), ("white", self.white)]:
            groups = self._find_groups(stones)
            
            for group in groups:
                if len(group) < 3:
                    continue
                
                metrics = self._calculate_thickness_score(group, color)
                thickness_type = self._classify_thickness_type(group, metrics)
                
                if thickness_type:
                    weak_coords = []
                    opp_stones = self.white if color == "black" else self.black
                    
                    for stone in group:
                        own_conn = sum(1 for n in self.get_neighbors(stone) if n in group)
                        opp_adjacent = sum(1 for n in self.get_neighbors(stone) if n in opp_stones)
                        diag_own = sum(1 for d in self.get_diagonals(stone) if d in group)
                        
                        if own_conn <= 1 and opp_adjacent >= 2:
                            if diag_own == 0:
                                weak_coords.append({"point": stone.coord, "level": "dangerous", "reason": "无保护断点"})
                            else:
                                weak_coords.append({"point": stone.coord, "level": "minor", "reason": "虎口保护但连接薄弱"})
                    
                    result = {
                        "type": thickness_type,
                        "color": color,
                        "stones": [s.coord for s in sorted(group, key=lambda p: (p.col, p.row))],
                        "stone_count": len(group),
                        "avg_line": round(metrics["avg_row"], 1),
                        "metrics": metrics,
                        "weak_points": weak_coords if weak_coords else None,
                        "description": self._generate_description(thickness_type, metrics, weak_coords)
                    }
                    results.append(result)
        
        results.sort(key=lambda x: x["metrics"]["total_score"], reverse=True)
        return results
    
    def _generate_description(self, thickness_type: str, metrics: Dict, weak_points: List) -> str:
        height_desc = f"{metrics['avg_row']:.0f}线"
        
        descriptions = {
            "铁壁": f"完美厚势，{height_desc}铁壁，向中央完全敞开（开放度{metrics['center_score']:.0%}），无断点，辐射力强",
            "厚势": f"强大外势，{height_desc}厚势，发展潜力良好（开放度{metrics['center_score']:.0%}）",
            "准厚势": f"具有一定厚势，{height_desc}，但存在缺陷（弱点指数{metrics['weakness_score']:.0%}）",
            "薄势": f"外势不稳固，{height_desc}，需要补强（开放度{metrics['center_score']:.0%}）"
        }
        
        desc = descriptions.get(thickness_type, "")
        
        if weak_points:
            danger_count = sum(1 for w in weak_points if w["level"] == "dangerous")
            if danger_count > 0:
                desc += f"，注意{danger_count}个危险断点"
        
        return desc


class GoAnalyzer:
    """围棋棋型分析主类（含厚势识别）"""
    
    def __init__(self, game_state: Dict):
        self.state = game_state
        self.black_stones = set(GoPoint(p) for p in game_state.get("black_stones", []))
        self.white_stones = set(GoPoint(p) for p in game_state.get("white_stones", []))
        self.last_move = None
        self.last_color = None
        
        last = game_state.get("last_move")
        if last and not last.get("is_pass"):
            self.last_move = GoPoint(last["point"])
            self.last_color = last["color"]
    
    def get_neighbors(self, point: GoPoint) -> List[GoPoint]:
        col, row = point.to_tuple()
        neighbors = []
        for dc, dr in [(0,1), (0,-1), (1,0), (-1,0)]:
            nc, nr = col + dc, row + dr
            if 0 <= nc < 19 and 0 <= nr < 19:
                neighbors.append(GoPoint.from_tuple(nc, nr))
        return neighbors
    
    def get_diagonals(self, point: GoPoint) -> List[GoPoint]:
        col, row = point.to_tuple()
        diagonals = []
        for dc, dr in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            nc, nr = col + dc, row + dr
            if 0 <= nc < 19 and 0 <= nr < 19:
                diagonals.append(GoPoint.from_tuple(nc, nr))
        return diagonals
    
    def _find_groups(self, stones: Set[GoPoint]) -> List[List[GoPoint]]:
        if not stones:
            return []
        remaining = set(stones)
        groups = []
        while remaining:
            start = remaining.pop()
            group = {start}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for n in self.get_neighbors(current):
                    if n in remaining:
                        remaining.remove(n)
                        group.add(n)
                        queue.append(n)
            groups.append(list(group))
        return groups
    
    def _normalize(self, stones: List[GoPoint]) -> List[Tuple[int, int]]:
        min_c = min(s.col for s in stones)
        min_r = min(s.row for s in stones)
        return sorted([(s.col - min_c, s.row - min_r) for s in stones])
    
    def analyze_board_positions(self) -> List[Dict]:
        results = []
        for pos_name, points in BOARD_POSITIONS.items():
            occupied = []
            for p in points:
                gp = GoPoint(p)
                if gp in self.black_stones:
                    occupied.append({"point": p, "color": "black"})
                elif gp in self.white_stones:
                    occupied.append({"point": p, "color": "white"})
            if occupied:
                results.append({
                    "position_type": pos_name,
                    "occupied": occupied,
                    "count": len(occupied),
                    "total": len(points)
                })
        return results
    
    def analyze_move_techniques(self) -> List[Dict]:
        if not self.last_move:
            return []
        
        techniques = []
        own = self.black_stones if self.last_color == "black" else self.white_stones
        opp = self.white_stones if self.last_color == "black" else self.black_stones
        
        lm = self.last_move
        neighbors = [n for n in self.get_neighbors(lm) if n in own]
        opp_neighbors = [n for n in self.get_neighbors(lm) if n in opp]
        diagonals = [d for d in self.get_diagonals(lm) if d in own]
        
        for n in neighbors:
            is_edge = n.col == 0 or n.col == 18 or n.row == 0 or n.row == 18
            if is_edge and ((n.col == lm.col) or (n.row == lm.row)):
                techniques.append({"name": "立", "description": "边线或底线向中腹立", "related_stones": [n.coord]})
            else:
                techniques.append({"name": "长", "description": "沿直线延伸", "related_stones": [n.coord]})
        
        for d in diagonals:
            techniques.append({"name": "尖", "description": "对角连接（小尖）", "related_stones": [d.coord]})
        
        for opp_s in opp_neighbors:
            if lm.row > opp_s.row:
                name, desc = "压", "从上方压迫"
            elif lm.row < opp_s.row:
                name, desc = "托", "从下方托住"
            elif lm.col < opp_s.col:
                name, desc = "靠", "从左侧贴住"
            else:
                name, desc = "靠", "从右侧贴住"
            
            if abs(lm.col - opp_s.col) + abs(lm.row - opp_s.row) == 1:
                techniques.append({"name": name, "description": desc, "related_stones": [opp_s.coord]})
        
        if len(opp_neighbors) >= 2:
            opp_groups = set()
            for opp_s in opp_neighbors:
                opp_group = self._find_group(opp_s, opp)
                opp_groups.add(tuple(sorted([p.coord for p in opp_group])))
            
            if len(opp_groups) > 1:
                techniques.append({"name": "断", "description": "切断对方棋子连接", "related_stones": [o.coord for o in opp_neighbors]})
        
        if len(opp_neighbors) == 1:
            opp_s = opp_neighbors[0]
            side_stones = [n for n in self.get_neighbors(opp_s) if n != lm and n in own]
            if side_stones:
                techniques.append({"name": "夹", "description": "夹击对方棋子", "related_stones": [opp_s.coord] + [s.coord for s in side_stones]})
        
        seen = set()
        unique_techniques = []
        for t in techniques:
            key = (t["name"], tuple(sorted(t["related_stones"])))
            if key not in seen:
                seen.add(key)
                unique_techniques.append(t)
        
        return [{"move": lm.coord, "color": self.last_color, "techniques": unique_techniques}]
    
    def _find_group(self, start: GoPoint, stones: Set[GoPoint]) -> Set[GoPoint]:
        group = {start}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for n in self.get_neighbors(current):
                if n in stones and n not in group:
                    group.add(n)
                    queue.append(n)
        return group
    
    def analyze_shapes(self) -> List[Dict]:
        """识别眼位形状（围成的空）- 使用滑动掩码方法"""
        results = []
        
        # 使用滑动掩码方法直接在棋盘上查找形状
        black_shapes = self._find_shapes_by_mask(self.black_stones, self.white_stones, "black")
        white_shapes = self._find_shapes_by_mask(self.white_stones, self.black_stones, "white")
        
        results.extend(black_shapes)
        results.extend(white_shapes)
        
        return results

    def _find_shapes_by_mask(self, own_stones: Set[GoPoint], opp_stones: Set[GoPoint], owner_color: str) -> List[Dict]:
        """
        使用滑动掩码方法查找特定形状
        直接在棋盘上滑动预定义的形状模板
        """
        results = []
        all_stones = own_stones | opp_stones
        
        # 定义所有要识别的形状模板
        shape_templates = self._get_shape_templates()
        
        # 在整个棋盘上滑动每个模板
        for shape_name, templates in shape_templates.items():
            for template in templates:
                # 获取模板的尺寸
                max_col = max(pos[0] for pos in template)
                max_row = max(pos[1] for pos in template)
                
                # 在棋盘上滑动模板
                for base_col in range(19 - max_col):
                    for base_row in range(19 - max_row):
                        # 检查模板是否匹配当前位置
                        if self._check_template_match(template, base_col, base_row, own_stones, opp_stones, all_stones):
                            # 转换为实际坐标
                            actual_points = []
                            for rel_col, rel_row in template:
                                actual_col = base_col + rel_col
                                actual_row = base_row + rel_row
                                point = GoPoint.from_tuple(actual_col, actual_row)
                                actual_points.append(point)
                            
                            # 创建结果 - 传入 own_stones 以便识别形成眼位的棋子
                            result = self._create_shape_result(shape_name, actual_points, owner_color, own_stones)
                            if result:
                                results.append(result)
        
        return results
    
    def _get_shape_templates(self) -> Dict[str, List[List[Tuple[int, int]]]]:
        """
        获取所有形状的模板定义
        每个模板是相对坐标的列表 [(col_offset, row_offset), ...]
        """
        return {
            "直四": [
                [(0,0), (0,1), (0,2), (0,3)],  # 垂直直四
                [(0,0), (1,0), (2,0), (3,0)],  # 水平直四
            ],
            "曲四": [
                [(0,0), (0,1), (0,2), (1,0)],
                [(0,0), (0,1), (0,2), (1,2)],
                [(0,0), (0,1), (1,0), (2,0)],
                [(0,0), (0,1), (1,1), (2,1)],
                [(0,0), (1,0), (1,1), (1,2)],
                [(0,0), (1,0), (2,0), (2,1)],
                [(0,1), (1,1), (2,0), (2,1)],
                [(0,2), (1,0), (1,1), (1,2)],
            ],
            "闪电四": [
                [(0,0), (1,0), (1,1), (2,1)],
                [(0,0), (0,1), (1,1), (1,2)],
                [(1,0), (1,1), (0,1), (0,2)],
                [(0,1), (1,1), (1,0), (2,0)],
            ],
            "直三": [
                [(0,0), (0,1), (0,2)],
                [(0,0), (1,0), (2,0)],
            ],
            "弯三": [
                [(0,0), (0,1), (1,1)],
                [(0,0), (1,0), (1,1)],
                [(0,0), (0,1), (1,0)],
                [(0,1), (1,0), (1,1)],
            ],
            "方四": [
                [(0,0), (0,1), (1,0), (1,1)],
            ],
            "丁四": [
                [(0,0), (0,1), (0,2), (1,1)],
                [(0,0), (1,0), (2,0), (1,1)],
                [(0,1), (1,0), (1,1), (1,2)],
                [(0,1), (1,0), (1,1), (2,1)],
            ],
            "板六": [
                [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
                [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)],
            ],
        }
    
    def _check_template_match(self, template: List[Tuple[int, int]], base_col: int, base_row: int, 
                            own_stones: Set[GoPoint], opp_stones: Set[GoPoint], all_stones: Set[GoPoint]) -> bool:
        """
        检查模板是否在指定位置匹配
        对于眼形识别，模板位置应该是空点，周围应该是己方棋子或棋盘边界
        """
        template_points = []
        for rel_col, rel_row in template:
            col = base_col + rel_col
            row = base_row + rel_row  # Fixed: was base_col + rel_row
            point = GoPoint.from_tuple(col, row)
            template_points.append(point)
        
        # 首先检查模板内的所有点都是空的
        for point in template_points:
            if point in all_stones:
                return False
        
        # 检查模板周围的所有相邻位置
        # 每个模板点的邻居必须是：己方棋子 或 模板内的其他点 或 棋盘边界外
        for point in template_points:
            neighbors = self.get_neighbors(point)
            for neighbor in neighbors:
                # 如果邻居不在模板内，那么它必须是己方棋子（不能是对方棋子或空点）
                if neighbor not in template_points:
                    if neighbor not in own_stones:
                        return False
        
        # 如果到达这里，说明所有外部邻居都是己方棋子
        return True
    
    def _is_corner_template(self, template: List[Tuple[int, int]], base_col: int, base_row: int) -> bool:
        """
        检查模板是否位于棋盘角落区域
        使用坐标范围检测标准角落区域 (3x3 areas at each corner)
        """
        # 获取模板覆盖的实际坐标范围
        actual_cols = []
        actual_rows = []
        for rel_col, rel_row in template:
            col = base_col + rel_col
            row = base_row + rel_row
            actual_cols.append(col)
            actual_rows.append(row)
        
        min_col, max_col = min(actual_cols), max(actual_cols)
        min_row, max_row = min(actual_rows), max(actual_rows)
        
        # 检查是否在四个角落之一的3x3区域内 (coordinates 0-2 or 16-18)
        if max_col <= 2 and max_row <= 2:  # 左下角 (A1-C3 area)
            return True
        if max_col <= 2 and min_row >= 16:  # 左上角 (A17-C19 area)
            return True
        if min_col >= 16 and max_row <= 2:  # 右下角 (Q1-S3 area)
            return True
        if min_col >= 16 and min_row >= 16:  # 右上角 (Q17-S19 area)
            return True
            
        return False
    
    def _create_shape_result(self, shape_name: str, points: List[GoPoint], owner_color: str, own_stones: Set[GoPoint]) -> Optional[Dict]:
        """
        创建形状识别结果
        
        特殊规则：
        - 如果是一般的曲四但在角落，升级为盘角曲四
        - 盘角曲四特指在角落区域的特殊弯曲四子形状
        
        返回结果包含：
        - eye_positions: 眼位（空点坐标）
        - forming_stones: 形成眼位的棋子（包围眼位的己方棋子）
        """
        # 保存原始形状名称
        original_shape = shape_name
        
        # 特殊处理：如果是一般的曲四但在角落，升级为盘角曲四
        if shape_name == "曲四" and self._is_points_in_corner(points):
            shape_name = "盘角曲四"
        
        status_map = {
            "直四": "活形",
            "盘角曲四": "劫活", 
            "直三": "先手活，后手死",
            "弯三": "先手活，后手死",
            "方四": "死形",
            "丁四": "劫活",
            "曲四": "活形",
            "闪电四": "活形",
            "板六": "活形",
        }
        
        if shape_name not in status_map:
            return None
            
        # 获取形成眼位的棋子（包围这些空点的己方棋子）
        own_stones = self.black_stones if owner_color == "black" else self.white_stones
        forming_stones = set()
        eye_set = set(points)
        
        for point in points:
            neighbors = self.get_neighbors(point)
            for neighbor in neighbors:
                if neighbor in own_stones and neighbor not in eye_set:
                    forming_stones.add(neighbor)
        
        result = {
            "name": shape_name,
            "eye_positions": [p.coord for p in sorted(points, key=lambda p: (p.col, p.row))],  # 眼位（空点）
            "forming_stones": [s.coord for s in sorted(forming_stones, key=lambda p: (p.col, p.row))],  # 形成眼位的棋子
            "status": status_map[shape_name],
            "owner": owner_color,
        }
        
        # 添加角落标记
        if shape_name == "盘角曲四":
            result["position"] = "corner"
            
        return result
    
    def _is_points_in_corner(self, points: List[GoPoint]) -> bool:
        """
        检查一组点是否位于棋盘角落区域
        
        角落定义：坐标范围在 0-2 或 16-18 的区域
        """
        if not points:
            return False
            
        cols = [p.col for p in points]
        rows = [p.row for p in points]
        min_col, max_col = min(cols), max(cols)
        min_row, max_row = min(rows), max(rows)
        
        # 左下角 (A1-C3 area): col 0-2, row 0-2
        if max_col <= 2 and max_row <= 2:
            return True
        # 左上角 (A17-C19 area): col 0-2, row 16-18
        if max_col <= 2 and min_row >= 16:
            return True
        # 右下角 (Q1-S3 area): col 16-18, row 0-2
        if min_col >= 16 and max_row <= 2:
            return True
        # 右上角 (Q17-S19 area): col 16-18, row 16-18
        if min_col >= 16 and min_row >= 16:
            return True
            
        return False

    def analyze_fuseki(self) -> List[Dict]:
        """
        识别布局流派 - 修正版
        
        修正要点：
        1. 小林流：星+同边小目+向对方星位挂角+高位结构
        2. 中国流：星+同边小目+边星下一路拆边（三线/四线）
        3. 迷你中国流：星+小目+高位拆边（四线），韩国流代表
        4. 星无忧角：星位+无忧角（小目+小飞/大飞守角）
        5. 错小目：对角交错小目（如右上+左下）
        """
        results = []

        star_pts = set(GoPoint(p) for p in BOARD_POSITIONS["星位"])
        komoku_pts = set(GoPoint(p) for p in BOARD_POSITIONS["小目"])
        taka_pts = set(GoPoint(p) for p in BOARD_POSITIONS["高目"])
        moku_pts = set(GoPoint(p) for p in BOARD_POSITIONS["目外"])
        # 需要边星位置来判断拆边
        side_star_pts = set(GoPoint(p) for p in BOARD_POSITIONS.get("边星", []))

        for color, stones in [("black", self.black_stones), ("white", self.white_stones)]:
            stars = list(stones & star_pts)
            komokus = list(stones & komoku_pts)
            takas = list(stones & taka_pts)
            mokus = list(stones & moku_pts)
            
            # 收集该方所有候选布局
            candidates = []

            # ========== 1. 三连星（最高优先级）==========
            if len(stars) >= 3:
                sanrensei_list = []
                # 横向三连星
                for row in range(19):
                    row_stars = sorted([s for s in stars if s.row == row], key=lambda p: p.col)
                    if len(row_stars) >= 3:
                        sanrensei_list.append({
                            "name": "三连星",
                            "stones": [s.coord for s in row_stars[:3]],
                            "description": "横向三连星布局（宇宙流）",
                            "priority": 1,
                            "score": len(row_stars)
                        })
                # 纵向三连星
                for col in range(19):
                    col_stars = sorted([s for s in stars if s.col == col], key=lambda p: p.row)
                    if len(col_stars) >= 3:
                        sanrensei_list.append({
                            "name": "三连星",
                            "stones": [s.coord for s in col_stars[:3]],
                            "description": "纵向三连星布局（宇宙流）",
                            "priority": 1,
                            "score": len(col_stars)
                        })
                
                if sanrensei_list:
                    best = max(sanrensei_list, key=lambda x: x["score"])
                    candidates.append(best)

            # ========== 2. 中国流 / 高中国流 / 迷你中国流 ==========
            # 核心：星+同边小目+拆边
            chinese_styles = []
            
            for star in stars:
                for komoku in komokus:
                    # 必须是同一边（都在左侧或都在右侧）
                    same_side = (star.col < 10 and komoku.col < 10) or (star.col > 8 and komoku.col > 8)
                    if not same_side:
                        continue
                        
                    # 检查是否有拆边（边星附近）
                    # 标准中国流：拆在边星下一路（三线）
                    # 高中国流/迷你中国流：拆在四线
                    for stone in stones:
                        if stone in star_pts or stone in komoku_pts:
                            continue
                        # 检查是否在边星位置附近（拆边）
                        if (star.col < 10 and 8 <= stone.col <= 12 and 2 <= stone.row <= 5) or \
                        (star.col > 8 and 8 <= stone.col <= 12 and 13 <= stone.row <= 16):
                            
                            row = stone.row
                            if 2 <= row <= 3:  # 三线 - 低中国流
                                chinese_styles.append({
                                    "name": "中国流",
                                    "stones": [star.coord, komoku.coord, stone.coord],
                                    "description": "星位+小目+三线拆边（低中国流）",
                                    "priority": 2,
                                    "score": 10,
                                    "side": "left" if star.col < 10 else "right"
                                })
                            elif row == 4:  # 四线 - 高中国流/迷你中国流
                                chinese_styles.append({
                                    "name": "高中国流",
                                    "stones": [star.coord, komoku.coord, stone.coord],
                                    "description": "星位+小目+四线拆边（高中国流）",
                                    "priority": 2,
                                    "score": 9,
                                    "side": "left" if star.col < 10 else "right"
                                })
            
            # 每侧只保留最佳中国流
            if chinese_styles:
                left_side = [c for c in chinese_styles if c["side"] == "left"]
                right_side = [c for c in chinese_styles if c["side"] == "right"]
                
                if left_side:
                    candidates.append(max(left_side, key=lambda x: x["score"]))
                if right_side:
                    candidates.append(max(right_side, key=lambda x: x["score"]))

            # ========== 3. 小林流 ==========
            # 定义：星+同边小目+向对方星位挂角+高位结构
            # 简化判断：星+同边小目，且小目位置偏向低位（row<4），有高位配合
            kobayashi_list = []
            for star in stars:
                for komoku in komokus:
                    # 同一边
                    same_side = (star.col < 10 and komoku.col < 10) or (star.col > 8 and komoku.col > 8)
                    if not same_side:
                        continue
                    
                    # 小林流特征：星位+小目，小目在低位（3-4线）
                    # 且星位与小目横向距离适中（挂角方向）
                    if komoku.row < 4:  # 低位小目
                        # 检查是否有高位结构配合（如四线拆或高挂）
                        has_high_structure = any(
                            s not in star_pts and s not in komoku_pts and s.row >= 4
                            for s in stones
                        )
                        
                        if has_high_structure or abs(star.col - komoku.col) <= 4:
                            kobayashi_list.append({
                                "name": "小林流",
                                "stones": [star.coord, komoku.coord],
                                "description": "星位+低位小目+高位结构（小林流）",
                                "priority": 3,
                                "score": 5,
                                "side": "left" if star.col < 10 else "right"
                            })
            
            # 小林流每侧最多一个
            if kobayashi_list:
                left_k = [k for k in kobayashi_list if k["side"] == "left"]
                right_k = [k for k in kobayashi_list if k["side"] == "right"]
                if left_k:
                    candidates.append(max(left_k, key=lambda x: x["score"]))
                if right_k:
                    candidates.append(max(right_k, key=lambda x: x["score"]))

            # ========== 4. 星无忧角 ==========
            # 定义：星位+无忧角（小目+小飞/大飞守角）
            star_wuyou_list = []
            for star in stars:
                # 寻找无忧角：两个小目相邻形成（小飞/大飞守角）
                for i, k1 in enumerate(komokus):
                    for k2 in komokus[i+1:]:
                        # 检查是否形成无忧角结构（相邻小目）
                        if abs(k1.col - k2.col) <= 2 and abs(k1.row - k2.row) <= 2:
                            # 检查星位是否与无忧角配合
                            if abs(star.col - k1.col) <= 4 or abs(star.col - k2.col) <= 4:
                                star_wuyou_list.append({
                                    "name": "星无忧角",
                                    "stones": [star.coord, k1.coord, k2.coord],
                                    "description": "星位+无忧角布局",
                                    "priority": 4,
                                    "score": 3
                                })
            
            if star_wuyou_list:
                candidates.append(star_wuyou_list[0])  # 只保留一个

            # ========== 5. 错小目 ==========
            # 定义：对角交错小目（如右上+左下，或左上+右下）
            if len(komokus) >= 2:
                for i, k1 in enumerate(komokus):
                    for k2 in komokus[i+1:]:
                        # 对角判断：一个在左上/右下区域，一个在右上/左下区域
                        k1_left = k1.col < 10
                        k1_top = k1.row > 8
                        k2_left = k2.col < 10
                        k2_top = k2.row > 8
                        
                        # 错小目：对角分布
                        is_diagonal = (k1_left != k2_left) and (k1_top != k2_top)
                        
                        if is_diagonal:
                            candidates.append({
                                "name": "错小目",
                                "stones": [k1.coord, k2.coord],
                                "description": "对角错小目布局（平行型）",
                                "priority": 5,
                                "score": 2
                            })

            # ========== 去重与排序 ==========
            # 按优先级排序
            candidates.sort(key=lambda x: (x["priority"], -x["score"]))
            
            # 去重策略：三连星可与其他共存，其他同优先级每方只保留一个
            seen_priorities = set()
            final_candidates = []
            
            for c in candidates:
                if c["priority"] == 1:  # 三连星可以共存
                    final_candidates.append(c)
                elif c["priority"] not in seen_priorities:
                    final_candidates.append(c)
                    seen_priorities.add(c["priority"])
            
            # 添加到结果
            for c in final_candidates:
                results.append({
                    "name": c["name"],
                    "color": color,
                    "stones": c["stones"],
                    "description": c["description"]
                })

        return results

    def _find_eye_spaces(self, own_stones: Set[GoPoint], opp_stones: Set[GoPoint]) -> List[Set[GoPoint]]:
        """
        找到一方棋子围成的眼位（空点）- 增强角部检测
        """
        all_stones = own_stones | opp_stones

        # 找到所有空点
        all_empty = set()
        for col in range(19):
            for row in range(19):
                p = GoPoint.from_tuple(col, row)
                if p not in all_stones:
                    all_empty.add(p)

        # 筛选可能被己方控制的眼位候选点 - 放宽条件
        candidate_empty = set()

        for empty_p in all_empty:
            col, row = empty_p.to_tuple()

            # 检查四邻
            neighbors = self.get_neighbors(empty_p)
            own_surround = sum(1 for n in neighbors if n in own_stones)
            opp_surround = sum(1 for n in neighbors if n in opp_stones)
            empty_surround = sum(1 for n in neighbors if n in all_empty)

            # 如果有对方棋子直接相邻，则不可能是眼
            if opp_surround > 0:
                continue

            # 角部特殊处理：角部眼位只需要至少1个己方包围（因为棋盘边缘算天然屏障）
            is_corner_pos = (col <= 2 and row <= 2) or (col <= 2 and row >= 16) or \
                           (col >= 16 and row <= 2) or (col >= 16 and row >= 16)

            # 边线特殊处理：边线眼位需要至少2个己方包围
            is_edge = col == 0 or col == 18 or row == 0 or row == 18

            if is_corner_pos:
                # 角部：至少1个己方 + 棋盘边缘 = 可能的眼位候选
                if own_surround >= 1:
                    candidate_empty.add(empty_p)
            elif is_edge:
                # 边线：至少2个己方 + 棋盘边缘 = 可能的眼位候选  
                if own_surround >= 2:
                    candidate_empty.add(empty_p)
            else:
                # 中腹：需要至少2个己方控制（放宽条件）
                if own_surround >= 2:
                    candidate_empty.add(empty_p)

        # 将连通的空点分组（BFS）
        if not candidate_empty:
            return []

        remaining = set(candidate_empty)
        eye_groups = []

        while remaining:
            start = remaining.pop()
            group = {start}
            queue = [start]

            while queue:
                current = queue.pop(0)
                for neighbor in self.get_neighbors(current):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        group.add(neighbor)
                        queue.append(neighbor)

            eye_groups.append(group)

        return eye_groups

    def _is_definitely_eye(self, empty_points: Set[GoPoint], own_stones: Set[GoPoint], 
                           opp_stones: Set[GoPoint]) -> bool:
        """
        判断一组空点是否确定是眼位（完全控制）
        根据修正规则：只要区块周围没有对方的棋就是（本方/边缘）
        """
        empty_set = set(empty_points)
        
        # 检查每个空点的所有邻居
        for p in empty_points:
            neighbors = self.get_neighbors(p)
            for neighbor in neighbors:
                # 如果邻居不是空点也不是己方棋子，那么必须是对方棋子（因为棋盘上只有三种状态：空、己方、对方）
                if neighbor not in empty_set and neighbor not in own_stones:
                    # 邻居是对方棋子，所以这不是确定的眼位
                    return False
        
        return True

    def analyze_thickness(self) -> List[Dict]:
        """分析厚势"""
        analyzer = ThickShapeAnalyzer(self.black_stones, self.white_stones)
        return analyzer.analyze_thickness()
    
    def analyze(self) -> Dict:
        """执行完整分析"""
        return {
            "board_positions": self.analyze_board_positions(),
            "move_techniques": self.analyze_move_techniques(),
            "shapes": self.analyze_shapes(),
            "fuseki": self.analyze_fuseki(),
            "thickness": self.analyze_thickness(),
            "summary": {
                "move_number": self.state.get("move_number", 0),
                "game_phase": self.state.get("game_phase", "unknown"),
                "to_play": self.state.get("to_play", "unknown"),
                "black_stones": len(self.black_stones),
                "white_stones": len(self.white_stones),
                "findings": self._generate_findings()
            }
        }
    
    def _generate_findings(self) -> List[str]:
        findings = []
        
        fuseki = self.analyze_fuseki()
        if fuseki:
            unique = []
            seen = set()
            for f in fuseki:
                key = f"{f['name']}({f['color']})"
                if key not in seen:
                    seen.add(key)
                    unique.append(key)
            findings.append(f"布局流派: {', '.join(unique)}")
        
        shapes = self.analyze_shapes()
        if shapes:
            unique_shapes = list(dict.fromkeys([s['name'] for s in shapes]))
            findings.append(f"棋形: {', '.join(unique_shapes)}")
        
        thickness = self.analyze_thickness()
        if thickness:
            thick_summary = []
            seen_types = set()
            for t in thickness:
                key = f"{t['color']}{t['type']}"
                if key not in seen_types:
                    seen_types.add(key)
                    thick_summary.append(f"{t['color']}{t['type']}{t['stone_count']}子")
            findings.append(f"厚势: {', '.join(thick_summary)}")
        
        if self.last_move:
            techs = self.analyze_move_techniques()
            if techs and techs[0]["techniques"]:
                names = [t["name"] for t in techs[0]["techniques"]]
                findings.append(f"最后一手({self.last_move.coord}): {', '.join(names)}")
        
        return findings


def analyze_go_game(json_input: str) -> str:
    """
    主入口函数：分析围棋局面
    
    参数:
        json_input: JSON字符串，包含state字段（move_number, to_play, game_phase, last_move, black_stones, white_stones）
        
    返回:
        JSON字符串，包含：
        - board_positions: 棋盘位置占领情况
        - move_techniques: 最后一手着法分析
        - shapes: 死活棋形识别
        - fuseki: 布局流派识别
        - thickness: 厚势识别（新增）
        - summary: 分析摘要
    """
    try:
        data = json.loads(json_input)
        state = data.get("state", data)
        
        analyzer = GoAnalyzer(state)
        result = analyzer.analyze()
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2)


# ========== 使用示例 ==========
if __name__ == "__main__":
    example = {
        "state": {
            "move_number": 40,
            "to_play": "black",
            "game_phase": "middle",
            "last_move": {
                "color": "white",
                "point": "K16",
                "is_pass": False
            },
            "black_stones": [
                #"A3", "B3", "B2", "C2", "D2", "D1"
                "D15", "D16", "D17", "E15", "E16", "F15", "F16", "G15"
            ],
            "white_stones": [
                #"E1", "E2", "F2", "G2", "H2", "J2", "K2", "K1" 
                "Q10", "Q11", "Q12", "Q13", "R10", "R11", "S10", "S11"
            ]
        }
    }
    
    result = analyze_go_game(json.dumps(example, ensure_ascii=False))
    print(result)
