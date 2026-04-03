#!/usr/bin/env python3
"""Simple web app for browsing Go game steps under outputs/."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
STEP_FILE_PATTERN = re.compile(r"step_(\d+)\.json$")
HOST = "127.0.0.1"
PORT = 8000


def json_response(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(
    handler: BaseHTTPRequestHandler,
    text: str,
    content_type: str = "text/plain; charset=utf-8",
    status: int = 200,
) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _step_num(path: Path) -> int:
    m = STEP_FILE_PATTERN.search(path.name)
    if not m:
        return -1
    return int(m.group(1))


@lru_cache(maxsize=1)
def discover_games() -> list[str]:
    if not OUTPUTS_DIR.exists():
        return []
    games: list[str] = []
    for directory in OUTPUTS_DIR.iterdir():
        if not directory.is_dir():
            continue
        if any(STEP_FILE_PATTERN.match(p.name) for p in directory.glob("step_*.json")):
            games.append(directory.name)
    games.sort()
    return games


@lru_cache(maxsize=64)
def step_files(game_name: str) -> list[Path]:
    game_dir = OUTPUTS_DIR / game_name
    if not game_dir.is_dir():
        return []
    files = [p for p in game_dir.glob("step_*.json") if STEP_FILE_PATTERN.match(p.name)]
    files.sort(key=_step_num)
    return files


@lru_cache(maxsize=4096)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_step(game_name: str, step_index: int) -> dict | None:
    files = step_files(game_name)
    if step_index < 1 or step_index > len(files):
        return None
    raw = load_json(str(files[step_index - 1]))
    return {
        "step_index": step_index,
        "step_count": len(files),
        "filename": files[step_index - 1].name,
        "payload": raw,
    }


def page_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>围棋棋局查看器</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 18px; background: #fafafa; color: #1f2328; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 16px; }
    .toolbar input, .toolbar select, .toolbar button { padding: 6px 10px; font-size: 14px; }
    .layout { display: block; }
    .panel { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    #board { width: 100%; max-width: 760px; aspect-ratio: 1 / 1; border-radius: 8px; background: #e7bf78; display: block; }
    .title { margin: 0 0 8px; font-size: 16px; font-weight: 700; }
    .muted { color: #57606a; font-size: 13px; }
    .hint-current { margin: 0 0 10px; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 8px 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .stats { margin: 6px 0; font-size: 14px; }
  </style>
</head>
<body>
  <div class="toolbar">
    <label>对局
      <select id="game-select"></select>
    </label>
    <button id="prev-btn">上一步</button>
    <button id="next-btn">下一步</button>
    <label>跳转到第
      <input id="jump-input" type="number" min="1" step="1" style="width: 90px;" />
      步
    </label>
    <button id="jump-btn">跳转</button>
    <span id="step-text"></span>
  </div>

  <div class="layout">
    <div class="panel">
      <h3 class="title">当前步着法提示</h3>
      <div id="hint-current" class="hint-current">-</div>
      <h3 class="title">棋盘</h3>
      <canvas id="board" width="760" height="760"></canvas>
      <div class="stats" id="game-meta"></div>
      <div class="muted" id="move-meta"></div>
    </div>
  </div>

  <script>
    const LETTERS = "ABCDEFGHJKLMNOPQRSTUVWX";
    const state = {
      game: "",
      step: 1,
      stepCount: 1,
      boardSize: 19,
      payload: null
    };

    async function getJson(url) {
      const res = await fetch(url);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Request failed: ${res.status} ${text}`);
      }
      return await res.json();
    }

    function parsePoint(point, boardSize) {
      if (!point || typeof point !== "string") return null;
      const text = point.trim().toUpperCase();
      const match = text.match(/^([A-Z]+)(\\d+)$/);
      if (!match) return null;
      const colLetters = match[1];
      const rowNum = Number(match[2]);
      if (!Number.isFinite(rowNum) || rowNum < 1 || rowNum > boardSize) return null;
      let col = 0;
      for (let i = 0; i < colLetters.length; i += 1) {
        const idx = LETTERS.indexOf(colLetters[i]);
        if (idx === -1) return null;
        col = col * LETTERS.length + (idx + 1);
      }
      col -= 1;
      if (col < 0 || col >= boardSize) return null;
      const x = col;
      const y = boardSize - rowNum;
      return { x, y };
    }

    function drawBoard() {
      const canvas = document.getElementById("board");
      const ctx = canvas.getContext("2d");
      const size = state.boardSize || 19;
      const W = canvas.width;
      const H = canvas.height;
      const pad = Math.round(W * 0.07);
      const span = W - pad * 2;
      const step = span / (size - 1);

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#e7bf78";
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = "#2b2113";
      ctx.lineWidth = 1;
      for (let i = 0; i < size; i += 1) {
        const pos = pad + i * step;
        ctx.beginPath();
        ctx.moveTo(pad, pos);
        ctx.lineTo(W - pad, pos);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(pos, pad);
        ctx.lineTo(pos, H - pad);
        ctx.stroke();
      }

      drawStarPoints(ctx, size, pad, step);

      const payload = state.payload || {};
      const boardState = payload.state || {};
      const blacks = Array.isArray(boardState.black_stones) ? boardState.black_stones : [];
      const whites = Array.isArray(boardState.white_stones) ? boardState.white_stones : [];
      const last = boardState.last_move || null;

      blacks.forEach((pt) => drawStone(ctx, pt, "black", size, pad, step));
      whites.forEach((pt) => drawStone(ctx, pt, "white", size, pad, step));
      if (last && !last.is_pass && last.point) {
        drawLastMoveMark(ctx, last.point, size, pad, step);
      }
    }

    function drawStarPoints(ctx, size, pad, step) {
      const stars = [];
      if (size === 19) {
        [3, 9, 15].forEach((x) => [3, 9, 15].forEach((y) => stars.push([x, y])));
      } else if (size === 13) {
        [3, 6, 9].forEach((x) => [3, 6, 9].forEach((y) => stars.push([x, y])));
      } else if (size === 9) {
        [2, 4, 6].forEach((x) => [2, 4, 6].forEach((y) => stars.push([x, y])));
      }
      ctx.fillStyle = "#2b2113";
      stars.forEach(([x, y]) => {
        const cx = pad + x * step;
        const cy = pad + y * step;
        ctx.beginPath();
        ctx.arc(cx, cy, Math.max(3, step * 0.08), 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function drawStone(ctx, point, color, size, pad, step) {
      const p = parsePoint(point, size);
      if (!p) return;
      const cx = pad + p.x * step;
      const cy = pad + p.y * step;
      const r = step * 0.45;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = color === "black" ? "#101010" : "#f6f6f6";
      ctx.fill();
      ctx.strokeStyle = color === "black" ? "#181818" : "#bbbbbb";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    function drawLastMoveMark(ctx, point, size, pad, step) {
      const p = parsePoint(point, size);
      if (!p) return;
      const cx = pad + p.x * step;
      const cy = pad + p.y * step;
      const r = step * 0.2;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = "#d1242f";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    async function loadGames() {
      const data = await getJson("/api/games");
      const select = document.getElementById("game-select");
      select.innerHTML = "";
      if (!data.games.length) {
        throw new Error("未发现 outputs 下的对局数据");
      }
      data.games.forEach((g) => {
        const option = document.createElement("option");
        option.value = g.id;
        option.textContent = `${g.id} (${g.step_count} 步文件)`;
        select.appendChild(option);
      });
      state.game = data.default_game || data.games[0].id;
      select.value = state.game;
    }

    async function loadStep(step) {
      const data = await getJson(`/api/game/${encodeURIComponent(state.game)}/step/${step}`);
      state.step = data.step_index;
      state.stepCount = data.step_count;
      state.payload = data.payload;
      state.boardSize = Number(data.payload?.rules?.board_size) || 19;

      const sem = data.payload.semantic_context || {};
      document.getElementById("hint-current").textContent = sem.move_location_hint || "(空)";
      document.getElementById("step-text").textContent = `第 ${state.step} / ${state.stepCount} 步`;
      document.getElementById("jump-input").value = String(state.step);

      const moveNumber = data.payload?.state?.move_number;
      const toPlay = data.payload?.state?.to_play || "-";
      document.getElementById("move-meta").textContent =
        `文件: ${data.filename} | move_number: ${moveNumber} | to_play: ${toPlay}`;
      document.getElementById("game-meta").textContent =
        `棋盘尺寸: ${state.boardSize}x${state.boardSize}`;

      drawBoard();
      updateButtons();
    }

    function updateButtons() {
      document.getElementById("prev-btn").disabled = state.step <= 1;
      document.getElementById("next-btn").disabled = state.step >= state.stepCount;
    }

    async function switchGame(game) {
      state.game = game;
      await loadStep(1);
    }

    function bindEvents() {
      document.getElementById("game-select").addEventListener("change", async (e) => {
        await switchGame(e.target.value);
      });
      document.getElementById("prev-btn").addEventListener("click", async () => {
        if (state.step > 1) await loadStep(state.step - 1);
      });
      document.getElementById("next-btn").addEventListener("click", async () => {
        if (state.step < state.stepCount) await loadStep(state.step + 1);
      });
      document.getElementById("jump-btn").addEventListener("click", async () => {
        const v = Number(document.getElementById("jump-input").value);
        if (!Number.isFinite(v)) return;
        const n = Math.max(1, Math.min(state.stepCount, Math.trunc(v)));
        await loadStep(n);
      });
      document.getElementById("jump-input").addEventListener("keydown", async (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          document.getElementById("jump-btn").click();
        }
      });
    }

    async function boot() {
      try {
        bindEvents();
        await loadGames();
        await switchGame(state.game);
      } catch (err) {
        alert(err.message || String(err));
      }
    }

    boot();
  </script>
</body>
</html>
"""


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            text_response(self, page_html(), content_type="text/html; charset=utf-8")
            return

        if path == "/api/games":
            games = discover_games()
            data = [{"id": name, "step_count": len(step_files(name))} for name in games]
            default_game = games[-1] if games else ""
            json_response(self, {"games": data, "default_game": default_game})
            return

        m_step = re.match(r"^/api/game/([^/]+)/step/(\d+)$", path)
        if m_step:
            game_name = unquote(m_step.group(1))
            step_index = int(m_step.group(2))
            info = load_step(game_name, step_index)
            if info is None:
                json_response(self, {"error": "step not found"}, status=HTTPStatus.NOT_FOUND)
                return
            json_response(self, info)
            return

        json_response(self, {"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    if not OUTPUTS_DIR.exists():
        raise SystemExit(f"outputs directory not found: {OUTPUTS_DIR}")
    with ThreadingHTTPServer((HOST, PORT), AppHandler) as httpd:
        print(f"Serving on http://{HOST}:{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
