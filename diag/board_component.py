"""Drag-and-drop chess board for Gradio (gr.HTML + a bit of JavaScript).

The board is rendered client side from a plain dict (see ``ChessGUI.board_value``).
When the user drops a piece on a legal square the JS calls
``trigger('click', {action: 'move', uci: 'e2e4'})`` which Python handles with a
normal ``board.click(...)`` listener.
"""

from __future__ import annotations

import base64
import json

import chess
import chess.svg


def _piece_icons() -> dict:
    icons = {}
    for color in (chess.WHITE, chess.BLACK):
        for pt in range(1, 7):
            piece = chess.Piece(pt, color)
            svg = chess.svg.piece(piece, size=90)
            icons[piece.symbol()] = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return icons


HTML_TEMPLATE = """
<div class="cb-wrap">
  <div class="cb-eval" title="Evaluation (white's point of view)"><div class="cb-eval-white"></div><span class="cb-eval-text"></span></div>
  <div class="cb-root"></div>
</div>
"""

CSS_TEMPLATE = """
.cb-wrap { display:flex; gap:8px; align-items:stretch; width:100%; max-width:560px; margin:0 auto; user-select:none; }
.cb-eval { position:relative; width:22px; flex:none; background:#403d39; border-radius:4px; overflow:hidden; }
.cb-eval-white { position:absolute; left:0; right:0; bottom:0; height:50%; background:#f4f1ea; transition:height .35s ease; }
.cb-eval.flipped .cb-eval-white { bottom:auto; top:0; }
.cb-eval-text { position:absolute; left:0; right:0; font:600 10px/1 sans-serif; text-align:center; mix-blend-mode:difference; color:#fff; }
.cb-eval:not(.flipped) .cb-eval-text { bottom:3px; }
.cb-eval.flipped .cb-eval-text { top:3px; }
.cb-root { flex:1; min-width:0; }
.cb-board { position:relative; width:100%; aspect-ratio:1/1; touch-action:none; border-radius:4px; overflow:hidden; box-shadow:0 2px 10px rgba(0,0,0,.25); }
.cb-sq { position:absolute; width:12.5%; height:12.5%; }
.cb-sq.light { background:#f0d9b5; } .cb-sq.dark { background:#b58863; }
.cb-sq.last::after { content:''; position:absolute; inset:0; background:rgba(255,235,59,.45); }
.cb-sq.check::after { content:''; position:absolute; inset:0; background:radial-gradient(circle, rgba(255,0,0,.85) 0%, rgba(255,0,0,0) 70%); }
.cb-sq.sel::after { content:''; position:absolute; inset:0; background:rgba(20,140,255,.45); }
.cb-sq.target::before { content:''; position:absolute; left:35%; top:35%; width:30%; height:30%; border-radius:50%; background:rgba(20,85,30,.5); z-index:3; pointer-events:none; }
.cb-sq.target.occ::before { left:0; top:0; width:100%; height:100%; border-radius:0; background:radial-gradient(circle, rgba(0,0,0,0) 58%, rgba(20,85,30,.5) 60%); }
.cb-sq.hover { box-shadow: inset 0 0 0 3px rgba(255,255,255,.85); }
.cb-coord { position:absolute; font:600 10px/1 sans-serif; pointer-events:none; z-index:1; }
.cb-coord.f { right:3px; bottom:2px; } .cb-coord.r { left:3px; top:2px; }
.cb-sq.light .cb-coord { color:#b58863; } .cb-sq.dark .cb-coord { color:#f0d9b5; }
.cb-wrap img, .cb-wrap .cb-piece { margin:0 !important; max-width:none !important; border:0 !important; }
.cb-piece { position:absolute; width:12.5%; height:12.5%; z-index:4; cursor:grab; transition:left .12s ease, top .12s ease; -webkit-user-drag:none; }
.cb-piece.dragging { cursor:grabbing; z-index:20; transition:none; filter:drop-shadow(0 4px 6px rgba(0,0,0,.4)); }
.cb-piece.locked { cursor:default; }
.cb-promo { position:absolute; z-index:30; background:#fff; border-radius:6px; box-shadow:0 4px 14px rgba(0,0,0,.4); display:flex; flex-direction:column; width:12.5%; }
.cb-promo img { width:100%; cursor:pointer; } .cb-promo img:hover { background:#dfeeff; }
.cb-banner { position:absolute; left:0; right:0; top:50%; transform:translateY(-50%); z-index:25; text-align:center; background:rgba(0,0,0,.72); color:#fff; padding:10px; font:700 18px sans-serif; pointer-events:none; }
"""

JS_ON_LOAD = r"""
const ICONS = __ICONS__;
const FILES = 'abcdefgh';
let drag = null;      // active drag: {from, el, ox, oy, moved, startLeft, startTop}
let selected = null;  // selected square for click-to-move

function boardEl() { return element.querySelector('.cb-board'); }
function orient() { return (props.value && props.value.orientation) === 'black' ? 'black' : 'white'; }
function pos(f, r) {  // file/rank (0..7) -> percent left/top
  return orient() === 'white' ? [f * 12.5, (7 - r) * 12.5] : [(7 - f) * 12.5, r * 12.5];
}
function squareAt(x, y) {
  const b = boardEl(); if (!b) return null;
  const rc = b.getBoundingClientRect();
  if (x < rc.left || x >= rc.right || y < rc.top || y >= rc.bottom) return null;
  let f = Math.floor((x - rc.left) / rc.width * 8), r = Math.floor((y - rc.top) / rc.height * 8);
  if (orient() === 'white') r = 7 - r; else f = 7 - f;
  return FILES[f] + (r + 1);
}
function clearMarks() {
  element.querySelectorAll('.cb-sq').forEach(s => s.classList.remove('sel', 'target', 'occ', 'hover'));
}
function showTargets(from) {
  clearMarks();
  const v = props.value; if (!v || !v.legal || !v.legal[from]) return;
  const src = element.querySelector('.cb-sq[data-sq="' + from + '"]'); if (src) src.classList.add('sel');
  new Set(v.legal[from].map(u => u.slice(2, 4))).forEach(t => {
    const s = element.querySelector('.cb-sq[data-sq="' + t + '"]');
    if (s) { s.classList.add('target'); if (v.pieces[t]) s.classList.add('occ'); }
  });
}
function sendMove(uci) {
  trigger('click', { action: 'move', uci: uci });
}
function commit(from, to) {
  const v = props.value;
  const ucis = (v.legal[from] || []).filter(u => u.slice(2, 4) === to);
  if (!ucis.length) return false;
  if (ucis.length === 1) { applyOptimistic(from, to); sendMove(ucis[0]); return true; }
  // promotion: let the user choose
  const b = boardEl(); const [l, t] = pos(FILES.indexOf(to[0]), parseInt(to[1]) - 1);
  const menu = document.createElement('div'); menu.className = 'cb-promo';
  menu.style.left = l + '%'; menu.style.top = (t > 50 ? t - 37.5 : t) + '%';
  const color = v.turn === 'w' ? 'w' : 'b';
  ['q', 'r', 'b', 'n'].forEach(p => {
    const img = document.createElement('img'); img.src = ICONS[color === 'w' ? p.toUpperCase() : p];
    img.addEventListener('pointerdown', ev => { ev.stopPropagation(); ev.preventDefault(); menu.remove(); applyOptimistic(from, to); sendMove(from + to + p); });
    menu.appendChild(img);
  });
  b.appendChild(menu);
  return true;
}
function applyOptimistic(from, to) {
  const mover = element.querySelector('.cb-piece[data-sq="' + from + '"]');
  const victim = element.querySelector('.cb-piece[data-sq="' + to + '"]');
  if (victim && victim !== mover) victim.remove();
  if (mover) {
    const [l, t] = pos(FILES.indexOf(to[0]), parseInt(to[1]) - 1);
    mover.style.left = l + '%'; mover.style.top = t + '%'; mover.dataset.sq = to; mover.classList.remove('dragging');
  }
  selected = null; clearMarks();
}

function render() {
  const v = props.value; const root = element.querySelector('.cb-root');
  if (!v || !root || !v.pieces) return;
  const o = orient();
  const evalBar = element.querySelector('.cb-eval');
  if (evalBar) {
    evalBar.classList.toggle('flipped', o === 'black');
    const ev = v.eval;
    element.querySelector('.cb-eval-white').style.height = (ev ? ev.white_pct : 50) + '%';
    element.querySelector('.cb-eval-text').textContent = ev ? ev.text : '';
  }
  const board = document.createElement('div'); board.className = 'cb-board';
  for (let f = 0; f < 8; f++) for (let r = 0; r < 8; r++) {
    const name = FILES[f] + (r + 1), [l, t] = pos(f, r);
    const sq = document.createElement('div');
    sq.className = 'cb-sq ' + ((f + r) % 2 === 0 ? 'dark' : 'light');
    sq.dataset.sq = name; sq.style.left = l + '%'; sq.style.top = t + '%';
    if (v.last && (v.last[0] === name || v.last[1] === name)) sq.classList.add('last');
    if (v.check === name) sq.classList.add('check');
    const bottomRank = o === 'white' ? 0 : 7, leftFile = o === 'white' ? 0 : 7;
    if (r === bottomRank) { const c = document.createElement('span'); c.className = 'cb-coord f'; c.textContent = FILES[f]; sq.appendChild(c); }
    if (f === leftFile) { const c = document.createElement('span'); c.className = 'cb-coord r'; c.textContent = String(r + 1); sq.appendChild(c); }
    board.appendChild(sq);
  }
  Object.keys(v.pieces).forEach(name => {
    const img = document.createElement('img'); img.className = 'cb-piece'; img.draggable = false;
    img.src = ICONS[v.pieces[name]]; img.dataset.sq = name;
    const [l, t] = pos(FILES.indexOf(name[0]), parseInt(name[1]) - 1);
    img.style.left = l + '%'; img.style.top = t + '%';
    if (!(v.legal && v.legal[name])) img.classList.add('locked');
    board.appendChild(img);
  });
  if (v.over) { const bn = document.createElement('div'); bn.className = 'cb-banner'; bn.textContent = v.over; board.appendChild(bn); }
  root.replaceChildren(board);
  if (selected && v.legal && v.legal[selected]) showTargets(selected); else selected = null;
}

function onMove(e) {
  if (!drag) return;
  const b = boardEl(); if (!b) return;
  const rc = b.getBoundingClientRect(), s = rc.width / 8;
  if (!drag.moved && Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) < 4) return;
  if (!drag.moved) { drag.moved = true; drag.el.classList.add('dragging'); }
  drag.el.style.left = (e.clientX - rc.left - s / 2) + 'px';
  drag.el.style.top = (e.clientY - rc.top - s / 2) + 'px';
  element.querySelectorAll('.cb-sq.hover').forEach(x => x.classList.remove('hover'));
  const sq = squareAt(e.clientX, e.clientY);
  if (sq) { const t = element.querySelector('.cb-sq[data-sq="' + sq + '"]'); if (t && t.classList.contains('target')) t.classList.add('hover'); }
}
function onUp(e) {
  document.removeEventListener('pointermove', onMove);
  document.removeEventListener('pointerup', onUp);
  document.removeEventListener('pointercancel', onUp);
  if (!drag) return;
  const d = drag; drag = null;
  element.querySelectorAll('.cb-sq.hover').forEach(x => x.classList.remove('hover'));
  if (!d.moved) return;                       // simple click: keep the selection for click-to-move
  const to = squareAt(e.clientX, e.clientY);
  d.el.classList.remove('dragging');
  if (to && to !== d.from && commit(d.from, to)) return;
  d.el.style.left = d.startLeft; d.el.style.top = d.startTop;   // illegal drop: snap back
  if (to === d.from) { selected = d.from; showTargets(d.from); }
}

if (!element.__cbInit) {
  element.__cbInit = true;
  element.addEventListener('pointerdown', e => {
    if (e.button !== undefined && e.button !== 0) return;
    if (e.target.closest && e.target.closest('.cb-promo')) return;
    const v = props.value; if (!v || !v.legal || v.over) return;
    const sq = squareAt(e.clientX, e.clientY); if (!sq) return;
    element.querySelectorAll('.cb-promo').forEach(m => m.remove());
    if (selected && selected !== sq && (v.legal[selected] || []).some(u => u.slice(2, 4) === sq)) {
      e.preventDefault(); commit(selected, sq); return;      // click-to-move (second click)
    }
    if (v.legal[sq]) {
      const el = element.querySelector('.cb-piece[data-sq="' + sq + '"]'); if (!el) return;
      e.preventDefault();
      selected = sq; showTargets(sq);
      drag = { from: sq, el: el, sx: e.clientX, sy: e.clientY, moved: false, startLeft: el.style.left, startTop: el.style.top };
      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', onUp);
      document.addEventListener('pointercancel', onUp);
    } else { selected = null; clearMarks(); }
  });
}
render();
watch('value', render);
"""


def js_on_load() -> str:
    return JS_ON_LOAD.replace("__ICONS__", json.dumps(_piece_icons()))
