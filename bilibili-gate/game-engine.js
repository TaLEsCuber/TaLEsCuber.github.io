(function () {
  'use strict';

  // ── Config ─────────────────────────────────────
  const WIN_VALUE = 2048;  // 修改此值即可改变通关条件

  // ── State ─────────────────────────────────────
  let grid = [];
  let score = 0;
  let bestScore = 0;
  let gameOver = false;
  let won = false;
  let keepPlaying = false;
  let tileIdCounter = 0;
  const history = [];  // max 3 entries: { grid, score, gameOver, won, tileIdCounter }

  // ── localStorage ──────────────────────────────
  const LS_KEY = 'bilibili-gate-2048-best';

  function loadBestScore() {
    try { bestScore = parseInt(localStorage.getItem(LS_KEY)) || 0; }
    catch (_) { bestScore = 0; }
  }

  function saveBestScore() {
    try { localStorage.setItem(LS_KEY, String(bestScore)); }
    catch (_) { /* private browsing */ }
  }

  // ── Tile creation ─────────────────────────────
  function createTile(value) {
    return { id: ++tileIdCounter, value: value };
  }

  // ── Grid utilities ────────────────────────────
  function emptyCells() {
    const cells = [];
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++)
        if (grid[r][c] === null) cells.push({ r, c });
    return cells;
  }

  function cloneGrid() {
    return grid.map(row => row.map(t => t ? { id: t.id, value: t.value } : null));
  }

  function gridsEqual(a, b) {
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++) {
        const ta = a[r][c], tb = b[r][c];
        if ((ta === null) !== (tb === null)) return false;
        if (ta && ta.id !== tb.id) return false;
      }
    return true;
  }

  function transpose(g) {
    const t = Array.from({ length: 4 }, () => [null, null, null, null]);
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++)
        t[c][r] = g[r][c];
    return t;
  }

  function reverseRows(g) {
    return g.map(row => [...row].reverse());
  }

  // ── Core algorithm ────────────────────────────
  function slideLine(line) {
    const tiles = [];
    for (let i = 0; i < 4; i++) { if (line[i] !== null) tiles.push(line[i]); }

    let points = 0;
    const consumedIds = [];

    for (let i = 0; i < tiles.length - 1; i++) {
      if (tiles[i].value === tiles[i + 1].value) {
        tiles[i].value *= 2;
        points += tiles[i].value;
        consumedIds.push(tiles[i + 1].id);
        tiles.splice(i + 1, 1);
      }
    }

    const result = [null, null, null, null];
    for (let i = 0; i < tiles.length; i++) result[i] = tiles[i];

    return { result, points, consumedIds };
  }

  function spawnTile() {
    const cells = emptyCells();
    if (cells.length === 0) return;
    const { r, c } = cells[Math.floor(Math.random() * cells.length)];
    grid[r][c] = createTile(Math.random() < 0.9 ? 2 : 4);
  }

  function canMove() {
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++) {
        if (grid[r][c] === null) return true;
        const v = grid[r][c].value;
        if (c < 3 && grid[r][c + 1] && grid[r][c + 1].value === v) return true;
        if (r < 3 && grid[r + 1][c] && grid[r + 1][c].value === v) return true;
      }
    return false;
  }

  function checkWin() {
    if (keepPlaying) return;
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++)
        if (grid[r][c] && grid[r][c].value >= WIN_VALUE) { won = true; }
  }

  function checkGameOver() {
    if (!canMove()) gameOver = true;
  }

  // ── Move ──────────────────────────────────────
  function move(direction) {
    if (gameOver) return { moved: false };
    if (won && !keepPlaying) return { moved: false };

    const oldGrid = cloneGrid();

    // Transform so the slide direction becomes "left"
    let workGrid;
    if (direction === 'left') {
      workGrid = cloneGrid();
    } else if (direction === 'right') {
      workGrid = reverseRows(cloneGrid());
    } else if (direction === 'up') {
      workGrid = transpose(cloneGrid());
    } else if (direction === 'down') {
      const src = cloneGrid();
      workGrid = Array.from({ length: 4 }, () => [null, null, null, null]);
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++)
          workGrid[c][3 - r] = src[r][c];
    }

    let totalPoints = 0;

    for (let r = 0; r < 4; r++) {
      const { result, points } = slideLine(workGrid[r]);
      workGrid[r] = result;
      totalPoints += points;
    }

    // Inverse transform
    if (direction === 'up') {
      grid = transpose(workGrid);
    } else if (direction === 'down') {
      const result = Array.from({ length: 4 }, () => [null, null, null, null]);
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++)
          result[r][c] = workGrid[c][3 - r];
      grid = result;
    } else if (direction === 'right') {
      grid = reverseRows(workGrid);
    } else {
      grid = workGrid;
    }

    if (gridsEqual(oldGrid, grid)) return { moved: false };

    // Save pre-move state for undo (before mutations)
    history.push({
      grid: oldGrid,
      score: score,
      gameOver: gameOver,
      won: won,
      tileIdCounter: tileIdCounter
    });
    if (history.length > 3) history.shift();

    score += totalPoints;
    if (score > bestScore) { bestScore = score; saveBestScore(); }

    spawnTile();
    checkWin();
    checkGameOver();

    return { moved: true, oldGrid: oldGrid };
  }

  // ── Undo ──────────────────────────────────────
  function undo() {
    if (history.length === 0) return null;
    const entry = history.pop();
    const currentGrid = cloneGrid();
    grid = entry.grid;
    score = entry.score;
    gameOver = entry.gameOver;
    won = entry.won;
    tileIdCounter = entry.tileIdCounter;
    return { oldGrid: currentGrid };
  }

  // ── Init ──────────────────────────────────────
  function init() {
    score = 0;
    gameOver = false;
    won = false;
    keepPlaying = false;
    tileIdCounter = 0;
    history.length = 0;
    grid = Array.from({ length: 4 }, () => [null, null, null, null]);
    loadBestScore();
    spawnTile();
    spawnTile();
  }

  // ── Public API ────────────────────────────────
  window.GameEngine = {
    get grid() { return grid; },
    get score() { return score; },
    get bestScore() { return bestScore; },
    get gameOver() { return gameOver; },
    get won() { return won; },
    get keepPlaying() { return keepPlaying; },
    set keepPlaying(v) { keepPlaying = v; },
    set won(v) { won = v; },

    init: init,
    move: move,
    undo: undo,
    get historyLength() { return history.length; },
    loadBestScore: loadBestScore,
    saveBestScore: saveBestScore,

    // Debug helpers
    createTile: createTile,
    emptyCells: emptyCells,
    slideLine: slideLine,
    cloneGrid: cloneGrid,
    gridsEqual: gridsEqual,
    transpose: transpose,
    reverseRows: reverseRows,
    spawnTile: spawnTile,
    canMove: canMove,
    checkWin: checkWin,
    checkGameOver: checkGameOver,
  };
})();
