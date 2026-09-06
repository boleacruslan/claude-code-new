// UI layer: rendering the board, handling input, and wiring up game modes.

let state = createInitialState();
let selectedSquare = null;
let legalMovesForSelected = [];
let lastMove = null;
let boardFlipped = false;
let gameOver = false;
let pendingPromotion = null; // {from, to} awaiting user's piece choice
let mode = "pvp"; // 'pvp' | 'ai'
let humanColor = "w";
let aiDepth = 3;
let aiThinking = false;

const boardEl = document.getElementById("board");
const turnIndicatorEl = document.getElementById("turnIndicator");
const statusMessageEl = document.getElementById("statusMessage");
const moveListEl = document.getElementById("moveList");
const capturedTopEl = document.getElementById("capturedTop");
const capturedBottomEl = document.getElementById("capturedBottom");
const promotionModal = document.getElementById("promotionModal");
const promotionChoices = document.getElementById("promotionChoices");
const gameOverModal = document.getElementById("gameOverModal");
const gameOverText = document.getElementById("gameOverText");
const aiOptions = document.getElementById("aiOptions");

document.querySelectorAll('input[name="mode"]').forEach((el) => {
  el.addEventListener("change", (e) => {
    mode = e.target.value;
    aiOptions.classList.toggle("hidden", mode !== "ai");
    startNewGame();
  });
});
document.getElementById("humanColor").addEventListener("change", (e) => {
  humanColor = e.target.value;
  startNewGame();
});
document.getElementById("difficulty").addEventListener("change", (e) => {
  aiDepth = parseInt(e.target.value, 10) + 1;
});
document.getElementById("newGameBtn").addEventListener("click", startNewGame);
document.getElementById("undoBtn").addEventListener("click", undoMove);
document.getElementById("flipBtn").addEventListener("click", () => {
  boardFlipped = !boardFlipped;
  renderBoard();
});
document.getElementById("gameOverCloseBtn").addEventListener("click", () => {
  gameOverModal.classList.add("hidden");
});

function startNewGame() {
  state = createInitialState();
  selectedSquare = null;
  legalMovesForSelected = [];
  lastMove = null;
  gameOver = false;
  aiThinking = false;
  boardFlipped = mode === "ai" && humanColor === "b";
  gameOverModal.classList.add("hidden");
  promotionModal.classList.add("hidden");
  moveListEl.innerHTML = "";
  renderBoard();
  updateStatus();
  maybeTriggerAiMove();
}

function displayRowCol(displayRow, displayCol) {
  // Converts on-screen grid position to board row/col, honoring the flip state.
  if (boardFlipped) {
    return { r: 7 - displayRow, c: 7 - displayCol };
  }
  return { r: displayRow, c: displayCol };
}

function renderBoard() {
  boardEl.innerHTML = "";
  const kingInCheckPos = (() => {
    const status = getGameStatus(state);
    if (!status.inCheck) return null;
    return findKing(state.board, state.turn);
  })();

  for (let displayRow = 0; displayRow < 8; displayRow++) {
    for (let displayCol = 0; displayCol < 8; displayCol++) {
      const { r, c } = displayRowCol(displayRow, displayCol);
      const square = document.createElement("div");
      square.className = "square " + ((r + c) % 2 === 0 ? "light" : "dark");
      square.dataset.r = r;
      square.dataset.c = c;

      if (selectedSquare && selectedSquare.r === r && selectedSquare.c === c) {
        square.classList.add("selected");
      }
      if (lastMove && ((lastMove.from.r === r && lastMove.from.c === c) || (lastMove.to.r === r && lastMove.to.c === c))) {
        square.classList.add("last-move");
      }
      if (kingInCheckPos && kingInCheckPos.r === r && kingInCheckPos.c === c) {
        square.classList.add("in-check");
      }

      const piece = state.board[r][c];
      if (piece) {
        const pieceEl = document.createElement("span");
        pieceEl.className = "piece";
        pieceEl.textContent = PIECE_UNICODE[piece.color][piece.type];
        square.appendChild(pieceEl);
      }

      const move = legalMovesForSelected.find((m) => m.to.r === r && m.to.c === c);
      if (move) {
        const marker = document.createElement("div");
        marker.className = move.captured ? "capture-ring" : "move-dot";
        square.appendChild(marker);
      }

      if (displayCol === 0) {
        const label = document.createElement("span");
        label.className = "square-label";
        label.textContent = 8 - r;
        square.appendChild(label);
      }

      square.addEventListener("click", () => onSquareClick(r, c));
      boardEl.appendChild(square);
    }
  }

  renderCapturedPieces();
}

function renderCapturedPieces() {
  const captured = { w: [], b: [] };
  for (const move of state.history) {
    if (move.captured) captured[move.captured.color].push(move.captured.type);
  }
  const order = { q: 0, r: 1, b: 2, n: 3, p: 4 };
  captured.w.sort((a, b) => order[a] - order[b]);
  captured.b.sort((a, b) => order[a] - order[b]);

  // Top row shows pieces captured FROM the player shown at top of board, i.e. pieces of the opposite color removed.
  const topIsWhite = boardFlipped; // if flipped, white sits at top
  const topCaptured = topIsWhite ? captured.b : captured.w;
  const bottomCaptured = topIsWhite ? captured.w : captured.b;
  const topColor = topIsWhite ? "w" : "b";
  const bottomColor = topIsWhite ? "b" : "w";

  capturedTopEl.innerHTML = topCaptured.map((t) => PIECE_UNICODE[topColor][t]).join(" ");
  capturedBottomEl.innerHTML = bottomCaptured.map((t) => PIECE_UNICODE[bottomColor][t]).join(" ");
}

function onSquareClick(r, c) {
  if (gameOver || aiThinking || pendingPromotion) return;
  if (mode === "ai" && state.turn !== humanColor) return;

  const piece = state.board[r][c];

  if (selectedSquare) {
    const move = legalMovesForSelected.find((m) => m.to.r === r && m.to.c === c);
    if (move) {
      if (move.promotion) {
        askPromotion(move);
      } else {
        performMove(move);
      }
      return;
    }
    // Clicking another own piece re-selects instead of attempting an illegal move.
    if (piece && piece.color === state.turn) {
      selectSquare(r, c);
    } else {
      clearSelection();
      renderBoard();
    }
    return;
  }

  if (piece && piece.color === state.turn) {
    selectSquare(r, c);
  }
}

function selectSquare(r, c) {
  selectedSquare = { r, c };
  const allLegal = generateLegalMoves(state, state.turn);
  legalMovesForSelected = allLegal.filter((m) => m.from.r === r && m.from.c === c);
  renderBoard();
}

function clearSelection() {
  selectedSquare = null;
  legalMovesForSelected = [];
}

function askPromotion(move) {
  pendingPromotion = move;
  promotionChoices.innerHTML = "";
  const color = move.piece.color;
  for (const type of ["q", "r", "b", "n"]) {
    const btn = document.createElement("button");
    btn.textContent = PIECE_UNICODE[color][type];
    btn.addEventListener("click", () => {
      const chosenMove = { ...move, promotion: type };
      promotionModal.classList.add("hidden");
      pendingPromotion = null;
      performMove(chosenMove);
    });
    promotionChoices.appendChild(btn);
  }
  promotionModal.classList.remove("hidden");
}

function performMove(move) {
  const colorMoving = state.turn;
  const moveNumber = state.fullMoveNumber;
  const legalForNotation = generateLegalMoves(state, colorMoving);
  const notation = moveToNotation(state, move, legalForNotation);

  applyMove(state, move);
  lastMove = move;
  clearSelection();

  const status = getGameStatus(state);
  let suffix = "";
  if (status.status === "checkmate") suffix = "#";
  else if (status.inCheck) suffix = "+";

  appendMoveToList(colorMoving, notation + suffix, moveNumber);
  renderBoard();
  updateStatus();

  if (status.status === "checkmate" || status.status === "stalemate" || status.status === "draw-50move") {
    endGame(status);
    return;
  }

  maybeTriggerAiMove();
}

function appendMoveToList(color, text, moveNumber) {
  if (color === "w") {
    const li = document.createElement("li");
    li.className = "move-pair";
    const num = document.createElement("span");
    num.textContent = moveNumber + ".";
    const whiteSpan = document.createElement("span");
    whiteSpan.className = "white-move";
    whiteSpan.textContent = text;
    li.appendChild(num);
    li.appendChild(whiteSpan);
    const blackSpan = document.createElement("span");
    blackSpan.className = "black-move";
    li.appendChild(blackSpan);
    moveListEl.appendChild(li);
  } else {
    const lastLi = moveListEl.lastElementChild;
    if (lastLi) {
      lastLi.querySelector(".black-move").textContent = text;
    } else {
      const li = document.createElement("li");
      li.className = "move-pair";
      const num = document.createElement("span");
      num.textContent = moveNumber + "...";
      const blackSpan = document.createElement("span");
      blackSpan.className = "black-move";
      blackSpan.textContent = text;
      li.appendChild(num);
      li.appendChild(document.createElement("span"));
      li.appendChild(blackSpan);
      moveListEl.appendChild(li);
    }
  }
  moveListEl.scrollTop = moveListEl.scrollHeight;
}

function updateStatus() {
  const status = getGameStatus(state);
  turnIndicatorEl.textContent = "Ход " + (state.turn === "w" ? "белых" : "чёрных");

  if (status.status === "check") {
    statusMessageEl.textContent = "Шах!";
  } else if (status.status === "checkmate") {
    statusMessageEl.textContent = "Мат!";
  } else if (status.status === "stalemate") {
    statusMessageEl.textContent = "Пат — ничья.";
  } else if (status.status === "draw-50move") {
    statusMessageEl.textContent = "Ничья по правилу 50 ходов.";
  } else {
    statusMessageEl.textContent = "";
  }
}

function endGame(status) {
  gameOver = true;
  let text;
  if (status.status === "checkmate") {
    text = "Мат! Победили " + (status.winner === "w" ? "белые" : "чёрные") + ".";
  } else if (status.status === "stalemate") {
    text = "Пат. Ничья.";
  } else {
    text = "Ничья по правилу 50 ходов.";
  }
  gameOverText.textContent = text;
  gameOverModal.classList.remove("hidden");
}

function undoMove() {
  if (aiThinking) return;
  if (state.history.length === 0) return;

  const movesToUndo = mode === "ai" && state.history.length >= 2 ? 2 : 1;
  const newHistory = state.history.slice(0, state.history.length - movesToUndo);

  const fresh = createInitialState();
  for (const move of newHistory) {
    applyMove(fresh, move);
  }
  state = fresh;
  lastMove = state.history.length > 0 ? state.history[state.history.length - 1] : null;
  clearSelection();
  gameOver = false;
  gameOverModal.classList.add("hidden");
  renderBoard();
  updateStatus();
}

function maybeTriggerAiMove() {
  if (mode !== "ai" || gameOver) return;
  if (state.turn === humanColor) return;

  aiThinking = true;
  statusMessageEl.textContent = "Компьютер думает...";
  setTimeout(() => {
    const move = findBestMove(state, aiDepth);
    aiThinking = false;
    if (!move) {
      updateStatus();
      return;
    }
    performMove(move);
  }, 50);
}

startNewGame();
