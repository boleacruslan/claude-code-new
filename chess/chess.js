// Chess rules engine: board representation, move generation, and game state.
// Board: 8x8 array, row 0 = rank 8 (top, black side), row 7 = rank 1 (bottom, white side).
// col 0 = file 'a', col 7 = file 'h'.

const PIECE_UNICODE = {
  w: { p: "♙", n: "♘", b: "♗", r: "♖", q: "♕", k: "♔" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
};

function createInitialState() {
  const back = ["r", "n", "b", "q", "k", "b", "n", "r"];
  const board = Array.from({ length: 8 }, () => Array(8).fill(null));

  for (let c = 0; c < 8; c++) {
    board[0][c] = { type: back[c], color: "b" };
    board[1][c] = { type: "p", color: "b" };
    board[6][c] = { type: "p", color: "w" };
    board[7][c] = { type: back[c], color: "w" };
  }

  return {
    board,
    turn: "w",
    castling: { wK: true, wQ: true, bK: true, bQ: true },
    enPassant: null, // {r, c} square that can be captured onto
    halfMoveClock: 0,
    fullMoveNumber: 1,
    history: [], // list of move records for undo + notation
  };
}

function cloneState(state) {
  return {
    board: state.board.map((row) => row.map((p) => (p ? { ...p } : null))),
    turn: state.turn,
    castling: { ...state.castling },
    enPassant: state.enPassant ? { ...state.enPassant } : null,
    halfMoveClock: state.halfMoveClock,
    fullMoveNumber: state.fullMoveNumber,
    history: state.history.slice(),
  };
}

function isInside(r, c) {
  return r >= 0 && r < 8 && c >= 0 && c < 8;
}

function opponent(color) {
  return color === "w" ? "b" : "w";
}

function squareName(r, c) {
  return "abcdefgh"[c] + (8 - r);
}

const KNIGHT_OFFSETS = [
  [-2, -1], [-2, 1], [-1, -2], [-1, 2],
  [1, -2], [1, 2], [2, -1], [2, 1],
];
const BISHOP_DIRS = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
const ROOK_DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const KING_OFFSETS = [...BISHOP_DIRS, ...ROOK_DIRS];

// Generates pseudo-legal moves for the piece at (r,c) — does not check for
// leaving the mover's own king in check (that filtering happens in generateLegalMoves).
function pieceMoves(state, r, c) {
  const board = state.board;
  const piece = board[r][c];
  if (!piece) return [];
  const moves = [];
  const color = piece.color;

  const addMove = (tr, tc, extra = {}) => {
    if (!isInside(tr, tc)) return;
    const target = board[tr][tc];
    if (target && target.color === color) return;
    moves.push({
      from: { r, c },
      to: { r: tr, c: tc },
      piece: { ...piece },
      captured: target ? { ...target } : null,
      ...extra,
    });
  };

  if (piece.type === "p") {
    const dir = color === "w" ? -1 : 1;
    const startRow = color === "w" ? 6 : 1;
    const promoRow = color === "w" ? 0 : 7;

    // forward move
    if (isInside(r + dir, c) && !board[r + dir][c]) {
      if (r + dir === promoRow) {
        for (const promo of ["q", "r", "b", "n"]) {
          moves.push({
            from: { r, c }, to: { r: r + dir, c },
            piece: { ...piece }, captured: null, promotion: promo,
          });
        }
      } else {
        moves.push({ from: { r, c }, to: { r: r + dir, c }, piece: { ...piece }, captured: null });
        // double move
        if (r === startRow && !board[r + 2 * dir][c]) {
          moves.push({
            from: { r, c }, to: { r: r + 2 * dir, c },
            piece: { ...piece }, captured: null, doubleStep: true,
          });
        }
      }
    }

    // captures
    for (const dc of [-1, 1]) {
      const tr = r + dir;
      const tc = c + dc;
      if (!isInside(tr, tc)) continue;
      const target = board[tr][tc];
      if (target && target.color !== color) {
        if (tr === promoRow) {
          for (const promo of ["q", "r", "b", "n"]) {
            moves.push({
              from: { r, c }, to: { r: tr, c: tc },
              piece: { ...piece }, captured: { ...target }, promotion: promo,
            });
          }
        } else {
          moves.push({ from: { r, c }, to: { r: tr, c: tc }, piece: { ...piece }, captured: { ...target } });
        }
      } else if (!target && state.enPassant && state.enPassant.r === tr && state.enPassant.c === tc) {
        moves.push({
          from: { r, c }, to: { r: tr, c: tc },
          piece: { ...piece }, captured: { type: "p", color: opponent(color) },
          isEnPassant: true,
        });
      }
    }
  } else if (piece.type === "n") {
    for (const [dr, dc] of KNIGHT_OFFSETS) addMove(r + dr, c + dc);
  } else if (piece.type === "k") {
    for (const [dr, dc] of KING_OFFSETS) addMove(r + dr, c + dc);
    addCastlingMoves(state, r, c, color, moves);
  } else {
    const dirs = piece.type === "b" ? BISHOP_DIRS : piece.type === "r" ? ROOK_DIRS : KING_OFFSETS;
    for (const [dr, dc] of dirs) {
      let tr = r + dr;
      let tc = c + dc;
      while (isInside(tr, tc)) {
        const target = board[tr][tc];
        if (!target) {
          moves.push({ from: { r, c }, to: { r: tr, c: tc }, piece: { ...piece }, captured: null });
        } else {
          if (target.color !== color) {
            moves.push({ from: { r, c }, to: { r: tr, c: tc }, piece: { ...piece }, captured: { ...target } });
          }
          break;
        }
        tr += dr;
        tc += dc;
      }
    }
  }

  return moves;
}

function addCastlingMoves(state, r, c, color, moves) {
  if (isSquareAttacked(state.board, r, c, opponent(color))) return; // king in check, can't castle
  const rights = state.castling;
  const row = color === "w" ? 7 : 0;
  if (r !== row || c !== 4) return;

  const canKingSide = color === "w" ? rights.wK : rights.bK;
  const canQueenSide = color === "w" ? rights.wQ : rights.bQ;
  const board = state.board;

  if (canKingSide && !board[row][5] && !board[row][6]) {
    const rook = board[row][7];
    if (rook && rook.type === "r" && rook.color === color) {
      if (!isSquareAttacked(board, row, 5, opponent(color)) && !isSquareAttacked(board, row, 6, opponent(color))) {
        moves.push({
          from: { r: row, c: 4 }, to: { r: row, c: 6 },
          piece: { type: "k", color }, captured: null, isCastle: "K",
        });
      }
    }
  }
  if (canQueenSide && !board[row][1] && !board[row][2] && !board[row][3]) {
    const rook = board[row][0];
    if (rook && rook.type === "r" && rook.color === color) {
      if (!isSquareAttacked(board, row, 3, opponent(color)) && !isSquareAttacked(board, row, 2, opponent(color))) {
        moves.push({
          from: { r: row, c: 4 }, to: { r: row, c: 2 },
          piece: { type: "k", color }, captured: null, isCastle: "Q",
        });
      }
    }
  }
}

function isSquareAttacked(board, r, c, byColor) {
  // Pawn attacks
  const pawnDir = byColor === "w" ? 1 : -1; // attacker's pawn sits "pawnDir" rows away vertically from target
  for (const dc of [-1, 1]) {
    const pr = r + pawnDir;
    const pc = c + dc;
    if (isInside(pr, pc)) {
      const p = board[pr][pc];
      if (p && p.color === byColor && p.type === "p") return true;
    }
  }
  // Knight attacks
  for (const [dr, dc] of KNIGHT_OFFSETS) {
    const nr = r + dr, nc = c + dc;
    if (isInside(nr, nc)) {
      const p = board[nr][nc];
      if (p && p.color === byColor && p.type === "n") return true;
    }
  }
  // King attacks
  for (const [dr, dc] of KING_OFFSETS) {
    const nr = r + dr, nc = c + dc;
    if (isInside(nr, nc)) {
      const p = board[nr][nc];
      if (p && p.color === byColor && p.type === "k") return true;
    }
  }
  // Sliding: bishop/queen diagonals
  for (const [dr, dc] of BISHOP_DIRS) {
    let tr = r + dr, tc = c + dc;
    while (isInside(tr, tc)) {
      const p = board[tr][tc];
      if (p) {
        if (p.color === byColor && (p.type === "b" || p.type === "q")) return true;
        break;
      }
      tr += dr; tc += dc;
    }
  }
  // Sliding: rook/queen orthogonals
  for (const [dr, dc] of ROOK_DIRS) {
    let tr = r + dr, tc = c + dc;
    while (isInside(tr, tc)) {
      const p = board[tr][tc];
      if (p) {
        if (p.color === byColor && (p.type === "r" || p.type === "q")) return true;
        break;
      }
      tr += dr; tc += dc;
    }
  }
  return false;
}

function findKing(board, color) {
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p && p.type === "k" && p.color === color) return { r, c };
    }
  }
  return null;
}

function isKingInCheck(state, color) {
  const king = findKing(state.board, color);
  if (!king) return false;
  return isSquareAttacked(state.board, king.r, king.c, opponent(color));
}

// Applies a move to state in place, updating castling rights, en passant, clocks, and history.
function applyMove(state, move) {
  const board = state.board;
  const { from, to, piece } = move;

  const movingPiece = board[from.r][from.c];
  board[from.r][from.c] = null;

  if (move.isEnPassant) {
    board[from.r][to.c] = null; // captured pawn sits beside the destination
  }

  let placed = { ...movingPiece };
  if (move.promotion) {
    placed = { type: move.promotion, color: piece.color };
  }
  board[to.r][to.c] = placed;

  if (move.isCastle === "K") {
    const row = to.r;
    board[row][5] = board[row][7];
    board[row][7] = null;
  } else if (move.isCastle === "Q") {
    const row = to.r;
    board[row][3] = board[row][0];
    board[row][0] = null;
  }

  // Update castling rights
  if (piece.type === "k") {
    if (piece.color === "w") { state.castling.wK = false; state.castling.wQ = false; }
    else { state.castling.bK = false; state.castling.bQ = false; }
  }
  const clearRookRight = (r, c) => {
    if (r === 7 && c === 0) state.castling.wQ = false;
    else if (r === 7 && c === 7) state.castling.wK = false;
    else if (r === 0 && c === 0) state.castling.bQ = false;
    else if (r === 0 && c === 7) state.castling.bK = false;
  };
  if (piece.type === "r") clearRookRight(from.r, from.c);
  if (move.captured) clearRookRight(to.r, to.c);

  // Update en passant target
  if (move.doubleStep) {
    state.enPassant = { r: (from.r + to.r) / 2, c: from.c };
  } else {
    state.enPassant = null;
  }

  // Halfmove clock (for 50-move rule)
  if (piece.type === "p" || move.captured) {
    state.halfMoveClock = 0;
  } else {
    state.halfMoveClock += 1;
  }

  if (piece.color === "b") {
    state.fullMoveNumber += 1;
  }

  state.turn = opponent(piece.color);
  state.history.push(move);
}

function generateLegalMoves(state, color) {
  const legal = [];
  const board = state.board;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (!p || p.color !== color) continue;
      const pseudo = pieceMoves(state, r, c);
      for (const move of pseudo) {
        const trial = cloneState(state);
        applyMove(trial, move);
        if (!isKingInCheck(trial, color)) {
          legal.push(move);
        }
      }
    }
  }
  return legal;
}

function getGameStatus(state) {
  const color = state.turn;
  const inCheck = isKingInCheck(state, color);
  const legalMoves = generateLegalMoves(state, color);

  if (legalMoves.length === 0) {
    if (inCheck) return { status: "checkmate", winner: opponent(color), inCheck: true };
    return { status: "stalemate", winner: null, inCheck: false };
  }
  if (state.halfMoveClock >= 100) {
    return { status: "draw-50move", winner: null, inCheck };
  }
  return { status: inCheck ? "check" : "playing", winner: null, inCheck };
}

function moveToNotation(state, move, legalMovesForColor) {
  if (move.isCastle === "K") return "O-O";
  if (move.isCastle === "Q") return "O-O-O";

  const pieceLetters = { p: "", n: "N", b: "B", r: "R", q: "Q", k: "K" };
  let notation = pieceLetters[move.piece.type];
  const isCapture = !!move.captured;

  if (move.piece.type === "p") {
    if (isCapture) notation += "abcdefgh"[move.from.c] + "x";
  } else {
    // Disambiguation: check if another piece of same type could move to same square
    const others = legalMovesForColor.filter(
      (m) => m.piece.type === move.piece.type &&
        !(m.from.r === move.from.r && m.from.c === move.from.c) &&
        m.to.r === move.to.r && m.to.c === move.to.c
    );
    if (others.length > 0) {
      const sameFile = others.some((m) => m.from.c === move.from.c);
      const sameRank = others.some((m) => m.from.r === move.from.r);
      if (!sameFile) notation += "abcdefgh"[move.from.c];
      else if (!sameRank) notation += (8 - move.from.r);
      else notation += "abcdefgh"[move.from.c] + (8 - move.from.r);
    }
    if (isCapture) notation += "x";
  }

  notation += squareName(move.to.r, move.to.c);
  if (move.promotion) notation += "=" + pieceLetters[move.promotion];
  return notation;
}
