// Simple chess AI: minimax with alpha-beta pruning, material + piece-square evaluation.

const PIECE_VALUES = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 20000 };

const PAWN_TABLE = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [50, 50, 50, 50, 50, 50, 50, 50],
  [10, 10, 20, 30, 30, 20, 10, 10],
  [5, 5, 10, 25, 25, 10, 5, 5],
  [0, 0, 0, 20, 20, 0, 0, 0],
  [5, -5, -10, 0, 0, -10, -5, 5],
  [5, 10, 10, -20, -20, 10, 10, 5],
  [0, 0, 0, 0, 0, 0, 0, 0],
];
const KNIGHT_TABLE = [
  [-50, -40, -30, -30, -30, -30, -40, -50],
  [-40, -20, 0, 0, 0, 0, -20, -40],
  [-30, 0, 10, 15, 15, 10, 0, -30],
  [-30, 5, 15, 20, 20, 15, 5, -30],
  [-30, 0, 15, 20, 20, 15, 0, -30],
  [-30, 5, 10, 15, 15, 10, 5, -30],
  [-40, -20, 0, 5, 5, 0, -20, -40],
  [-50, -40, -30, -30, -30, -30, -40, -50],
];
const BISHOP_TABLE = [
  [-20, -10, -10, -10, -10, -10, -10, -20],
  [-10, 0, 0, 0, 0, 0, 0, -10],
  [-10, 0, 5, 10, 10, 5, 0, -10],
  [-10, 5, 5, 10, 10, 5, 5, -10],
  [-10, 0, 10, 10, 10, 10, 0, -10],
  [-10, 10, 10, 10, 10, 10, 10, -10],
  [-10, 5, 0, 0, 0, 0, 5, -10],
  [-20, -10, -10, -10, -10, -10, -10, -20],
];
const ROOK_TABLE = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [5, 10, 10, 10, 10, 10, 10, 5],
  [-5, 0, 0, 0, 0, 0, 0, -5],
  [-5, 0, 0, 0, 0, 0, 0, -5],
  [-5, 0, 0, 0, 0, 0, 0, -5],
  [-5, 0, 0, 0, 0, 0, 0, -5],
  [-5, 0, 0, 0, 0, 0, 0, -5],
  [0, 0, 0, 5, 5, 0, 0, 0],
];
const QUEEN_TABLE = [
  [-20, -10, -10, -5, -5, -10, -10, -20],
  [-10, 0, 0, 0, 0, 0, 0, -10],
  [-10, 0, 5, 5, 5, 5, 0, -10],
  [-5, 0, 5, 5, 5, 5, 0, -5],
  [0, 0, 5, 5, 5, 5, 0, -5],
  [-10, 5, 5, 5, 5, 5, 0, -10],
  [-10, 0, 5, 0, 0, 0, 0, -10],
  [-20, -10, -10, -5, -5, -10, -10, -20],
];
const KING_TABLE = [
  [-30, -40, -40, -50, -50, -40, -40, -30],
  [-30, -40, -40, -50, -50, -40, -40, -30],
  [-30, -40, -40, -50, -50, -40, -40, -30],
  [-30, -40, -40, -50, -50, -40, -40, -30],
  [-20, -30, -30, -40, -40, -30, -30, -20],
  [-10, -20, -20, -20, -20, -20, -20, -10],
  [20, 20, 0, 0, 0, 0, 20, 20],
  [20, 30, 10, 0, 0, 10, 30, 20],
];

const PIECE_TABLES = { p: PAWN_TABLE, n: KNIGHT_TABLE, b: BISHOP_TABLE, r: ROOK_TABLE, q: QUEEN_TABLE, k: KING_TABLE };

function evaluateBoard(state) {
  let score = 0;
  const board = state.board;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (!p) continue;
      const value = PIECE_VALUES[p.type];
      const tableRow = p.color === "w" ? r : 7 - r;
      const positional = PIECE_TABLES[p.type][tableRow][c];
      const total = value + positional;
      score += p.color === "w" ? total : -total;
    }
  }
  return score;
}

function orderMoves(moves) {
  // Captures first (MVV-LVA-ish) to improve alpha-beta pruning efficiency.
  return moves.slice().sort((a, b) => {
    const aScore = a.captured ? PIECE_VALUES[a.captured.type] - PIECE_VALUES[a.piece.type] / 10 : 0;
    const bScore = b.captured ? PIECE_VALUES[b.captured.type] - PIECE_VALUES[b.piece.type] / 10 : 0;
    return bScore - aScore;
  });
}

function minimax(state, depth, alpha, beta, maximizing) {
  const color = maximizing ? "w" : "b";
  const moves = generateLegalMoves(state, color);

  if (moves.length === 0) {
    if (isKingInCheck(state, color)) {
      // Checkmate: heavily penalize, but prefer faster mates via depth bonus.
      return maximizing ? -100000 - depth : 100000 + depth;
    }
    return 0; // stalemate
  }

  if (depth === 0) {
    return evaluateBoard(state);
  }

  const ordered = orderMoves(moves);

  if (maximizing) {
    let best = -Infinity;
    for (const move of ordered) {
      const next = cloneState(state);
      applyMove(next, move);
      const score = minimax(next, depth - 1, alpha, beta, false);
      best = Math.max(best, score);
      alpha = Math.max(alpha, score);
      if (beta <= alpha) break;
    }
    return best;
  } else {
    let best = Infinity;
    for (const move of ordered) {
      const next = cloneState(state);
      applyMove(next, move);
      const score = minimax(next, depth - 1, alpha, beta, true);
      best = Math.min(best, score);
      beta = Math.min(beta, score);
      if (beta <= alpha) break;
    }
    return best;
  }
}

// Returns the best move for the side to move, searching to the given depth.
function findBestMove(state, depth) {
  const color = state.turn;
  const moves = orderMoves(generateLegalMoves(state, color));
  if (moves.length === 0) return null;

  const maximizing = color === "w";
  let bestMoves = [];
  let bestScore = maximizing ? -Infinity : Infinity;

  for (const move of moves) {
    const next = cloneState(state);
    applyMove(next, move);
    const score = minimax(next, depth - 1, -Infinity, Infinity, !maximizing);

    if (maximizing ? score > bestScore : score < bestScore) {
      bestScore = score;
      bestMoves = [move];
    } else if (score === bestScore) {
      bestMoves.push(move);
    }
  }

  // Add slight randomness among equally good moves so the AI isn't fully deterministic.
  return bestMoves[Math.floor(Math.random() * bestMoves.length)];
}
