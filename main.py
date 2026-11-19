import time
import numpy as np
import chess
import chess.engine


class ChessGame:
    def __init__(self, mode="normal", startTime=600, extra=5, path="stockfish"):
        self.fishpath = path
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci(path)

        self.winner = None
        self.lastMoveAt = time.time()
        self.moveHistory = []
        self.white = True

        self.mode = mode
        self.whiteTime = startTime
        self.blackTime = startTime
        self.extra = extra

    def move(self, moved):
        if self.winner:
            return self.winner
        self.white = self.board.turn == chess.WHITE

        now = time.time()
        taken = now - self.lastMoveAt

        try:
            self.board.push_san(moved)
        except ValueError:
            print("illegal move:", moved)
            return False

        if self.white:
            self.whiteTime -= taken - self.extra
        else:
            self.blackTime -= taken - self.extra
        self.lastMoveAt = now

        self.moveHistory.append(moved)

        if self.board.is_checkmate():
            self.winner = "white" if self.white else "black"
            print("Done")

        return True

    def eval(self, depth=12, top_moves=5):
        latest = {}
        with self.engine.analysis(
            self.board, chess.engine.Limit(depth=depth), multipv=top_moves
        ) as a:
            for info in a:
                if "pv" in info and "score" in info and "multipv" in info:
                    latest[info["multipv"]] = info

        results = []
        for rank in sorted(latest):
            info = latest[rank]
            move = self.board.san(info["pv"][0])
            score = info["score"].pov(self.board.turn).score(mate_score=10000)
            results.append((move, score))
        return results

    def close(self):
        self.engine.quit()


def softmax_dist(results, T=100):
    cps = np.clip([s for _, s in results], -1000, 1000) / T
    e = np.exp(cps - cps.max())
    return e / e.sum()


def encode(board):
    """Board -> (12, 8, 8) array. Planes 0-5 = side to move's pieces, 6-11 = opponent's."""
    if board.turn == chess.BLACK:
        board = board.mirror()  # flip so the side to move always looks like White
    x = np.zeros((12, 8, 8), np.float32)
    for sq, piece in board.piece_map().items():
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        x[plane, sq // 8, sq % 8] = 1
    return x


def move_idx(move, board):
    #turns moves ino slots
    f, t = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f * 64 + t




if __name__ == "__main__":


    larp = ChessGame(
        path=r"Projects\antithink\stockfish\stockfish-windows-x86-64-universal.exe"
    )
    try:
        larp.move("e4")
        larp.move("e5")
        larp.move("Nf3")

        res = larp.eval()
        for T in (10, 100, 1000):
            probs = softmax_dist(res, T)
            print(f"\nT={T}")
            for (m, s), p in zip(res, probs):
                print(f"  {m:6} {s:5}  {p:.2%}")
    finally:
        larp.close()