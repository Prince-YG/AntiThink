import random
import time
import numpy as np
import chess
import chess.engine
from main import encode, move_idx   # reuse your encoder

SF_PATH = r"Projects\antithink\stockfish\stockfish-windows-x86-64-universal.exe"
N_POSITIONS = 5000
DEPTH = 8      
TOP = 5

#make random pos/boradas
def random_position():
    
    while True:
        b = chess.Board()
        #plays like 6-80 moves
        for _ in range(random.randint(6, 80)):
            if b.is_game_over():
                break
            #dues a random LEGAL move
            b.push(random.choice(list(b.legal_moves)))
        if not b.is_game_over() and b.legal_moves.count() >= TOP:
            return b


def label(engine, board):
    #evualates moves, as slot numbers via stockfishy baka +move_idx
    latest = {}
    with engine.analysis(board, chess.engine.Limit(depth=DEPTH), multipv=TOP) as a:
        for info in a:
            #gets stockfishs finished answer
            if "pv" in info and "score" in info and "multipv" in info:
                latest[info["multipv"]] = info
    if len(latest) < TOP:
        return None
    slots, cps = [], []
    #for each rank,
    for rank in sorted(latest):
        info = latest[rank]
        #assigns to slot
        slots.append(move_idx(info["pv"][0], board))
        cps.append(info["score"].pov(board.turn).score(mate_score=10000))
    return slots, cps

#just filters only legal slots
def legal_mask(board):
    m = np.zeros(4096, dtype=bool)
    for mv in board.legal_moves:
        m[move_idx(mv, board)] = True
    return m


if __name__ == "__main__":
    engine = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    X, SLOTS, CPS, MASKS = [], [], [], []
    t0 = time.time()
    try:
        #make the boards and moves
        while len(X) < N_POSITIONS:
            #get random board 
            board = random_position()
            #get moves (in slots) for it
            out = label(engine, board)
            if out is None:
                continue
            slots, cps = out
            #esssentaily deacitvate illegail moves
            mask = legal_mask(board)
            assert all(mask[s] for s in slots)   # check

            X.append(encode(board).astype(np.uint8))
            SLOTS.append(slots)
            CPS.append(cps)
            MASKS.append(mask)

            if len(X) % 100 == 0:
                print(f"{len(X)}/{N_POSITIONS}  ({time.time() - t0:.0f}s)")
    finally:
        engine.quit()

    np.savez_compressed(
        "data.npz",
        x=np.array(X),                       # (N, 12, 8, 8)
        slots=np.array(SLOTS),               # (N, 5)
        cps=np.array(CPS, dtype=np.float32), # (N, 5)
        masks=np.array(MASKS),               # (N, 4096)
    )
    print("saved data.npz", np.array(X).shape)