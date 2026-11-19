import os
import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from main import encode, move_idx

HERE = os.path.dirname(os.path.abspath(__file__))
SQ = 72  # square size in pixels


# ---------- same model as train.py (must match exactly) ----------
class Net(nn.Module):
    def __init__(self, ch=64, blocks=2):
        super().__init__()
        self.inp = nn.Conv2d(12, ch, 3, padding=1)
        self.res = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
            ) for _ in range(blocks)
        ])
        self.head = nn.Linear(ch * 64, 4096)

    def forward(self, x):
        x = F.relu(self.inp(x))
        for block in self.res:
            x = F.relu(x + block(x))
        return self.head(x.flatten(1))


model = Net()
model.load_state_dict(torch.load(os.path.join(HERE, "model.pt"), map_location="cpu"))
model.eval()


@torch.no_grad()
def model_move(board):
    """Encode board -> 4096 scores -> kill illegal slots -> best slot -> real move."""
    x = torch.tensor(encode(board)).unsqueeze(0)
    logits = model(x)[0]
    mask = torch.zeros(4096, dtype=torch.bool)
    for mv in board.legal_moves:
        mask[move_idx(mv, board)] = True
    logits = logits.masked_fill(~mask, -1e9)
    slot = logits.argmax().item()
    f, t = slot // 64, slot % 64
    if board.turn == chess.BLACK:                      # undo the flip
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    mv = chess.Move(f, t)
    if mv not in board.legal_moves:                    # pawn promotion
        mv = chess.Move(f, t, chess.QUEEN)
    return mv


# ---------- the window ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("antithink")
        self.canvas = tk.Canvas(root, width=8 * SQ, height=8 * SQ)
        self.canvas.pack()
        self.status = tk.Label(root, text="", font=("Arial", 12))
        self.status.pack(pady=4)
        tk.Button(root, text="Switch side + new game", command=self.switch).pack(pady=(0, 8))
        self.canvas.bind("<Button-1>", self.on_click)

        self.user = chess.WHITE
        self.new_game()

    def new_game(self):
        self.board = chess.Board()
        self.sel = None
        self.targets = set()
        self.draw()
        if self.user == chess.BLACK:
            self.root.after(300, self.bot_move)

    def switch(self):
        self.user = not self.user
        self.new_game()

    def sq_at(self, r, c):
        if self.user == chess.WHITE:
            return chess.square(c, 7 - r)
        return chess.square(7 - c, r)

    def draw(self):
        self.canvas.delete("all")
        for r in range(8):
            for c in range(8):
                sq = self.sq_at(r, c)
                color = "#c8a97e" if (r + c) % 2 == 0 else "#8b6a46"
                if sq == self.sel:
                    color = "#e8d44d"
                elif sq in self.targets:
                    color = "#7fb069"
                x0, y0 = c * SQ, r * SQ
                self.canvas.create_rectangle(x0, y0, x0 + SQ, y0 + SQ, fill=color, outline="")
                p = self.board.piece_at(sq)
                if p:
                    glyph = chess.Piece(p.piece_type, chess.WHITE).unicode_symbol(invert_color=True)  # solid glyph
                    fill = "#ffffff" if p.color == chess.WHITE else "#111111"
                    self.canvas.create_text(x0 + SQ / 2, y0 + SQ / 2, text=glyph, fill=fill, font=("Arial", 40))
        if self.board.is_game_over():
            self.status.config(text=f"Game over: {self.board.result()}")
        elif self.board.turn == self.user:
            self.status.config(text="Your move" + (" (check!)" if self.board.is_check() else ""))
        else:
            self.status.config(text="Model is thinking...")

    def on_click(self, e):
        if self.board.is_game_over() or self.board.turn != self.user:
            return
        c, r = e.x // SQ, e.y // SQ
        if not (0 <= c < 8 and 0 <= r < 8):
            return
        sq = self.sq_at(r, c)
        piece = self.board.piece_at(sq)

        if self.sel is not None and sq in self.targets:
            mv = chess.Move(self.sel, sq)
            if mv not in self.board.legal_moves:
                mv = chess.Move(self.sel, sq, chess.QUEEN)   # auto-queen
            self.board.push(mv)
            self.sel, self.targets = None, set()
            self.draw()
            self.root.after(200, self.bot_move)
        elif piece and piece.color == self.user:
            self.sel = sq
            self.targets = {m.to_square for m in self.board.legal_moves if m.from_square == sq}
            self.draw()
        else:
            self.sel, self.targets = None, set()
            self.draw()

    def bot_move(self):
        if self.board.is_game_over():
            return
        self.board.push(model_move(self.board))
        self.draw()


root = tk.Tk()
App(root)
root.mainloop()