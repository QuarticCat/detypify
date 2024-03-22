import numpy as np
import psycopg
from torch.utils.data import Dataset

from train.strokes import Stroke


class TypSamples(Dataset):
    def __init__(self):
        self.conn = psycopg.connect("dbname=detypify")
        self.len = self.conn.execute("SELECT COUNT(*) FROM typ_samples").fetchone()[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.conn.execute(
            """
            SELECT strokes, typ
            FROM typ_samples
            WHERE id = %s
            """,
            (idx + 1,),
        ).fetchone()


def main():
    data = TypSamples()
    strokes, label = data[2745]
    strokes = [
        np.array([[x, y] for x, y, _ in s], dtype=np.float64).view(Stroke)
        for s in strokes
    ]
    print(strokes[0])
    print(strokes[0].redistribute(10))
    print(strokes[0].redistribute(10).dominant(2 * np.pi * 15 / 360))
