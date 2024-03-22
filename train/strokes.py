from typing import Self

import numpy as np


class Stroke(np.ndarray):
    """Array of 2D points"""

    def normalize(self) -> Self:
        """Fit points into [(0, 0), (1, 1)]"""
        p1 = np.min(self, axis=0)
        p2 = np.max(self, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (self - p1) / (p2 - p1)
            return np.nan_to_num(x, copy=False, nan=0.5, posinf=0.5, neginf=0.5)

    def dedup(self) -> Self:
        """Remove successive similar points"""
        shift1 = np.roll(self, 1, axis=0)
        shift1[0] = [-1, -1]
        return self[np.linalg.norm(self - shift1, axis=1) > 1e-10]

    def smooth(self) -> Self:
        """As the name suggests"""
        if len(self) < 3:
            return self
        shift1 = np.roll(self, -1, axis=0)
        shift2 = np.roll(self, -2, axis=0)
        shift1[-1] = shift2[-1] = self[-1]
        shift1[-2] = shift2[-2] = self[-2]
        return (self.data + shift1 + shift2) / 3

    def redistribute(self, num) -> Self:
        """Redistribute to equidistant series by number of points"""
        if len(self) < 2:
            return self

        sum_len = sum(np.linalg.norm(self[:-1] - self[1:], axis=1))
        seg_len = sum_len / (num - 1)

        new_stroke = [self[0]]
        rem_len = seg_len
        for point in self[1:]:
            while True:
                delta = point - new_stroke[-1]
                dist = np.linalg.norm(delta)
                if dist < rem_len:
                    rem_len -= dist
                    break
                new_stroke.append(new_stroke[-1] + delta * (rem_len / dist))
                rem_len = seg_len
        new_stroke.append(self[-1])

        return np.array(new_stroke).view(Stroke)

    def dominant(self, alpha) -> Self:
        if len(self) < 3:
            return self

        new_stroke = [self[0]]
        for p1, p2 in zip(self[1:], self[2:]):
            v = p1 - new_stroke[-1]
            w = p2 - p1
            angle = (v @ w) / (np.linalg.norm(v) * np.linalg.norm(w))
            angle = np.arccos(min(1, max(-1, angle)))
            if angle > alpha:
                new_stroke.append(p1)
        new_stroke.append(self[-1])

        return np.array(new_stroke).view(Stroke)
