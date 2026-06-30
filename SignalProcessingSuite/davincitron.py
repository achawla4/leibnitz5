# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:15:49 2025

@author: acer
"""

"""
davincitron.py — A hierarchical generative artist

The DavinciTron is a meta-designer inspired by the idea of nested heuristics.
It coordinates multiple worker designtrons (each generating local motifs)
and a higher-level master designer that arranges these motifs on a canvas.
Each layer has its own 'inspiration' process and perception of the audience.
"""

import numpy as np
from PIL import Image, ImageDraw
import random, time, json, os
from datetime import datetime

# ===============================================================
# Perceptron — structural feasibility and sanity check
# ===============================================================

class Perceptron:
    def __init__(self, canvas_size=(256, 256), margin=5):
        self.canvas_size = canvas_size
        self.margin = margin

    def approve(self, x, y, r, centers):
        """Check if a new element fits within bounds and doesn't overlap."""
        if x - r < 0 or y - r < 0 or x + r > self.canvas_size[0] or y + r > self.canvas_size[1]:
            return False
        for c in centers:
            cx, cy, cr = c
            if np.hypot(x - cx, y - cy) < (r + cr + self.margin):
                return False
        return True


# ===============================================================
# InspirationField — stochastic driver for design decisions
# ===============================================================

class InspirationField:
    def __init__(self, lambda_rate=0.1, seed=None):
        self.lambda_rate = lambda_rate
        self.rng = np.random.default_rng(seed)
        self.last_time = time.time()

    def sample(self):
        """Occasionally emit a strong burst of inspiration."""
        if self.rng.random() < self.lambda_rate:
            return self.rng.normal(0.5, 0.2)
        else:
            return self.rng.normal(0, 0.05)


# ===============================================================
# Audience — collection of observer functions
# ===============================================================

class Audience:
    def __init__(self, size=3, seed=None):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.funcs = [self._make_func() for _ in range(size)]

    def _make_func(self):
        """Each observer has a slightly different sensitivity."""
        a, b = self.rng.random(), self.rng.random()
        return lambda arr: float(np.mean(np.sin(a * arr + b)))

    def evaluate(self, img_arr):
        vals = [f(img_arr) for f in self.funcs]
        return float(np.mean(vals))


# ===============================================================
# WorkerDesigntron — a subtask designer (places motif)
# ===============================================================

class WorkerDesigntron:
    def __init__(self, perceptron, inspiration_field, audience):
        self.p = perceptron
        self.i = inspiration_field
        self.a = audience

    def design_motif(self, n_shapes=13):
        """Place variable-sized, variable-color circles under inspiration control."""
        centers = []
        canvas = Image.new("RGB", self.p.canvas_size, (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        for _ in range(n_shapes):
            r = random.randint(5, 20)
            x, y = random.randint(0, 255), random.randint(0, 255)
            dx, dy = self.i.sample(), self.i.sample()
            x = np.clip(x + dx * 50, 0, 255)
            y = np.clip(y + dy * 50, 0, 255)

            if self.p.approve(x, y, r, centers):
                color = tuple(np.clip(np.random.normal(128, 50, 3), 0, 255).astype(int))
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))
                centers.append((x, y, r))

        score = self.a.evaluate(np.asarray(canvas) / 255.0)
        return canvas, centers, score


# ===============================================================
# MasterDesigner — arranges motifs on the canvas
# ===============================================================

class MasterDesigner:
    def __init__(self, canvas_size=(512, 512), n_workers=3, seed=None):
        self.canvas_size = canvas_size
        self.workers = []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        for i in range(n_workers):
            p = Perceptron(canvas_size=(256, 256))
            i_seed = None if seed is None else seed + i + 1
            a_seed = None if seed is None else seed + i + 10
            i_field = InspirationField(lambda_rate=0.2, seed=i_seed)
            aud = Audience(seed=a_seed)
            self.workers.append(WorkerDesigntron(p, i_field, aud))
        m_aud_seed = None if seed is None else seed + 100
        self.audience = Audience(size=5, seed=m_aud_seed)

    def compose(self):
        """Request motifs from workers and place them on the global canvas."""
        canvas = Image.new("RGB", self.canvas_size, (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        offsets = [(0, 0), (256, 0), (0, 256), (256, 256)]
        global_score = 0.0
        placed = 0

        for idx, w in enumerate(self.workers):
            motif, _, score = w.design_motif()
            ox, oy = random.choice(offsets)
            canvas.paste(motif, (ox, oy))
            global_score += score
            placed += 1

        arr = np.asarray(canvas) / 255.0
        audience_score = self.audience.evaluate(arr)
        final_score = 0.5 * global_score / placed + 0.5 * audience_score
        return canvas, final_score, placed


# ===============================================================
# DavinciTron — orchestrates everything
# ===============================================================

class DavinciTron:
    def __init__(self, outdir="./davincitron_outputs"):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.master = MasterDesigner()
        self.run_log = []

    def run(self, canvases=3):
        for i in range(canvases):
            c, score, placed = self.master.compose()
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(self.outdir, f"davincitron_canvas_{i:02d}_score{score:.3f}_{ts}.png")
            c.save(path)
            self.run_log.append(dict(path=path, score=score, placed=placed))
            print(f"Saved {path} score={score:.3f} placed={placed}")
            time.sleep(1)

        with open(os.path.join(self.outdir, "davincitron_run_log.json"), "w") as f:
            json.dump(self.run_log, f, indent=2)


# ===============================================================
# MAIN
# ===============================================================

if __name__ == "__main__":
    d = DavinciTron()
    d.run(canvases=3)
    print("DavinciTron run complete.")
