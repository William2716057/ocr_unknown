import argparse
import os
import sys
import cv2
import numpy as np

SIZE = 32          # glyphs are normalised to SIZE x SIZE
ASPECT_W = 2.0     # weight of the aspect-ratio term in the match distance

# Pages
class Source:
    """One or more PDFs / images presented as a flat list of pages."""

    def __init__(self, paths, dpi=200):
        self.dpi, self.entries, self.pdfs = dpi, [], {}
        for p in paths:
            if p.lower().endswith(".pdf"):
                import fitz  # PyMuPDF
                doc = fitz.open(p)
                self.pdfs[p] = doc
                self.entries += [(p, k) for k in range(len(doc))]
            else:
                self.entries.append((p, None))
        self.n = len(self.entries)
        self._last = (None, None)

    def get(self, i):
        if self._last[0] == i:
            return self._last[1]
        path, k = self.entries[i]
        if k is None:
            img = cv2.imread(path)
            if img is None:
                sys.exit(f"Cannot read {path}")
        else:
            import fitz
            pix = self.pdfs[path][k].get_pixmap(dpi=self.dpi, colorspace=fitz.csRGB, alpha=False)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._last = (i, img)
        return img


def ink_mask(img):
    """255 where there is dark ink (any colour darker than the paper)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink

# Glyph features + sample library

def normalize(ink_crop):
    """Tight-crop the ink, pad to a square, resize. Returns (SIZExSIZE uint8, aspect) or None."""
    pts = cv2.findNonZero(ink_crop)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    crop = ink_crop[y:y + h, x:x + w]
    s = max(w, h)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA), w / h


def feat(img):
    return (cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32) / 255.0).ravel()


class Library:
    def __init__(self, path):
        self.path = path
        self.imgs, self.aspects, self.labels = [], [], []
        self._mat = None
        if os.path.exists(path):
            d = np.load(path)
            self.imgs = list(d["imgs"])
            self.aspects = [float(a) for a in d["aspects"]]
            self.labels = [str(s) for s in d["labels"]]

    def add(self, img, aspect, label):
        self.imgs.append(img)
        self.aspects.append(aspect)
        self.labels.append(label)
        self._mat = None
        self.save()

    def undo(self):
        if self.labels:
            self.imgs.pop(), self.aspects.pop(), self.labels.pop()
            self._mat = None
            self.save()

    def save(self):
        np.savez_compressed(
            self.path,
            imgs=np.array(self.imgs, np.uint8).reshape(-1, SIZE, SIZE),
            aspects=np.array(self.aspects, np.float32),
            labels=np.array(self.labels, dtype=str),
        )

    def matrix(self):
        if not self.imgs:
            return None
        if self._mat is None:
            self._mat = (np.stack([feat(i) for i in self.imgs]),
                         np.log(np.array(self.aspects, np.float32)))
        return self._mat


def classify(lib, img, aspect, max_dist):
    m = lib.matrix()
    if m is None:
        return "?", 999.0
    M, log_a = m
    d = np.linalg.norm(M - feat(img), axis=1) + ASPECT_W * np.abs(log_a - np.log(aspect))
    j = int(d.argmin())
    return (lib.labels[j] if d[j] <= max_dist else "?"), float(d[j])

# Segmentation + recognition
def recognise(img, lib, max_dist=6.0, dilate=1, gap_factor=0.0, ink=None):
    """Returns (rows, ref_height). rows = list of lines; each line a list of
    dicts {box, label, dist}; label ' ' marks a word break."""
    if ink is None:
        ink = ink_mask(img)
    dil = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=dilate) if dilate > 0 else ink
    _, _, stats, _ = cv2.connectedComponentsWithStats(dil)
    comps = [tuple(int(v) for v in s[:4]) for s in stats[1:] if s[4] >= 8]
    tall = [c[3] for c in comps if c[3] >= 20]
    if not tall:
        return [], 0
    ref = float(np.median(tall))                       # typical glyph height
    glyphs = [c for c in comps if 0.5 * ref <= c[3] <= 1.6 * ref and c[2] <= 1.6 * ref]
    dots = [c for c in comps if c[3] < 0.35 * ref and c[2] < 0.35 * ref]

    # group glyphs into lines by vertical centre
    glyphs.sort(key=lambda c: c[1] + c[3] / 2)
    lines = []
    for g in glyphs:
        cy = g[1] + g[3] / 2
        if lines and abs(cy - np.mean([c[1] + c[3] / 2 for c in lines[-1]])) < 0.6 * ref:
            lines[-1].append(g)
        else:
            lines.append([g])

    rows = []
    for ln in lines:
        mid = np.mean([c[1] + c[3] / 2 for c in ln])
        items = [(g, "g") for g in ln] + [(d, "d") for d in dots if abs(d[1] + d[3] / 2 - mid) < 0.8 * ref]
        items.sort(key=lambda t: t[0][0])
        row, prev_right = [], None
        for (x, y, w, h), kind in items:
            if kind == "d":
                if row and row[-1]["label"] != " ":
                    row.append(dict(box=(x, y, w, h), label=" ", dist=0.0))
                prev_right = x + w
                continue
            if (gap_factor and prev_right is not None and x - prev_right > gap_factor * ref
                    and row and row[-1]["label"] != " "):
                row.append(dict(box=(x, y, 0, 0), label=" ", dist=0.0))
            n = normalize(ink[y:y + h, x:x + w])
            if n is None:
                continue
            label, dist = classify(lib, n[0], n[1], max_dist)
            row.append(dict(box=(x, y, w, h), label=label, dist=dist))
            prev_right = x + w
        rows.append(row)
    return rows, ref


def page_text(rows):
    return "\n".join("".join(it["label"] for it in row).strip() for row in rows)

# Training GUI
def train_gui(args):
    import tkinter as tk
    from tkinter import simpledialog
    from PIL import Image, ImageTk

    src = Source(args.inputs, args.dpi)
    lib = Library(args.lib)

    class App:
        def __init__(self, root):
            self.root = root
            root.title("glyph_ocr - drag a box on a glyph, type its letter")
            root.geometry("1250x880")
            self.page, self.zoom = 0, 1.0
            self.max_dist = tk.DoubleVar(value=args.max_dist)
            self.show_rec = tk.BooleanVar(value=False)
            self.taught, self.hist, self.rec = {}, [], None

            bar = tk.Frame(root)
            bar.pack(side="top", fill="x")
            for txt, cmd in [("< Prev", lambda: self.goto(-1)), ("Next >", lambda: self.goto(1)),
                             ("Zoom -", lambda: self.set_zoom(self.zoom / 1.25)),
                             ("Zoom +", lambda: self.set_zoom(self.zoom * 1.25)),
                             ("Fit width", self.fit), ("Undo (Ctrl+Z)", self.undo),
                             ("Page text", self.show_text)]:
                tk.Button(bar, text=txt, command=cmd).pack(side="left", padx=2, pady=2)
            tk.Checkbutton(bar, text="Show recognition", variable=self.show_rec,
                           command=self.refresh).pack(side="left", padx=8)
            tk.Label(bar, text="Max distance:").pack(side="left")
            sp = tk.Spinbox(bar, from_=1, to=30, increment=0.5, width=5,
                            textvariable=self.max_dist, command=self.refresh)
            sp.pack(side="left")
            sp.bind("<Return>", lambda e: self.refresh())
            self.status = tk.Label(root, anchor="w")
            self.status.pack(side="bottom", fill="x")

            fr = tk.Frame(root)
            fr.pack(fill="both", expand=True)
            self.c = tk.Canvas(fr, bg="#555", cursor="crosshair")
            vs = tk.Scrollbar(fr, orient="vertical", command=self.c.yview)
            hs = tk.Scrollbar(fr, orient="horizontal", command=self.c.xview)
            self.c.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
            vs.pack(side="right", fill="y")
            hs.pack(side="bottom", fill="x")
            self.c.pack(fill="both", expand=True)

            self.c.bind("<ButtonPress-1>", self.press)
            self.c.bind("<B1-Motion>", self.drag)
            self.c.bind("<ButtonRelease-1>", self.release)
            self.c.bind("<MouseWheel>", self.wheel)
            self.c.bind("<Button-4>", lambda e: self.wheel(e, 1))
            self.c.bind("<Button-5>", lambda e: self.wheel(e, -1))
            root.bind("<Control-z>", lambda e: self.undo())
            root.bind("<Left>", lambda e: self.goto(-1))
            root.bind("<Right>", lambda e: self.goto(1))

            root.update()
            self.load()
            self.fit()

        # ---- page handling
        def load(self):
            self.img = src.get(self.page)
            self.ink = ink_mask(self.img)
            self.rec = None
            self.redraw()

        def goto(self, d):
            p = self.page + d
            if 0 <= p < src.n:
                self.page = p
                self.load()

        def set_zoom(self, z):
            self.zoom = max(0.1, min(z, 8.0))
            self.redraw()

        def fit(self):
            self.zoom = max(0.1, (self.c.winfo_width() - 4) / self.img.shape[1])
            self.redraw()

        def redraw(self):
            h, w = self.img.shape[:2]
            rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            interp = cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_NEAREST
            disp = cv2.resize(rgb, (int(w * self.zoom), int(h * self.zoom)), interpolation=interp)
            self.photo = ImageTk.PhotoImage(Image.fromarray(disp))
            self.c.delete("all")
            self.c.create_image(0, 0, anchor="nw", image=self.photo)
            self.c.configure(scrollregion=(0, 0, disp.shape[1], disp.shape[0]))
            self.overlays()

        def refresh(self):
            self.rec = None
            self.overlays()

        def run_rec(self):
            self.rec, _ = recognise(self.img, lib, self.max_dist.get(), args.dilate, ink=self.ink)

        def bc(self, b):
            x, y, w, h = b
            z = self.zoom
            return x * z, y * z, (x + w) * z, (y + h) * z

        def overlays(self):
            self.c.delete("ov")
            unknown = 0
            if self.show_rec.get():
                if self.rec is None:
                    self.run_rec()
                for row in self.rec:
                    for it in row:
                        if it["label"] == " ":
                            continue
                        bad = it["label"] == "?"
                        unknown += bad
                        col = "#ff3030" if bad else "#40a0ff"
                        x0, y0, x1, y1 = self.bc(it["box"])
                        self.c.create_rectangle(x0, y0, x1, y1, outline=col, tags="ov")
                        self.c.create_text(x0, y0 - 1, text=it["label"], anchor="sw", fill=col,
                                           font=("TkDefaultFont", 11, "bold"), tags="ov")
            for b, lab in self.taught.get(self.page, []):
                x0, y0, x1, y1 = self.bc(b)
                self.c.create_rectangle(x0, y0, x1, y1, outline="#30d030", width=2, tags="ov")
                self.c.create_text(x0, y0 - 1, text=lab, anchor="sw", fill="#30d030",
                                   font=("TkDefaultFont", 11, "bold"), tags="ov")
            extra = f" | unknown glyphs on page: {unknown}" if self.show_rec.get() else ""
            self.status.config(text=f"Page {self.page + 1}/{src.n} | zoom {self.zoom:.0%} | "
                                    f"{len(lib.labels)} samples, {len(set(lib.labels))} distinct letters{extra}")

        #mouse functions
        def press(self, e):
            self.start = (self.c.canvasx(e.x), self.c.canvasy(e.y))
            self.rect = self.c.create_rectangle(*self.start, *self.start, outline="orange",
                                                width=2, tags="drag")

        def drag(self, e):
            self.c.coords(self.rect, self.start[0], self.start[1],
                          self.c.canvasx(e.x), self.c.canvasy(e.y))

        def release(self, e):
            x0, y0 = self.start
            x1, y1 = self.c.canvasx(e.x), self.c.canvasy(e.y)
            self.c.delete("drag")
            if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
                return
            h, w = self.ink.shape
            z = self.zoom
            X0, X1 = sorted((int(x0 / z), int(x1 / z)))
            Y0, Y1 = sorted((int(y0 / z), int(y1 / z)))
            X0, Y0, X1, Y1 = max(0, X0), max(0, Y0), min(w, X1), min(h, Y1)
            n = normalize(self.ink[Y0:Y1, X0:X1])
            if n is None:
                self.status.config(text="No ink inside that box.")
                return
            label = simpledialog.askstring("Letter", "Letter(s) for this glyph:", parent=self.root)
            if not label or not label.strip():
                return
            label = label.strip()
            lib.add(n[0], n[1], label)
            self.taught.setdefault(self.page, []).append(((X0, Y0, X1 - X0, Y1 - Y0), label))
            self.hist.append(self.page)
            self.refresh()

        def wheel(self, e, d=None):
            if d is None:
                d = 1 if e.delta > 0 else -1
            if e.state & 0x4:                       # Ctrl = zoom
                self.set_zoom(self.zoom * (1.15 if d > 0 else 1 / 1.15))
            elif e.state & 0x1:                     # Shift = horizontal
                self.c.xview_scroll(-d * 3, "units")
            else:
                self.c.yview_scroll(-d * 3, "units")

        #actions
        def undo(self):
            if not self.hist:
                return
            pg = self.hist.pop()
            lib.undo()
            self.taught[pg].pop()
            self.refresh()

        def show_text(self):
            self.run_rec()
            win = tk.Toplevel(self.root)
            win.title(f"Recognised text - page {self.page + 1}")
            t = tk.Text(win, width=80, height=25, font=("TkFixedFont", 13), wrap="none")
            t.pack(fill="both", expand=True)
            t.insert("1.0", page_text(self.rec))

    root = tk.Tk()
    App(root)
    root.mainloop()

# Batch OCR
def parse_pages(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a) - 1, int(b))
        else:
            out.append(int(part) - 1)
    return [i for i in out if 0 <= i < n]


def ocr_cmd(args):
    lib = Library(args.lib)
    if not lib.labels:
        sys.exit("Library is empty - run the 'train' command first.")
    src = Source(args.inputs, args.dpi)
    chunks = []
    for i in parse_pages(args.pages, src.n):
        img = src.get(i)
        rows, _ = recognise(img, lib, args.max_dist, args.dilate, args.gap_factor)
        unk = sum(it["label"] == "?" for row in rows for it in row)
        total = sum(it["label"] != " " for row in rows for it in row)
        print(f"page {i + 1}: {total} glyphs, {unk} unknown", file=sys.stderr)
        chunks.append(f"--- page {i + 1} ---\n{page_text(rows)}")
        if args.debug:
            os.makedirs("debug", exist_ok=True)
            dbg = img.copy()
            for row in rows:
                for it in row:
                    if it["label"] == " ":
                        continue
                    x, y, w, h = it["box"]
                    col = (0, 0, 255) if it["label"] == "?" else (0, 160, 0)
                    cv2.rectangle(dbg, (x, y), (x + w, y + h), col, 1)
                    cv2.putText(dbg, it["label"].encode("ascii", "replace").decode(), (x, max(10, y - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cv2.imwrite(f"debug/page_{i + 1}.png", dbg)
    text = "\n\n".join(chunks)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"written to {args.out}", file=sys.stderr)
    else:
        print(text)


def main():
    ap = argparse.ArgumentParser(description="Teach-by-example OCR for unfamiliar fonts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "ocr"):
        p = sub.add_parser(name)
        p.add_argument("inputs", nargs="+", help="PDF(s) and/or image(s)")
        p.add_argument("--lib", default="glyph_library.npz", help="sample library file")
        p.add_argument("--dpi", type=int, default=200, help="PDF render resolution")
        p.add_argument("--max-dist", type=float, default=6.0,
                       help="match threshold; glyphs further than this become '?'")
        p.add_argument("--dilate", type=int, default=1,
                       help="dilation passes when joining broken glyph strokes (0 = off)")
    sub.choices["ocr"].add_argument("--pages", help="e.g. 1-3,7 (default: all)")
    sub.choices["ocr"].add_argument("--out", help="output text file")
    sub.choices["ocr"].add_argument("--debug", action="store_true", help="save annotated pages to ./debug")
    sub.choices["ocr"].add_argument("--gap-factor", type=float, default=0.0,
                                    help="also insert a space for gaps wider than this x glyph height "
                                         "(for texts without word dots; try 0.5)")
    args = ap.parse_args()
    train_gui(args) if args.cmd == "train" else ocr_cmd(args)


if __name__ == "__main__":
    main()
