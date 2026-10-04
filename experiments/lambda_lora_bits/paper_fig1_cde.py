#!/usr/bin/env python3
"""Figure 1 in the colours of Contextual Document Embeddings (Morris and Rush), as an editable PPTX.

Pastel fills (lavender, pink, mint) inside thick black outlines, flat model icons, gray connectors and gray secondary
labels, dark-hatched bars, generous white space. Model outputs are response bars (hatched thinking segment, mint
answer box), not sentences.

Layout is driven by one spec (the constants below) so that every block, gap, arrow and font size is the same
wherever it appears:
  * one row grid: the six teachers in (a) and the six adapters in (b) share row centres, and one dashed line marks
    the switch in both panels; the three edits in (c) each occupy two rows of the same grid;
  * one block size for adapter squares, response bars and key swatches; one gap between blocks;
  * one arrow length for every gray connector; one outline width; one connector width;
  * three font sizes, chosen for the printed size: the slide is 13.33 in wide and is set at 6.3 in in the paper
    (x 0.47), so 20/17/15 pt print at about 9.5/8/7 pt.
Text widths are measured with the Arial metrics (LibreOffice renders Liberation Sans, which has the same metrics),
and the script fails if any label overlaps a shape or leaves the canvas.
Run inside the experiment container; writes materials/main_figure_pptx/cde/fig1_cde.pptx.
"""

import json
import os
from pathlib import Path

from lxml import etree
from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path("/home/lab/workspace/cot-distill-lora-bit/materials/main_figure_pptx/cde")
OUT = Path(os.environ.get("FIG1_OUT", HERE))
FONT_DIR = HERE / ".fonts"

# ---------------------------------------------------------------- canvas and units
W, H = 2000, 920                     # canvas units; 150 units = 1 inch on the slide
W_EMU = 12192000
K = W_EMU / W
E = lambda v: Emu(int(round(v * K)))
PT = 150 / 72                         # canvas units per point

# ---------------------------------------------------------------- palette (Contextual Document Embeddings)
BLACK = RGBColor(0x11, 0x11, 0x11)
LINE_GRAY = RGBColor(0xB3, 0xB3, 0xB3)
LABEL_GRAY = RGBColor(0x9A, 0x9A, 0x9A)
LAVENDER = RGBColor(0xD9, 0xCE, 0xF9)
PINK = RGBColor(0xFB, 0xA9, 0xCB)
HOT_PINK = RGBColor(0xF2, 0x5C, 0x93)
MINT = RGBColor(0xA6, 0xE6, 0xBA)
HATCH_BG = RGBColor(0xE9, 0xE1, 0xFB)
SHARED_FILL = RGBColor(0x9C, 0x86, 0xF0)
SHARED_BAND = RGBColor(0xEE, 0xE9, 0xFD)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"

# ---------------------------------------------------------------- the spec
OUTLINE, CONNECT = 2.5, 2.0           # pt
T_HEAD, T_LABEL, T_SMALL = 20, 17, 15  # pt
T_NUM = 13                            # pt, the lambda values inside the teacher squares
MARGIN = 30                           # canvas edge to content
PANEL_GAP = 92                        # between the content of neighbouring panels
ARROW = 50                            # length of every gray connector arrow
TIP = 6                               # gap between an arrow tip and its target
B = 30                                # block side: adapter squares, response-bar height, key swatches
BG = 6                                # gap between blocks
T_SQ = 56                             # teacher squares in (a)
ROWGAP = 108                          # row pitch
ROWS = [205 + i * ROWGAP for i in range(6)]
LAMS = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
THINK = {"0": 160, "0.2": 145}        # thinking segment of the teacher's response; none after the switch
ROW_W = 16 * B + 15 * BG              # width of a 16-block adapter row
GROW = 26                             # how much taller an amplified block is (half above, half below)
HEAD_Y = (20, 72)                     # header band
KEY_Y = ROWS[5] + 65 + 14 + 42.5 + 22 + 15  # key row centre: its own row below the "Instruct model" label
SWY = (ROWS[1] + ROWS[2]) / 2         # the switch, between lambda = 0.2 and 0.4

SHARED = {1, 2, 4, 7, 8, 11}          # same 6 of the 16 directions in every pre-switch adapter
USED = {"0": {0, 5, 9, 13, 14}, "0.2": {0, 5, 10, 13, 15}, "0.4": {0, 3, 6, 10, 12, 13, 15},
        "0.6": {3, 5, 6, 9, 12, 13, 14}, "0.8": {2, 3, 6, 9, 10, 12, 14, 15}, "1": {2, 3, 5, 6, 9, 10, 12, 14, 15}}

_FONTS = {}


def text_w(t, size, bold):
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(str(FONT_DIR / ("Arial_Bold.ttf" if bold else "Arial.ttf")), size * PT * 10)
    return _FONTS[key].getlength(t) / 10


def text_h(size):
    return size * PT * 1.2


# ---------------------------------------------------------------- drawing primitives (all record their boxes)
prs = Presentation()
prs.slide_width = Emu(W_EMU)
prs.slide_height = E(H)
s = prs.slides.add_slide(prs.slide_layouts[6])
BOXES = []                            # (kind, name, x0, y0, x1, y1)
FONTS_USED = []


def strip_style(sh):
    st = sh._element.find(qn("p:style"))
    if st is not None:
        sh._element.remove(st)


def shape(name, x0, y0, x1, y1, fill=WHITE, line=BLACK, w=OUTLINE, dash=None, record=True):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, E(x0), E(y0), E(x1 - x0), E(y1 - y0))
    sh.name = name
    strip_style(sh)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(w)
        if dash:
            sh.line.dash_style = dash
    if record:
        BOXES.append(("shape", name, x0, y0, x1, y1))
    return sh


def line(name, x0, y0, x1, y1, color=LINE_GRAY, w=CONNECT, head=False, dash=None):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x0), E(y0), E(x1), E(y1))
    c.name = name
    strip_style(c)
    c.line.color.rgb = color; c.line.width = Pt(w)
    if dash:
        c.line.dash_style = dash
    if head:
        ln = c.line._get_or_add_ln()
        te = etree.SubElement(ln, qn("a:tailEnd")); te.set("type", "triangle"); te.set("w", "med"); te.set("len", "med")
    BOXES.append(("line", name, min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return c


def arrow(name, x0, y, x1=None):
    """A horizontal gray arrow from x0 that stops TIP short of x1 (default: one standard ARROW long)."""
    x1 = x0 + ARROW if x1 is None else x1
    line(name, x0, y, x1 - TIP, y, head=True)


def text(name, x, cy, t, size, color=BLACK, bold=False, align="center"):
    """Text anchored at x (left, centre or right edge) and vertically centred on cy; the box is the measured text.
    t may be a list of lines."""
    lines = t if isinstance(t, list) else [t]
    w = max(text_w(l, size, bold) for l in lines)
    h = text_h(size) * len(lines)
    x0 = {"left": x, "center": x - w / 2, "right": x - w}[align]
    tb = s.shapes.add_textbox(E(x0 - 2), E(cy - h / 2), E(w + 4), E(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, l in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
        r = p.add_run(); r.text = l
        f = r.font; f.size = Pt(size); f.bold = bold; f.name = FONT; f.color.rgb = color
    ink = h - text_h(size) + size * PT          # glyph extent: all line pitches but the last, plus one em
    BOXES.append(("text", name, x0, cy - ink / 2, x0 + w, cy + ink / 2))
    FONTS_USED.append((name, size, bold, "\n".join(lines)))
    return x0, x0 + w


def hatched(name, x0, y0, x1, y1, step=16, faint=False):
    """CDE hatch: light fill, thick dark diagonal stripes clipped to the box, black outline on top.
    faint: gray stripes and outline, for the response before an edit."""
    ink = LINE_GRAY if faint else BLACK
    shape(name + " fill", x0, y0, x1, y1, fill=HATCH_BG, line=None, record=False)
    h = y1 - y0
    x = x0 - h
    while x < x1:
        b0, b1 = max(x, x0), min(x + h, x1)
        if b1 > b0:
            c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(b0), E(y1 - (b0 - x)), E(b1), E(y1 - (b1 - x)))
            strip_style(c); c.line.color.rgb = ink; c.line.width = Pt(2.5)
        x += step
    return shape(name, x0, y0, x1, y1, fill=None, line=ink)


def trapezoid(name, x0, y0, x1, y1, fill):
    inset = (y1 - y0) * 0.22
    pts = [(x0, y0), (x1, y0 + inset), (x1, y1 - inset), (x0, y1)]
    fb = s.shapes.build_freeform(E(pts[0][0]), E(pts[0][1]), scale=1.0)
    fb.add_line_segments([(E(x), E(y)) for x, y in pts[1:]], close=True)
    sh = fb.convert_to_shape()
    sh.name = name
    strip_style(sh)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = BLACK; sh.line.width = Pt(OUTLINE)
    BOXES.append(("shape", name, x0, y0, x1, y1))


def mix(c0, c1, t):
    return RGBColor(*[round(a + (b - a) * t) for a, b in zip(c0, c1)])


def tint(lam):
    return mix(LAVENDER, PINK, LAMS.index(lam) / 5)


def adapter(name, x0, cy, lam, mode="plain"):
    """16 blocks. mode: plain (shared blocks only before the switch), remove, graft, amplify."""
    for k in range(16):
        x = x0 + k * (B + BG)
        y0, y1 = cy - B / 2, cy + B / 2
        if k in SHARED and (mode in ("graft", "amplify") or (mode == "plain" and lam in ("0", "0.2"))):
            g = GROW / 2 if mode == "amplify" else 0     # amplified blocks grow equally up and down
            shape(f"{name} dir {k + 1}", x, y0 - g, x + B, y1 + g, fill=SHARED_FILL)
        elif k in SHARED and mode == "remove":
            shape(f"{name} dir {k + 1}", x, y0, x + B, y1, fill=SHARED_BAND, line=SHARED_FILL, w=1.75,
                  dash=MSO_LINE_DASH_STYLE.DASH)
        else:
            shape(f"{name} dir {k + 1}", x, y0, x + B, y1, fill=tint(lam) if k in USED[lam] else WHITE)


def key_item(name, x, swatch, label):
    """A key entry: swatch (B tall) then label; returns the x where the next entry may start."""
    w = swatch(name, x, KEY_Y)
    _, x1 = text(f"{name} label", x + w + 12, KEY_Y, label, T_SMALL, color=LABEL_GRAY, align="left")
    return x1 + 40


# ---------------------------------------------------------------- x layout, left to right
A_MODEL_W, A_MODEL_H = 120, 130
A_LABEL_W = max(text_w("Reasoning model", T_LABEL, True), text_w("Instruct model", T_LABEL, True))
A_MODEL_CX = MARGIN + A_LABEL_W / 2                      # labels centred on the models, flush with the margin
A_MX0, A_MX1 = A_MODEL_CX - A_MODEL_W / 2, A_MODEL_CX + A_MODEL_W / 2
A_BUS = A_MX1 + ARROW
A_SX = A_BUS + ARROW                                     # teacher squares
A_BX = A_SX + T_SQ + TIP + ARROW                               # response bars
A_END = A_BX + THINK["0"] + B                            # right edge of (a)
B_X0 = A_END + PANEL_GAP                                 # (b): adapters
B_X1 = B_X0 + ROW_W
C_TRUNK = B_X1 + PANEL_GAP / 2
C_X0 = C_TRUNK + ARROW                                   # (c): edited adapters
C_X1 = C_X0 + ROW_W
assert C_X1 <= W - MARGIN, f"(c) ends at {C_X1:.0f}, past the right margin {W - MARGIN}"

# ---------------------------------------------------------------- headers, centred over each panel's content
for name, x0, x1, t in [("a", MARGIN, A_END, "(a) Interpolated teachers"), ("b", B_X0, B_X1, "(b) Student LoRA adapters"),
                        ("c", C_X0, C_X1, "(c) Editing the shared region")]:
    text(f"header {name}", (x0 + x1) / 2, sum(HEAD_Y) / 2, t, T_HEAD, bold=True)

# ---------------------------------------------------------------- (a) interpolated teachers
LABEL_GAP = 14
text("label reasoning", A_MODEL_CX, ROWS[0] - A_MODEL_H / 2 - LABEL_GAP - text_h(T_LABEL) / 2, "Reasoning model",
     T_LABEL, bold=True)
trapezoid("reasoning model", A_MX0, ROWS[0] - A_MODEL_H / 2, A_MX1, ROWS[0] + A_MODEL_H / 2, LAVENDER)
trapezoid("instruct model", A_MX0, ROWS[5] - A_MODEL_H / 2, A_MX1, ROWS[5] + A_MODEL_H / 2, PINK)
text("label instruct", A_MODEL_CX, ROWS[5] + A_MODEL_H / 2 + LABEL_GAP + text_h(T_LABEL) / 2, "Instruct model",
     T_LABEL, bold=True)
line("reasoning to bus", A_MX1 + TIP, ROWS[0], A_BUS, ROWS[0])     # clear of the model outline
line("instruct to bus", A_MX1 + TIP, ROWS[5], A_BUS, ROWS[5])
line("bus", A_BUS, ROWS[0], A_BUS, ROWS[5])
text("lambda header", A_SX + T_SQ / 2, ROWS[0] - T_SQ / 2 - LABEL_GAP - text_h(T_LABEL) / 2, "λ", T_LABEL,
     color=LABEL_GRAY, bold=True)
for lam, cy in zip(LAMS, ROWS):
    arrow(f"branch {lam}", A_BUS, cy, A_SX)
    shape(f"teacher {lam}", A_SX, cy - T_SQ / 2, A_SX + T_SQ, cy + T_SQ / 2, fill=tint(lam))
    BOXES.pop()                                          # the label sits inside its square on purpose
    text(f"teacher label {lam}", A_SX + T_SQ / 2, cy, lam, T_NUM, bold=True)
    BOXES.pop()
    BOXES.append(("shape", f"teacher {lam}", A_SX, cy - T_SQ / 2, A_SX + T_SQ, cy + T_SQ / 2))
    arrow(f"to response {lam}", A_SX + T_SQ + TIP, cy, A_BX)     # starts clear of the square, like every arrow
    x = A_BX
    if lam in THINK:
        hatched(f"thinking {lam}", x, cy - B / 2, x + THINK[lam], cy + B / 2)
        x += THINK[lam]
    shape(f"answer {lam}", x, cy - B / 2, x + B, cy + B / 2, fill=MINT)

# ---------------------------------------------------------------- (b) student adapters, on the same rows
text("distill label", (A_END + B_X0) / 2, ROWS[0] - B / 2 - LABEL_GAP - text_h(T_SMALL) / 2, "distill", T_SMALL,
     color=LABEL_GRAY, bold=True)
for k in sorted(SHARED):                                 # the shared directions sit at the same positions
    x = B_X0 + k * (B + BG)
    shape(f"shared band {k + 1}", x - 4, ROWS[0] - B / 2 - 8, x + B + 4, ROWS[1] + B / 2 + 8, fill=SHARED_BAND,
          line=None, record=False)
for lam, cy in zip(LAMS, ROWS):
    arrow(f"distill {lam}", B_X0 - ARROW, cy, B_X0)
    adapter(f"adapter {lam}", B_X0, cy, lam)

# the switch: one dashed line through (a) and (b); its labels sit in the free band above it
_, sa1 = text("switch label a", A_BUS - 12, SWY, ["teacher stops", "thinking"], T_SMALL, color=HOT_PINK, bold=True,
              align="right")
line("switch", sa1 + 10, SWY, B_X1, SWY, color=HOT_PINK, w=2.25, dash=MSO_LINE_DASH_STYLE.DASH)
BOXES.pop()                                              # the dashed line crosses the bus and the arrows by design
text("switch label b", B_X1, SWY - 6 - text_h(T_SMALL) / 2, "adapters change direction", T_SMALL, color=HOT_PINK,
     bold=True, align="right")

# ---------------------------------------------------------------- (c) editing the shared region
# Each edit takes two rows of the grid: its name, the edited adapter, then the response before the edit (faint) and
# after it (solid), left-aligned, with a dashed guide at the end of the "before" response. Remove ends before the
# guide (thinking off), Graft runs past it with no answer (thinking on, never finishes), Amplify ends on it (no more).
EDITS = [("Remove", "0", "remove", 0), ("Graft", "1", "graft", 2), ("Amplify", "0", "amplify", 4)]
SLOT_TOP = lambda r: ROWS[r] - ROWGAP / 2
A_OFF, BEFORE_OFF, AFTER_OFF = 82, 140, 187             # adapter, before and after centres below the slot top
BAND_Y = SLOT_TOP(0) + A_OFF                            # enters the trunk level with the Remove branch
line("to edits", B_X1 + TIP, BAND_Y, C_TRUNK, BAND_Y)
line("trunk", C_TRUNK, BAND_Y, C_TRUNK, SLOT_TOP(4) + A_OFF)
TAG_W = max(text_w(t, T_SMALL, False) for t in ("before", "after"))
RX = C_X0 + TAG_W + 16                                  # responses start here
GRAFT_THINK = 300                                       # longer than any teacher's thinking


def response(name, x, cy, think, answer, faint=False):
    """think: thinking length (0 = none); answer: 'yes' or 'missing' (dashed empty box). Returns the right end."""
    if think:
        hatched(f"{name} thinking", x, cy - B / 2, x + think, cy + B / 2, faint=faint)
        x += think
    if answer == "yes":
        shape(f"{name} answer", x, cy - B / 2, x + B, cy + B / 2, fill=MINT, line=LINE_GRAY if faint else BLACK)
    else:
        shape(f"{name} no answer", x, cy - B / 2, x + B, cy + B / 2, fill=WHITE, w=1.75,
              dash=MSO_LINE_DASH_STYLE.SQUARE_DOT)
    return x + B


for name, lam, mode, r in EDITS:
    top = SLOT_TOP(r)
    ay, by, fy = top + A_OFF, top + BEFORE_OFF, top + AFTER_OFF
    _, lx1 = text(f"{name} label", C_X0, top + 22, name, T_LABEL, bold=True, align="left")
    arrow(f"to {name}", C_TRUNK, ay, C_X0)
    adapter(f"{name} adapter", C_X0, ay, lam, mode)
    text(f"{name} before tag", C_X0, by, "before", T_SMALL, color=LABEL_GRAY, align="left")
    text(f"{name} after tag", C_X0, fy, "after", T_SMALL, color=LABEL_GRAY, align="left")
    before_end = response(f"{name} before", RX, by, THINK.get(lam, 0), "yes", faint=True)
    if mode == "remove":
        response(f"{name} after", RX, fy, 0, "yes")
    elif mode == "graft":
        response(f"{name} after", RX, fy, GRAFT_THINK, "missing")
    else:
        response(f"{name} after", RX, fy, THINK["0"], "yes")
    line(f"{name} guide", before_end, by - B / 2 - 6, before_end, fy + B / 2 + 6, color=HOT_PINK, w=1.75,
         dash=MSO_LINE_DASH_STYLE.DASH)
    BOXES.pop()

# ---------------------------------------------------------------- key: one row of its own, centred under all panels
# Every swatch is one B x B block and every entry is separated by the same gap.
KEY_ITEMS = [
    ("thinking", lambda n, x: hatched(n, x, KEY_Y - B / 2, x + B, KEY_Y + B / 2, step=12)),
    ("answer", lambda n, x: shape(n, x, KEY_Y - B / 2, x + B, KEY_Y + B / 2, fill=MINT)),
    ("shared region", lambda n, x: shape(n, x, KEY_Y - B / 2, x + B, KEY_Y + B / 2, fill=SHARED_FILL)),
    ("other directions", lambda n, x: shape(n, x, KEY_Y - B / 2, x + B, KEY_Y + B / 2, fill=tint("0.4"))),
    ("no answer", lambda n, x: shape(n, x, KEY_Y - B / 2, x + B, KEY_Y + B / 2, fill=WHITE, w=1.75,
                                     dash=MSO_LINE_DASH_STYLE.SQUARE_DOT)),
]
KEY_SW_GAP, KEY_ITEM_GAP = 12, 60
widths = [B + KEY_SW_GAP + text_w(lbl, T_SMALL, False) for lbl, _ in KEY_ITEMS]
x = (W - (sum(widths) + KEY_ITEM_GAP * (len(widths) - 1))) / 2
for (lbl, draw), w_ in zip(KEY_ITEMS, widths):
    draw(f"key {lbl}", x)
    text(f"key {lbl} label", x + B + KEY_SW_GAP, KEY_Y, lbl, T_SMALL, color=LABEL_GRAY, align="left")
    x += w_ + KEY_ITEM_GAP

key_boxes = sorted([b_ for b_ in BOXES if b_[1].startswith("key ")], key=lambda b_: b_[2])
panel_bottom = max(b_[5] for b_ in BOXES if not b_[1].startswith("key "))
assert min(b_[3] for b_ in key_boxes) > panel_bottom + 10, "key row touches the panels"

# ---------------------------------------------------------------- checks


def overlaps(a, b, pad=2):
    return a[2] < b[4] - pad and b[2] < a[4] - pad and a[3] < b[5] - pad and b[3] < a[5] - pad


problems = []
for bx in BOXES:
    if bx[2] < MARGIN - 1 or bx[4] > W - MARGIN + 1 or bx[3] < 0 or bx[5] > H:
        problems.append(f"outside canvas: {bx[1]} {bx[2:]}")
texts = [b for b in BOXES if b[0] == "text"]
for t in texts:
    for o in BOXES:
        if o is not t and overlaps(t, o):
            problems.append(f"overlap: '{t[1]}' with '{o[1]}'")
if problems:
    raise SystemExit("layout problems:\n  " + "\n  ".join(sorted(set(problems))))

# size audit: every object of one kind must have the same size
import collections
audit = collections.defaultdict(set)
for kind, name, x0, y0, x1, y1 in BOXES:
    w_, h_ = round(x1 - x0, 1), round(y1 - y0, 1)
    if " dir " in name:
        if name.startswith(("Remove",)) and int(name.split()[-1]) - 1 in SHARED:
            audit["removed block"].add((w_, h_))
        elif name.startswith("Amplify") and int(name.split()[-1]) - 1 in SHARED:
            audit["amplified block"].add((w_, h_))
        else:
            audit["adapter block"].add((w_, h_))
    elif name.startswith("teacher ") and kind == "shape":
        audit["teacher square"].add((w_, h_))
    elif name.startswith("key ") and kind == "shape":
        audit["key swatch"].add((w_, h_))
    elif name.startswith(("answer", "Remove answer", "Amplify answer")):
        audit["answer box"].add((w_, h_))
    elif kind == "line" and (name.startswith(("branch", "to response", "distill ", "to Remove", "to Graft", "to Amplify"))):
        audit["arrow length"].add((w_,))
for k_, v_ in audit.items():
    if len(v_) != 1:
        raise SystemExit(f"size audit failed for {k_}: {sorted(v_)}")
# block gaps along every adapter row
rows_ = collections.defaultdict(list)
for kind, name, x0, y0, x1, y1 in BOXES:
    if " dir " in name:
        rows_[name.rsplit(" dir ", 1)[0]].append((x0, x1))
gaps_ = {round(b[0] - a[1], 1) for r in rows_.values() for a, b in zip(sorted(r), sorted(r)[1:])}
font_sizes = sorted({f[1] for f in FONTS_USED})
report = [f"{k_}: {sorted(v_)[0]}" for k_, v_ in sorted(audit.items())]
report += [f"gaps between blocks: {sorted(gaps_)}",
           f"font sizes (pt): {font_sizes}", f"objects: {len(s.shapes)} separate shapes, none grouped"]
report += [f"text '{t}': {sz} pt{' bold' if b else ''}" for n, sz, b, t in FONTS_USED]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "fig1_cde_audit.txt").write_text("\n".join(report) + "\n")
prs.save(OUT / "fig1_cde.pptx")
spec = dict(W=W, H=H, A_END=A_END, B_X0=B_X0, B_X1=B_X1, C_TRUNK=C_TRUNK, C_X0=C_X0, C_X1=C_X1, ROWS=ROWS, SWY=SWY,
            fonts=[T_HEAD, T_LABEL, T_SMALL], block=B, gap=BG, arrow=ARROW, panel_gap=PANEL_GAP)
(OUT / "fig1_cde_layout.json").write_text(json.dumps(spec, indent=1))
print("saved", OUT / "fig1_cde.pptx", "| right margin", round(W - C_X1), "| checks passed")
