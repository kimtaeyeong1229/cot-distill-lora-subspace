#!/usr/bin/env python3
"""Figure 1 of the manuscript as an editable PowerPoint slide, in the style of Jack Morris's overview figures.

Layout follows the Gemini draft plan/figures/fig1_morris/m1_pipeline.png and the Morris figures it was prompted
with (Language Model Inversion, Vec2Text, Contextual Document Embeddings): thick black outlines, models as
gradient trapezoids, adapters as rows of outlined squares, monospace example text with the reflective words in
dark red and underlined, bold stage headers, black elbow arrows. Colours match Figures 2-5 (navy, slate, yellow).
Every shape is native, so labels, arrows and colours stay editable. Coordinates are in a 2000 x 848 canvas that
maps onto a 13.333 x 5.653 inch slide. Run inside the experiment container (python-pptx); writes
materials/main_figure_pptx/fig1_overview.pptx.
"""

from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

OUT = Path("/home/lab/workspace/cot-distill-lora-bit/materials/main_figure_pptx")
W_EMU = 12192000
K = W_EMU / 2000
E = lambda v: Emu(int(round(v * K)))

BLACK = RGBColor(0, 0, 0)
INK = RGBColor(34, 34, 40)
NAVY = RGBColor(0x1F, 0x2C, 0x56)
BLUE_LIGHT = RGBColor(0xC9, 0xD6, 0xF2)
BLUE_TEXT = RGBColor(0x2B, 0x4C, 0x9B)
RED = RGBColor(0xA3, 0x24, 0x2F)
RED_LIGHT = RGBColor(0xF6, 0xD3, 0xD3)
MUSTARD = RGBColor(0xF2, 0xCF, 0x3A)
GREY = RGBColor(0xD6, 0xD6, 0xD6)
GREY_TEXT = RGBColor(0x55, 0x55, 0x55)
TAG = RGBColor(0xE6, 0xE6, 0xE6)
WHITE = RGBColor(255, 255, 255)
SANS, MONO = "Arial", "Courier New"
OUTLINE = 2.25

prs = Presentation()
prs.slide_width = Emu(W_EMU)
prs.slide_height = E(848)
s = prs.slides.add_slide(prs.slide_layouts[6])


def strip_style(sh):
    st = sh._element.find(qn("p:style"))
    if st is not None:
        sh._element.remove(st)


def outline(sh, color=BLACK, w=OUTLINE, dash=None):
    if color is None:
        sh.line.fill.background()
        return
    sh.line.color.rgb = color
    sh.line.width = Pt(w)
    if dash:
        sh.line.dash_style = dash


def fill(sh, color):
    if color is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = color


def gradient(sh, c0, c1, angle=0):
    sh.fill.gradient(); sh.fill.gradient_angle = angle
    st = sh.fill.gradient_stops
    st[0].color.rgb = c0; st[0].position = 0.0
    st[1].color.rgb = c1; st[1].position = 1.0


def box(name, x0, y0, x1, y1, kind=MSO_SHAPE.RECTANGLE, color=WHITE, line=BLACK, w=OUTLINE, dash=None):
    sh = s.shapes.add_shape(kind, E(x0), E(y0), E(x1 - x0), E(y1 - y0))
    sh.name = name
    strip_style(sh)
    fill(sh, color)
    outline(sh, line, w, dash)
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = 0.06
    return sh


def runs(sh, paragraphs, size, font=SANS, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE,
         margin=12, spacing=None):
    """paragraphs: list of lines; a line is a string or a list of (text, style) with style in {'', 'red'}."""
    tf = sh.text_frame
    tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = E(margin); tf.margin_top = tf.margin_bottom = E(4)
    tf.vertical_anchor = anchor
    for i, ln in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.line_spacing = spacing
        for t, style in ([(ln, "")] if isinstance(ln, str) else ln):
            r = p.add_run(); r.text = t
            f = r.font; f.size = Pt(size); f.bold = bold; f.name = font
            f.color.rgb = RED if style == "red" else color
            f.underline = style == "red"
    return sh


def text(name, x0, y0, x1, y1, paragraphs, size, **kw):
    tb = s.shapes.add_textbox(E(x0), E(y0), E(x1 - x0), E(y1 - y0))
    tb.name = name
    kw.setdefault("margin", 0)
    nb = lambda t: t.replace(" ", "\u00a0")                 # keep one-line labels on one line in every renderer
    paragraphs = [nb(p) if isinstance(p, str) else [(nb(t), st) for t, st in p] for p in paragraphs]
    sh = runs(tb, paragraphs, size, **kw)
    sh.text_frame.word_wrap = False
    return sh


def line(name, x0, y0, x1, y1, color=BLACK, w=OUTLINE, head=False, dash=None):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x0), E(y0), E(x1), E(y1))
    c.name = name
    strip_style(c)
    c.line.color.rgb = color; c.line.width = Pt(w)
    if dash:
        c.line.dash_style = dash
    if head:
        ln = c.line._get_or_add_ln()
        te = etree.SubElement(ln, qn("a:tailEnd")); te.set("type", "triangle"); te.set("w", "med"); te.set("len", "med")
    return c


def elbow(name, pts):
    """Polyline of straight segments with an arrow head on the last one."""
    for i in range(len(pts) - 1):
        line(f"{name} {i + 1}", *pts[i], *pts[i + 1], head=i == len(pts) - 2)


def trapezoid(name, x0, y0, x1, y1, wide_left, c_dark, c_light):
    """Model icon: a trapezoid with its wide side facing left or right, in a soft gradient."""
    inset = (y1 - y0) * 0.2
    if wide_left:
        pts = [(x0, y0), (x1, y0 + inset), (x1, y1 - inset), (x0, y1)]
    else:
        pts = [(x0, y0 + inset), (x1, y0), (x1, y1), (x0, y1 - inset)]
    fb = s.shapes.build_freeform(E(pts[0][0]), E(pts[0][1]), scale=1.0)
    fb.add_line_segments([(E(x), E(y)) for x, y in pts[1:]], close=True)
    sh = fb.convert_to_shape()
    sh.name = name
    strip_style(sh)
    gradient(sh, c_dark, c_light, angle=0 if wide_left else 180)
    outline(sh)
    return sh


# Adapter rows: which of the 16 directions are the shared region, and how the rest look per lambda.
SHARED = {1, 2, 4, 7, 8, 11}
OTHER = {                                    # light-grey squares = other directions the adapter uses
    "0": {0, 5, 9, 13, 14}, "0.2": {0, 5, 10, 13, 15},
    "0.4": {0, 3, 6, 10, 12, 13, 15}, "0.6": {3, 5, 6, 9, 12, 13, 14}, "0.8": {2, 3, 6, 9, 10, 12, 14, 15},
    "1": {2, 3, 5, 6, 9, 10, 12, 14, 15},
}


def adapter(name, x0, y0, sq_w, sq_h, gap, lam, shared="fill", scale=1.0):
    """A LoRA adapter as 16 outlined squares. shared: 'fill' (mustard), 'none', 'removed' (dashed hollow)."""
    for k in range(16):
        x = x0 + k * (sq_w + gap)
        if k in SHARED and shared != "none":
            if shared == "removed":
                box(f"{name} dir {k + 1}", x, y0, x + sq_w, y0 + sq_h, color=WHITE, w=1.5,
                    dash=MSO_LINE_DASH_STYLE.DASH)
                continue
            dy = sq_h * (scale - 1) / 2
            box(f"{name} dir {k + 1}", x, y0 - dy, x + sq_w, y0 + sq_h + dy, color=MUSTARD, w=2.0)
            continue
        box(f"{name} dir {k + 1}", x, y0, x + sq_w, y0 + sq_h, color=GREY if k in OTHER[lam] else WHITE, w=1.5)


# ------------------------------------------------------------------ stage 1: interpolated teachers
text("header teachers", 20, 18, 680, 78, ["Interpolated teachers"], 20, bold=True, align=PP_ALIGN.CENTER)
text("label reasoning", 20, 92, 300, 160, ["Reasoning model"], 16, bold=True, color=BLUE_TEXT,
     align=PP_ALIGN.CENTER)
text("label instruct", 400, 92, 680, 160, ["Instruct model"], 16, bold=True, color=RED, align=PP_ALIGN.CENTER)
trapezoid("reasoning model", 90, 165, 230, 315, True, NAVY, BLUE_LIGHT)
trapezoid("instruct model", 470, 165, 610, 315, False, RED, RED_LIGHT)
text("interpolate", 240, 205, 460, 275, ["interpolate", "weights"], 14, color=GREY_TEXT, align=PP_ALIGN.CENTER)
line("interpolate left", 300, 290, 240, 290, w=1.5, head=True)
line("interpolate right", 400, 290, 460, 290, w=1.5, head=True)
SL0, SL1, SY = 160, 540, 385
TX = [SL0 + i * (SL1 - SL0) / 5 for i in range(6)]
elbow("reasoning to slider", [(160, 315), (160, SY - 12)])
elbow("instruct to slider", [(540, 315), (540, SY - 12)])
bar = box("lambda slider", SL0, SY - 8, SL1, SY + 8, kind=MSO_SHAPE.ROUNDED_RECTANGLE, line=BLACK, w=1.75)
bar.adjustments[0] = 0.5
gradient(bar, NAVY, RED)
for x, lab in zip(TX, ["0", "0.2", "0.4", "0.6", "0.8", "1"]):
    line(f"tick {lab}", x, SY + 8, x, SY + 22, w=1.5)
    text(f"tick label {lab}", x - 40, SY + 24, x + 40, SY + 54, [lab], 14, align=PP_ALIGN.CENTER)
text("lambda", SL0 - 70, SY - 16, SL0 - 18, SY + 16, ["λ"], 16, bold=True, align=PP_ALIGN.RIGHT)
SW = (TX[1] + TX[2]) / 2
line("switch line", SW, SY - 40, SW, SY + 92, color=RED, w=2.0, dash=MSO_LINE_DASH_STYLE.DASH)
text("switch label", SW + 10, SY + 62, SW + 330, SY + 92, ["teacher stops thinking"], 14, bold=True, color=RED)
out0 = box("lambda0 output", 20, 520, 330, 740)
runs(out0, ["Okay, let me", [("think.", "")], "12 × 15 = 180.", [("Wait", "red"), (", let me", "")],
            "re-check..."], 14, font=MONO, bold=True, margin=14)
out1 = box("lambda1 output", 370, 520, 680, 740)
runs(out1, ["12 × 15 = 180.", "The answer is", "180."], 14, font=MONO, bold=True, margin=14)
text("output label 0", 20, 745, 330, 780, ["λ = 0 teacher"], 13, color=GREY_TEXT, align=PP_ALIGN.CENTER)
text("output label 1", 370, 745, 680, 780, ["λ = 1 teacher"], 13, color=GREY_TEXT, align=PP_ALIGN.CENTER)

# ------------------------------------------------------------------ stage 2: LoRA adapters of the students
text("header adapters", 715, 18, 1265, 78, ["Student LoRA adapters"], 20, bold=True, align=PP_ALIGN.CENTER)
text("lambda column", 770, 72, 828, 106, ["λ"], 14, bold=True, align=PP_ALIGN.RIGHT)
elbow("distil arrow", [(688, 472), (765, 472)])
text("distil label", 688, 434, 765, 466, ["distil"], 14, bold=True, align=PP_ALIGN.CENTER)
SQW, SQH, GAP, AX = 24, 38, 3, 840
ROWY = [110 + i * 98 for i in range(6)]
for lam, y in zip(["0", "0.2", "0.4", "0.6", "0.8", "1"], ROWY):
    text(f"row label {lam}", 770, y, 828, y + SQH, [lam], 14, align=PP_ALIGN.RIGHT)
    adapter(f"adapter lambda {lam}", AX, y, SQW, SQH, GAP, lam, shared="fill" if lam in ("0", "0.2") else "none")
SWY = (ROWY[1] + SQH + ROWY[2]) / 2
line("switch line adapters", 735, SWY, AX + 16 * (SQW + GAP), SWY, color=RED, w=2.0, dash=MSO_LINE_DASH_STYLE.DASH)
text("switch label adapters", 900, SWY + 2, AX + 16 * (SQW + GAP), SWY + 30, ["adapters change direction"], 12,
     bold=True, color=RED, align=PP_ALIGN.RIGHT)
ly = 718
box("key shared", 790, ly + 4, 812, ly + 30, color=MUSTARD, w=2.0)
text("key shared label", 822, ly, 1010, ly + 34, ["shared region"], 14)
box("key other", 1030, ly + 4, 1052, ly + 30, color=GREY, w=1.5)
text("key other label", 1062, ly, 1290, ly + 34, ["other directions"], 14)

# ------------------------------------------------------------------ stage 3: editing the shared region
text("header edits", 1290, 18, 1990, 78, ["Editing the shared region"], 20, bold=True, align=PP_ALIGN.CENTER)
EX, ESW, ESH, EGAP = 1320, 19, 32, 2
EROW = [205, 435, 665]
TRUNK = 1290
line("trunk", AX + 16 * (SQW + GAP) + 8, 430, TRUNK, 430)
line("trunk vertical", TRUNK, EROW[0] + ESH / 2, TRUNK, EROW[2] + ESH / 2)
EDITS = [
    ("Remove", "0", "removed", 1.0,
     ["12 × 15 = 180.", "The answer is", "180."], "short and complete"),
    ("Graft into λ = 1", "1", "fill", 1.0,
     [[("Wait", "red"), ("...", "")], [("wait", "red"), ("...", "")], [("wait", "red"), ("...", "")]],
     "never finishes"),
    ("Amplify", "0", "fill", 1.45,
     ["Okay, let me", "think.", [("Wait", "red"), (", let me", "")]], "no more thinking"),
]
AEND = EX + 16 * (ESW + EGAP)
for (name, lam, mode, scale, out, tag), y in zip(EDITS, EROW):
    cy = y + ESH / 2
    elbow(f"to {name}", [(TRUNK, cy), (EX - 8, cy)])
    text(f"{name} label", EX, y - 62, EX + 360, y - 22, [name], 16, bold=True)
    adapter(f"{name} adapter", EX, y, ESW, ESH, EGAP, lam, shared=mode, scale=scale)
    elbow(f"{name} out", [(AEND + 8, cy), (1688, cy)])
    ob = box(f"{name} output", 1695, cy - 82, 1990, cy + 82)
    runs(ob, out, 14, font=MONO, bold=True, margin=14)
    t = box(f"{name} tag", 1765, cy - 100, 1990, cy - 70, kind=MSO_SHAPE.ROUNDED_RECTANGLE, color=TAG, line=None)
    t.adjustments[0] = 0.5
    runs(t, [tag], 12, color=INK, align=PP_ALIGN.CENTER, margin=4)
    t.text_frame.word_wrap = False

OUT.mkdir(parents=True, exist_ok=True)
prs.save(OUT / "fig1_overview.pptx")
print("saved", OUT / "fig1_overview.pptx")
