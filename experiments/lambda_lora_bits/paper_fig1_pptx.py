#!/usr/bin/env python3
"""Figure 1 of the manuscript as an editable PowerPoint slide.

Layout follows the Gemini draft d1 (three panels: interpolated teachers, LoRA students, editing the shared region),
rebuilt from native shapes so every label, arrow and colour can be edited. Coordinates are in a 2000 x 848 canvas
that maps onto a 13.333 x 5.653 inch slide. Run inside the experiment container (python-pptx); writes
materials/main_figure_pptx/fig1_overview.pptx.
"""

import math
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt
from lxml import etree

OUT = Path("/home/lab/workspace/cot-distill-lora-bit/materials/main_figure_pptx")
W_EMU = 12192000
K = W_EMU / 2000
E = lambda v: Emu(int(round(v * K)))

INK = RGBColor(34, 34, 40)
GREY = RGBColor(140, 140, 140)
LIGHT = RGBColor(225, 225, 225)
BLUE = RGBColor(92, 120, 166)        # thinking
CORAL = RGBColor(238, 120, 92)       # short answer
MUSTARD = RGBColor(204, 160, 40)     # shared region
SAND = RGBColor(242, 228, 186)       # shared region fill
RED = RGBColor(178, 34, 52)
WHITE = RGBColor(255, 255, 255)
FONT = "Arial"

prs = Presentation()
prs.slide_width = Emu(W_EMU)
prs.slide_height = E(848)
s = prs.slides.add_slide(prs.slide_layouts[6])


def strip_style(sh):
    st = sh._element.find(qn("p:style"))
    if st is not None:
        sh._element.remove(st)


def shape(name, kind, x0, y0, x1, y1, fill=None, line=None, lw=1.0, dash=None):
    sh = s.shapes.add_shape(kind, E(x0), E(y0), E(x1 - x0), E(y1 - y0))
    sh.name = name
    strip_style(sh)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(lw)
        if dash:
            sh.line.dash_style = dash
    return sh


def write(sh, lines, size, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, italic=False):
    tf = sh.text_frame
    tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = E(14); tf.margin_top = tf.margin_bottom = E(6)
    tf.vertical_anchor = anchor
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = ln
        f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic; f.name = FONT; f.color.rgb = color
    return sh


def text(name, x0, y0, x1, y1, lines, size, **kw):
    tb = s.shapes.add_textbox(E(x0), E(y0), E(x1 - x0), E(y1 - y0))
    tb.name = name
    w = write(tb, lines, size, **kw)
    tb.text_frame.margin_left = tb.text_frame.margin_right = 0
    return w


def line(name, x0, y0, x1, y1, color=GREY, w=1.5, head=False, dash=None):
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


def fan(name, cx, cy, mustard=6, angle0=-80, spread=150, length=70, shared_angle=None, grey_dirs=None, extra=0,
        hide_shared=False, scale_shared=1.0):
    """A LoRA adapter drawn as 16 direction arrows from one point.

    The first `mustard` directions are the shared region (mustard, clustered toward shared_angle); the rest are grey.
    """
    n = 16
    for k in range(n):
        if k < mustard:
            if hide_shared:
                continue
            a = (shared_angle if shared_angle is not None else -55) + (k - mustard / 2) * 7
            col, ln = MUSTARD, length * scale_shared
            w = 2.6 if scale_shared > 1 else 2.0
        else:
            a = angle0 + (k - mustard) * spread / (n - mustard) if grey_dirs is None else grey_dirs[k - mustard]
            col, ln, w = GREY, length * 0.92, 1.2
        r = math.radians(a)
        line(f"{name} dir {k + 1}", cx, cy, cx + ln * math.cos(r), cy + ln * math.sin(r), color=col, w=w, head=True)


# ------------------------------------------------------------------ (a) interpolated teachers
text("title a", 40, 30, 700, 80, ["(a) Interpolated teachers"], 20, bold=True)
text("label reasoning", 60, 100, 420, 140, ["reasoning model (λ = 0)"], 15)
text("label instruct", 850, 100, 1190, 140, ["instruct model (λ = 1)"], 15, align=PP_ALIGN.RIGHT)
bar = shape("lambda slider", MSO_SHAPE.ROUNDED_RECTANGLE, 135, 158, 1145, 172, fill=BLUE, line=GREY, lw=0.75)
bar.fill.gradient(); bar.fill.gradient_angle = 0
stops = bar.fill.gradient_stops
stops[0].color.rgb = BLUE; stops[0].position = 0.0
stops[1].color.rgb = RGBColor(245, 245, 245); stops[1].position = 1.0
shape("slider knob", MSO_SHAPE.OVAL, 122, 148, 154, 182, fill=WHITE, line=GREY, lw=1.25)
TX = [140, 337, 534, 731, 928, 1125]
for x, lab in zip(TX, ["0", ".2", ".4", ".6", ".8", "1"]):
    line(f"tick {lab}", x, 172, x, 192, color=GREY, w=1.25)
    text(f"tick label {lab}", x - 40, 196, x + 40, 236, [lab], 15, align=PP_ALIGN.CENTER)
line("switch line", 455, 112, 455, 805, color=RED, w=1.75, dash=MSO_LINE_DASH_STYLE.DASH)
text("switch label", 468, 240, 700, 320, ["teacher stops", "thinking"], 15, color=RED)
card = shape("lambda0 output", MSO_SHAPE.ROUNDED_RECTANGLE, 40, 262, 420, 418, fill=BLUE)
write(card, ["Okay, let me think…", "12 × 15 = 180.", "Wait, let me re-check…"], 15, color=WHITE)
card = shape("lambda1 output", MSO_SHAPE.ROUNDED_RECTANGLE, 880, 262, 1198, 418, fill=CORAL)
write(card, ["12 × 15 = 180.", "The answer is 180."], 15, color=WHITE)

# ------------------------------------------------------------------ (b) LoRA students
text("title b", 40, 462, 700, 512, ["(b) LoRA students"], 20, bold=True)
shape("shared region fill", MSO_SHAPE.OVAL, 60, 540, 450, 745, fill=SAND)
lab_sr = text("shared region label", 30, 752, 480, 790, ["shared region (5–8 of 16 directions)"], 14,
     align=PP_ALIGN.CENTER, color=RGBColor(120, 90, 10))
lab_sr.text_frame.word_wrap = False
FX = [160, 345, 540, 730, 920, 1110]
for i, (x, lab) in enumerate(zip(FX, ["0", ".2", ".4", ".6", ".8", "1"])):
    pre = i < 2
    fan(f"adapter lambda {lab}", x, 650, mustard=6 if pre else 0, angle0=-170 if pre else -150 + 25 * i,
        spread=200 if pre else 230, length=72)
    if not pre:
        text(f"adapter label {lab}", x - 50, 752, x + 50, 792, [f"λ = {lab}"], 13, align=PP_ALIGN.CENTER,
             color=GREY)
for x, lab in zip(FX[:2], ["0", ".2"]):
    text(f"adapter label {lab}", x - 50, 698, x + 50, 732, [f"λ = {lab}"], 13, align=PP_ALIGN.CENTER, color=GREY)

# ------------------------------------------------------------------ (c) editing the region
text("title c", 1290, 30, 1990, 80, ["(c) Editing the shared region"], 20, bold=True)
line("bracket stem", 1185, 645, 1258, 645, color=GREY, w=1.5)
line("bracket vertical", 1258, 205, 1258, 685, color=GREY, w=1.5)
ROWS = [
    ("remove", 205, dict(hide_shared=True), CORAL, ["12 × 15 = 180.", "The answer is 180."], "short, complete"),
    ("graft", 445, dict(), BLUE, ["Wait… wait… wait…"], "never finishes"),
    ("amplify", 685, dict(scale_shared=1.35), BLUE, ["Okay, let me think…", "Wait, let me re-check…"],
     "no more thinking"),
]
for name, y, kw, col, out, tag in ROWS:
    line(f"to {name}", 1258, y, 1300, y, color=GREY, w=1.5, head=True)
    shape(f"{name} region", MSO_SHAPE.OVAL, 1310, y - 88, 1450, y + 88, fill=SAND if not kw.get("hide_shared") else None,
          line=MUSTARD if kw.get("hide_shared") else None, lw=1.25,
          dash=MSO_LINE_DASH_STYLE.DASH if kw.get("hide_shared") else None)
    fan(f"{name} adapter", 1372, y + 10, angle0=-170, spread=200, length=64, **kw)
    line(f"{name} arrow", 1462, y, 1615, y, color=GREY, w=1.75, head=True)
    text(f"{name} label", 1462, y - 48, 1615, y - 8, [name], 15, align=PP_ALIGN.CENTER)
    if name == "graft":
        text("graft sublabel", 1462, y + 6, 1615, y + 40, ["into λ = 1"], 12, align=PP_ALIGN.CENTER, color=GREY)
    card = shape(f"{name} output", MSO_SHAPE.ROUNDED_RECTANGLE, 1625, y - 62, 1990, y + 62, fill=col)
    write(card, out, 15, color=WHITE)
    text(f"{name} tag", 1625, y + 66, 1990, y + 100, [tag], 13, align=PP_ALIGN.RIGHT, color=col, italic=True)

OUT.mkdir(parents=True, exist_ok=True)
prs.save(OUT / "fig1_overview.pptx")
print("saved", OUT / "fig1_overview.pptx")
