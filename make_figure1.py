"""Generate Figure 1 (classifier pipeline + locked-prompt validation design).

Regenerates figure1_pipeline.png. The original v1 image had no source
script; this recreates it with one wording change in the Panel B results
box: "3-way κ" -> "Three-category κ", "human κ" -> "Inter-adjudicator
benchmark κ", to match the manuscript's terminology and avoid reading as
a three-rater comparison.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "Arial"

FILL = "#f2f2f2"
FILL_DARK = "#d9d9d9"
EDGE = "#4d4d4d"

fig = plt.figure(figsize=(1631 / 300, 1307 / 300), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1631)
ax.set_ylim(0, 1307)
ax.invert_yaxis()
ax.axis("off")


def box(cx, cy, w, h, title, lines, dark=False, lw=1.0, title_size=6.2,
        body_size=5.4, gap=18):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=14",
        facecolor=FILL_DARK if dark else FILL,
        edgecolor="black" if dark else EDGE,
        linewidth=lw, mutation_aspect=1))
    n_body = len(lines)
    # vertical layout: title block then body lines
    title_lines = title.count("\n") + 1
    total = title_lines * 1.35 * 16 + gap + n_body * 1.35 * 16
    y = cy - h / 2 + (h - total) / 2
    ax.text(cx, y + (title_lines * 1.35 * 16) / 2, title, ha="center",
            va="center", fontsize=title_size, fontweight="bold",
            linespacing=1.35)
    y += title_lines * 1.35 * 16 + gap
    for ln in lines:
        ax.text(cx, y + 0.675 * 16, ln, ha="center", va="center",
                fontsize=body_size, linespacing=1.3)
        y += 1.35 * 16


def arrow(x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_width=2.2,head_length=4.5",
        linewidth=1.1, color="black", shrinkA=0, shrinkB=0))


# ---------------- Panel A ----------------
ax.text(107, 62, "A", fontsize=13, fontweight="bold", ha="center",
        va="center")
ax.text(172, 62, "Classification pipeline (one API call per encounter)",
        fontsize=9.5, ha="left", va="center")

ay = 265  # panel A box center y
aw, ah = 268, 340
xs = [243, 546, 849, 1152, 1458]
panels = [
    ("LEO-flagged\nED encounters",
     ["n = 3,121", "six NYC EDs", "2012–2024", "36,197 note rows"]),
    ("Note\npreprocessing",
     ["drop boilerplate;", "reassemble split", "notes; chronological;",
      "labeled headers"]),
    ("Prompt +\nschema",
     ["verbatim codebook;", "NYC framing; rules", "R1–R8; confidence",
      "rubric; enum schema"]),
    ("GPT-5\nAzure OpenAI",
     ["institutional BAA —", "no PHI leaves", "boundary; hash-",
      "stamped metadata"]),
    ("Structured\noutput",
     ["category (0/1/2);", "audit tag;", "evidence quote;", "confidence"]),
]
for x, (t, lines) in zip(xs, panels):
    box(x, ay, aw, ah, t, lines)
for x1, x2 in zip(xs[:-1], xs[1:]):
    arrow(x1 + aw / 2 + 4, ay, x2 - aw / 2 - 4, ay)

# ---------------- Panel B ----------------
ax.text(107, 533, "B", fontsize=13, fontweight="bold", ha="center",
        va="center")
ax.text(172, 533, "Locked-prompt validation design", fontsize=9.5,
        ha="left", va="center")

box(240, 730, 250, 180, "Random 10%\nsample", ["n = 312"])
box(240, 1020, 250, 190, "Analyzable\ncharts", ["n = 309", "(3 excluded)"])
box(655, 720, 355, 195, "Development set",
    ["n = 50, stratified;", "prompt iteration +", "error review"])
box(1025, 715, 220, 195, "PROMPT\nLOCKED", ["version 1.0"], dark=True,
    lw=2.2)
box(655, 1055, 355, 195, "Held-out\nevaluation set",
    ["n = 259; includes all", "62 double-coded charts"])
box(1025, 1055, 220, 195, "Single\nscoring pass",
    ["no prompt", "changes"])
box(1420, 1000, 360, 350, "Agreement vs\nprimary adjudicator",
    ["Three-category κ 0.653", "(0.570–0.731)",
     "Binary (any police vs none)", "κ 0.734 (0.619–0.834)",
     "Inter-adjudicator", "benchmark κ 0.761"], gap=14)

arrow(240, 730 + 90 + 4, 240, 1020 - 95 - 4)                    # sample -> analyzable
arrow(240 + 125 + 4, 970, 655 - 177 - 4, 745, rad=-0.35)        # analyzable -> dev
arrow(240 + 125 + 4, 1045, 655 - 177 - 4, 1060, rad=0.0)        # analyzable -> heldout
arrow(655 + 177 + 4, 718, 1025 - 110 - 4, 716)                  # dev -> locked
arrow(1025, 715 + 97 + 4, 1025, 1055 - 97 - 4)                  # locked -> scoring
arrow(655 + 177 + 4, 1055, 1025 - 110 - 4, 1055)                # heldout -> scoring
arrow(1025 + 110 + 4, 1035, 1420 - 175 - 4, 1005, rad=0.0)      # scoring -> results

ax.text(35, 1230,
        "reference standard: primary adjudicator;\n"
        "62 charts double-coded, κ = 0.761",
        fontsize=6.4, fontstyle="italic", ha="left", va="center",
        linespacing=1.5)

fig.savefig("figure1_pipeline.png", dpi=300, facecolor="white")
print("wrote figure1_pipeline.png")
