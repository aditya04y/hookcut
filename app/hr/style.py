"""Visual system for the reviewer pages: a marked-up shooting script.

White script paper and ink on a cool desk; two highlighters (hook = yellow, cliffhanger = pink) and the red
editing pencil an editor uses for in/out marks. Archivo for the interface, Courier Prime for script lines
(screenplays are written in Courier).
"""
from __future__ import annotations

import streamlit as st

PAPER = "#FFFFFF"
DESK = "#E8ECF1"
INK = "#18223A"
GRAPHITE = "#5A6378"
RULE = "#CDD4DE"
HOOK = "#FFE45E"        # highlighter yellow
CLIFF = "#FF7EB6"       # highlighter pink
PENCIL = "#C8102E"      # red editing pencil: in/out marks, primary action
HOOK_DARK = "#A67C00"   # hook colour for lines and text on white
CLIFF_DARK = "#C2306D"  # cliffhanger colour for lines and text on white

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&family=Courier+Prime:wght@400;700&display=swap');

.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stMarkdownContainer"] {{
  font-family: 'Archivo', system-ui, sans-serif;
}}
.stApp {{ color: {INK}; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: {DESK} !important; }}
.block-container {{ padding-top: 2.6rem; max-width: 1180px; }}

.stApp h1.hc-title {{
  font-family: 'Archivo', sans-serif !important; font-stretch: 125%; font-variation-settings: "wdth" 125;
  font-weight: 800 !important; font-size: clamp(2.4rem, 4.8vw, 3.5rem) !important; letter-spacing: -0.015em;
  line-height: 1.02 !important; color: {INK}; margin: 0 0 .55rem 0 !important; padding: 0 !important;
}}
.stApp h2.hc-h2 {{
  font-family: 'Archivo', sans-serif !important; font-stretch: 110%; font-variation-settings: "wdth" 110;
  font-weight: 700 !important; font-size: 1.22rem !important; line-height: 1.3 !important;
  color: {INK}; margin: 1.7rem 0 .45rem 0 !important; padding: 0 !important; letter-spacing: 0;
}}
.hc-lede {{ font-size: 1.12rem; line-height: 1.5; color: {GRAPHITE}; max-width: 64ch; margin: 0 0 1.2rem 0; }}
.hc-note {{ font-size: .92rem; color: {GRAPHITE}; max-width: 70ch; }}
.hc-mark-hook {{ background: {HOOK}; padding: 0 .2em; border-radius: 2px; }}
.hc-mark-cliff {{ background: {CLIFF}; padding: 0 .2em; border-radius: 2px; }}
.hc-pencil {{ color: {PENCIL}; font-weight: 700; }}

/* the five steps */
.hc-steps {{ list-style: none; padding: 0; margin: .4rem 0 1.2rem 0; }}
.hc-steps li {{ display: grid; grid-template-columns: 2.1rem 13.5rem 1fr; align-items: baseline;
  padding: .42rem 0; border-bottom: 1px solid {RULE}; }}
.hc-steps li:last-child {{ border-bottom: 0; }}
.hc-num {{ display: inline-grid; place-items: center; width: 1.45rem; height: 1.45rem; border-radius: 50%;
  border: 1.5px solid {GRAPHITE}; color: {GRAPHITE}; font-size: .8rem; font-weight: 700; }}
.hc-steps li.done .hc-num {{ background: {INK}; border-color: {INK}; color: {PAPER}; }}
.hc-steps li.working .hc-num {{ border-color: {PENCIL}; color: {PENCIL}; }}
.hc-step-title {{ font-weight: 650; }}
.hc-step-result {{ color: {GRAPHITE}; }}
.hc-steps li.working .hc-step-result {{ color: {PENCIL}; }}

/* the timeline strip */
.hc-timeline {{ width: 100%; height: auto; display: block; border: 1px solid {RULE}; border-radius: 6px; background: {PAPER}; }}
.hc-legend {{ font-size: .88rem; color: {GRAPHITE}; margin: .35rem 0 0 0; }}

/* the script page */
.hc-page {{ background: {PAPER}; border: 1px solid {RULE}; border-radius: 6px; padding: .9rem 1rem .9rem 0;
  font-family: 'Courier Prime', 'Courier New', monospace; font-size: .92rem; line-height: 1.45; color: {INK};
  max-height: 560px; overflow-y: auto; }}
.hc-line {{ display: grid; grid-template-columns: 5.4rem 1fr; column-gap: .9rem; padding: .08rem .4rem .08rem 0;
  border-left: 3px solid transparent; }}
.hc-gutter {{ text-align: right; color: {GRAPHITE}; font-size: .78rem; padding-top: .12rem; }}
.hc-line.in-cut {{ border-left-color: {PENCIL}; }}
.hc-line.outside {{ opacity: .45; }}
.hc-line .who {{ font-weight: 700; letter-spacing: .02em; }}
.hc-line .how {{ color: {GRAPHITE}; }}
.hc-line.camera {{ color: {GRAPHITE}; font-style: italic; font-size: .82rem; }}
.hc-gutter.in, .hc-gutter.out {{ color: {PENCIL}; font-weight: 700; }}
.hc-text.hook {{ background: rgba(255,228,94,var(--a)); }}
.hc-text.cliff {{ background: rgba(255,126,182,var(--a)); }}

/* the vertical promo frame */
.hc-phone {{ width: 100%; max-width: 270px; aspect-ratio: 9 / 16; border: 2px solid {INK}; border-radius: 30px;
  background: {PAPER}; padding: 1.4rem 1rem 1.2rem; display: flex; flex-direction: column; justify-content: space-between;
  margin: 0 auto; }}
.hc-phone .slot {{ font-size: .78rem; color: {GRAPHITE}; margin-bottom: .25rem; }}
.hc-phone .line {{ font-family: 'Courier Prime', monospace; font-size: .9rem; line-height: 1.4; }}
.hc-phone .mid {{ color: {GRAPHITE}; font-family: 'Courier Prime', monospace; font-size: .82rem; text-align: center; }}
.hc-phone .cut {{ color: {PENCIL}; font-weight: 700; font-size: .8rem; margin-top: .3rem; }}
.hc-facts {{ margin: .9rem 0 0 0; font-size: .95rem; }}
.hc-facts dt {{ color: {GRAPHITE}; font-size: .82rem; margin-top: .55rem; }}
.hc-facts dd {{ margin: 0; }}
.hc-verdict {{ display: inline-block; padding: .05rem .55rem; border-radius: 999px; font-size: .86rem; font-weight: 600; }}
.hc-verdict.pass {{ background: {INK}; color: {PAPER}; }}
.hc-verdict.borderline {{ border: 1.5px solid {INK}; color: {INK}; }}
.hc-verdict.fail {{ color: {GRAPHITE}; text-decoration: line-through; border: 1.5px solid {RULE}; }}
.hc-rules {{ list-style: none; padding: 0; margin: .2rem 0 0 0; font-size: .9rem; }}
.hc-rules li::before {{ content: "✓ "; color: {INK}; font-weight: 700; }}
.hc-rules li.miss::before {{ content: "✗ "; color: {PENCIL}; }}

/* how-it-works flow */
.hc-flow {{ list-style: none; padding: 0; margin: .6rem 0; counter-reset: step; }}
.hc-flow li {{ display: grid; grid-template-columns: 2.1rem 6.2rem 1fr; padding: .5rem 0; border-bottom: 1px solid {RULE}; counter-increment: step; }}
.hc-flow li::before {{ content: counter(step); font-weight: 700; color: {GRAPHITE}; }}
.hc-who {{ font-weight: 700; }}
.hc-who.ai {{ color: {CLIFF_DARK}; }}
.hc-who.code {{ color: {INK}; }}
.hc-who.human {{ color: {PENCIL}; }}
.hc-layer {{ padding: .55rem 0 .55rem .9rem; border-left: 3px solid {RULE}; margin: .5rem 0; }}
.hc-layer.real {{ border-left-color: {INK}; }}
.hc-layer.sim {{ border-left-style: dashed; border-left-color: {GRAPHITE}; }}
.hc-tag {{ font-size: .78rem; font-weight: 700; color: {GRAPHITE}; margin-left: .4rem; }}

@media (max-width: 640px) {{
  .hc-steps li {{ grid-template-columns: 2rem 1fr; }}
  .hc-step-result {{ grid-column: 2; }}
  .hc-line {{ grid-template-columns: 3.8rem 1fr; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior: auto !important; }} }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def title(text: str, lede: str | None = None) -> None:
    st.markdown(f'<h1 class="hc-title">{text}</h1>', unsafe_allow_html=True)
    if lede:
        st.markdown(f'<p class="hc-lede">{lede}</p>', unsafe_allow_html=True)


def h2(text: str) -> None:
    st.markdown(f'<h2 class="hc-h2">{text}</h2>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<p class="hc-note">{text}</p>', unsafe_allow_html=True)
