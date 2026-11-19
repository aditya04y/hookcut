"""HTML/SVG pieces for the reviewer pages: the five steps, the timeline strip, the marked-up script page and the
vertical promo frame, plus the season chart, story graph and evaluation chart."""
from __future__ import annotations

import html
from typing import Iterable, Optional

import altair as alt
import pandas as pd

from hookcut.schema import Beat, Episode, EpisodeAnalysis, Promo, Segment
from hookcut.textfmt import tc

from .style import CLIFF, CLIFF_DARK, GRAPHITE, HOOK, HOOK_DARK, INK, PENCIL, RULE

VERDICT_WORDS = {
    "pass": "A stranger followed it",
    "borderline": "A stranger half-followed it",
    "fail": "A stranger got lost",
}
WAVE_WORDS = {1: "Post first, to new viewers", 2: "Post next", 3: "Save for people who know the show"}


def esc(text: str) -> str:
    return html.escape(str(text or ""), quote=True)


def short_tc(seconds: float) -> str:
    m, s = divmod(max(0.0, float(seconds)), 60)
    return f"{int(m):02d}:{int(s):02d}"


def cut_label(segments: Iterable[Segment]) -> str:
    return " + ".join(f"E{s.episode:02d} {short_tc(s.start)}–{short_tc(s.end)}" for s in segments)


# ------------------------------------------------------------------------------------------------- steps
def steps_html(steps: list[dict]) -> str:
    """steps: [{"title", "result", "state": done|working|waiting}] in order."""
    rows = []
    for i, s in enumerate(steps, 1):
        state = s.get("state", "waiting")
        result = s.get("result") or ("Working…" if state == "working" else "")
        rows.append(f'<li class="{state}"><span class="hc-num">{i}</span><span class="hc-step-title">{esc(s["title"])}'
                    f'</span><span class="hc-step-result">{esc(result)}</span></li>')
    return f'<ol class="hc-steps">{"".join(rows)}</ol>'


# ---------------------------------------------------------------------------------------------- timeline
def timeline_svg(ep: Episode, analysis: Optional[EpisodeAnalysis], segments: Iterable[Segment] = ()) -> str:
    """A mirrored waveform: hook strength rises above the line, cliffhanger strength hangs below it,
    every line of the script is a tick, and the chosen cut is bracketed in red pencil."""
    W, H, MID, UNIT, PAD = 1000, 136, 66, 10.5, 30
    dur = max(ep.duration, 1.0)
    x = lambda t: round(PAD + (W - 2 * PAD) * min(max(t, 0.0), dur) / dur, 1)
    scores = {s.beat_id: s for s in (analysis.beats if analysis else [])}
    out = [f'<svg class="hc-timeline" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Episode {ep.number}: hook strength above the line, cliffhanger strength below, chosen cut bracketed">']
    for seg in segments:
        if seg.episode != ep.number:
            continue
        a, b = x(seg.start), x(seg.end)
        out.append(f'<rect x="{a}" y="8" width="{max(b - a, 1)}" height="{H - 30}" fill="{PENCIL}" fill-opacity="0.06"/>')
        for xx, label, anchor, dx in ((a, "In", "start", 4), (b, "Out", "end", -4)):
            out.append(f'<line x1="{xx}" y1="8" x2="{xx}" y2="{H - 22}" stroke="{PENCIL}" stroke-width="2"/>')
            out.append(f'<line x1="{xx}" y1="8" x2="{xx + (8 if anchor == "start" else -8)}" y2="8" stroke="{PENCIL}" stroke-width="2"/>')
            out.append(f'<line x1="{xx}" y1="{H - 22}" x2="{xx + (8 if anchor == "start" else -8)}" y2="{H - 22}" stroke="{PENCIL}" stroke-width="2"/>')
            out.append(f'<text x="{xx + dx}" y="22" fill="{PENCIL}" font-size="12" font-weight="700" '
                       f'font-family="Archivo, sans-serif" text-anchor="{anchor}">{label}</text>')
    for b in ep.beats:
        out.append(f'<line x1="{x(b.start)}" y1="{MID - 2}" x2="{x(b.start)}" y2="{MID + 2}" stroke="{RULE}" stroke-width="1"/>')
    for b in ep.beats:
        s = scores.get(b.id)
        if not s:
            continue
        w = max(3.0, min(9.0, (b.end - b.start) / dur * (W - 2 * PAD)))
        if s.hook:
            out.append(f'<rect x="{x(b.start)}" y="{MID - s.hook * UNIT}" width="{w:.1f}" height="{s.hook * UNIT}" '
                       f'fill="{HOOK}" stroke="{HOOK_DARK}" stroke-width="0.6"><title>{esc(tc(b.start))} hook {s.hook}/5</title></rect>')
        if s.cliff:
            out.append(f'<rect x="{x(b.start)}" y="{MID}" width="{w:.1f}" height="{s.cliff * UNIT}" '
                       f'fill="{CLIFF}" stroke="{CLIFF_DARK}" stroke-width="0.6"><title>{esc(tc(b.start))} cliffhanger {s.cliff}/5</title></rect>')
    out.append(f'<line x1="{PAD}" y1="{MID}" x2="{W - PAD}" y2="{MID}" stroke="{INK}" stroke-width="1"/>')
    step = 10 if dur <= 150 else 20
    t = 0
    while t <= dur:
        out.append(f'<text x="{x(t)}" y="{H - 6}" fill="{GRAPHITE}" font-size="11" text-anchor="middle" '
                   f'font-family="Courier Prime, monospace">{short_tc(t)}</text>')
        t += step
    out.append("</svg>")
    return "".join(out)


def legend_html() -> str:
    return ('<p class="hc-legend"><span class="hc-mark-hook">Yellow, above the line</span>: how hard a line would stop '
            'someone scrolling. <span class="hc-mark-cliff">Pink, below</span>: how badly you would need to know what '
            'happens next if the clip cut there. <span class="hc-pencil">Red brackets</span>: the chosen cut.</p>')


# ------------------------------------------------------------------------------------------- script page
def _line_html(b: Beat, score, in_cut: bool, gutter: str, gutter_cls: str, dim: bool, force: str = "") -> str:
    cls = ["hc-line"]
    if in_cut:
        cls.append("in-cut")
    if dim:
        cls.append("outside")
    if b.kind == "camera":
        cls.append("camera")
    text_cls, alpha = "", 0
    if force:
        text_cls, alpha = force, 0.95
    elif score is not None:
        level = max(score.hook, score.cliff)
        if level >= 3:
            text_cls = "hook" if score.hook >= score.cliff else "cliff"
            alpha = {3: 0.4, 4: 0.7, 5: 0.95}[min(level, 5)]
    if b.kind in ("dialogue", "vo"):
        how = f' <span class="how">({esc(b.delivery)})</span>' if b.delivery else ""
        body = f'<span class="who">{esc((b.speaker or "").upper())}</span>{how} {esc(b.text)}'
    elif b.kind == "camera":
        body = f"[Camera] {esc(b.text)}"
    elif b.kind == "sfx":
        body = f"[Sound] {esc(b.text)}"
    else:
        body = esc(b.text)
    style = f' style="--a:{alpha}"' if text_cls else ""
    return (f'<div class="{" ".join(cls)}"><span class="hc-gutter {gutter_cls}">{esc(gutter)}</span>'
            f'<span class="hc-text {text_cls}"{style}>{body}</span></div>')


def script_page_html(ep: Episode, analysis: Optional[EpisodeAnalysis], segments: Iterable[Segment] = (),
                     context_s: Optional[float] = 12.0, show_camera: bool = True, hook_beat: str = "",
                     cliff_beat: str = "") -> str:
    """The episode as a marked-up script page. With context_s set, only the cut and that many seconds around it
    are shown (outside lines dimmed); None shows the whole episode."""
    segs = [s for s in segments if s.episode == ep.number]
    scores = {s.beat_id: s for s in (analysis.beats if analysis else [])}
    inside = lambda b: any(s.start - 0.01 <= b.start < s.end - 0.01 for s in segs)
    lo = min((s.start for s in segs), default=0.0) - (context_s or 0)
    hi = max((s.end for s in segs), default=ep.duration) + (context_s or 0)
    rows, marked_in, marked_out = [], set(), set()
    beats = [b for b in ep.beats if show_camera or b.kind != "camera"]
    for b in beats:
        if context_s is not None and segs and not (lo <= b.start <= hi):
            continue
        in_cut = inside(b)
        gutter, gcls = tc(b.start)[:-1], ""
        for s in segs:
            if in_cut and s.start - 0.01 <= b.start and id(s) not in marked_in:
                gutter, gcls = f"In {short_tc(s.start)}", "in"
                marked_in.add(id(s))
        force = "hook" if b.id == hook_beat else "cliff" if b.id == cliff_beat else ""
        rows.append(_line_html(b, scores.get(b.id), in_cut, gutter, gcls, dim=bool(segs) and not in_cut, force=force))
        for s in segs:
            nxt = next((x for x in beats if x.start > b.start), None)
            if in_cut and id(s) not in marked_out and (nxt is None or nxt.start >= s.end - 0.01) and b.start < s.end:
                rows.append(f'<div class="hc-line in-cut"><span class="hc-gutter out">Out {short_tc(s.end)}</span>'
                            f'<span class="hc-text"></span></div>')
                marked_out.add(id(s))
    return f'<div class="hc-page" aria-label="Episode {ep.number} script">{"".join(rows)}</div>'


# ----------------------------------------------------------------------------------------- promo frame
def _middle_line(promo: Promo, ep_by_num: dict[int, Episode]) -> str:
    seg = promo.segments[len(promo.segments) // 2]
    ep = ep_by_num.get(seg.episode)
    if not ep:
        return ""
    mid = (seg.start + seg.end) / 2
    lines = [b for b in ep.beats if b.kind in ("dialogue", "vo") and seg.start <= b.start <= seg.end]
    if not lines:
        return ""
    b = min(lines, key=lambda b: abs(b.start - mid))
    return f"{(b.speaker or '').upper()}: {b.text}"


def phone_html(promo: Promo, ep_by_num: dict[int, Episode]) -> str:
    mid = _middle_line(promo, ep_by_num)
    return (f'<div class="hc-phone" aria-label="The promo as a vertical clip">'
            f'<div><div class="slot">First 3 seconds</div><div class="line"><span class="hc-mark-hook">{esc(promo.hook_text)}</span></div></div>'
            f'<div class="mid">{esc(mid[:140])}</div>'
            f'<div><div class="slot">Last line before the cut</div><div class="line"><span class="hc-mark-cliff">'
            f'{esc(promo.cliffhanger_text)}</span></div><div class="cut">Cut here</div></div></div>')


def facts_html(promo: Promo, issues: list[dict]) -> str:
    v = promo.cold_viewer
    verdict = (f'<span class="hc-verdict {v.verdict}">{VERDICT_WORDS[v.verdict]}</span>'
               f'<div class="hc-note">They thought: “{esc(v.premise)}”</div>') if v else '<span class="hc-note">Not tested</span>'
    mine = {i.get("check") for i in issues if i.get("candidate_id") == promo.candidate_id}
    rules = [(("duration", "duration_field"), "20–60 seconds long"),
             (("hook_window", "hook_beat", "hook_beat_id"), "The hook lands in the first 3 seconds"),
             (("mid_line_cut", "dialogue"), "No line is cut in half"),
             (("answer_inside", "cliff_at_end", "cliff_beat"), "Ends on the cliffhanger, with the answer after the cut")]
    rule_items = "".join(f'<li class="{"miss" if mine & set(keys) else ""}">{t}</li>' for keys, t in rules)
    why = esc(promo.why_it_works) if promo.why_it_works else "—"
    return (f'<dl class="hc-facts"><dt>Cut</dt><dd style="font-family:Courier Prime,monospace">{esc(cut_label(promo.segments))} '
            f'({promo.duration:.0f} s)</dd><dt>Why it works</dt><dd>{why}</dd><dt>Stranger test</dt><dd>{verdict}</dd>'
            f'<dt>When to post</dt><dd>{esc(WAVE_WORDS.get(promo.post_wave, ""))}</dd>'
            f'<dt>Rules</dt><dd><ul class="hc-rules">{rule_items}</ul></dd></dl>')


# ------------------------------------------------------------------------------------------------ charts
def season_chart(report_rows: list[dict]) -> Optional[alt.Chart]:
    rows = [r for r in report_rows if r.get("opening_hook_score") is not None]
    if not rows:
        return None
    df = pd.DataFrame([{"Episode": r["episode"], "Score": r["opening_hook_score"], "Measure": "Opening hook"} for r in rows] +
                      [{"Episode": r["episode"], "Score": r.get("ending_cliff_score", 0), "Measure": "Ending cliffhanger"} for r in rows])
    color = alt.Color("Measure:N", scale=alt.Scale(domain=["Opening hook", "Ending cliffhanger"], range=[HOOK_DARK, CLIFF_DARK]),
                      legend=alt.Legend(orient="top", title=None))
    base = alt.Chart(df).encode(x=alt.X("Episode:Q", axis=alt.Axis(tickMinStep=1, format="d")),
                                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5]), title="Strength (0–5)"), color=color)
    return (base.mark_line(strokeWidth=1.6) + base.mark_point(filled=True, size=46)).properties(height=240)


def eval_chart(stats: dict) -> alt.Chart:
    names = {"tool": "HookCut's top 20", "last_window": "Last 45 s of each episode", "random": "A random 25–45 s cut"}
    df = pd.DataFrame([{"Set": names[k], "Pass rate": stats[k]["pass_rate"]["mean"] * 100,
                        "lo": stats[k]["pass_rate"]["ci90"][0] * 100, "hi": stats[k]["pass_rate"]["ci90"][1] * 100,
                        "ours": k == "tool"} for k in names if k in stats])
    order = list(names.values())
    y = alt.Y("Set:N", sort=order, title=None, axis=alt.Axis(labelLimit=280, labelOverlap=False, labelFontSize=12))
    bars = alt.Chart(df).mark_bar(size=20).encode(
        y=y, x=alt.X("Pass rate:Q", scale=alt.Scale(domain=[0, 100]), title="Pass the stranger test (%)"),
        color=alt.condition("datum.ours", alt.value(INK), alt.value("#AEB6C4")))
    ci = alt.Chart(df).mark_rule(color=PENCIL, strokeWidth=2).encode(y=y, x="lo:Q", x2="hi:Q")
    return (bars + ci).properties(height=alt.Step(36))


def story_dot(graph, max_people: int = 9) -> str:
    """Main characters and the bonds between them; red edges are feuds (conflict), width = scenes shared."""
    people = sorted(graph.characters, key=lambda c: (-len(c.episodes_present), -c.promo_pull))[:max_people]
    ids = {c.id for c in people}
    lines = ['graph G {', 'graph [layout=neato, overlap=false, splines=true, sep="+12", size="7.5,4.2", ratio=compress, '
             'bgcolor="transparent", pad="0.15", fontname="Archivo"];',
             f'node [shape=box, style="rounded", color="{INK}", fontcolor="{INK}", fontname="Archivo", fontsize=11, penwidth=1.3, margin="0.12,0.06", height=0.3];',
             f'edge [fontname="Archivo", fontsize=9, fontcolor="{GRAPHITE}", len=1.6];']
    for c in people:
        lines.append(f'"{c.id}" [label="{esc(c.name)}"];')
    for e in graph.relationships:
        if e.a in ids and e.b in ids and (e.shared_scenes >= 2 or e.label):
            feud = e.core or e.conflict_intensity >= 50
            color = PENCIL if feud else INK
            width = 1 + min(4.0, e.shared_scenes / 4)
            label = esc(e.label) if e.label else ""
            lines.append(f'"{e.a}" -- "{e.b}" [color="{color}", penwidth={width:.1f}, label="{label}"];')
    lines.append("}")
    return "\n".join(lines)
