"""The reviewer-facing pages run on the saved sample run with no API calls."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.hr import components as C
from app.hr import data as D

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "app" / "hr"


def test_try_page_replays_a_sample_episode_without_models(monkeypatch):
    monkeypatch.setattr(D, "live_models_available", lambda: False)
    at = AppTest.from_file(str(PAGES / "page_try.py"), default_timeout=60).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Find the promo").click().run()
    assert not at.exception and not at.error
    html = "".join(m.value for m in at.markdown)
    assert "hc-steps" in html and "hc-timeline" in html and "hc-page" in html and "hc-phone" in html


def test_pasted_script_without_models_explains_what_is_needed(monkeypatch):
    monkeypatch.setattr(D, "live_models_available", lambda: False)
    at = AppTest.from_file(str(PAGES / "page_try.py"), default_timeout=60).run()
    at.segmented_control[0].set_value("Paste a script").run()
    at.text_area[0].input((ROOT / "samples" / "episode-001-sample.txt").read_text()).run()
    next(b for b in at.button if b.label == "Find the promo").click().run()
    assert any("needs a model key" in e.value for e in at.error)


def test_show_and_how_pages_render():
    for page in ("page_show.py", "page_how.py"):
        at = AppTest.from_file(str(PAGES / page), default_timeout=60).run()
        assert not at.exception, page


def test_episode_slice_and_components_on_the_sample_run():
    r = D.sample_result()
    one = D.episode_slice(r, 1)
    assert [e.number for e in one.episodes] == [1] and one.promos and all(
        s.episode == 1 for p in one.promos for s in p.segments)
    p, ep = one.promos[0], one.episodes[0]
    svg = C.timeline_svg(ep, one.analyses[0], p.segments)
    page = C.script_page_html(ep, one.analyses[0], p.segments, hook_beat=p.hook_beat_id, cliff_beat=p.cliff_beat_id)
    assert svg.startswith("<svg") and ">In<" in svg and ">Out<" in svg
    assert "In 00:" in page and "Out 00:" in page and "hc-text cliff" in page
    assert len(D.step_summaries(one)) == 5
