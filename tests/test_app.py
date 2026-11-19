from pathlib import Path

from streamlit.testing.v1 import AppTest

from hookcut import audience, pipeline
from hookcut.llm import registry
from hookcut.schema import RunResult
from hookcut.validate import validate_promos

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "workbench.py"
DEMO = ROOT / "app" / "fixtures" / "demo_run.json"
TAB_NAMES = ["Promos", "Episode report", "Story genome", "Engagement (simulated)", "Audience studio", "Toolkit (AI Adda)", "Timeline", "All candidates", "Run log & cost", "Downloads"]




def tab(app, label):
    """Find a tab by its label (positions shift when tabs are added)."""
    return next(t for t in app.tabs if t.label == label)

def button(app, label):
    """Find a button by its label (indices shift when buttons are added)."""
    return next(b for b in app.button if b.label == label)

def demo_result() -> RunResult:
    return RunResult.model_validate_json(DEMO.read_bytes())


def test_app_loads():
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.title[0].value == "HookCut"
    assert [control.label for control in app.sidebar.text_input] == ["Analyze model", "Judge model", "Writer model",
                                                                      "Genome model"]
    assert "Open a saved run" in app.radio[0].options


def test_demo_renders_all_tabs(monkeypatch):
    def forbid_pipeline(*args, **kwargs):
        raise AssertionError("Opening a demo must never call the pipeline.")

    monkeypatch.setattr(pipeline, "run_pipeline", forbid_pipeline)
    app = AppTest.from_file(str(APP)).run()
    button(app, "Demo run (no API calls)").click().run()
    assert not app.exception
    assert not app.error
    assert [tab.label for tab in app.tabs] == TAB_NAMES
    (promos, episodes, genome, engagement, studio, toolkit, timeline, candidates, log, downloads) = app.tabs
    assert any("Simulated audience activity" in i.value for i in studio.info)
    assert len(promos.expander) == 5
    assert len(episodes.dataframe[0].value) == 4
    for chart_tab in (episodes, timeline):
        assert chart_tab.get("vega_lite_chart") or chart_tab.get("arrow_vega_lite_chart")
    assert len(candidates.dataframe[0].value) == 8
    assert len(log.dataframe[0].value) == 3
    assert len(downloads.get("download_button")) == 2
    promos.selectbox[0].select(3).run()
    assert not app.exception
    assert len(tab(app, "Promos").expander) == 1


def test_demo_fixture_validates():
    result = demo_result()
    assert len(result.episodes) == 4
    assert all(not episode.warnings for episode in result.episodes)
    assert [promo.rank for promo in result.promos] == [1, 2, 3, 4, 5]
    assert validate_promos(result.promos, result.episodes, {}, result.analyses) == []


def test_partial_run_renders():
    result = demo_result()
    result.verdicts = []
    result.analyses = result.analyses[1:]
    result.episode_report = []
    for promo in result.promos:
        promo.cold_viewer = None
    app = AppTest.from_file(str(APP))
    app.session_state["active_run"] = result, None, "partial"
    app.run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == TAB_NAMES
    assert tab(app, "Timeline").info[0].value == "No beat scores are available for this episode."


def test_audience_studio_changes_the_first_pick_for_each_cohort():
    app = AppTest.from_file(str(APP)).run()
    button(app, "Demo run (no API calls)").click().run()
    tab(app, "Audience studio").selectbox[0].select("hold").run()
    tab(app, "Audience studio").selectbox[1].select("new").run()
    new_top = tab(app, "Audience studio").dataframe[0].value.iloc[0]["Cut"]
    tab(app, "Audience studio").selectbox[1].select("returning").run()
    assert not app.exception
    assert new_top == "C0001" and tab(app, "Audience studio").dataframe[0].value.iloc[0]["Cut"] == "C0003"
    tab(app, "Audience studio").selectbox[2].select("C0006").run()
    assert any("no impressions" in text.value for text in tab(app, "Audience studio").markdown)
    assert any("Character relationships" in text.value for text in tab(app, "Audience studio").markdown)


def test_audience_studio_uses_reported_csv_and_handles_invalid_activity():
    result = demo_result()
    app = AppTest.from_file(str(APP))
    app.session_state["active_run"] = result, None, "uploaded-run"
    app.run()
    tab(app, "Audience studio").radio[0].set_value("Upload activity CSV").run()
    assert len(tab(app, "Audience studio").get("download_button")) == 2
    file = tab(app, "Audience studio").file_uploader[0]
    file.upload("bad.csv", b"clip_key\nunknown\n", "text/csv").run()
    assert not app.exception
    assert "Missing CSV columns" in tab(app, "Audience studio").error[0].value
    keys = audience.clip_keys(result)
    activity = audience.events_csv([audience.Exposure("i1", "v1", "subscribers", keys["C0001"], 4.0, False)])
    tab(app, "Audience studio").file_uploader[0].set_value(("events.csv", activity.encode(), "text/csv")).run()
    assert not app.exception
    assert not tab(app, "Audience studio").error
    assert any("uploaded CSV" in text.value for text in tab(app, "Audience studio").caption)
    frame = tab(app, "Audience studio").dataframe[0].value
    assert frame.loc[frame["Cut"] == "C0001", "Impressions"].iloc[0] == 1


def test_sample_run_uses_sidebar_settings(monkeypatch):
    calls = []

    def fake_pipeline(*, inputs, cfg, out_dir, progress, use_cache, show):
        calls.append((inputs, cfg, out_dir, use_cache, show))
        progress("Sample progress", 0.5)
        return demo_result()

    monkeypatch.setattr(pipeline, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(registry, "load_secrets", lambda: {})
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value("Use the synthetic sample show")
    app.sidebar.selectbox[0].select("Fast (Groq only)").run()
    app.sidebar.number_input[0].set_value(5)
    app.sidebar.number_input[1].set_value(8)
    app.sidebar.toggle[0].set_value(False)
    button(app, "Find promos").click().run()

    assert not app.exception
    assert not app.error
    [(inputs, cfg, out_dir, use_cache, show)] = calls
    assert len(inputs) == 4
    assert inputs[0][0] == "episode-001-sample.txt"
    assert inputs[0][1].startswith("EPISODE NUMBER: Episode 1")
    assert cfg["models"] == {"analyze": "groq:openai/gpt-oss-120b", "judge": "groq:qwen/qwen3.8-27b",
                             "writer": "groq:openai/gpt-oss-120b", "genome": "groq:qwen/qwen3.8-27b"}
    assert all(not cfg["fallbacks"].get(stage) for stage in cfg["models"])
    assert cfg["promo"]["final_top_n"] == 5
    assert cfg["promo"]["judge_top_k"] == 8
    assert out_dir.parent == ROOT / "outputs" / "app_runs"
    assert use_cache is False
    assert show == "The Marlowe After Midnight"


def test_run_reports_missing_key(monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("no usable model for stage 'analyze' (tried ['groq:openai/gpt-oss-120b'])")

    monkeypatch.setattr(pipeline, "run_pipeline", unavailable)
    monkeypatch.setattr(registry, "load_secrets", lambda: {})
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value("Use the synthetic sample show")
    app.sidebar.selectbox[0].select("Fast (Groq only)").run()
    button(app, "Find promos").click().run()
    assert not app.exception
    assert "Set GROQ_API_KEY" in app.error[0].value


def test_real_sample_run_fixture_opens_without_api_calls():
    from hookcut.schema import RunResult
    fixture = ROOT / "app" / "fixtures" / "sample_run.json"
    result = RunResult.model_validate_json(fixture.read_text())
    assert result.config.get("fixture") == "real-run" and result.promos and result.analyses
