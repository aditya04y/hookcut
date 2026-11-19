"""HookCut's local and Streamlit Community Cloud entry point."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.views import (
    show_candidates, show_downloads, show_episode_report, show_promos, show_run_log, show_timeline,
)
from app.audience_studio import show_audience_studio
from app.engagement_view import show_engagement
from app.genome_view import show_genome
from app.toolkit_view import show_toolkit
from hookcut import pipeline
from hookcut.config import STAGES, load_config
from hookcut.llm.registry import PROVIDERS, load_secrets
from hookcut.schema import RunResult

OUTPUTS = ROOT / "outputs"
DEMO = ROOT / "app" / "fixtures" / "demo_run.json"
SAMPLE_RUN = ROOT / "app" / "fixtures" / "sample_run.json"   # a real pipeline run on the synthetic sample show
SAMPLE_SHOW = "The Marlowe After Midnight"
SECRET_NAMES = (
    "GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    "OPENROUTER_API_KEY", "HOOKCUT_BASE_URL", "HOOKCUT_API_KEY",
)
FREE = "Free tier (Gemini + Groq)"
PRESETS = {
    FREE: None,
    "Fast (Groq only)": ("groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b",
                         "groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b"),
    "Claude": ("anthropic:claude-opus-5-5",) * len(STAGES),
    "Custom gateway": ("custom:<model>",) * len(STAGES),
}


def copy_cloud_secrets() -> None:
    try:
        for name in SECRET_NAMES:
            if value := st.secrets.get(name):
                os.environ[name] = str(value)
    except Exception:
        # Local Streamlit runs may have no secrets file.
        pass


def redact(text: str, secrets: dict | None = None) -> str:
    values = {os.environ.get(name) for name in SECRET_NAMES}
    if secrets:
        values.update(secrets.get(name) for name in SECRET_NAMES)
    for value in sorted(filter(None, values), key=len, reverse=True):
        text = text.replace(value, "[redacted]")
    return text


def apply_preset(default_models: dict) -> None:
    specs = PRESETS[st.session_state.model_preset]
    for index, stage in enumerate(STAGES):
        if stage in default_models:
            st.session_state[f"model_{stage}"] = specs[index] if specs else default_models[stage]


def sidebar(defaults: dict) -> tuple[dict, dict, bool]:
    with st.sidebar:
        st.header("Run settings")
        st.selectbox("Model preset", PRESETS.keys(), key="model_preset", on_change=apply_preset,
                     args=(defaults["models"],))
        models = {}
        for stage in (s for s in STAGES if s in defaults["models"]):
            key = f"model_{stage}"
            st.session_state.setdefault(key, defaults["models"][stage])
            models[stage] = st.text_input(f"{stage.title()} model", key=key, placeholder="provider:model").strip()
        if any(spec.startswith("custom:") for spec in models.values()):
            st.caption("Replace <model> with your gateway's model name. Set HOOKCUT_BASE_URL and, for authenticated gateways, HOOKCUT_API_KEY.")
        promo = defaults["promo"].copy()
        promo["final_top_n"] = st.number_input("Final promos", min_value=1, value=promo["final_top_n"], step=1)
        promo["judge_top_k"] = st.number_input("Candidates to judge", min_value=1, value=promo["judge_top_k"], step=1)
        cache = st.toggle("Use cached model responses", value=True)
    return models, promo, cache


def run_error(exc: Exception, cfg: dict, secrets: dict) -> str:
    message = str(exc)
    for stage in STAGES:
        if f"no usable model for stage {stage!r}" not in message:
            continue
        specs = [cfg["models"][stage], *cfg.get("fallbacks", {}).get(stage, [])]
        key_names = {PROVIDERS.get(spec.split(":", 1)[0], {}).get("key_env") for spec in specs}
        if key_names and None not in key_names and all(not secrets.get(key) for key in key_names):
            names = " or ".join(sorted(key_names))
            return f"{stage.title()} could not start. Set {names} in your environment or Streamlit secrets."
        break
    return redact(message, secrets)[:3000] or type(exc).__name__


def run(inputs: list[tuple[str, str]], models: dict, promo: dict, cache: bool, default_models: dict,
        show: str | None = None) -> None:
    copy_cloud_secrets()
    cfg, secrets = {}, {}
    bar = st.progress(0.0)
    status = st.empty()
    status.caption("Starting the run…")
    try:
        secrets = load_secrets()
        for stage, spec in models.items():
            provider, _, name = spec.partition(":")
            if provider not in PROVIDERS or not name or name == "<model>":
                raise ValueError(f"Enter a provider:model for {stage}. Available providers are {', '.join(PROVIDERS)}.")
            if provider == "custom" and not secrets.get("HOOKCUT_BASE_URL"):
                raise ValueError("Set HOOKCUT_BASE_URL in your environment or Streamlit secrets for the custom gateway.")
        # Keep the configured fallbacks for untouched defaults. Explicit picks use only that model.
        overrides = {stage: spec for stage, spec in models.items()
                     if st.session_state.model_preset != FREE or spec != default_models[stage]
                     or os.environ.get(f"HOOKCUT_{stage.upper()}_MODEL")}
        cfg = load_config(overrides={**overrides, "promo": promo})
        out_dir = OUTPUTS / "app_runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")

        def progress(message: str, fraction: float) -> None:
            bar.progress(max(0.0, min(1.0, float(fraction))))
            status.caption(redact(message, secrets))

        result = pipeline.run_pipeline(inputs=inputs, cfg=cfg, out_dir=out_dir, progress=progress, use_cache=cache,
                                       show=show)
        st.session_state.active_run = result, out_dir, out_dir.name
        progress("Run complete.", 1.0)
    except Exception as exc:
        status.empty()
        bar.empty()
        st.error(run_error(exc, cfg, secrets))


def open_saved(data: bytes, folder: Path | None, source: str) -> None:
    try:
        result = RunResult.model_validate_json(data)
    except ValidationError as exc:
        st.error(f"This file is not a valid HookCut result.json ({exc.error_count()} validation error(s)).")
        return
    st.session_state.active_run = result, folder, source


def load_layer(folder: Path | None, name: str, kind: str):
    """A saved layer output next to result.json (story_graph.json / engagement.json), or None to build it live."""
    if folder is None or not (Path(folder) / name).exists():
        return None
    path = Path(folder) / name
    if kind == "graph":
        from hookcut.story_graph import load_story_graph
        return load_story_graph(path)
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def render_layer(label: str, render) -> None:
    """Layers read a finished run; a failure in one tab must never take down the promo views."""
    try:
        render()
    except Exception as exc:  # noqa: BLE001 - shown to the editor, logged nowhere else
        st.warning(f"The {label} view could not render for this run: {redact(str(exc))[:300]}")


def choose_saved_run() -> None:
    runs = sorted(
        [*OUTPUTS.glob("*/result.json"), *OUTPUTS.glob("app_runs/*/result.json")],
        reverse=True,
    )
    saved = st.selectbox("Saved run", runs, index=None,
                         format_func=lambda path: str(path.relative_to(OUTPUTS).parent),
                         placeholder="Choose a local run")
    upload = st.file_uploader("Or upload a result.json", type=["json"])
    if st.button("Open run", disabled=saved is None and upload is None):
        try:
            if upload is not None:
                open_saved(upload.getvalue(), None, Path(upload.name).stem)
            elif saved is not None:
                open_saved(saved.read_bytes(), saved.parent, saved.parent.name)
        except OSError as exc:
            st.error(f"Unable to open the saved run. {redact(str(exc))}")


def main() -> None:
    try:  # the multipage entrypoint already set it; standalone runs set it here
        st.set_page_config(page_title="HookCut workbench", layout="wide")
    except Exception:
        pass
    st.title("HookCut")
    st.caption("Workbench: every tab, for editors and reviewers who want the detail.")
    st.write("Find promo cuts in a screenplay and choose which opening to test with your audience.")
    with st.expander("How it works"):
        st.write("Code parses the script. An LLM scores each episode’s beats. Code builds 20–60 second clips. "
                 "A different model tests whether a new viewer can follow them. Code ranks clips for variety, "
                 "and an LLM writes a line about why each works. Audience studio connects these cuts to "
                 "viewer activity and the story beats they share.")

    defaults = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    models, promo, cache = sidebar(defaults)
    left, right = st.columns(2)
    if left.button("Sample show: real HookCut run (no API calls)", type="primary"):
        try:
            open_saved(SAMPLE_RUN.read_bytes(), None, "marlowe-real-run")
        except OSError as exc:
            st.error(f"Unable to open the sample run. {redact(str(exc))}")
    if right.button("Demo run (no API calls)"):
        try:
            open_saved(DEMO.read_bytes(), None, "marlowe-demo")
        except OSError as exc:
            st.error(f"Unable to open the demo run. {redact(str(exc))}")
    mode = st.radio("Input", ("Upload episodes", "Use the synthetic sample show", "Open a saved run"),
                    horizontal=True)
    if mode == "Open a saved run":
        choose_saved_run()
    else:
        sample = mode == "Use the synthetic sample show"
        if sample:
            scripts = sorted((ROOT / "samples").glob("*.txt"))
            st.caption(f"{SAMPLE_SHOW} · {len(scripts)} episodes")
        else:
            scripts = st.file_uploader("Episode scripts (.txt)", type=["txt"], accept_multiple_files=True)
        if st.button("Find promos", disabled=not scripts):
            try:
                inputs = [(script.name, (script.read_bytes() if sample else script.getvalue())
                           .decode("utf-8-sig", errors="replace")) for script in scripts]
                run(inputs, models, promo, cache, defaults["models"], show=SAMPLE_SHOW if sample else None)
            except OSError as exc:
                st.error(f"Unable to read the episode scripts. {redact(str(exc))}")

    if "active_run" not in st.session_state:
        return
    result, folder, source = st.session_state.active_run
    st.subheader(result.show)
    st.caption(f"{len(result.episodes)} episodes · {len(result.promos)} ranked promos · {len(result.candidates)} candidates")
    if result.config.get("fixture") == "hand-authored":
        st.caption("Hand-authored demo. Call counts and costs are illustrative.")
    elif result.config.get("fixture") == "real-run":
        st.caption(result.config.get("note", "Real HookCut run, loaded from cache."))
    (promos, report, genome, engagement, studio, toolkit, timeline, candidates, log, downloads) = st.tabs([
        "Promos", "Episode report", "Story genome", "Engagement (simulated)", "Audience studio", "Toolkit (AI Adda)",
        "Timeline", "All candidates", "Run log & cost", "Downloads",
    ])
    with promos:
        show_promos(result)
    with report:
        show_episode_report(result)
    with genome:
        render_layer("story genome", lambda: show_genome(result, load_layer(folder, "story_graph.json", "graph")))
    with engagement:
        render_layer("engagement", lambda: show_engagement(result, load_layer(folder, "engagement.json", "json")))
    with studio:
        show_audience_studio(result)
    with toolkit:
        render_layer("toolkit", lambda: show_toolkit(result, run_dir=folder))
    with timeline:
        show_timeline(result)
    with candidates:
        show_candidates(result)
    with log:
        show_run_log(result, defaults.get("prices_usd_per_mtok", {}), redact)
    with downloads:
        show_downloads(result, folder, source)


main()  # runs as a page of the multipage app and as a standalone script
