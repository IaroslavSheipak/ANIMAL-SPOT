#!/usr/bin/env python3
"""Streamlit demo UI for browsing ANIMAL-SPOT prediction results."""

from __future__ import annotations

import base64
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audio_timeline_component import audio_timeline

GROUND_TRUTH_SUFFIX = "_predict_output.log.annotation.result.txt"
MODEL_PREDICTION_SUFFIX = "_predictions.txt"
AUDIO_SUFFIXES = (".wav", ".WAV")

RAVEN_COLUMNS = [
    "Selection",
    "View",
    "Channel",
    "Begin time (s)",
    "End time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Sound type",
    "Comments",
    "Confidence",
]

DEFAULT_CLASS_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


def extract_expected_labels(recording_id: str) -> list[str]:
    """Extract expected class tags (Kxx) from a file stem."""
    labels = {token.upper() for token in re.findall(r"k\d+", recording_id, flags=re.IGNORECASE)}
    return sorted(labels)


def audio_duration_seconds(audio_path: Path | None) -> float | None:
    """Read WAV duration in seconds."""
    if audio_path is None or not audio_path.exists():
        return None
    try:
        with wave.open(str(audio_path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except (wave.Error, OSError):
        return None


def parse_annotation_file(path: Path) -> pd.DataFrame:
    """Parse one Raven annotation table into a normalized dataframe."""
    table = pd.read_csv(path, sep="\t", dtype=str, engine="python")
    table.columns = [col.strip() for col in table.columns]

    for column in RAVEN_COLUMNS:
        if column not in table.columns:
            table[column] = ""

    table = table[RAVEN_COLUMNS].copy()
    table["Sound type"] = table["Sound type"].fillna("").astype(str).str.strip()
    table = table[table["Sound type"] != ""].copy()

    numeric_columns = [
        "Selection",
        "Begin time (s)",
        "End time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Confidence",
    ]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=["Begin time (s)", "End time (s)"]).copy()

    table["duration_s"] = (table["End time (s)"] - table["Begin time (s)"]).clip(lower=0)
    table["label"] = table["Sound type"].astype(str)
    table["event_idx"] = np.arange(1, len(table) + 1)
    return table.rename(
        columns={
            "Selection": "selection",
            "Begin time (s)": "begin_s",
            "End time (s)": "end_s",
            "Low Freq (Hz)": "low_freq_hz",
            "High Freq (Hz)": "high_freq_hz",
            "Confidence": "confidence",
        }
    )


def pair_audio_with_prediction(prediction_path: Path) -> Path | None:
    """Find matching WAV for a prediction result file."""
    recording_id = prediction_path.name[: -len(GROUND_TRUTH_SUFFIX)]
    for suffix in AUDIO_SUFFIXES:
        candidate = prediction_path.parent / f"{recording_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_dataset_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata and all events from one folder."""
    prediction_files = sorted(data_dir.glob(f"*{GROUND_TRUTH_SUFFIX}"))
    metadata_rows: list[dict[str, Any]] = []
    event_tables: list[pd.DataFrame] = []

    for prediction_path in prediction_files:
        recording_id = prediction_path.name[: -len(GROUND_TRUTH_SUFFIX)]
        audio_path = pair_audio_with_prediction(prediction_path)
        expected_labels = extract_expected_labels(recording_id)

        events = parse_annotation_file(prediction_path)
        events["recording_id"] = recording_id
        events["prediction_file"] = prediction_path.name
        events["audio_file"] = audio_path.name if audio_path is not None else ""
        events["expected_labels"] = ",".join(expected_labels)
        event_tables.append(events)

        detected_labels = sorted(events["label"].astype(str).unique().tolist())
        metadata_rows.append(
            {
                "recording_id": recording_id,
                "prediction_file": prediction_path.name,
                "prediction_path": str(prediction_path),
                "audio_file": audio_path.name if audio_path is not None else "",
                "audio_path": str(audio_path) if audio_path is not None else "",
                "audio_duration_s": audio_duration_seconds(audio_path),
                "event_count": int(len(events)),
                "detected_duration_s": float(events["duration_s"].sum()),
                "expected_labels": ", ".join(expected_labels) if expected_labels else "-",
                "detected_labels": ", ".join(detected_labels) if detected_labels else "-",
            }
        )

    metadata_df = pd.DataFrame(metadata_rows)
    events_df = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    return metadata_df, events_df


def recording_id_from_file(file_name: str, suffix: str) -> str:
    """Strip a known suffix from a filename."""
    return file_name[: -len(suffix)] if file_name.endswith(suffix) else Path(file_name).stem


def load_source_tables(data_dir: Path, suffix: str, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load events from one source folder and suffix pattern."""
    files = sorted(data_dir.glob(f"*{suffix}")) if data_dir.exists() else []
    metadata_rows: list[dict[str, Any]] = []
    event_tables: list[pd.DataFrame] = []

    for source_path in files:
        recording_id = recording_id_from_file(source_path.name, suffix)
        events = parse_annotation_file(source_path)
        events["recording_id"] = recording_id
        events["source"] = source
        events["source_file"] = source_path.name
        event_tables.append(events)

        labels = sorted(events["label"].astype(str).unique().tolist())
        metadata_rows.append(
            {
                "recording_id": recording_id,
                "source": source,
                "source_file": source_path.name,
                "source_path": str(source_path),
                "event_count": int(len(events)),
                "detected_duration_s": float(events["duration_s"].sum()),
                "labels": ", ".join(labels) if labels else "-",
            }
        )

    metadata_df = pd.DataFrame(metadata_rows)
    events_df = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    return metadata_df, events_df


def build_audio_catalog(
    gt_metadata: pd.DataFrame,
    pred_metadata: pd.DataFrame,
    extra_audio_dirs: list[Path] | None = None,
) -> pd.DataFrame:
    """Create unified index of audio files with optional GT and prediction tables."""
    candidate_dirs: set[Path] = set()
    for metadata in (gt_metadata, pred_metadata):
        if metadata.empty:
            continue
        for source_path in metadata["source_path"].astype(str).tolist():
            path_obj = Path(source_path)
            if path_obj.exists():
                candidate_dirs.add(path_obj.parent)
    if extra_audio_dirs:
        for directory in extra_audio_dirs:
            if directory.exists() and directory.is_dir():
                candidate_dirs.add(directory)

    audio_files: list[Path] = []
    for directory in sorted(candidate_dirs):
        for suffix in AUDIO_SUFFIXES:
            audio_files.extend(sorted(directory.glob(f"*{suffix}")))

    audio_by_key: dict[str, Path] = {path.stem.lower(): path for path in audio_files}
    gt_by_key: dict[str, str] = {}
    pred_by_key: dict[str, str] = {}
    gt_id_by_key: dict[str, str] = {}
    pred_id_by_key: dict[str, str] = {}

    if not gt_metadata.empty:
        for row in gt_metadata.itertuples(index=False):
            key = str(row.recording_id).lower()
            gt_by_key[key] = str(row.source_path)
            gt_id_by_key[key] = str(row.recording_id)

    if not pred_metadata.empty:
        for row in pred_metadata.itertuples(index=False):
            key = str(row.recording_id).lower()
            pred_by_key[key] = str(row.source_path)
            pred_id_by_key[key] = str(row.recording_id)

    keys = sorted(set(audio_by_key) | set(gt_by_key) | set(pred_by_key))
    rows: list[dict[str, Any]] = []
    for key in keys:
        audio_path = audio_by_key.get(key)
        recording_id = (
            audio_path.stem
            if audio_path is not None
            else gt_id_by_key.get(key, pred_id_by_key.get(key, key))
        )
        rows.append(
            {
                "recording_id": recording_id,
                "audio_path": str(audio_path) if audio_path is not None else "",
                "audio_file": audio_path.name if audio_path is not None else "",
                "audio_duration_s": audio_duration_seconds(audio_path),
                "has_ground_truth": key in gt_by_key,
                "has_predictions": key in pred_by_key,
                "ground_truth_path": gt_by_key.get(key, ""),
                "prediction_path": pred_by_key.get(key, ""),
            }
        )

    return pd.DataFrame(rows).sort_values("recording_id").reset_index(drop=True)


def load_audio_base64(audio_path: Path) -> str:
    """Read audio as base64 string for the timeline component."""
    data = audio_path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def compute_audio_peaks(audio_path: Path, target_peaks: int = 2000) -> list[float]:
    """Compute waveform peaks for WaveSurfer (abs max per chunk)."""
    with wave.open(str(audio_path), "rb") as handle:
        n_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)

    dtype_by_width = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_by_width:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    signal = np.frombuffer(raw, dtype=dtype_by_width[sample_width])
    if n_channels > 1:
        signal = signal.reshape(-1, n_channels).mean(axis=1)

    if sample_width == 1:
        signal = (signal.astype(np.float32) - 128.0) / 128.0
    else:
        signal = signal.astype(np.float32) / float(2 ** (8 * sample_width - 1))

    total_samples = len(signal)
    if total_samples == 0:
        return []
    target_peaks = max(100, min(target_peaks, total_samples))
    chunk = max(1, total_samples // target_peaks)
    peaks: list[float] = []
    for idx in range(0, total_samples, chunk):
        window = signal[idx : idx + chunk]
        if window.size == 0:
            continue
        peaks.append(float(np.max(np.abs(window))))

    duration_s = total_samples / float(sample_rate)
    if duration_s and len(peaks) < 200:
        # Ensure minimum resolution for short files.
        stride = max(1, total_samples // 500)
        peaks = [float(np.max(np.abs(signal[i : i + stride]))) for i in range(0, total_samples, stride)]
    return peaks


def rgba_to_css(color: Any) -> str:
    """Convert matplotlib RGBA tuple to CSS color."""
    try:
        r, g, b, a = color
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a:.2f})"
    except Exception:
        return "#6b7280"


def aggregate_class_stats(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute count and duration by class."""
    if events_df.empty:
        return pd.DataFrame(columns=["label", "detections", "detected_seconds"])
    grouped = (
        events_df.groupby("label", as_index=False)
        .agg(
            detections=("label", "size"),
            detected_seconds=("duration_s", "sum"),
        )
        .sort_values(["detections", "detected_seconds"], ascending=False)
        .reset_index(drop=True)
    )
    grouped["detected_seconds"] = grouped["detected_seconds"].round(3)
    return grouped


def detection_density(events_df: pd.DataFrame, bin_size: float = 10.0) -> pd.DataFrame:
    """Number of detections per fixed time bin."""
    if events_df.empty:
        return pd.DataFrame(columns=["time_s", "detections"]).set_index("time_s")
    max_time = float(events_df["end_s"].max())
    bins = np.arange(0.0, max_time + bin_size, bin_size)
    if len(bins) < 2:
        bins = np.array([0.0, bin_size], dtype=float)
    counts, edges = np.histogram(events_df["begin_s"].values, bins=bins)
    return pd.DataFrame({"time_s": edges[:-1], "detections": counts}).set_index("time_s")


def build_timeline_figure(events_df: pd.DataFrame):
    """Create a class-vs-time segment timeline."""
    import matplotlib.pyplot as plt

    labels = sorted(events_df["label"].unique().tolist())
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(i % 20) for i, label in enumerate(labels)}

    fig_height = max(2.6, 0.55 * len(labels) + 1.3)
    fig, ax = plt.subplots(figsize=(12.0, fig_height))

    for idx, label in enumerate(labels):
        class_events = events_df[events_df["label"] == label]
        bars = [
            (float(row.begin_s), float(row.duration_s))
            for row in class_events.itertuples(index=False)
            if float(row.duration_s) > 0
        ]
        if bars:
            ax.broken_barh(bars, (idx - 0.35, 0.7), facecolors=colors[label], alpha=0.9)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Class")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_title("Detection Timeline (non-overlapping view)")
    fig.tight_layout()
    return fig


def render_styles(st_module) -> None:
    """Inject visual styling."""
    st_module.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg-a: #f2f7f6;
  --bg-b: #fff6ea;
  --ink: #102a43;
  --teal: #0f766e;
  --amber: #d97706;
}

.stApp {
  background:
    radial-gradient(circle at 15% 10%, rgba(15,118,110,0.13), transparent 45%),
    radial-gradient(circle at 88% 2%, rgba(217,119,6,0.14), transparent 37%),
    linear-gradient(160deg, var(--bg-a), var(--bg-b));
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
}

/* Keep Streamlit / Material icon ligatures from being converted to plain text. */
.material-icons,
.material-icons-round,
.material-icons-outlined,
.material-symbols-rounded,
.material-symbols-outlined,
[class*="material-icons"],
[class*="material-symbols"] {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
}

.mono {
  font-family: "IBM Plex Mono", monospace;
}

.hero-panel {
  padding: 1.1rem 1.25rem;
  border-radius: 14px;
  background: linear-gradient(130deg, rgba(15,118,110,0.16), rgba(217,119,6,0.14));
  border: 1px solid rgba(16,42,67,0.14);
  margin-bottom: 0.9rem;
  animation: slide-fade .55s ease-out;
}

[data-testid="stMetric"] {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(16,42,67,0.1);
  border-radius: 12px;
  padding: 0.35rem 0.6rem;
  animation: metric-fade .7s ease-out;
}

@keyframes slide-fade {
  from { transform: translateY(10px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes metric-fade {
  from { transform: translateY(6px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def format_seconds(seconds: float | None) -> str:
    """Format seconds for display."""
    if seconds is None or pd.isna(seconds):
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - (minutes * 60)
    return f"{minutes}m {rem:04.1f}s"


def run_app() -> None:
    """Entrypoint for Streamlit."""
    try:
        import streamlit as st
    except ModuleNotFoundError:
        raise SystemExit("Missing dependency: streamlit. Install with `pip install streamlit`.")

    st.set_page_config(
        page_title="ANIMAL-SPOT IJCAI 2026 Demo",
        page_icon=":material/graphic_eq:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_styles(st)

    app_dir = Path(__file__).resolve().parent
    default_gt_dir = app_dir / "new_testing_files_2016"
    default_pred_dir = app_dir / "predictions"

    @st.cache_data(show_spinner=False)
    def cached_load_source(folder: str, suffix: str, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        return load_source_tables(Path(folder), suffix, source)

    @st.cache_data(show_spinner=False)
    def cached_audio_base64(path: str) -> str:
        return load_audio_base64(Path(path))

    @st.cache_data(show_spinner=False)
    def cached_audio_peaks(path: str, target_peaks: int) -> list[float]:
        return compute_audio_peaks(Path(path), target_peaks=target_peaks)

    st.markdown(
        """
<div class="hero-panel">
  <h1>Real-Time Killer Whale Call Type Identification</h1>
  <p>Interactive localhost demo for IJCAI 2026 Demo Track: compare model predictions
  against annotated ground truth on the waveform, filter by class, and review class summaries.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Data")
    gt_dir_input = st.sidebar.text_input("Ground Truth folder", str(default_gt_dir))
    pred_dir_input = st.sidebar.text_input("Prediction folder", str(default_pred_dir))
    gt_load_clicked = st.sidebar.button("Load GT")
    refresh_cache = st.sidebar.button("Reload From Disk")
    if refresh_cache:
        cached_load_source.clear()
        cached_audio_base64.clear()
        cached_audio_peaks.clear()

    gt_dir = Path(gt_dir_input)
    pred_dir = Path(pred_dir_input)
    gt_load_state_key = "gt_load_enabled"
    if gt_load_state_key not in st.session_state:
        st.session_state[gt_load_state_key] = False
    if gt_load_clicked:
        st.session_state[gt_load_state_key] = True

    gt_load_enabled = bool(st.session_state.get(gt_load_state_key, False))
    if not gt_load_enabled:
        st.sidebar.caption("Ground Truth is not loaded. Click `Load GT` to include GT events.")

    if not (gt_dir.exists() and gt_dir.is_dir()) and not (pred_dir.exists() and pred_dir.is_dir()):
        st.error(
            "Neither Ground Truth nor Prediction folder exists. "
            "Set at least one valid folder in the sidebar."
        )
        return

    gt_metadata, gt_events = (
        cached_load_source(str(gt_dir), GROUND_TRUTH_SUFFIX, "ground_truth")
        if gt_load_enabled and gt_dir.exists() and gt_dir.is_dir()
        else (pd.DataFrame(), pd.DataFrame())
    )
    pred_metadata, pred_events = (
        cached_load_source(str(pred_dir), MODEL_PREDICTION_SUFFIX, "prediction")
        if pred_dir.exists() and pred_dir.is_dir()
        else (pd.DataFrame(), pd.DataFrame())
    )

    catalog_df = build_audio_catalog(gt_metadata, pred_metadata, extra_audio_dirs=[gt_dir, pred_dir])
    if catalog_df.empty:
        st.warning(
            "No audio/prediction data found for the selected folders. "
            "Check your audio, ground truth, and prediction paths in the sidebar."
        )
        return

    source_events = []
    if not gt_events.empty:
        source_events.append(gt_events.copy())
    if not pred_events.empty:
        source_events.append(pred_events.copy())
    all_events_df = pd.concat(source_events, ignore_index=True) if source_events else pd.DataFrame()

    total_audio_s = float(pd.to_numeric(catalog_df["audio_duration_s"], errors="coerce").dropna().sum())
    gt_total = int(gt_events.shape[0]) if not gt_events.empty else 0
    pred_total = int(pred_events.shape[0]) if not pred_events.empty else 0
    all_non_noise = (
        all_events_df[all_events_df["label"].str.lower() != "noise"] if not all_events_df.empty else pd.DataFrame()
    )
    class_count = int(all_non_noise["label"].nunique()) if not all_non_noise.empty else 0

    metric_cols = st.columns(6)
    metric_cols[0].metric("Recordings", int(catalog_df.shape[0]))
    metric_cols[1].metric("Ground Truth Events", gt_total)
    metric_cols[2].metric("Prediction Events", pred_total)
    metric_cols[3].metric("Classes (non-noise)", class_count)
    metric_cols[4].metric("Audio Duration", format_seconds(total_audio_s))
    metric_cols[5].metric("GT / Pred Files", f"{int(gt_metadata.shape[0])} / {int(pred_metadata.shape[0])}")

    tabs = st.tabs(["Audio Page", "Overview", "All Events"])

    with tabs[0]:
        st.subheader("Audio + Ground Truth + Predictions")

        recording_ids = catalog_df["recording_id"].tolist()
        selected_recording = st.selectbox("Select audio file", recording_ids, index=0)
        selected_row = catalog_df[catalog_df["recording_id"] == selected_recording].iloc[0]
        empty_events = pd.DataFrame(
            columns=[
                "recording_id",
                "source",
                "source_file",
                "event_idx",
                "begin_s",
                "end_s",
                "duration_s",
                "label",
                "low_freq_hz",
                "high_freq_hz",
            ]
        )

        gt_record_events = (
            gt_events[gt_events["recording_id"].str.lower() == selected_recording.lower()].copy()
            if not gt_events.empty
            else empty_events.copy()
        )
        pred_record_events = (
            pred_events[pred_events["recording_id"].str.lower() == selected_recording.lower()].copy()
            if not pred_events.empty
            else empty_events.copy()
        )

        toggle_cols = st.columns((1, 1))
        with toggle_cols[0]:
            show_ground_truth = st.checkbox(
                "Show Ground Truth",
                value=False,
                disabled=gt_record_events.empty,
            )
        with toggle_cols[1]:
            show_predictions = st.checkbox(
                "Show Predictions",
                value=not pred_record_events.empty,
                disabled=pred_record_events.empty,
            )
        peak_bins = 5000

        class_values = sorted(
            set(gt_record_events["label"].astype(str).tolist()) | set(pred_record_events["label"].astype(str).tolist())
        )
        selected_classes: list[str] = []
        if class_values:
            st.markdown("**Classes (checkbox per class)**")
            action_cols = st.columns((1, 1, 1))
            class_key_map = {
                class_name: f"class_visible_{selected_recording}_{class_name}" for class_name in class_values
            }
            if action_cols[0].button("Select all", key=f"select_all_{selected_recording}"):
                for key in class_key_map.values():
                    st.session_state[key] = True
            if action_cols[1].button("Select none", key=f"select_none_{selected_recording}"):
                for key in class_key_map.values():
                    st.session_state[key] = False
            if action_cols[2].button("Hide noise", key=f"hide_noise_{selected_recording}"):
                for class_name, key in class_key_map.items():
                    st.session_state[key] = class_name.lower() != "noise"

            class_cols = st.columns(6)
            for idx, class_name in enumerate(class_values):
                default_enabled = class_name.lower() != "noise"
                checked = class_cols[idx % 6].checkbox(
                    class_name,
                    value=default_enabled,
                    key=class_key_map[class_name],
                )
                if checked:
                    selected_classes.append(class_name)

        visible_gt = (
            gt_record_events[gt_record_events["label"].isin(selected_classes)].copy()
            if show_ground_truth and selected_classes
            else empty_events.copy()
        )
        visible_pred = (
            pred_record_events[pred_record_events["label"].isin(selected_classes)].copy()
            if show_predictions and selected_classes
            else empty_events.copy()
        )

        summary_cols = st.columns(4)
        summary_cols[0].metric("Visible GT Events", int(visible_gt.shape[0]))
        summary_cols[1].metric("Visible Pred Events", int(visible_pred.shape[0]))
        summary_cols[2].metric("Visible Classes", len(selected_classes))
        summary_cols[3].metric("Audio Duration", format_seconds(selected_row["audio_duration_s"]))

        class_palette = {
            name: DEFAULT_CLASS_COLORS[i % len(DEFAULT_CLASS_COLORS)] for i, name in enumerate(selected_classes)
        }

        audio_path = str(selected_row["audio_path"])
        if audio_path:
            st.caption(f"Audio file: `{selected_row['audio_file']}`")
            try:
                audio_base64 = cached_audio_base64(audio_path)
                peaks = cached_audio_peaks(audio_path, peak_bins)
                duration_s = (
                    float(selected_row["audio_duration_s"]) if pd.notna(selected_row["audio_duration_s"]) else 0.0
                )

                gt_payload = [
                    {
                        "id": f"gt-{row.event_idx}",
                        "begin_s": float(row.begin_s),
                        "end_s": float(row.end_s),
                        "label": str(row.label),
                        "source": "Ground Truth",
                        "confidence": float(row.confidence) if not pd.isna(row.confidence) else None,
                    }
                    for row in visible_gt.itertuples(index=False)
                ]
                pred_payload = [
                    {
                        "id": f"pred-{row.event_idx}",
                        "begin_s": float(row.begin_s),
                        "end_s": float(row.end_s),
                        "label": str(row.label),
                        "source": "Predictions",
                        "confidence": float(row.confidence) if not pd.isna(row.confidence) else None,
                    }
                    for row in visible_pred.itertuples(index=False)
                ]

                lanes = []
                if show_ground_truth:
                    lanes.append({"id": "ground_truth", "label": "Ground Truth", "events": gt_payload})
                if show_predictions:
                    lanes.append({"id": "prediction", "label": "Predictions", "events": pred_payload})

                audio_timeline(
                    audio_base64=audio_base64,
                    audio_mime="audio/wav",
                    duration_s=duration_s,
                    peaks=peaks,
                    lanes=lanes,
                    class_colors=class_palette,
                    show_ground_truth=show_ground_truth,
                    show_predictions=show_predictions,
                    selected_classes=selected_classes,
                    key=f"audio_timeline_{selected_recording}",
                )
            except Exception as exc:
                st.warning(f"Could not render timeline for `{selected_row['audio_file']}`: {exc}")
        else:
            st.warning("No matching WAV file for this recording id in the selected folders.")

        class_distribution = pd.DataFrame(index=selected_classes)
        if not visible_gt.empty:
            class_distribution["ground_truth"] = visible_gt["label"].value_counts()
        if not visible_pred.empty:
            class_distribution["prediction"] = visible_pred["label"].value_counts()
        class_distribution = class_distribution.fillna(0).astype(int)
        if not class_distribution.empty:
            class_distribution["total"] = class_distribution.sum(axis=1)
            st.markdown("**Sound Summary (selected file)**")
            st.dataframe(class_distribution.reset_index().rename(columns={"index": "label"}), width="stretch", hide_index=True)
            st.bar_chart(class_distribution[["total"]], width="stretch")

        visible_events = pd.concat([visible_gt, visible_pred], ignore_index=True)
        if not visible_events.empty:
            event_table = visible_events[
                [
                    "source",
                    "event_idx",
                    "begin_s",
                    "end_s",
                    "duration_s",
                    "label",
                    "source_file",
                ]
            ].sort_values(["begin_s", "source"])
            st.markdown("**Visible Events**")
            st.dataframe(event_table, width="stretch", hide_index=True)
            st.download_button(
                "Download Visible Events (.tsv)",
                data=event_table.to_csv(index=False, sep="\t").encode("utf-8"),
                file_name=f"{selected_recording}_visible_events.tsv",
                mime="text/tab-separated-values",
            )

    with tabs[1]:
        st.subheader("Overview")
        if all_events_df.empty:
            st.info("No event tables loaded.")
        else:
            non_noise_events = all_events_df[all_events_df["label"].str.lower() != "noise"].copy()
            class_stats_df = aggregate_class_stats(non_noise_events) if not non_noise_events.empty else pd.DataFrame()

            left, right = st.columns((1, 1))
            with left:
                st.markdown("**Class Distribution (non-noise)**")
                if class_stats_df.empty:
                    st.info("No non-noise classes in loaded events.")
                else:
                    st.bar_chart(class_stats_df.set_index("label")[["detections"]], width="stretch")
                    st.dataframe(class_stats_df, width="stretch", hide_index=True)

            with right:
                st.markdown("**Audio Availability**")
                availability = catalog_df[
                    ["recording_id", "audio_file", "has_ground_truth", "has_predictions", "audio_duration_s"]
                ].copy()
                availability["audio_duration_s"] = pd.to_numeric(
                    availability["audio_duration_s"], errors="coerce"
                ).round(2)
                st.dataframe(availability, width="stretch", hide_index=True)

    with tabs[2]:
        st.subheader("All Parsed Events")
        if all_events_df.empty:
            st.info("No events loaded.")
        else:
            all_labels = sorted(all_events_df["label"].unique().tolist())
            selected_global_labels = st.multiselect("Filter Classes", all_labels, default=all_labels)
            source_values = sorted(all_events_df["source"].unique().tolist())
            selected_sources = st.multiselect("Filter Sources", source_values, default=source_values)

            global_filtered = all_events_df[
                (all_events_df["label"].isin(selected_global_labels))
                & (all_events_df["source"].isin(selected_sources))
            ].copy()
            st.dataframe(
                global_filtered[
                    [
                        "recording_id",
                        "source",
                        "event_idx",
                        "begin_s",
                        "end_s",
                        "duration_s",
                        "label",
                        "source_file",
                    ]
                ].sort_values(["recording_id", "begin_s", "source"]),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                f"Ground truth suffix: `{GROUND_TRUTH_SUFFIX}` | "
                f"Prediction suffix: `{MODEL_PREDICTION_SUFFIX}`"
            )


if __name__ == "__main__":
    run_app()
