from __future__ import annotations

from pathlib import Path

import numpy as np

from bci.bundle import (
    Bundle,
    collect_package_versions,
    find_bundles,
    find_sessions,
    load_bundle,
    load_session_recording,
    make_metadata,
    make_recording_metadata,
    save_bundle,
    save_session_recording,
)
from bci.config import PreprocessingConfig
from bci.models.csp_lda import CSPLDADecoder
from tests._synth import generate_synthetic_mi


def _make_fitted_bundle() -> tuple[Bundle, np.ndarray]:
    X, y = generate_synthetic_mi(n_trials_per_class=20, n_channels=2, seed=0)
    decoder = CSPLDADecoder(n_components=4).fit(X, y)
    pre_cfg = PreprocessingConfig(fs=128.0, channel_order=["C3", "C4"])
    metadata = make_metadata(
        subject="A",
        cal_acc=0.78,
        raw_data_path="data/A_2026-05-06.npz",
        trial_len_s=4.0,
        cue_s=1.0,
        mi_s=4.0,
        rest_s=2.0,
        n_trials_per_class=20,
        fs=128.0,
        emotiv_n_channels=2,
        emotiv_total_channels=7,
        emotiv_channel_names=["FC3", "FC4"],
        physical_electrodes=["FC3", "FC4"],
        logical_channels=["C3", "C4"],
    )
    bundle = Bundle(
        decoder=decoder,
        decoder_name="csp_lda",
        decoder_params=decoder.get_params(),
        preprocessing=pre_cfg,
        label_map={0: "left", 1: "right"},
        window_samples=256,
        stride_samples=64,
        smoother_name="passthrough",
        smoother_params={},
        metadata=metadata,
    )
    return bundle, X[:5]


def test_round_trip_predict_proba_matches(tmp_path: Path) -> None:
    bundle, X_eval = _make_fitted_bundle()
    proba_before = bundle.decoder.predict_proba(X_eval)

    path = save_bundle(bundle, tmp_path / "b.joblib")
    assert path.exists()
    loaded = load_bundle(path)
    proba_after = loaded.decoder.predict_proba(X_eval)

    np.testing.assert_allclose(proba_before, proba_after)


def test_round_trip_preserves_static_fields(tmp_path: Path) -> None:
    bundle, _ = _make_fitted_bundle()
    path = save_bundle(bundle, tmp_path / "b.joblib")
    loaded = load_bundle(path)

    assert loaded.decoder_name == "csp_lda"
    assert loaded.decoder_params == bundle.decoder_params
    assert loaded.label_map == {0: "left", 1: "right"}
    assert loaded.window_samples == 256
    assert loaded.stride_samples == 64
    assert loaded.smoother_name == "passthrough"
    assert loaded.smoother_params == {}
    assert loaded.preprocessing.channel_order == ["C3", "C4"]
    assert loaded.preprocessing.apply_car is False
    assert loaded.preprocessing.bandpass == (8.0, 30.0)


def test_session_recording_folder_round_trip(tmp_path: Path) -> None:
    raw = np.zeros((2, 128), dtype=np.float64)
    ts = np.arange(128, dtype=np.float64) / 128.0
    metadata = make_recording_metadata(
        session_id="A_20260509T123000",
        subject="A",
        recorded_at=None,
        fs=128.0,
        channel_order=["C3", "C4"],
        n_trials_per_class=1,
        n_total_trials=2,
        cue_s=1.0,
        mi_s=4.0,
        rest_s=2.0,
    )

    path = save_session_recording(
        tmp_path,
        "A_20260509T123000",
        raw=raw,
        timestamps=ts,
        marker_times=np.array([0.0, 6.0]),
        marker_labels=np.array([0, 1]),
        metadata=metadata,
    )
    loaded = load_session_recording(tmp_path, "A_20260509T123000")

    assert path.name == "recording.npz"
    np.testing.assert_allclose(loaded["raw"], raw)
    assert loaded["metadata"]["session_id"] == "A_20260509T123000"
    assert [s["session_id"] for s in find_sessions(tmp_path)] == [
        "A_20260509T123000"
    ]


def test_bundle_folder_round_trip_and_scanner(tmp_path: Path) -> None:
    bundle, X_eval = _make_fitted_bundle()
    bundle.metadata.update(
        {"session_id": "A_20260509T123000", "decoder_name": "csp_lda", "cv_acc": 0.78}
    )

    path = save_bundle(bundle, tmp_path / "A_20260509T123000" / "csp_lda")
    loaded = load_bundle(path.parent)

    assert path.name == "bundle.joblib"
    np.testing.assert_allclose(
        loaded.decoder.predict_proba(X_eval),
        bundle.decoder.predict_proba(X_eval),
    )
    found = find_bundles(tmp_path)
    assert found[0]["session_id"] == "A_20260509T123000"
    assert found[0]["decoder_name"] == "csp_lda"
    assert found[0]["acc"] == 0.78


def test_metadata_contains_required_fields(tmp_path: Path) -> None:
    bundle, _ = _make_fitted_bundle()
    path = save_bundle(bundle, tmp_path / "b.joblib")
    loaded = load_bundle(path)
    md = loaded.metadata

    for key in (
        "subject",
        "date",
        "cal_acc",
        "trial_len_s",
        "fs",
        "physical_electrodes",
        "logical_channels",
        "package_versions",
        "git_sha",
    ):
        assert key in md, f"missing metadata key: {key}"

    pkgs = md["package_versions"]
    for pkg in ("numpy", "scipy", "scikit-learn", "mne", "python"):
        assert pkg in pkgs
        assert pkgs[pkg] != ""


def test_smoother_internal_state_not_saved(tmp_path: Path) -> None:
    """Bundle records only smoother init params, never an instance.

    This protects against carrying realtime state (e.g. an EMA running
    average) across calibration sessions.
    """
    bundle, _ = _make_fitted_bundle()
    path = save_bundle(bundle, tmp_path / "b.joblib")
    import joblib  # local: matches what save_bundle uses

    raw_payload = joblib.load(path)
    assert "smoother" not in raw_payload, (
        "Bundle payload must not contain a smoother instance; "
        "only smoother_name + smoother_params are allowed."
    )
    assert raw_payload["smoother_name"] == "passthrough"
    assert raw_payload["smoother_params"] == {}


def test_collect_package_versions_returns_known_keys() -> None:
    versions = collect_package_versions()
    for pkg in ("numpy", "scipy", "scikit-learn", "mne", "python"):
        assert pkg in versions
