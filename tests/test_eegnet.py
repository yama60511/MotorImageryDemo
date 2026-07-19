from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from bci.models.eegnet import EEGNetDecoder  # noqa: E402
from bci.models.registry import build, list_available  # noqa: E402


def test_eegnet_registers_and_predicts_probabilities() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 4, 128)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    decoder = EEGNetDecoder(
        epochs=2,
        batch_size=4,
        kernel_length=16,
        random_state=0,
    )
    decoder.fit(X, y)

    proba = decoder.predict_proba(X[:3])
    pred = decoder.predict(X[:3])

    assert "eegnet" in list_available()
    assert isinstance(build("eegnet", epochs=1), EEGNetDecoder)
    assert proba.shape == (3, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), rtol=1e-6)
    assert pred.shape == (3,)
