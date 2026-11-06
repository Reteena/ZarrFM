import numpy as np
import pytest

from zarrfm import canonical
from zarrfm.foundation import channel_ids, model_patches


def batch():
    ids = channel_ids(["C3", "C4"])
    x = np.zeros((2, canonical.n_canonical(), 400), dtype=np.float32)
    x[:, ids[0]] = 100e-6
    x[:, ids[1]] = 200e-6
    mask = np.zeros(x.shape[:2], dtype=bool)
    mask[:, ids] = True
    return {"signals": x, "mask": mask}


def test_order_units_and_patches():
    x = model_patches(batch(), ["C4", "C3"], sfreq=200)
    assert x.shape == (2, 2, 2, 200)
    np.testing.assert_allclose(x[:, 0], 2.)
    np.testing.assert_allclose(x[:, 1], 1.)
    uv = batch()
    uv["signals"] *= 1e6
    np.testing.assert_allclose(model_patches(uv, ["C4", "C3"], sfreq=200, unit="uV"), x)


def test_missing_bad_and_invalid_input_rejected():
    b = batch()
    b["mask"][1, channel_ids(["C3"])[0]] = False
    with pytest.raises(ValueError, match="missing or bad"):
        model_patches(b, ["C3"], sfreq=200)
    with pytest.raises(ValueError, match="200 Hz"):
        model_patches(batch(), ["C3"], sfreq=256)
    with pytest.raises(ValueError, match="integer number"):
        b = batch()
        b["signals"] = b["signals"][..., :-1]
        model_patches(b, ["C3"], sfreq=200)
    with pytest.raises(ValueError, match="unique"):
        channel_ids(["C3", "C3"])
