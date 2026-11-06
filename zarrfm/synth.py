from __future__ import annotations

import numpy as np

MONTAGE_10_20 = {
    "Fp1": (-0.0294, 0.0839, 0.0209), "Fp2": (0.0298, 0.0848, 0.0210),
    "F7": (-0.0702, 0.0424, -0.0111), "F3": (-0.0502, 0.0531, 0.0424),
    "Fz": (0.0003, 0.0582, 0.0668), "F4": (0.0518, 0.0543, 0.0409),
    "F8": (0.0731, 0.0429, -0.0120), "T7": (-0.0842, -0.0157, -0.0090),
    "C3": (-0.0653, -0.0116, 0.0642), "Cz": (0.0004, -0.0097, 0.1009),
    "C4": (0.0654, -0.0100, 0.0645), "T8": (0.0850, -0.0150, -0.0090),
    "P7": (-0.0724, -0.0734, -0.0022), "P3": (-0.0530, -0.0787, 0.0567),
    "Pz": (0.0003, -0.0817, 0.0812), "P4": (0.0555, -0.0785, 0.0546),
    "P8": (0.0730, -0.0730, -0.0027), "O1": (-0.0294, -0.1122, 0.0154),
    "O2": (0.0298, -0.1121, 0.0155),
}


def _pink(n_channels: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal((n_channels, n_samples))
    spectrum = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spectrum * scale, n=n_samples, axis=1)
    return shaped / (np.std(shaped, axis=1, keepdims=True) + 1e-30)


def _lowpass(data: np.ndarray, sfreq: float, cutoff: float, order: int = 8) -> np.ndarray:
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)
    spectrum = np.fft.rfft(data, axis=1)
    spectrum *= 1.0 / (1.0 + (freqs / cutoff) ** order)
    return np.fft.irfft(spectrum, n=data.shape[1], axis=1)


def synth_eeg(
    n_channels: int = 19,
    duration_s: float = 60.0,
    sfreq: float = 256.0,
    seed: int = 0,
    n_events: int = 40,
    lowpass_hz: float | None = None,
):
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_s * sfreq))
    names = list(MONTAGE_10_20)
    if n_channels <= len(names):
        names = names[:n_channels]
    else:
        names = names + [f"EXG{i:02d}" for i in range(n_channels - len(names))]
    pos = np.array(
        [MONTAGE_10_20.get(n, (np.nan, np.nan, np.nan)) for n in names], dtype=np.float64
    )

    t = np.arange(n_samples) / sfreq

    n_sources = max(4, n_channels // 3)
    sources = _pink(n_sources, n_samples, rng)
    src_pos = rng.normal(scale=0.05, size=(n_sources, 3))
    known = np.isfinite(pos).all(axis=1)
    sensor_pos = np.where(known[:, None], pos, rng.normal(scale=0.07, size=pos.shape))
    dist = np.linalg.norm(sensor_pos[:, None, :] - src_pos[None, :, :], axis=2)
    lead_field = 1.0 / (dist + 0.03) ** 2
    lead_field /= np.linalg.norm(lead_field, axis=1, keepdims=True)
    data = 15e-6 * (lead_field @ sources)
    data += 4e-6 * _pink(1, n_samples, rng)
    data += 1.5e-6 * rng.standard_normal((n_channels, n_samples))

    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2 * np.pi))
    alpha = np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi)) * envelope
    posterior = np.array([1.0 if n[0] in "OP" else 0.15 for n in names])
    data += 25e-6 * posterior[:, None] * alpha[None, :]

    frontal = np.array([1.0 if n.startswith("Fp") else (0.3 if n[0] == "F" else 0.0) for n in names])
    for onset in rng.uniform(0, max(duration_s - 1.0, 0.5), size=max(int(duration_s // 8), 1)):
        i = int(onset * sfreq)
        w = int(0.3 * sfreq)
        if i + w <= n_samples:
            shape = np.hanning(w)
            data[:, i : i + w] += 120e-6 * frontal[:, None] * shape[None, :]

    events = {}
    if n_events:
        onsets = np.sort(rng.uniform(1.0, max(duration_s - 1.0, 1.0), size=n_events))
        codes = rng.choice([1, 2], size=n_events)
        events = {
            "onset": onsets,
            "duration": np.zeros(n_events),
            "label": ["target" if c == 1 else "standard" for c in codes],
            "value": codes,
        }
        erp_w = int(0.4 * sfreq)
        erp = np.sin(np.linspace(0, np.pi, erp_w)) * np.linspace(1, 0.2, erp_w)
        central = np.array([1.0 if n in ("Cz", "Pz", "C3", "C4") else 0.2 for n in names])
        for onset, code in zip(onsets, codes):
            i = int(onset * sfreq)
            if i + erp_w <= n_samples:
                amp = 8e-6 if code == 1 else 3e-6
                data[:, i : i + erp_w] += amp * central[:, None] * erp[None, :]

    cutoff = lowpass_hz if lowpass_hz is not None else min(0.35 * sfreq, 100.0)
    data = _lowpass(data, sfreq, cutoff)

    meta = {
        "sfreq": sfreq,
        "lowpass_hz": cutoff,
        "ch_names": names,
        "ch_types": ["eeg"] * n_channels,
        "positions": pos,
        "events": events,
    }
    return data.astype(np.float64), meta
