from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

CANONICAL_VERSION = "1.0"

UNMAPPED = -1

HEAD_RADIUS_M = 0.0875

_ROW_INCLINATION = {
    "Fp": 72.0, "AF": 54.0, "F": 36.0, "FC": 18.0, "C": 0.0,
    "CP": -18.0, "P": -36.0, "PO": -54.0, "O": -72.0,
}
_RING_AZIMUTH = {
    "Fp": 18.0, "AF": 36.0, "F": 54.0, "FC": 72.0, "C": 90.0,
    "CP": 108.0, "P": 126.0, "PO": 144.0, "O": 162.0,
}
_RING_NAME = {"FC": "FT", "C": "T", "CP": "TP"}


def _sph(inclination_deg: float, azimuth_deg: float, right: bool = False) -> np.ndarray:
    th, az = np.deg2rad(inclination_deg), np.deg2rad(azimuth_deg)
    x = -np.sin(th) * np.sin(az)
    if right:
        x = -x
    return np.array([x, np.sin(th) * np.cos(az), np.cos(th)], dtype=np.float64)


def _slerp(a: np.ndarray, b: np.ndarray, f: float) -> np.ndarray:
    omega = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if omega < 1e-12:
        return a.copy()
    return (np.sin((1 - f) * omega) * a + np.sin(f * omega) * b) / np.sin(omega)


def _build_scalp_table() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []

    def add(name, vec):
        out.append((name, vec))

    add("Nz", _sph(90.0, 0.0))
    for row in ("Fp", "AF", "F", "FC", "C", "CP", "P", "PO", "O"):
        incl = _ROW_INCLINATION[row]
        mid = _sph(abs(incl), 0.0 if incl >= 0 else 180.0)
        ring_az = _RING_AZIMUTH[row]
        ring_name = _RING_NAME.get(row, row)
        if row in ("Fp", "O"):
            add(f"{row}1", _sph(72.0, ring_az))
            add(f"{row}z", mid)
            add(f"{row}2", _sph(72.0, ring_az, right=True))
            if row == "O":
                add("O9", _sph(90.0, ring_az))
                add("O10", _sph(90.0, ring_az, right=True))
            continue
        lateral = (7, 5, 3, 1)
        add(f"{ring_name}9", _sph(90.0, ring_az))
        for k in lateral:
            f = (k + 1) / 8.0
            name = f"{ring_name if k == 7 else row}{k}"
            add(name, _slerp(mid, _sph(72.0, ring_az), f))
        add(f"{row}z", mid)
        for k in lateral[::-1]:
            f = (k + 1) / 8.0
            name = f"{ring_name if k == 7 else row}{k + 1}"
            add(name, _slerp(mid, _sph(72.0, ring_az, right=True), f))
        add(f"{ring_name}10", _sph(90.0, ring_az, right=True))
    add("Iz", _sph(90.0, 180.0))
    add("A1", _sph(100.0, 95.0))
    add("A2", _sph(100.0, 95.0, right=True))
    add("M1", _sph(105.0, 112.0))
    add("M2", _sph(105.0, 112.0, right=True))
    return out


_AUX_TABLE: list[tuple[str, str]] = [
    ("eog:heog_l", "eog"), ("eog:heog_r", "eog"), ("eog:veog_l", "eog"),
    ("eog:veog_r", "eog"), ("eog:heog", "eog"), ("eog:veog", "eog"),
    ("eog:generic", "eog"),
    ("ecg:generic", "ecg"),
    ("emg:chin", "emg"), ("emg:leg_l", "emg"), ("emg:leg_r", "emg"),
    ("emg:generic", "emg"),
    ("resp:thoracic", "resp"), ("resp:abdominal", "resp"), ("resp:airflow", "resp"),
    ("resp:spo2", "resp"), ("resp:generic", "resp"),
    ("gsr:generic", "gsr"), ("temp:generic", "temp"),
    ("stim:generic", "stim"), ("misc:generic", "misc"),
]

CHANNEL_TYPES = (
    "eeg", "eog", "ecg", "emg", "seeg", "ecog", "dbs",
    "resp", "gsr", "temp", "stim", "misc",
)


@lru_cache(maxsize=1)
def _table():
    scalp = _build_scalp_table()
    names = [n for n, _ in scalp] + [n for n, _ in _AUX_TABLE]
    types = ["eeg"] * len(scalp) + [t for _, t in _AUX_TABLE]
    pos = np.full((len(names), 3), np.nan, dtype=np.float64)
    pos[: len(scalp)] = np.stack([v for _, v in scalp]) * HEAD_RADIUS_M
    return tuple(names), tuple(types), pos


def canonical_names() -> list[str]:
    return list(_table()[0])


def canonical_types() -> list[str]:
    return list(_table()[1])


def canonical_positions() -> np.ndarray:
    return _table()[2].copy()


def n_canonical() -> int:
    return len(_table()[0])


_EEG_ALIASES = {
    "t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8",
    "fpz": "Fpz", "fp1": "Fp1", "fp2": "Fp2", "oz": "Oz", "o1": "O1", "o2": "O2",
    "nas": "Nz", "nasion": "Nz", "inion": "Iz", "iz": "Iz",
    "lpa": "A1", "rpa": "A2", "ref": None, "gnd": None, "ground": None,
    "left mastoid": "M1", "right mastoid": "M2", "lm": "M1", "rm": "M2",
}

_AUX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(loc|e1|eog[-_ ]?loc)$"), "eog:heog_l"),
    (re.compile(r"^(roc|e2|eog[-_ ]?roc)$"), "eog:heog_r"),
    (re.compile(r"^(h?eog)[-_ ]?(l|left|1)$"), "eog:heog_l"),
    (re.compile(r"^(h?eog)[-_ ]?(r|right|2)$"), "eog:heog_r"),
    (re.compile(r"^(l|left)[-_ ]?h?eog$"), "eog:heog_l"),
    (re.compile(r"^(r|right)[-_ ]?h?eog$"), "eog:heog_r"),
    (re.compile(r"^veog[-_ ]?(l|left|1)$"), "eog:veog_l"),
    (re.compile(r"^veog[-_ ]?(r|right|2)$"), "eog:veog_r"),
    (re.compile(r"^(l|left)[-_ ]?veog$"), "eog:veog_l"),
    (re.compile(r"^(r|right)[-_ ]?veog$"), "eog:veog_r"),
    (re.compile(r"^(heog|eog[-_ ]?h|horizontal[-_ ]?eog|eog[-_ ]?horiz\w*)$"), "eog:heog"),
    (re.compile(r"^(veog|eog[-_ ]?v|vertical[-_ ]?eog|eog[-_ ]?vert\w*)$"), "eog:veog"),
    (re.compile(r"^(eog\w*|eyes?|blink)$"), "eog:generic"),
    (re.compile(r"^(ecg\w*|ekg\w*|heart|cardiac)$"), "ecg:generic"),
    (re.compile(r"^(emg[-_ ]?chin|chin\w*|submental\w*)$"), "emg:chin"),
    (re.compile(r"^(emg[-_ ]?(l|left)[-_ ]?leg|(l|left)[-_ ]?leg\w*|leg[-_ ]?(l|left|1))$"), "emg:leg_l"),
    (re.compile(r"^(emg[-_ ]?(r|right)[-_ ]?leg|(r|right)[-_ ]?leg\w*|leg[-_ ]?(r|right|2))$"), "emg:leg_r"),
    (re.compile(r"^(emg\w*|muscle)$"), "emg:generic"),
    (re.compile(r"^((resp|rip)?[-_ ]?thor\w*|chest)$"), "resp:thoracic"),
    (re.compile(r"^((resp|rip)?[-_ ]?abdo\w*|belly)$"), "resp:abdominal"),
    (re.compile(r"^(airflow|flow|nasal\w*|cannula|thermistor|pressure)$"), "resp:airflow"),
    (re.compile(r"^(spo2|sao2|oxsat|saturation)$"), "resp:spo2"),
    (re.compile(r"^(resp\w*|breath\w*)$"), "resp:generic"),
    (re.compile(r"^(gsr|eda|skin[-_ ]?cond\w*)$"), "gsr:generic"),
    (re.compile(r"^(temp\w*|body[-_ ]?temp\w*)$"), "temp:generic"),
    (re.compile(r"^(sti\s?[0-9]+|stim\w*|status|trig\w*|marker\w*|event\w*|dc[0-9]+|photic)$"), "stim:generic"),
]

_TYPE_FALLBACK = {
    "eog": "eog:generic", "ecg": "ecg:generic", "emg": "emg:generic",
    "resp": "resp:generic", "gsr": "gsr:generic", "temp": "temp:generic",
    "stim": "stim:generic", "misc": "misc:generic",
}

_MODALITY_PREFIX = re.compile(r"^(eeg|eog|ecg|ekg|emg|seeg|ecog|dbs|meg|ref)[\s_:.-]+", re.I)
_SPLIT_REF = re.compile(r"^(.+?)[-](.+)$")


@lru_cache(maxsize=4096)
def _eeg_lookup(token: str) -> str | None:
    names, _, _ = _table()
    key = token.strip().lower()
    if not key:
        return None
    if key in _EEG_ALIASES:
        return _EEG_ALIASES[key]
    by_lower = {n.lower(): n for n in names}
    if key in by_lower:
        return by_lower[key]
    if key.endswith("z") and key[:-1] + "z" in by_lower:
        return by_lower[key[:-1] + "z"]
    return None


def _parse_one(token: str, ch_type: str) -> tuple[str | None, str | None]:
    m = _SPLIT_REF.match(token)
    if m and ch_type in ("eeg", "eog", "ecg", "emg"):
        head, tail = m.group(1).strip(), m.group(2).strip()
        head_eeg = _eeg_lookup(head)
        if head_eeg is not None:
            ref_eeg = _eeg_lookup(tail)
            return head_eeg, (ref_eeg if ref_eeg is not None else tail)
    if ch_type in ("eeg", "eog", "ecg", "emg"):
        eeg = _eeg_lookup(token)
        if eeg is not None:
            return eeg, None
    key = re.sub(r"[^a-z0-9]+", " ", token.lower()).strip()
    key_tight = key.replace(" ", "")
    for pat, canon in _AUX_PATTERNS:
        if pat.match(key) or pat.match(key_tight):
            return canon, None
    return None, None


def parse_channel_name(name: str, ch_type: str = "eeg") -> tuple[str | None, str | None]:
    raw = str(name).strip()
    ch_type = str(ch_type).lower()
    raw = raw.rstrip(".").strip()
    m = re.match(r"^(.+-[^-\s]+)-\d+$", raw)
    if m:
        raw = m.group(1)
    stripped = _MODALITY_PREFIX.sub("", raw)
    for cand in dict.fromkeys((stripped, raw)):
        canon, ref = _parse_one(cand, ch_type)
        if canon is not None:
            return canon, ref
    if ch_type != "eeg":
        return _TYPE_FALLBACK.get(ch_type), None
    return None, None


def map_channels(names, types=None):
    names = [str(n) for n in names]
    if types is None:
        types = ["eeg"] * len(names)
    if isinstance(types, str):
        types = [types] * len(names)
    index = {n: i for i, n in enumerate(_table()[0])}
    ids = np.full(len(names), UNMAPPED, dtype=np.int32)
    refs = []
    for i, (n, t) in enumerate(zip(names, types)):
        canon, ref = parse_channel_name(n, str(t).lower())
        if canon is not None and canon in index:
            ids[i] = index[canon]
        refs.append(ref or "")
    return ids, refs


def canonical_index(name: str) -> int:
    try:
        return _table()[0].index(name)
    except ValueError:
        return UNMAPPED


def project(data: np.ndarray, canonical_id: np.ndarray, n_out: int | None = None,
            fill: float = 0.0, bad: np.ndarray | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(canonical_id, dtype=np.int64)
    k = n_canonical() if n_out is None else int(n_out)
    out = np.full((k,) + tuple(data.shape[1:]), fill, dtype=data.dtype)
    present = np.zeros(k, dtype=bool)
    source = np.full(k, -1, dtype=np.int64)
    keep = np.flatnonzero((ids >= 0) & (ids < k))
    if keep.size:
        badness = np.zeros(ids.shape[0], dtype=bool) if bad is None else np.asarray(bad, bool)
        order = keep[np.lexsort((keep, badness[keep]))[::-1]]
        out[ids[order]] = data[order]
        source[ids[order]] = order
        present[ids[order]] = True
    return out, present, source
