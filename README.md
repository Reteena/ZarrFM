# ZarrFM

ZarrFM is an execution-oriented storage layer for EEG Foundation-Model Pretraining that complements BIDS by storing a corpus as a single [Zarr](https://zarr.dev) v3 hierarchy with indexed metadata, canonical channel spaces, explicit masks, hash-stable subject splits, verified event tables, and block-aligned shading. 

On real PhysioNet recordings and synthetic corpora, we compare ZarrFM with EDF, FIF, NPY, HDF5, and Parquet. Across 200 recordings, its two-level index reduces pretraining queries from 601 requests to 14. Preserving native signal codes reduces storage from 42.4 MB/hour in EDF to 22.7 MB/hour, while block-aligned sampling reaches 3, 100 windows/s, up to 11x faster than Parquet. Reads remain bit-identical across independent Zarr implementations, and concurrent failure-injected appends recover without data loss.


## Version

The ```zarrfm``` python package's version is `0.1.0`.

## Requirements

- Python ≥ 3.11
- Core: `zarr`, `numpy`, `numcodecs`
- Optional: `mne`, `edfio`, `torch` (see extras in `pyproject.toml`)

## Setup

```bash
git clone <this-repo> && cd zarrfm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# or, for the pinned benchmark stack:
# pip install -r requirements.txt && pip install -e .
```

## Quick start

```python
import numpy as np
import zarrfm

# Write a recording from an array (channels × samples)
data = np.random.randn(19, 256 * 60).astype(np.float64) * 1e-5
zarrfm.from_array(
    data,
    "study.zarr",
    sfreq=256.0,
    ch_names=[f"E{i}" for i in range(19)],
    subject="sub-01",
    session="ses-01",
    task="rest",
)

# Validate, read a window, epoch around events
assert not zarrfm.validate("study.zarr")
r = zarrfm.ZarrFMReader("study.zarr")
window = r.get_data("eeg", tmin=10.0, tmax=20.0)

# Optional MNE / EDF bridges
# raw = zarrfm.to_mne(r)
# zarrfm.from_edf("recording.edf", "out.zarr")
```

End-to-end walkthrough on synthetic data:

```bash
python demo.py
```

Creates `demo_study.zarr` and `demo_corpus.zarr` (remove with `rm -rf *.zarr`).

## Tests

```bash
pytest -q
```

## Benchmarks

Results land under `bench_results/` (gitignored by default; regenerate locally).

### Recording-level: ZarrFM vs EDF vs FIF

```bash
python bench.py                              # 64 ch × 10 min @ 256 Hz
python bench.py --channels 128 --minutes 30
python bench.py --output results.json
```

### Training-path

```bash
python bench_ml.py                           # all experiments
python bench_ml.py --exp metadata chunks formats latency
python bench_ml.py --exp workers gpu
```

Experiments: `metadata`, `chunks`, `formats`, `latency`, `workers`, `gpu`.

### One public EDF (~40 MB)

```bash
mkdir -p data/real
curl -sSL -o data/real/chb01_03.edf \
  https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf

python bench.py --edf data/real/chb01_03.edf
python bench_ml.py --exp formats chunks latency --edf data/real/chb01_03.edf
```

### PhysioNet corpus (~250 GB free disk recommended)

```bash
python bench_physionet.py fetch --jobs 12
python bench_physionet.py run --workers 16 --follow
python bench_physionet.py summary            # summary.json / summary.md
python bench_physionet.py bytype

# subset
python bench_physionet.py fetch --datasets eegmat chbmit
python bench_physionet.py run --datasets eegmat chbmit
```

Sleep-EDF Expanded (197 PSG / hypnogram pairs):

```bash
python bench_physionet.py fetch --datasets sleep-edfx --jobs 12
python bench_physionet.py run --datasets sleep-edfx --workers 8
python bench_physionet.py summary
```

Recording-level Sleep-EDF latency (three native 100 Hz channels + hypnogram):

```bash
python bench.py \
  --edf data/real/physionet/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf \
  --annotations data/real/physionet/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf \
  --picks "EEG Fpz-Cz" "EEG Pz-Oz" "EOG horizontal" \
  --native-edf --repeats 5 \
  --output bench_results/physionet/sleep_edf_latency.json
```

### MOABB

```bash
MNE_DATASETS_BNCI_PATH=$PWD/data/real/moabb MNE_DATA=$PWD/data/real/moabb \
  python bench_moabb.py --dataset BNCI2014_009 --subject 1
```

### CBraMod integration

`bench_cbramod.py` feeds matched NPY and ZarrFM windows into the
[official CBraMod](https://github.com/wjq-learning/CBraMod) reconstruction head.
It checks round trips, model outputs, and (optionally) backward gradients; it does
not claim downstream decoding accuracy.

```bash
git clone https://github.com/wjq-learning/CBraMod.git /tmp/zarrfm-cbramod-upstream
git -C /tmp/zarrfm-cbramod-upstream checkout b9e961003214326972c567eff390e75b0287e32a
curl -fL https://huggingface.co/weighting666/CBraMod/resolve/500543c7e30bda1b22bfd51a49301b238dee21fd/pretrained_weights.pth \
  -o /tmp/zarrfm-cbramod-pretrained.pth

python bench_cbramod.py \
  --model-repo /tmp/zarrfm-cbramod-upstream \
  --checkpoint /tmp/zarrfm-cbramod-pretrained.pth \
  --seconds 32 --max-batches 4 \
  --output bench_results/cbramod_synthetic.json
```

EDF path: band-pass 0.3–75 Hz (Nyquist-limited), optional notch (`--notch 50` or `60`), resample to 200 Hz, then map into CBraMod’s patch layout via `zarrfm/foundation.py`.

## Layout

```
zarrfm/            library (reader, writer, editor, dataset, sampler, I/O)
demo.py             end-to-end synthetic walkthrough
bench.py            recording-level format comparison
bench_ml.py         training-path benchmarks
bench_physionet.py  corpus fetch / run / summary
bench_moabb.py      MOABB size / quantisation benchmark
bench_cbramod.py    matched NPY vs ZarrFM + official CBraMod
tests/              pytest suite
pyproject.toml      package metadata and extras
requirements.txt    pinned benchmark dependencies
```

Local downloads (`data/real/`), generated stores (`*.zarr`), and benchmark outputs (`bench_results/`) should stay out of version control.

## Citation

If you use ZarrFM in academic work, please cite the accompanying paper (citation entry forthcoming).
