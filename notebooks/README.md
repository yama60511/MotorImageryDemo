# EEG Decoding Notebooks

Teaching materials for Motor Imagery (MI) EEG decoding.
Recorded with an Emotiv EPOC X (14-channel, 128 Hz) during "left-hand / right-hand motor imagery,"
two notebooks build and compare decoders using two different approaches.

| Notebook | Method | Contents |
|---|---|---|
| `01_csp_ml_classical_decoder.ipynb` | **CSP + Machine Learning** (classical) | Bandpass → CSP (spatial filter) → LDA/SVM, Riemannian geometry (FgMDM). ERD visualization, CSP topomaps, Leave-One-Session-Out evaluation. |
| `02_eegnet_dl_decoder.ipynb` | **EEGNet** (deep learning) | End-to-end decoding from preprocessed signals. PyTorch implementation, learning curves, comparison with classical methods. |

We recommend working through `①` (classical) before `②` (deep learning).

## Running on Google Colab

### 1. Upload to Google Drive
Place the `notebooks` folder and `data` folder in your Google Drive, **keeping them at the same directory level**:

```
MyDrive/
└── MotorImageryDemo/
    ├── notebooks/
    │   ├── 01_csp_ml_classical_decoder.ipynb
    │   ├── 02_eegnet_dl_decoder.ipynb
    │   └── utils.py
    └── data/
        ├── (session 1)/   recording.npz, metadata.json
        ├── (session 2)/   ...
        └── ...
```

### 2. Open in Colab
Right-click the `.ipynb` file on Drive → "Open with" → "Google Colaboratory".

### 3. Run the Setup cell
The second cell in each notebook (marked `# === Setup ===`) automatically:

- Installs any extra packages (Notebook ① needs `mne` + `pyriemann`; Notebook ② needs
  nothing beyond Colab's pre-installed PyTorch)
- Mounts Google Drive (you will be prompted to grant access)
- Changes to the `notebooks` directory (needed to import `utils.py` and find `../data`)

> If you renamed the folder, update the `%cd` path in the Setup cell accordingly.

### 4. Run cells top to bottom
Notebook `②` (EEGNet) runs faster on a **GPU runtime** (Runtime → Change runtime type →
Hardware accelerator: GPU). It also works on CPU.

## Data Location

Each notebook has a `DATA_DIR` cell near the top where you specify the data location:

```python
DATA_DIR = None          # auto-search: checks MI_DATA_DIR env var, then ../data
DATA_DIR = "/path/..."   # point directly to the folder containing recording.npz files
```

With the folder structure above, `DATA_DIR = None` will automatically find `../data`.
Each session folder contains `recording.npz` (continuous EEG, timestamps, markers) and
`metadata.json`.

## Shared Module `utils.py`

A shared module used by both notebooks for data loading and preprocessing (no external
dependencies beyond what the notebooks already install). The preprocessing pipeline
(notch → bandpass → CAR) and epoching logic are identical to the real-time BCI service.

- `load_dataset()` — loads all sessions, applies preprocessing, and extracts 4-second MI epochs.
- `load_erd_time_courses()` — loads baseline-to-MI epochs and computes configurable band ERD% curves.
- `plot_erd_time_courses()` — plots the configured bands for FC5/FC6.
- `sliding_windows()` — expands a set of trials into overlapping windows (same data augmentation as real-time inference). Returns a per-window source-trial index so windows are split **after** the train/test split (Notebook ② applies it inside each CV fold to avoid leakage).

## Note on Electrode Placement

The standard BCI electrodes C3/C4 (directly above the hand motor area) are absent on the EPOC X,
so we rely on nearby channels **FC5 / FC6 / T7 / T8**. Accuracy is therefore modest (slightly
above the 50% chance level in some sessions). Part of the learning experience is that electrode
placement and data volume matter as much as — or more than — algorithmic choices.

## Running Locally (for developers)

```bash
conda activate MotorImageryDemo
pip install mne pyriemann torch
export MI_DATA_DIR=/path/to/recordings   # used when DATA_DIR=None
jupyter lab
```

> The Setup cell is automatically skipped outside Colab. `utils` sets `KMP_DUPLICATE_LIB_OK`
> on import to prevent OpenMP double-load crashes on macOS when using PyTorch with numpy/MNE.
