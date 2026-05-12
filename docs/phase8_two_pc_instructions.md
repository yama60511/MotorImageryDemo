# Phase 8 Two-PC Instructions

Phase 8 is the first real Emotiv calibration pass. Phase 7 proved the Unity,
Python service, LSL command/status/probability streams, and bundle saving work
with dummy EEG. The next goal is to replace dummy EEG with the real Emotiv EEG
stream and confirm calibration can finish with an acceptable CV score.

Current code path:

- Unity runs on PC2.
- Emotiv app and Python service run on PC1.
- Emotiv/LSL stream is read by Python service on PC1.
- Unity on PC2 publishes `Markers` and `Commands` over LSL.
- Python service on PC1 publishes `Status` and `BCI_Proba` over LSL.
- PC1 and PC2 must be on the same LAN, and the OS firewall must allow LSL,
  Python, and Unity network traffic on both machines.
- Current Python receiver supports the EPOC X-shaped stream: 19 total channels,
  14 EEG channels. You can choose any subset of supported channel names through
  `configs/default.yaml` or the service `--channels` argument.
- Do not commit generated `data/`, `bundles/`, or `logs/` files.

Supported EPOC X EEG labels:

```text
AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4
```

Channel selection priority:

1. `python -m apps.service --channels ...`
2. `configs/default.yaml` -> `channels.selection`
3. `null`, which means use all 14 EEG channels

## Goal

Finish Phase 8.1 to 8.3:

- Emotiv EEG stream is visible over LSL.
- `python -m apps.service --channels ...` connects to the real stream with the
  selected channels.
- Unity data collection reaches `collection_done:session=...`.
- Pressing `Train Model` writes `bundles/<session>/csp_lda/bundle.joblib`.
- Trial-level GroupKFold accuracy is at least `0.60`.

If accuracy is below `0.60`, Phase 8.4 starts.

## Before Starting

On PC1:

```bash
cd /Users/yutayamamura/Projects/MotorImageryDemo
git pull
conda activate MotorImageryDemo
pytest -q
```

On PC2:

```powershell
cd C:\Users\yuta0\Projects\MotorImageryDemo
git pull
conda activate MotorImageryDemo
pytest -q
```

Expected:

```text
100% passed
```

If Unity files changed after opening Unity, check:

```powershell
git status --short
```

Only commit source/config files. Do not commit:

- `data/`
- `bundles/`
- `logs/`
- `Library/`
- `Temp/`
- `UserSettings/`
- `.csproj`
- `.sln`
- `.slnx`

## 8.1 Emotiv LSL Stream Check

1. Start the Emotiv app that exposes LSL.
2. Wear the headset and confirm signal quality is stable.
3. Enable the EEG LSL outlet in the Emotiv app.
4. In PC1 terminal:

```bash
conda activate MotorImageryDemo
python realTimePlotEmotivLSL.py
```

Expected:

- The script finds the Emotiv EEG stream.
- EEG traces move continuously.
- Signal is not flat, saturated, or full of huge spikes.

If this script cannot find the stream:

- Confirm the Emotiv app is still running.
- Confirm LSL outlet/export is enabled.
- Restart the Emotiv app.
- Restart the terminal and try again.
- Do not move to 8.2 until the stream is visible.

## 8.2 Python Service Connection Check

Keep the Emotiv LSL stream running.
Start Unity Play on PC2 first, so the Unity `Markers` and `Commands` LSL
streams are already available.

In a terminal on PC1:

```bash
cd /Users/yutayamamura/Projects/MotorImageryDemo
conda activate MotorImageryDemo
python -m apps.service --channels FC5 FC6
```

Replace `FC5 FC6` with the channels you want to test. For example:

```bash
python -m apps.service --channels F3 F4
python -m apps.service --channels FC5 FC6
python -m apps.service
```

The last command uses `configs/default.yaml`. To set the default there:

```yaml
channels:
  selection: ["F3", "F4"]
```

Expected log near startup:

```text
EEGReceiver connected: {... 'n_channels': 2, 'channel_names': ['F3', 'F4'], ...}
MarkerReceiver connected to type=Markers
CommandReceiver connected to type=Commands
status: state:IDLE
```

The `channel_names` value should match your selected channels.

Important:

- If it says `No LSL stream of type='Markers'`, start Unity Play first, then
  rerun the service on PC1. If Unity is already in Play mode, check firewall and
  make sure PC1 and PC2 are on the same network.
- If it says `No LSL stream of type='Commands'`, start Unity Play first, then
  rerun the service on PC1. If Unity is already in Play mode, check firewall and
  make sure PC1 and PC2 are on the same network.
- If it says the EEG layout is unexpected, copy the full `EEGReceiver connected`
  or error log and stop. The receiver code may need to support that Emotiv
  stream layout before calibration.

## 8.3 Real Calibration With Unity

Use:

- PC2: Unity Editor
- PC1: Terminal running `apps.service`

Steps:

1. Open Unity project:

```text
C:\Users\yuta0\Projects\MotorImageryDemo\unity\MotorImageryUnity
```

2. Open `Assets/MainMenu.unity`.
3. Press Play in Unity.
4. Start the service on PC1 after Unity Play is active:

```bash
cd /Users/yutayamamura/Projects/MotorImageryDemo
conda activate MotorImageryDemo
python -m apps.service --channels FC5 FC6
```

Use the same channel selection that passed 8.2.

5. Wait until Unity shows:

```text
state:IDLE
```

6. Press `Acquire Data` in Unity.
7. Follow the cue text:

- `LEFT`: imagine left-hand movement.
- `RIGHT`: imagine right-hand movement.
- `REST`: relax and avoid movement.

Do not press keys, click around, talk, move shoulders, blink hard, or adjust the
headset during trials if avoidable.

8. Wait for:

```text
Collection 40 / 40
collection_done:session=pc2_test_YYYYMMDDTHHMMSS
state:IDLE
```

9. Press `Train Model` in Unity.

Expected service log:

```text
cmd: train {'session': 'pc2_test_YYYYMMDDTHHMMSS', 'decoder': 'csp_lda'}
status: training_done:session=pc2_test_YYYYMMDDTHHMMSS,decoder=csp_lda,acc=...
```

10. Confirm the bundle exists on PC1:

```bash
ls bundles/pc2_test_YYYYMMDDTHHMMSS/csp_lda/bundle.joblib
```

## Pass Criteria

Phase 8.3 passes if:

- Collection reaches `collection_done:session=...`.
- `training_done:session=...,decoder=csp_lda,acc=...` appears.
- `training_done` accuracy is `0.60` or higher.

Example pass:

```text
status: collection_done:session=pc2_test_YYYYMMDDTHHMMSS
status: training_done:session=pc2_test_YYYYMMDDTHHMMSS,decoder=csp_lda,acc=0.650
status: bundles_available:pc2_test_YYYYMMDDTHHMMSS/csp_lda:0.650
```

## If Accuracy Is Below 0.60

This is not a code failure. Move to Phase 8.4.

Record these values:

- `training_done:...acc=...`
- headset/contact quality
- whether the selected channels looked clean in `realTimePlotEmotivLSL.py`
- whether you moved, blinked, or adjusted the headset during trials

Then try one improvement at a time:

1. Re-seat the headset/electrodes and improve contact quality.
2. Repeat calibration once with the same settings.
3. If still low, try calmer, more consistent motor imagery.
4. If still low, ask Codex to adjust calibration settings, for example longer
   MI duration or a different bandpass.

Do not change several things at once, because then we cannot tell what helped.

## After The Test

Stop running processes with `Ctrl + C`.

Check git:

```powershell
git status --short
```

Usually, `data/`, `bundles/`, and `logs/` should not be committed.

If source files changed intentionally:

```powershell
git add <changed-source-files>
git commit -m "Describe the Phase 8 change"
git push origin main
```

Send Codex:

- `collection_done:session=...`
- `training_done:...acc=...`
- `git status --short`
- any red Unity Console errors, if present
