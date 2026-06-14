# Unity BCI UI

Expected Unity project path:

```text
unity/MotorImageryUnity
```

The main scene is:

```text
unity/MotorImageryUnity/Assets/MainMenu.unity
```

For the two-PC Emotiv flow, see:

```text
docs/phase8_two_pc_instructions.md
```

## Fresh PC Setup

This repo is meant to be portable across PCs. Commit Unity source/config files,
then let each PC regenerate its own `Library`, `Temp`, `UserSettings`,
`.csproj`, and `.slnx` files.

1. Install Unity `6000.4.5f1` or open the project with a compatible Unity 6
   editor.
2. Clone or pull this repo.
3. Open `unity/MotorImageryUnity` in Unity Hub.
4. Wait for Package Manager to restore packages from `Packages/manifest.json`
   and `Packages/packages-lock.json`.
5. Confirm LSL4Unity is installed. If Package Manager did not restore it, add:
   `https://github.com/labstreaminglayer/LSL4Unity.git`
6. Open `Assets/MainMenu.unity`.
7. Confirm `BciLslPanel` is attached to `Canvas`.
8. Press Play once and check that the Unity Console has no red compile errors.

Python and conda are not required to edit or run the Unity scene by itself.
They are required on whichever PC runs `apps.service` or the fake LSL scripts.

## Script Roles

The runtime UI is scene-authored. The scripts should not create replacement UI
at Play time.

- `BciLslPanel.cs`: thin wiring component on the Canvas. It resolves scene UI
  references and passes them to the focused controllers.
- `BciLslStreams.cs`: LSL transport. It publishes `Commands` and `Markers`,
  and subscribes to `Status` and `BCI_Proba`.
- `BciScreenNavigator.cs`: screen switching between Main Menu, Calibration,
  and Realtime panels.
- `BciCalibrationController.cs`: subject ID, `Acquire Data`, marker
  sequence, collection progress, warnings, `Train Model`, and `Shutdown`.
- `BciRealtimeController.cs`: bundle load-on-start, `Start Realtime`,
  `Stop Realtime`, probability text, and the realtime probability indicator.

## Editing UI

The editable UI lives in `Assets/MainMenu.unity`.

- Edit layout in Unity Edit Mode, not Play Mode.
- `CalibrationPanel` and `RealtimePanel` are scene objects. If a panel is
  inactive, select it in the Hierarchy and enable it temporarily in the
  Inspector while editing.
- Save the scene after layout changes.
- Commit `Assets/MainMenu.unity` when the layout should be shared with other
  PCs.

If the editable UI needs to be rebuilt from the project default, use:

```text
Motor Imagery > Rebuild Editable Main Menu UI
```

That rebuild action recreates the panels, so use it only when you are okay with
overwriting manual layout edits.

## Smoke Tests

Run Python commands from the repository root in a `MotorImageryDemo` conda
environment.

### BCI_Proba Display

Start Unity Play, then run:

```powershell
python scripts/dev_lsl_loopback.py --duration-s 30
```

Expected Unity Console messages include:

```text
Connected to BCI_Proba stream: LoopbackProba
```

Expected UI:

```text
p_left: 0.xxx
p_right: 0.xxx
```

### Commands Button

Terminal 1:

```powershell
python scripts/dev_fake_lsl_stream.py --mode alternating
```

Terminal 2:

```powershell
python scripts/dev_fake_markers.py --trials-per-class 2 --cue-s 10 --mi-s 1.5 --rest-s 0.4
```

Start Unity Play.

Terminal 3:

```powershell
python -m apps.service --channels FC5 FC6
```

When Unity shows `state:IDLE`, press `Shutdown`. The service should log
`cmd: shutdown {}` and then `service stopped`.

### Calibration

`BciLslStreams` publishes a Unity `Markers` stream while Play is running. Enter
a Subject ID on the Calibration screen, then press `Acquire Data`.

Unity sends:

```text
start_collection:subject=S001
```

Use simple IDs such as `S001` or `subject_001`; spaces are converted to
underscores.

Unity then publishes balanced left/right cue markers plus rest markers:

- left cue: `0`
- right cue: `1`
- rest: `99`

During data collection, the service should report `state:COLLECTING`,
`collection_progress:x/40`, then `collection_done:session=...` and
`state:IDLE`. `BciCalibrationController` mirrors `collection_progress:x/40`
into a progress bar and `Collection x / 40` text. `error:*` messages are shown
in red in the warning text.

When collection is done, press `Train Model`. Unity sends:

```text
train:session=S001_YYYYMMDDTHHMMSS,decoder=csp_lda
```

Expected service status:

```text
training_done:session=S001_YYYYMMDDTHHMMSS,decoder=csp_lda,acc=...
bundles_available:S001_YYYYMMDDTHHMMSS/csp_lda:...
```

For a quick progress UI check without running a full calibration, enter Play
mode and run:

```powershell
python scripts/dev_status_progress.py
```

Expected Unity display:

- `Collection 5 / 40` fills the progress bar to about 12.5%.
- `error:test_warning` appears in red.

### Realtime

After training completes, press `Start Realtime`. If the service is still
`state:IDLE`, Unity first sends:

```text
load_bundle:session=S001_YYYYMMDDTHHMMSS,decoder=csp_lda
```

Once the service reports `state:READY`, Unity starts realtime automatically.

Expected service status:

```text
bundle_loaded:session=...,decoder=csp_lda
state:RUNNING
realtime_warming_up
realtime_ready
now_running:session=...,decoder=csp_lda,acc=...
```

Expected Unity display:

- `p_left` / `p_right` update continuously.
- The probability indicator moves left or right with the stronger class.

Press `Stop Realtime` to send `stop_realtime` and return the service to
`state:READY`.

## Two-PC Notes

For PC1 Emotiv plus PC2 Unity testing, keep the detailed checklist in
`docs/phase8_two_pc_instructions.md`. That file includes the expected command
order, channel selection examples, and the firewall/LSL troubleshooting path.

## What To Commit

Keep these in git:

- `Assets/`
- `Packages/manifest.json`
- `Packages/packages-lock.json`
- `ProjectSettings/`
- `docs/`

Do not commit generated/local files:

- `Library/`
- `Temp/`
- `UserSettings/`
- `Logs/`
- `.csproj`
- `.sln`
- `.slnx`
- `.vscode/`
