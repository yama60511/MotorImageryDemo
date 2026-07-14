using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class BciCalibrationController : MonoBehaviour
{
    public enum CalibrationEditorPreviewView
    {
        Menu,
        AcquireData,
        AcquireCue,
        TrainModel,
        CheckPerformance
    }

    public RectTransform calibrationPanel;
    public BciScreenNavigator screenNavigator;

    public TMP_Text cueText;
    public Button startCalibrationButton;
    public Button shutdownButton;
    public Button saveBundleButton;
    public TMP_InputField subjectIdInput;
    public RectTransform calibrationProgressTrack;
    public RectTransform calibrationProgressFill;
    public TMP_Text calibrationProgressText;
    public TMP_Text warningText;

    public Color warningColor = new Color(1.0f, 0.25f, 0.25f);
    public Color leftCueColor = new Color(0.35f, 0.85f, 1.0f);
    public Color rightCueColor = new Color(1.0f, 0.55f, 0.35f);
    public Color restCueColor = new Color(0.8f, 0.8f, 0.8f);
    public Color fixationCueColor = new Color(0.85f, 0.85f, 0.85f);
    public Color readyCueColor = Color.white;
    public Color realtimeCueColor = new Color(0.45f, 1.0f, 0.55f);

    public int trialsPerClass = 20;
    public float cueSeconds = 1.0f;
    public float motorImagerySeconds = 4.0f;
    public float restSeconds = 2.0f;
    public float fixationSeconds = 2.0f;
    // Trial phase markers. Motor-imagery onset (leftMarker/rightMarker)
    // stays 0/1 because those are the labels the decoder trains on; the
    // fixation/cue markers use distinct integers the service ignores for
    // training but still streams so they can be recovered offline.
    public int leftMarker = 0;         // left motor-imagery onset (GO)
    public int rightMarker = 1;        // right motor-imagery onset (GO)
    public int cueLeftMarker = 10;     // left cue (READY)
    public int cueRightMarker = 11;    // right cue (READY)
    public int fixationMarker = 98;    // fixation cross
    public int restMarker = 99;
    public bool shuffleCalibrationTrials = true;
    // When true, each collection run draws a fresh seed so the left/right
    // order differs every session. When false, trialShuffleSeed is used for a
    // reproducible order (useful for demos/tests). The effective seed is always
    // logged so any session's order can be reconstructed offline.
    public bool randomizeSeedPerSession = true;
    public int trialShuffleSeed = 1;
    public string decoderName = "csp_lda";
    public float serviceConfigWaitSeconds = 3.0f;
    public CalibrationEditorPreviewView editorPreviewView = CalibrationEditorPreviewView.Menu;

    private BciLslStreams streams;
    private string serviceState = "DISCONNECTED";
    private string currentSubjectId = "pc2_test";
    private string selectedSessionId = "";
    private string pendingUseSession = "";
    private string pendingUseDecoder = "";
    private readonly List<string> availableSessions = new List<string>();
    private readonly List<string> availableDecoders = new List<string>();
    private readonly Dictionary<string, Toggle> modelToggles = new Dictionary<string, Toggle>();
    private readonly List<PerformanceRecord> performanceRecords = new List<PerformanceRecord>();
    private int calibrationProgressCurrent;
    private int calibrationProgressTotal = 40;
    private Coroutine calibrationRoutine;
    private bool initialized;
    private bool serviceConfigReceived;

    private RectTransform workflowRoot;
    private RectTransform menuView;
    private RectTransform acquireSetupView;
    private RectTransform acquireCueView;
    private RectTransform trainView;
    private RectTransform performanceView;
    private TMP_InputField acquireSubjectInput;
    private TMP_Text cueTaskText;
    private TMP_Text acquireProgressText;
    private RectTransform acquireProgressTrack;
    private RectTransform acquireProgressFill;
    private Button acquireStartButton;
    private Button menuBackButton;
    private Button trainSelectedModelsButton;
    private Button showPerformanceButton;
    private Button trainSessionDropdownButton;
    private TMP_Text trainSessionDropdownLabel;
    private RectTransform trainSessionDropdownList;
    private RectTransform trainSessionDropdownContent;
    private RectTransform trainModelList;
    private TMP_Text performanceBestText;
    private RectTransform performanceList;
    private TMP_Text performanceDetailsText;
    private TMP_Text localWarningText;
    private bool trainSessionDropdownOpen;
    private string selectedPerformanceSessionId = "";
    private Button performanceSessionDropdownButton;
    private TMP_Text performanceSessionDropdownLabel;
    private RectTransform performanceSessionDropdownList;
    private RectTransform performanceSessionDropdownContent;
    private bool performanceSessionDropdownOpen;

    private const float FallbackCalibrationTrackWidth = 420.0f;
    private static readonly Vector2 BottomBackButtonPosition = new Vector2(0.0f, -345.0f);

    private class PerformanceRecord
    {
        public string session = "";
        public string decoder = "";
        public string acc = "";
        public string foldMin = "";
        public string foldMax = "";
        public string folds = "";
        public string trials = "";
        public string trainedAt = "";
        public string confusion = "";
        public string channels = "";
        public string windowSamples = "";
        public string bundlePath = "";
        public string rawDataPath = "";
        public string bandpass = "";
    }

    public void Initialize(BciLslStreams lslStreams)
    {
        if (initialized)
        {
            return;
        }

        streams = lslStreams;
        if (streams != null)
        {
            streams.StatusReceived += HandleStatusMessage;
        }

        if (calibrationPanel == null)
        {
            calibrationPanel = ResolveCalibrationPanel();
        }

        HideLegacyControls();
        BuildWorkflowUi();

        initialized = true;
        ShowMenu();
        UpdateCollectionProgress(0, 40);
        UpdateControlInteractivity();
    }

    public void BuildEditorPreview(RectTransform previewPanel)
    {
        if (Application.isPlaying)
        {
            return;
        }

        if (previewPanel != null)
        {
            calibrationPanel = previewPanel;
        }
        if (calibrationPanel == null)
        {
            calibrationPanel = ResolveCalibrationPanel();
        }

        HideLegacyControls();
        BuildWorkflowUi();
        ShowEditorPreviewView();
        UpdateCollectionProgress(0, trialsPerClass * 2);
        UpdateControlInteractivity();
    }

    private RectTransform ResolveCalibrationPanel()
    {
        if (subjectIdInput != null)
        {
            return subjectIdInput.GetComponentInParent<RectTransform>();
        }
        if (startCalibrationButton != null)
        {
            return startCalibrationButton.GetComponentInParent<RectTransform>();
        }
        return GetComponent<RectTransform>();
    }

    private void HideLegacyControls()
    {
        SetActive(subjectIdInput, false);
        SetActive(startCalibrationButton, false);
        SetActive(saveBundleButton, false);
        SetActive(calibrationProgressTrack, false);
        SetActive(calibrationProgressFill, false);
        SetActive(calibrationProgressText, false);
        SetActive(warningText, false);
        SetActive(cueText, false);
        SetLegacyChildActive("CalibrationTitle", false);
        SetLegacyChildActive("SubjectIdLabel", false);
        SetLegacyChildActive("BackToMenuFromCalibrationButton", false);
    }

    private void BuildWorkflowUi()
    {
        if (calibrationPanel == null)
        {
            return;
        }

        Transform existingWorkflowRoot = FindChildByName(calibrationPanel, "CalibrationWorkflowRoot");
        if (existingWorkflowRoot != null)
        {
            DestroyGameObject(existingWorkflowRoot.gameObject);
        }

        workflowRoot = CreatePanel(calibrationPanel, "CalibrationWorkflowRoot", true);
        workflowRoot.anchorMin = Vector2.zero;
        workflowRoot.anchorMax = Vector2.one;
        workflowRoot.offsetMin = Vector2.zero;
        workflowRoot.offsetMax = Vector2.zero;

        menuView = CreatePanel(workflowRoot, "CalibrationMenuView", true);
        acquireSetupView = CreatePanel(workflowRoot, "AcquireSetupView", false);
        acquireCueView = CreatePanel(workflowRoot, "AcquireCueView", false);
        trainView = CreatePanel(workflowRoot, "TrainModelView", false);
        performanceView = CreatePanel(workflowRoot, "CheckPerformanceView", false);

        BuildMenuView();
        BuildAcquireViews();
        BuildTrainView();
        BuildPerformanceView();
    }

    private void BuildMenuView()
    {
        CreateText(menuView, "CalibrationMenuTitle", "Calibration", 46.0f, new Vector2(0.0f, 235.0f), new Vector2(720.0f, 66.0f));
        CreateButton(menuView, "AcquireDataMenuButton", "Acquire Data", new Vector2(0.0f, 120.0f), ShowAcquireSetup, new Vector2(300.0f, 66.0f), 24.0f);
        CreateButton(menuView, "TrainModelMenuButton", "Train Model", new Vector2(0.0f, 35.0f), ShowTrainModel, new Vector2(300.0f, 66.0f), 24.0f);
        CreateButton(menuView, "CheckPerformanceMenuButton", "Check Performance", new Vector2(0.0f, -50.0f), ShowPerformance, new Vector2(300.0f, 66.0f), 24.0f);
        menuBackButton = CreateButton(menuView, "CalibrationMenuBackButton", "Back", BottomBackButtonPosition, ShowMainMenu);
    }

    private void BuildAcquireViews()
    {
        CreateText(acquireSetupView, "AcquireTitle", "Acquire Data", 44.0f, new Vector2(0.0f, 245.0f), new Vector2(720.0f, 62.0f));
        CreateText(acquireSetupView, "AcquireSubjectLabel", "Subject ID", 24.0f, new Vector2(-310.0f, 145.0f), new Vector2(170.0f, 40.0f));
        acquireSubjectInput = CreateInputField(acquireSetupView, "AcquireSubjectInput", currentSubjectId, new Vector2(-55.0f, 145.0f), new Vector2(360.0f, 54.0f));
        acquireSubjectInput.onEndEdit.AddListener(value => currentSubjectId = SanitizeSubjectId(value));
        acquireStartButton = CreateButton(acquireSetupView, "AcquireStartButton", "Start", new Vector2(250.0f, 145.0f), StartCollection, new Vector2(150.0f, 54.0f), 22.0f);
        CreateButton(acquireSetupView, "AcquireBackButton", "Back", BottomBackButtonPosition, ShowMenu);

        CreateText(acquireCueView, "AcquireCueTitle", "Acquire Data", 34.0f, new Vector2(0.0f, 300.0f), new Vector2(720.0f, 50.0f));
        acquireProgressText = CreateText(acquireCueView, "AcquireProgressText", "Collection 0 / 40", 26.0f, new Vector2(0.0f, -195.0f), new Vector2(620.0f, 42.0f));
        acquireProgressTrack = CreateBarTrack(acquireCueView, "AcquireProgressTrack", new Vector2(560.0f, 24.0f), new Vector2(0.0f, -230.0f));
        acquireProgressFill = CreateBarFill(acquireProgressTrack, "AcquireProgressFill", leftCueColor, new Vector2(0.0f, 24.0f));
        cueTaskText = CreateText(acquireCueView, "AcquireCueText", "", 86.0f, new Vector2(0.0f, 45.0f), new Vector2(860.0f, 150.0f));
        cueTaskText.fontStyle = FontStyles.Bold;
        CreateButton(acquireCueView, "AbortCollectionButton", "Abort", new Vector2(0.0f, -295.0f), AbortCollection, new Vector2(260.0f, 60.0f), 22.0f);
    }

    private void BuildTrainView()
    {
        CreateText(trainView, "TrainTitle", "Train Model", 44.0f, new Vector2(0.0f, 315.0f), new Vector2(720.0f, 62.0f));
        CreateText(trainView, "TrainSessionLabel", "EEG Data", 22.0f, new Vector2(-430.0f, 255.0f), new Vector2(150.0f, 36.0f));
        trainSessionDropdownButton = CreateButton(trainView, "TrainSessionDropdownButton", "No EEG Data", new Vector2(30.0f, 255.0f), ToggleTrainSessionDropdown, new Vector2(620.0f, 52.0f), 18.0f);
        trainSessionDropdownLabel = trainSessionDropdownButton.GetComponentInChildren<TMP_Text>();
        if (trainSessionDropdownLabel != null)
        {
            trainSessionDropdownLabel.alignment = TextAlignmentOptions.MidlineLeft;
        }
        trainSessionDropdownList = CreateScrollPanel(trainView, "TrainSessionDropdownList", new Vector2(30.0f, 125.0f), new Vector2(620.0f, 170.0f), out trainSessionDropdownContent);
        trainSessionDropdownList.gameObject.SetActive(false);
        trainModelList = CreatePanel(trainView, "TrainModelList", true);
        trainModelList.sizeDelta = new Vector2(820.0f, 190.0f);
        trainModelList.anchoredPosition = new Vector2(0.0f, 65.0f);
        trainSelectedModelsButton = CreateButton(trainView, "TrainSelectedModelsButton", "Train Selected Models", new Vector2(0.0f, -80.0f), TrainSelectedModels, new Vector2(320.0f, 62.0f), 22.0f);
        CreateButton(trainView, "TrainBackButton", "Back", BottomBackButtonPosition, ShowMenu);
        RebuildTrainSessionList();
        RebuildModelToggleList();
    }

    private void BuildPerformanceView()
    {
        CreateText(performanceView, "PerformanceTitle", "Check Performance", 44.0f, new Vector2(0.0f, 320.0f), new Vector2(760.0f, 62.0f));
        CreateText(performanceView, "PerformanceSessionLabel", "EEG Data", 22.0f, new Vector2(-430.0f, 265.0f), new Vector2(150.0f, 36.0f));
        performanceSessionDropdownButton = CreateButton(performanceView, "PerformanceSessionDropdownButton", "No EEG Data", new Vector2(-80.0f, 265.0f), TogglePerformanceSessionDropdown, new Vector2(480.0f, 48.0f), 18.0f);
        performanceSessionDropdownLabel = performanceSessionDropdownButton.GetComponentInChildren<TMP_Text>();
        if (performanceSessionDropdownLabel != null)
        {
            performanceSessionDropdownLabel.alignment = TextAlignmentOptions.MidlineLeft;
        }
        performanceSessionDropdownList = CreateScrollPanel(performanceView, "PerformanceSessionDropdownList", new Vector2(-80.0f, 140.0f), new Vector2(480.0f, 165.0f), out performanceSessionDropdownContent);
        performanceSessionDropdownList.gameObject.SetActive(false);
        showPerformanceButton = CreateButton(performanceView, "ShowPerformanceButton", "Show Results", new Vector2(310.0f, 265.0f), QueryPerformance, new Vector2(220.0f, 48.0f), 18.0f);
        performanceBestText = CreateText(performanceView, "PerformanceBestText", "Best: -", 22.0f, new Vector2(0.0f, 218.0f), new Vector2(820.0f, 38.0f));
        performanceList = CreatePanel(performanceView, "PerformanceList", true);
        performanceList.sizeDelta = new Vector2(900.0f, 300.0f);
        performanceList.anchoredPosition = new Vector2(0.0f, 55.0f);
        performanceDetailsText = CreateText(performanceView, "PerformanceDetailsText", "", 17.0f, new Vector2(0.0f, -220.0f), new Vector2(900.0f, 150.0f));
        performanceDetailsText.alignment = TextAlignmentOptions.TopLeft;
        CreateButton(performanceView, "PerformanceBackButton", "Back", BottomBackButtonPosition, ShowMenu);
    }

    private void ShowMenu()
    {
        ShowOnly(menuView);
        UpdateControlInteractivity();
    }

    private void ShowAcquireSetup()
    {
        ShowOnly(acquireSetupView);
        if (acquireSubjectInput != null)
        {
            acquireSubjectInput.SetTextWithoutNotify(currentSubjectId);
        }
        UpdateControlInteractivity();
    }

    private void ShowAcquireCue()
    {
        ShowOnly(acquireCueView);
        UpdateControlInteractivity();
    }

    private void ShowTrainModel()
    {
        ShowOnly(trainView);
        streams?.PushCommand("query_config");
        RebuildTrainSessionList();
        RebuildModelToggleList();
        UpdateControlInteractivity();
    }

    private void ShowPerformance()
    {
        ShowOnly(performanceView);
        streams?.PushCommand("query_config");
        RebuildPerformanceSessionList();
        RebuildPerformanceList();
        UpdateControlInteractivity();
    }

    private void ShowMainMenu()
    {
        screenNavigator?.ShowMainMenu();
    }

    private void ShowOnly(RectTransform view)
    {
        SetActive(menuView, view == menuView);
        SetActive(acquireSetupView, view == acquireSetupView);
        SetActive(acquireCueView, view == acquireCueView);
        SetActive(trainView, view == trainView);
        SetActive(performanceView, view == performanceView);
        SetWarningText("");
    }

    private void ShowEditorPreviewView()
    {
        switch (editorPreviewView)
        {
            case CalibrationEditorPreviewView.AcquireData:
                ShowOnly(acquireSetupView);
                break;
            case CalibrationEditorPreviewView.AcquireCue:
                ShowOnly(acquireCueView);
                SetCueText("LEFT", leftCueColor);
                break;
            case CalibrationEditorPreviewView.TrainModel:
                ShowOnly(trainView);
                RebuildTrainSessionList();
                RebuildModelToggleList();
                break;
            case CalibrationEditorPreviewView.CheckPerformance:
                ShowOnly(performanceView);
                RebuildPerformanceSessionList();
                RebuildPerformanceList();
                break;
            default:
                ShowOnly(menuView);
                break;
        }
    }

    private void StartCollection()
    {
        if (!IsIdle())
        {
            SetWarningText($"Acquire ignored while service state is {serviceState}");
            UpdateControlInteractivity();
            return;
        }
        if (calibrationRoutine != null)
        {
            Debug.LogWarning("Collection marker sequence is already running.");
            return;
        }

        currentSubjectId = GetSubjectIdFromInput(acquireSubjectInput);
        selectedSessionId = "";
        streams?.PushCommand($"start_collection:subject={currentSubjectId}");
        serviceState = "COLLECTING";
        UpdateCollectionProgress(0, trialsPerClass * 2);
        UpdateControlInteractivity();
        SetCueText("GET READY", readyCueColor);
        ShowAcquireCue();
        calibrationRoutine = StartCoroutine(PublishCollectionMarkers());
    }

    private void AbortCollection()
    {
        StopLocalCollection("ABORTED");
        streams?.PushCommand("abort_collection");
        serviceState = "IDLE";
        UpdateControlInteractivity();
    }

    private void Shutdown()
    {
        if (calibrationRoutine != null)
        {
            StopCoroutine(calibrationRoutine);
            calibrationRoutine = null;
        }

        SetCueText("SHUTDOWN", restCueColor);
        streams?.PushCommand("shutdown");
    }

    private void StopLocalCollection(string cueLabel)
    {
        if (calibrationRoutine != null)
        {
            StopCoroutine(calibrationRoutine);
            calibrationRoutine = null;
        }
        SetCueText(cueLabel, warningColor);
    }

    private void TrainSelectedModels()
    {
        if (!IsIdle())
        {
            SetWarningText($"Train ignored while service state is {serviceState}");
            return;
        }
        if (string.IsNullOrEmpty(selectedSessionId))
        {
            SetWarningText("Select EEG data first.");
            return;
        }

        var selectedModels = modelToggles
            .Where(pair => pair.Value != null && pair.Value.isOn)
            .Select(pair => pair.Key)
            .ToList();
        if (selectedModels.Count == 0)
        {
            SetWarningText("Select at least one model.");
            return;
        }

        string decoders = string.Join("|", selectedModels);
        streams?.PushCommand($"train_many:session={selectedSessionId},decoders={decoders}");
        serviceState = "TRAINING";
        SetWarningText("");
        UpdateControlInteractivity();
    }

    private void QueryPerformance()
    {
        if (!HasServiceState())
        {
            SetWarningText("Connect to the Python service first.");
            UpdateControlInteractivity();
            return;
        }
        if (string.IsNullOrEmpty(selectedPerformanceSessionId))
        {
            SetWarningText("Select EEG data first.");
            UpdateControlInteractivity();
            return;
        }

        currentSubjectId = SubjectFromSessionId(selectedPerformanceSessionId);
        performanceRecords.Clear();
        RebuildPerformanceList();
        performanceBestText.text = "Best: loading...";
        streams?.PushCommand($"query_performance:subject={currentSubjectId}");
    }

    private IEnumerator PublishCollectionMarkers()
    {
        float waitStart = Time.realtimeSinceStartup;
        while (!serviceConfigReceived
               && Time.realtimeSinceStartup - waitStart < serviceConfigWaitSeconds)
        {
            yield return null;
        }
        if (!serviceConfigReceived)
        {
            Debug.LogWarning(
                $"service_config not received within {serviceConfigWaitSeconds}s; " +
                $"using Inspector defaults (fixation={fixationSeconds}s, cue={cueSeconds}s, " +
                $"mi={motorImagerySeconds}s, rest={restSeconds}s, trials/class={trialsPerClass})."
            );
        }

        int seed = randomizeSeedPerSession
            ? unchecked((int)System.DateTime.Now.Ticks)
            : trialShuffleSeed;
        var labels = BuildCollectionLabels(seed);
        Debug.Log(
            $"Collection marker sequence started: {labels.Count} MI-onset markers "
            + $"(shuffle={shuffleCalibrationTrials}, seed={seed})");

        for (int index = 0; index < labels.Count; index++)
        {
            int label = labels[index];
            bool isLeft = label == leftMarker;
            Color sideColor = isLeft ? leftCueColor : rightCueColor;
            string sideName = isLeft ? "LEFT" : "RIGHT";
            int cueMarker = isLeft ? cueLeftMarker : cueRightMarker;

            // Every trial follows the same structure:
            //   rest -> fixation -> cue (READY) -> motor imagery (GO).
            // A distinct marker is streamed at each phase; the cue and MI
            // markers encode the side so left/right are separable offline.
            SetCueText("REST", restCueColor);
            streams?.PushMarker(restMarker);
            Debug.Log($"Collection trial {index + 1}/{labels.Count} rest: marker={restMarker}");
            yield return new WaitForSecondsRealtime(restSeconds);

            SetCueText("+", fixationCueColor);
            streams?.PushMarker(fixationMarker);
            Debug.Log($"Collection trial {index + 1}/{labels.Count} fixation: marker={fixationMarker}");
            yield return new WaitForSecondsRealtime(fixationSeconds);

            SetCueText($"{sideName}\nREADY", sideColor);
            streams?.PushMarker(cueMarker);
            Debug.Log($"Collection trial {index + 1}/{labels.Count} cue: marker={cueMarker}");
            yield return new WaitForSecondsRealtime(cueSeconds);

            SetCueText($"{sideName}\nGO", sideColor);
            streams?.PushMarker(label);
            Debug.Log($"Collection trial {index + 1}/{labels.Count} MI onset: marker={label}");
            yield return new WaitForSecondsRealtime(motorImagerySeconds);
        }

        Debug.Log("Collection marker sequence complete.");
        SetCueText("WAITING", readyCueColor);
        calibrationRoutine = null;
        UpdateControlInteractivity();
    }

    private List<int> BuildCollectionLabels(int seed)
    {
        int nPerClass = Mathf.Max(1, trialsPerClass);
        var labels = new List<int>(nPerClass * 2);
        for (int i = 0; i < nPerClass; i++)
        {
            labels.Add(leftMarker);
            labels.Add(rightMarker);
        }

        if (shuffleCalibrationTrials)
        {
            var rng = new System.Random(seed);
            for (int i = labels.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                int tmp = labels[i];
                labels[i] = labels[j];
                labels[j] = tmp;
            }
        }

        return labels;
    }

    private void HandleStatusMessage(string message)
    {
        if (message.StartsWith("state:"))
        {
            serviceState = message.Substring("state:".Length).Trim();
            if (serviceState == "DISCONNECTED")
            {
                StopLocalCollection("DISCONNECTED");
                pendingUseSession = "";
                pendingUseDecoder = "";
            }
            UpdateControlInteractivity();
        }
        else if (message.StartsWith("service_config:"))
        {
            ParseServiceConfig(message);
        }
        else if (message.StartsWith("decoders_available:"))
        {
            ParseSimpleList(message, "decoders_available:", availableDecoders);
            RebuildModelToggleList();
        }
        else if (message.StartsWith("sessions_available:"))
        {
            ParseSimpleList(message, "sessions_available:", availableSessions);
            if (string.IsNullOrEmpty(selectedSessionId) && availableSessions.Count > 0)
            {
                selectedSessionId = availableSessions[availableSessions.Count - 1];
            }
            RebuildTrainSessionList();
            RebuildPerformanceSessionList();
        }
        else if (message.StartsWith("collection_progress:"))
        {
            SetWarningText("");
            if (TryParseProgress(message, "collection_progress:", out int current, out int total))
            {
                UpdateCollectionProgress(current, total);
            }
        }
        else if (message.StartsWith("collection_done:"))
        {
            SetWarningText("");
            if (TryGetStatusValue(message, "session", out string session))
            {
                selectedSessionId = session;
                if (!availableSessions.Contains(session))
                {
                    availableSessions.Add(session);
                }
            }
            UpdateCollectionProgress(calibrationProgressTotal, calibrationProgressTotal);
            SetCueText("", readyCueColor);
            RebuildTrainSessionList();
            RebuildPerformanceSessionList();
            ShowAcquireSetup();
        }
        else if (message.StartsWith("training_started:"))
        {
            SetWarningText("");
        }
        else if (message.StartsWith("training_done:"))
        {
            if (TryGetStatusValue(message, "session", out string session))
            {
                selectedSessionId = session;
            }
            UpdateControlInteractivity();
        }
        else if (message.StartsWith("training_failed:"))
        {
            string decoder = TryGetStatusValue(message, "decoder", out string d) ? d : "?";
            string reason = TryGetStatusValue(message, "reason", out string r) ? r : "failed";
            SetWarningText($"{decoder}: failed {reason}");
        }
        else if (message.StartsWith("performance_best:"))
        {
            if (IsSelectedPerformanceSession(message))
            {
                HandlePerformanceBest(message);
            }
        }
        else if (message.StartsWith("performance:"))
        {
            PerformanceRecord record = ParsePerformanceRecord(message);
            if (string.IsNullOrEmpty(selectedPerformanceSessionId)
                || record.session == selectedPerformanceSessionId)
            {
                performanceRecords.Add(record);
                RebuildPerformanceList();
            }
        }
        else if (message.StartsWith("performance_done:"))
        {
            if (performanceRecords.Count == 0)
            {
                performanceBestText.text = "Best: no trained models";
            }
            else
            {
                UpdateBestPerformanceFromRecords();
            }
            RebuildPerformanceList();
        }
        else if (message.StartsWith("bundle_loaded:"))
        {
            if (TryGetStatusValue(message, "session", out string session)
                && TryGetStatusValue(message, "decoder", out string decoder)
                && session == pendingUseSession
                && decoder == pendingUseDecoder)
            {
                pendingUseSession = "";
                pendingUseDecoder = "";
                if (performanceDetailsText != null)
                {
                    performanceDetailsText.text = $"Selected for Realtime: {session} / {decoder}";
                }
            }
        }
        else if (message.StartsWith("error:"))
        {
            SetWarningText(message);
            pendingUseSession = "";
            pendingUseDecoder = "";
            UpdateControlInteractivity();
        }
    }

    private void HandlePerformanceBest(string message)
    {
        string session = TryGetStatusValue(message, "session", out string s) ? s : "?";
        string decoder = TryGetStatusValue(message, "decoder", out string d) ? d : "?";
        string acc = TryGetStatusValue(message, "acc", out string a) ? a : "?";
        performanceBestText.text = $"Best: {session} / {decoder}  CV Acc {acc}";
    }

    private void UpdateBestPerformanceFromRecords()
    {
        PerformanceRecord best = null;
        float bestAcc = float.NegativeInfinity;
        foreach (PerformanceRecord record in performanceRecords)
        {
            if (!float.TryParse(record.acc, NumberStyles.Float, CultureInfo.InvariantCulture, out float acc))
            {
                acc = float.NegativeInfinity;
            }
            if (best == null || acc > bestAcc)
            {
                best = record;
                bestAcc = acc;
            }
        }

        if (best != null)
        {
            performanceBestText.text = $"Best: {best.session} / {best.decoder}  CV Acc {best.acc}";
        }
    }

    private bool IsSelectedPerformanceSession(string message)
    {
        return string.IsNullOrEmpty(selectedPerformanceSessionId)
            || (TryGetStatusValue(message, "session", out string session)
                && session == selectedPerformanceSessionId);
    }

    private PerformanceRecord ParsePerformanceRecord(string message)
    {
        return new PerformanceRecord
        {
            session = GetStatusValueOrEmpty(message, "session"),
            decoder = GetStatusValueOrEmpty(message, "decoder"),
            acc = GetStatusValueOrEmpty(message, "acc"),
            foldMin = GetStatusValueOrEmpty(message, "fold_min"),
            foldMax = GetStatusValueOrEmpty(message, "fold_max"),
            folds = GetStatusValueOrEmpty(message, "folds"),
            trials = GetStatusValueOrEmpty(message, "trials"),
            trainedAt = GetStatusValueOrEmpty(message, "trained_at"),
            confusion = GetStatusValueOrEmpty(message, "confusion"),
            channels = GetStatusValueOrEmpty(message, "channels"),
            windowSamples = GetStatusValueOrEmpty(message, "window_samples"),
            bundlePath = GetStatusValueOrEmpty(message, "bundle_path"),
            rawDataPath = GetStatusValueOrEmpty(message, "raw_data_path"),
            bandpass = GetStatusValueOrEmpty(message, "bandpass")
        };
    }

    private void UsePerformanceRecord(PerformanceRecord record)
    {
        if (record == null || string.IsNullOrEmpty(record.session) || string.IsNullOrEmpty(record.decoder))
        {
            return;
        }

        pendingUseSession = record.session;
        pendingUseDecoder = record.decoder;
        streams?.PushCommand($"load_bundle:session={record.session},decoder={record.decoder}");
        SetWarningText("");
        performanceDetailsText.text = $"Loading {record.session} / {record.decoder}...";
    }

    private void ShowPerformanceDetails(PerformanceRecord record)
    {
        if (record == null)
        {
            performanceDetailsText.text = "";
            return;
        }

        string folds = string.IsNullOrEmpty(record.folds)
            ? "-"
            : record.folds.Replace("|", ", ");
        string channels = string.IsNullOrEmpty(record.channels)
            ? "-"
            : record.channels.Replace("|", ", ");
        string confusion = FormatConfusion(record.confusion);
        performanceDetailsText.text =
            $"Session: {record.session}\n" +
            $"Model: {record.decoder}\n" +
            $"CV Acc: {record.acc}\n" +
            $"Fold Acc: {folds}\n" +
            $"Trials Used: {record.trials}\n" +
            $"Trained At: {record.trainedAt}\n" +
            $"Confusion: {confusion}\n" +
            $"Channels: {channels}\n" +
            $"Window Samples: {record.windowSamples}  Bandpass: {record.bandpass}\n" +
            $"Bundle: {record.bundlePath}";
    }

    private string FormatConfusion(string raw)
    {
        if (string.IsNullOrEmpty(raw))
        {
            return "-";
        }
        return raw.Replace("|", " ").Replace(";", " / ");
    }

    private void RebuildTrainSessionList()
    {
        if (trainSessionDropdownButton == null || trainSessionDropdownLabel == null)
        {
            return;
        }

        if (availableSessions.Count == 0)
        {
            selectedSessionId = "";
            trainSessionDropdownLabel.text = "No EEG Data";
            trainSessionDropdownButton.interactable = false;
            CloseTrainSessionDropdown();
            return;
        }

        var sessions = availableSessions.AsEnumerable().Reverse().ToList();
        if (string.IsNullOrEmpty(selectedSessionId) || !sessions.Contains(selectedSessionId))
        {
            selectedSessionId = sessions[0];
        }

        trainSessionDropdownLabel.text = selectedSessionId;
        trainSessionDropdownButton.interactable = IsIdle();
        if (trainSessionDropdownOpen)
        {
            RebuildTrainSessionOptions();
        }
    }

    private void ToggleTrainSessionDropdown()
    {
        if (!IsIdle() || availableSessions.Count == 0)
        {
            UpdateControlInteractivity();
            return;
        }

        trainSessionDropdownOpen = !trainSessionDropdownOpen;
        if (trainSessionDropdownList != null)
        {
            trainSessionDropdownList.gameObject.SetActive(trainSessionDropdownOpen);
            if (trainSessionDropdownOpen)
            {
                trainSessionDropdownList.SetAsLastSibling();
            }
        }
        if (trainSessionDropdownOpen)
        {
            RebuildTrainSessionOptions();
        }
    }

    private void CloseTrainSessionDropdown()
    {
        trainSessionDropdownOpen = false;
        if (trainSessionDropdownList != null)
        {
            trainSessionDropdownList.gameObject.SetActive(false);
        }
    }

    private void RebuildTrainSessionOptions()
    {
        if (trainSessionDropdownContent == null)
        {
            return;
        }

        ClearChildren(trainSessionDropdownContent);
        var sessions = availableSessions.AsEnumerable().Reverse().ToList();
        float itemHeight = 34.0f;
        trainSessionDropdownContent.sizeDelta = new Vector2(0.0f, sessions.Count * itemHeight);
        for (int i = 0; i < sessions.Count; i++)
        {
            string session = sessions[i];
            string sessionForButton = session;
            string label = session == selectedSessionId ? "* " + session : session;
            Button button = CreateButton(
                trainSessionDropdownContent,
                $"SessionOption_{i}",
                label,
                new Vector2(0.0f, -i * itemHeight - itemHeight * 0.5f),
                () =>
                {
                    selectedSessionId = sessionForButton;
                    trainSessionDropdownLabel.text = selectedSessionId;
                    CloseTrainSessionDropdown();
                    UpdateControlInteractivity();
                },
                new Vector2(520.0f, 30.0f),
                13.0f
            );
            var rect = button.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.5f, 1.0f);
            rect.anchorMax = new Vector2(0.5f, 1.0f);
            TMP_Text optionText = button.GetComponentInChildren<TMP_Text>();
            if (optionText != null)
            {
                optionText.alignment = TextAlignmentOptions.MidlineLeft;
            }
        }
    }

    private void RebuildPerformanceSessionList()
    {
        if (performanceSessionDropdownButton == null || performanceSessionDropdownLabel == null)
        {
            return;
        }

        if (availableSessions.Count == 0)
        {
            selectedPerformanceSessionId = "";
            performanceSessionDropdownLabel.text = "No EEG Data";
            performanceSessionDropdownButton.interactable = false;
            ClosePerformanceSessionDropdown();
            return;
        }

        var sessions = availableSessions.AsEnumerable().Reverse().ToList();
        if (string.IsNullOrEmpty(selectedPerformanceSessionId) || !sessions.Contains(selectedPerformanceSessionId))
        {
            selectedPerformanceSessionId = sessions[0];
        }

        performanceSessionDropdownLabel.text = selectedPerformanceSessionId;
        performanceSessionDropdownButton.interactable = HasServiceState();
        if (performanceSessionDropdownOpen)
        {
            RebuildPerformanceSessionOptions();
        }
    }

    private void TogglePerformanceSessionDropdown()
    {
        if (!HasServiceState() || availableSessions.Count == 0)
        {
            UpdateControlInteractivity();
            return;
        }

        performanceSessionDropdownOpen = !performanceSessionDropdownOpen;
        if (performanceSessionDropdownList != null)
        {
            performanceSessionDropdownList.gameObject.SetActive(performanceSessionDropdownOpen);
            if (performanceSessionDropdownOpen)
            {
                performanceSessionDropdownList.SetAsLastSibling();
            }
        }
        if (performanceSessionDropdownOpen)
        {
            RebuildPerformanceSessionOptions();
        }
    }

    private void ClosePerformanceSessionDropdown()
    {
        performanceSessionDropdownOpen = false;
        if (performanceSessionDropdownList != null)
        {
            performanceSessionDropdownList.gameObject.SetActive(false);
        }
    }

    private void RebuildPerformanceSessionOptions()
    {
        if (performanceSessionDropdownContent == null)
        {
            return;
        }

        ClearChildren(performanceSessionDropdownContent);
        var sessions = availableSessions.AsEnumerable().Reverse().ToList();
        float itemHeight = 34.0f;
        performanceSessionDropdownContent.sizeDelta = new Vector2(0.0f, sessions.Count * itemHeight);
        for (int i = 0; i < sessions.Count; i++)
        {
            string session = sessions[i];
            string sessionForButton = session;
            string label = session == selectedPerformanceSessionId ? "* " + session : session;
            Button button = CreateButton(
                performanceSessionDropdownContent,
                $"PerformanceSessionOption_{i}",
                label,
                new Vector2(0.0f, -i * itemHeight - itemHeight * 0.5f),
                () =>
                {
                    selectedPerformanceSessionId = sessionForButton;
                    performanceSessionDropdownLabel.text = selectedPerformanceSessionId;
                    ClosePerformanceSessionDropdown();
                    UpdateControlInteractivity();
                },
                new Vector2(370.0f, 30.0f),
                12.0f
            );
            var rect = button.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.5f, 1.0f);
            rect.anchorMax = new Vector2(0.5f, 1.0f);
            TMP_Text optionText = button.GetComponentInChildren<TMP_Text>();
            if (optionText != null)
            {
                optionText.alignment = TextAlignmentOptions.MidlineLeft;
            }
        }
    }

    private void RebuildModelToggleList()
    {
        if (trainModelList == null)
        {
            return;
        }
        ClearChildren(trainModelList);
        modelToggles.Clear();

        if (availableDecoders.Count == 0)
        {
            string message = HasServiceState()
                ? "No models available"
                : "Connect to Python service";
            CreateText(trainModelList, "NoModelsText", message, 22.0f, new Vector2(0.0f, 25.0f), new Vector2(620.0f, 40.0f));
            UpdateControlInteractivity();
            return;
        }

        for (int i = 0; i < availableDecoders.Count; i++)
        {
            string modelName = availableDecoders[i];
            float y = 65.0f - i * 44.0f;
            Toggle toggle = CreateToggle(trainModelList, $"ModelToggle_{modelName}", modelName, new Vector2(-170.0f, y));
            toggle.isOn = modelName == decoderName || i == 0;
            modelToggles[modelName] = toggle;
        }
    }

    private void RebuildPerformanceList()
    {
        if (performanceList == null)
        {
            return;
        }
        ClearChildren(performanceList);

        float tableWidth = 860.0f;
        float tableHeight = 250.0f;
        float top = 124.0f;
        float rowHeight = 42.0f;
        float left = -tableWidth * 0.5f;
        float[] columnLines = { left, -235.0f, -75.0f, 40.0f, 155.0f, 245.0f, tableWidth * 0.5f };
        for (int i = 0; i < columnLines.Length; i++)
        {
            CreateDivider(performanceList, $"PerformanceColumnLine_{i}", new Vector2(columnLines[i], top - tableHeight * 0.5f), new Vector2(2.0f, tableHeight));
        }

        for (int i = 0; i <= 6; i++)
        {
            float y = top - i * rowHeight;
            CreateDivider(performanceList, $"PerformanceRowLine_{i}", new Vector2(0.0f, y), new Vector2(tableWidth, 2.0f));
        }

        CreateTableText(performanceList, "PerformanceHeaderSession", "Session", 17.0f, new Vector2(-330.0f, 103.0f), new Vector2(170.0f, 30.0f));
        CreateTableText(performanceList, "PerformanceHeaderModel", "Model", 17.0f, new Vector2(-155.0f, 103.0f), new Vector2(130.0f, 30.0f));
        CreateTableText(performanceList, "PerformanceHeaderAcc", "CV Acc", 17.0f, new Vector2(-18.0f, 103.0f), new Vector2(90.0f, 30.0f));
        CreateTableText(performanceList, "PerformanceHeaderFolds", "Folds", 17.0f, new Vector2(98.0f, 103.0f), new Vector2(90.0f, 30.0f));
        CreateTableText(performanceList, "PerformanceHeaderTrials", "Trials", 17.0f, new Vector2(200.0f, 103.0f), new Vector2(76.0f, 30.0f));
        CreateTableText(performanceList, "PerformanceHeaderAction", "Action", 17.0f, new Vector2(335.0f, 103.0f), new Vector2(170.0f, 30.0f));

        int maxRows = Mathf.Min(performanceRecords.Count, 5);
        for (int i = 0; i < maxRows; i++)
        {
            PerformanceRecord record = performanceRecords[i];
            PerformanceRecord recordForButton = record;
            float y = 61.0f - i * rowHeight;
            string folds = (!string.IsNullOrEmpty(record.foldMin) && !string.IsNullOrEmpty(record.foldMax))
                ? $"{record.foldMin}-{record.foldMax}"
                : "-";
            CreateTableText(performanceList, $"PerformanceSession_{i}", record.session, 15.0f, new Vector2(-330.0f, y), new Vector2(174.0f, 30.0f), TextAlignmentOptions.MidlineLeft);
            CreateTableText(performanceList, $"PerformanceModel_{i}", record.decoder, 15.0f, new Vector2(-155.0f, y), new Vector2(130.0f, 30.0f), TextAlignmentOptions.MidlineLeft);
            CreateTableText(performanceList, $"PerformanceAcc_{i}", record.acc, 15.0f, new Vector2(-18.0f, y), new Vector2(90.0f, 30.0f));
            CreateTableText(performanceList, $"PerformanceFolds_{i}", folds, 15.0f, new Vector2(98.0f, y), new Vector2(90.0f, 30.0f));
            CreateTableText(performanceList, $"PerformanceTrials_{i}", record.trials, 15.0f, new Vector2(200.0f, y), new Vector2(76.0f, 30.0f));
            CreateButton(performanceList, $"UseButton_{i}", "Use", new Vector2(290.0f, y), () => UsePerformanceRecord(recordForButton), new Vector2(74.0f, 30.0f), 13.0f);
            CreateButton(performanceList, $"DetailsButton_{i}", "Details", new Vector2(382.0f, y), () => ShowPerformanceDetails(recordForButton), new Vector2(82.0f, 30.0f), 13.0f);
        }
    }

    private void UpdateControlInteractivity()
    {
        bool idle = IsIdle();
        bool connected = HasServiceState();
        if (acquireStartButton != null)
        {
            acquireStartButton.interactable = idle && calibrationRoutine == null;
        }
        if (trainSelectedModelsButton != null)
        {
            trainSelectedModelsButton.interactable = idle
                && !string.IsNullOrEmpty(selectedSessionId)
                && modelToggles.Count > 0;
        }
        if (trainSessionDropdownButton != null)
        {
            trainSessionDropdownButton.interactable = idle && availableSessions.Count > 0;
        }
        if (showPerformanceButton != null)
        {
            showPerformanceButton.interactable = connected
                && !string.IsNullOrEmpty(selectedPerformanceSessionId);
        }
        if (performanceSessionDropdownButton != null)
        {
            performanceSessionDropdownButton.interactable = connected
                && availableSessions.Count > 0;
        }
    }

    private bool IsIdle()
    {
        return serviceState == "IDLE";
    }

    private bool HasServiceState()
    {
        return serviceState == "IDLE"
            || serviceState == "COLLECTING"
            || serviceState == "TRAINING"
            || serviceState == "READY"
            || serviceState == "RUNNING";
    }

    private void ParseServiceConfig(string message)
    {
        string payload = message.Substring("service_config:".Length).Trim();
        var culture = CultureInfo.InvariantCulture;
        foreach (string pair in payload.Split(','))
        {
            string[] kv = pair.Split('=');
            if (kv.Length != 2)
            {
                continue;
            }
            string key = kv[0].Trim();
            string value = kv[1].Trim();
            switch (key)
            {
                case "cue_s":
                    if (float.TryParse(value, NumberStyles.Float, culture, out float cs))
                    {
                        cueSeconds = cs;
                    }
                    break;
                case "mi_s":
                    if (float.TryParse(value, NumberStyles.Float, culture, out float mis))
                    {
                        motorImagerySeconds = mis;
                    }
                    break;
                case "rest_s":
                    if (float.TryParse(value, NumberStyles.Float, culture, out float rs))
                    {
                        restSeconds = rs;
                    }
                    break;
                case "fixation_s":
                    if (float.TryParse(value, NumberStyles.Float, culture, out float fs))
                    {
                        fixationSeconds = fs;
                    }
                    break;
                case "trials_per_class":
                    if (int.TryParse(value, NumberStyles.Integer, culture, out int tpc))
                    {
                        trialsPerClass = tpc;
                    }
                    break;
            }
        }
        serviceConfigReceived = true;
        UpdateCollectionProgress(calibrationProgressCurrent, trialsPerClass * 2);
    }

    private void ParseSimpleList(string message, string prefix, List<string> target)
    {
        target.Clear();
        string payload = message.Substring(prefix.Length).Trim();
        foreach (string item in payload.Split(','))
        {
            string value = item.Trim();
            if (!string.IsNullOrEmpty(value))
            {
                target.Add(value);
            }
        }
    }

    private bool TryParseProgress(string message, string prefix, out int current, out int total)
    {
        current = 0;
        total = 0;
        string payload = message.Substring(prefix.Length).Trim();
        string[] parts = payload.Split('/');
        if (parts.Length != 2)
        {
            return false;
        }

        return int.TryParse(parts[0], out current)
            && int.TryParse(parts[1], out total)
            && total > 0;
    }

    private void UpdateCollectionProgress(int current, int total)
    {
        int safeTotal = Mathf.Max(1, total);
        int safeCurrent = Mathf.Clamp(current, 0, safeTotal);
        calibrationProgressCurrent = safeCurrent;
        calibrationProgressTotal = safeTotal;
        float ratio = Mathf.Clamp01((float)safeCurrent / safeTotal);

        if (acquireProgressFill != null)
        {
            float trackWidth = GetRectWidth(acquireProgressTrack, FallbackCalibrationTrackWidth);
            acquireProgressFill.sizeDelta = new Vector2(trackWidth * ratio, acquireProgressFill.sizeDelta.y);
        }

        if (acquireProgressText != null)
        {
            acquireProgressText.text = $"Collection {safeCurrent} / {safeTotal}";
        }
    }

    private void SetWarningText(string message)
    {
        if (localWarningText == null && workflowRoot != null)
        {
            localWarningText = CreateText(workflowRoot, "CalibrationWarningText", "", 18.0f, new Vector2(0.0f, -360.0f), new Vector2(760.0f, 34.0f));
            localWarningText.color = warningColor;
        }
        if (localWarningText == null)
        {
            return;
        }

        localWarningText.text = message;
        localWarningText.color = warningColor;
        localWarningText.gameObject.SetActive(!string.IsNullOrEmpty(message));
    }

    private void SetCueText(string text, Color color)
    {
        if (cueTaskText == null)
        {
            return;
        }

        cueTaskText.text = text;
        cueTaskText.color = color;
    }

    private string GetSubjectIdFromInput(TMP_InputField input)
    {
        string raw = input != null ? input.text : currentSubjectId;
        currentSubjectId = SanitizeSubjectId(raw);
        if (input != null && input.text != currentSubjectId)
        {
            input.SetTextWithoutNotify(currentSubjectId);
        }
        return currentSubjectId;
    }

    private string SanitizeSubjectId(string raw)
    {
        string value = (raw ?? "").Trim();
        if (string.IsNullOrEmpty(value))
        {
            return "pc2_test";
        }

        var chars = new List<char>(value.Length);
        foreach (char ch in value)
        {
            if (char.IsLetterOrDigit(ch) || ch == '_' || ch == '-')
            {
                chars.Add(ch);
            }
            else if (char.IsWhiteSpace(ch))
            {
                chars.Add('_');
            }
        }

        return chars.Count == 0 ? "pc2_test" : new string(chars.ToArray());
    }

    private string SubjectFromSessionId(string sessionId)
    {
        string value = (sessionId ?? "").Trim();
        int split = value.IndexOf('_');
        if (split <= 0)
        {
            return SanitizeSubjectId(value);
        }
        return SanitizeSubjectId(value.Substring(0, split));
    }

    private bool TryGetStatusValue(string message, string key, out string value)
    {
        value = "";
        int payloadStart = message.IndexOf(':');
        if (payloadStart < 0 || payloadStart + 1 >= message.Length)
        {
            return false;
        }

        string target = key + "=";
        string[] pairs = message.Substring(payloadStart + 1).Split(',');
        foreach (string rawPair in pairs)
        {
            string pair = rawPair.Trim();
            if (pair.StartsWith(target))
            {
                value = pair.Substring(target.Length).Trim();
                return !string.IsNullOrEmpty(value);
            }
        }

        return false;
    }

    private string GetStatusValueOrEmpty(string message, string key)
    {
        return TryGetStatusValue(message, key, out string value) ? value : "";
    }

    private float GetRectWidth(RectTransform rect, float fallback)
    {
        if (rect == null)
        {
            return fallback;
        }

        if (rect.rect.width > 0.1f)
        {
            return rect.rect.width;
        }

        return Mathf.Abs(rect.sizeDelta.x) > 0.1f ? Mathf.Abs(rect.sizeDelta.x) : fallback;
    }

    private RectTransform CreatePanel(Transform parent, string objectName, bool active)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = new Vector2(800.0f, 700.0f);
        rect.anchoredPosition = Vector2.zero;
        go.SetActive(active);
        return rect;
    }

    private TMP_Text CreateText(Transform parent, string objectName, string text, float fontSize, Vector2 anchoredPosition, Vector2 size)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var tmp = go.AddComponent<TextMeshProUGUI>();
        tmp.text = text;
        tmp.fontSize = fontSize;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.color = Color.white;
        tmp.raycastTarget = false;
        var rect = tmp.rectTransform;
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = size;
        rect.anchoredPosition = anchoredPosition;
        return tmp;
    }

    private TMP_Text CreateTableText(Transform parent, string objectName, string text, float fontSize, Vector2 anchoredPosition, Vector2 size, TextAlignmentOptions alignment = TextAlignmentOptions.Center)
    {
        TMP_Text tmp = CreateText(parent, objectName, text, fontSize, anchoredPosition, size);
        tmp.alignment = alignment;
        tmp.overflowMode = TextOverflowModes.Ellipsis;
        return tmp;
    }

    private Button CreateButton(Transform parent, string objectName, string label, Vector2 anchoredPosition, UnityEngine.Events.UnityAction action)
    {
        return CreateButton(parent, objectName, label, anchoredPosition, action, new Vector2(220.0f, 52.0f), 20.0f);
    }

    private Button CreateButton(Transform parent, string objectName, string label, Vector2 anchoredPosition, UnityEngine.Events.UnityAction action, Vector2 size, float fontSize)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = size;
        rect.anchoredPosition = anchoredPosition;
        var image = go.AddComponent<Image>();
        image.color = new Color(0.12f, 0.12f, 0.12f, 0.92f);
        var button = go.AddComponent<Button>();
        button.targetGraphic = image;
        if (action != null)
        {
            button.onClick.AddListener(action);
        }
        CreateText(go.transform, $"{objectName}Text", label, fontSize, Vector2.zero, size);
        return button;
    }

    private TMP_InputField CreateInputField(Transform parent, string objectName, string text, Vector2 anchoredPosition, Vector2 size)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = size;
        rect.anchoredPosition = anchoredPosition;
        var image = go.AddComponent<Image>();
        image.color = new Color(0.08f, 0.08f, 0.08f, 0.92f);
        var input = go.AddComponent<TMP_InputField>();
        input.targetGraphic = image;

        var textComponent = CreateText(go.transform, "Text", text, 18.0f, Vector2.zero, size);
        textComponent.alignment = TextAlignmentOptions.MidlineLeft;
        textComponent.rectTransform.offsetMin = new Vector2(12.0f, 4.0f);
        textComponent.rectTransform.offsetMax = new Vector2(-12.0f, -4.0f);
        var placeholder = CreateText(go.transform, "Placeholder", "S001", 18.0f, Vector2.zero, size);
        placeholder.alignment = TextAlignmentOptions.MidlineLeft;
        placeholder.color = new Color(1.0f, 1.0f, 1.0f, 0.45f);
        placeholder.rectTransform.offsetMin = new Vector2(12.0f, 4.0f);
        placeholder.rectTransform.offsetMax = new Vector2(-12.0f, -4.0f);
        input.textComponent = textComponent;
        input.placeholder = placeholder;
        input.text = text;
        return input;
    }

    private RectTransform CreateScrollPanel(Transform parent, string objectName, Vector2 anchoredPosition, Vector2 size, out RectTransform content)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = size;
        rect.anchoredPosition = anchoredPosition;

        var image = go.AddComponent<Image>();
        image.color = new Color(0.06f, 0.06f, 0.06f, 0.98f);
        var scrollRect = go.AddComponent<ScrollRect>();

        var viewportGo = new GameObject("Viewport", typeof(RectTransform));
        viewportGo.transform.SetParent(go.transform, false);
        var viewport = viewportGo.GetComponent<RectTransform>();
        viewport.anchorMin = Vector2.zero;
        viewport.anchorMax = Vector2.one;
        viewport.offsetMin = Vector2.zero;
        viewport.offsetMax = Vector2.zero;
        var viewportImage = viewportGo.AddComponent<Image>();
        viewportImage.color = Color.white;
        viewportImage.raycastTarget = false;
        viewportGo.AddComponent<Mask>().showMaskGraphic = false;

        var contentGo = new GameObject("Content", typeof(RectTransform));
        contentGo.transform.SetParent(viewport, false);
        content = contentGo.GetComponent<RectTransform>();
        content.anchorMin = new Vector2(0.0f, 1.0f);
        content.anchorMax = new Vector2(1.0f, 1.0f);
        content.pivot = new Vector2(0.5f, 1.0f);
        content.sizeDelta = Vector2.zero;
        content.anchoredPosition = Vector2.zero;

        scrollRect.content = content;
        scrollRect.viewport = viewport;
        scrollRect.horizontal = false;
        scrollRect.vertical = true;
        scrollRect.movementType = ScrollRect.MovementType.Clamped;
        return rect;
    }

    private RectTransform CreateBarTrack(Transform parent, string objectName, Vector2 size, Vector2 anchoredPosition)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = size;
        rect.anchoredPosition = anchoredPosition;
        var image = go.AddComponent<Image>();
        image.color = new Color(0.20f, 0.20f, 0.20f, 0.85f);
        image.raycastTarget = false;
        return rect;
    }

    private RectTransform CreateDivider(Transform parent, string objectName, Vector2 anchoredPosition, Vector2 size)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = size;
        rect.anchoredPosition = anchoredPosition;
        var image = go.AddComponent<Image>();
        image.color = new Color(1.0f, 1.0f, 1.0f, 0.78f);
        image.raycastTarget = false;
        return rect;
    }

    private RectTransform CreateBarFill(RectTransform parent, string objectName, Color color, Vector2 size)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.0f, 0.5f);
        rect.anchorMax = new Vector2(0.0f, 0.5f);
        rect.pivot = new Vector2(0.0f, 0.5f);
        rect.sizeDelta = size;
        rect.anchoredPosition = Vector2.zero;
        var image = go.AddComponent<Image>();
        image.color = color;
        image.raycastTarget = false;
        return rect;
    }

    private Toggle CreateToggle(Transform parent, string objectName, string label, Vector2 anchoredPosition)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = new Vector2(420.0f, 38.0f);
        rect.anchoredPosition = anchoredPosition;

        var toggle = go.AddComponent<Toggle>();
        var box = new GameObject("CheckmarkBox", typeof(RectTransform));
        box.transform.SetParent(go.transform, false);
        var boxRect = box.GetComponent<RectTransform>();
        boxRect.anchorMin = new Vector2(0.0f, 0.5f);
        boxRect.anchorMax = new Vector2(0.0f, 0.5f);
        boxRect.sizeDelta = new Vector2(28.0f, 28.0f);
        boxRect.anchoredPosition = new Vector2(14.0f, 0.0f);
        var boxImage = box.AddComponent<Image>();
        boxImage.color = new Color(0.18f, 0.18f, 0.18f, 0.95f);
        toggle.targetGraphic = boxImage;

        var check = new GameObject("Checkmark", typeof(RectTransform));
        check.transform.SetParent(box.transform, false);
        var checkRect = check.GetComponent<RectTransform>();
        checkRect.anchorMin = Vector2.zero;
        checkRect.anchorMax = Vector2.one;
        checkRect.offsetMin = new Vector2(4.0f, 4.0f);
        checkRect.offsetMax = new Vector2(-4.0f, -4.0f);
        var checkImage = check.AddComponent<Image>();
        checkImage.color = realtimeCueColor;
        toggle.graphic = checkImage;

        TMP_Text text = CreateText(go.transform, "Label", label, 21.0f, new Vector2(125.0f, 0.0f), new Vector2(340.0f, 34.0f));
        text.alignment = TextAlignmentOptions.MidlineLeft;
        return toggle;
    }

    private void ClearChildren(Transform parent)
    {
        for (int i = parent.childCount - 1; i >= 0; i--)
        {
            DestroyGameObject(parent.GetChild(i).gameObject);
        }
    }

    private void DestroyGameObject(GameObject go)
    {
        if (go == null)
        {
            return;
        }
        if (Application.isPlaying)
        {
            Destroy(go);
        }
        else
        {
            DestroyImmediate(go);
        }
    }

    private void SetActive(UnityEngine.Object obj, bool active)
    {
        if (obj is Component component)
        {
            component.gameObject.SetActive(active);
        }
        else if (obj is GameObject go)
        {
            go.SetActive(active);
        }
    }

    private void SetLegacyChildActive(string objectName, bool active)
    {
        if (calibrationPanel == null)
        {
            return;
        }

        Transform target = FindChildByName(calibrationPanel, objectName);
        if (target != null)
        {
            target.gameObject.SetActive(active);
        }
    }

    private Transform FindChildByName(Transform root, string objectName)
    {
        if (root.name == objectName)
        {
            return root;
        }

        for (int i = 0; i < root.childCount; i++)
        {
            Transform found = FindChildByName(root.GetChild(i), objectName);
            if (found != null)
            {
                return found;
            }
        }

        return null;
    }

    private void OnDestroy()
    {
        if (calibrationRoutine != null)
        {
            StopCoroutine(calibrationRoutine);
            calibrationRoutine = null;
        }

        if (streams != null)
        {
            streams.StatusReceived -= HandleStatusMessage;
        }
    }
}
