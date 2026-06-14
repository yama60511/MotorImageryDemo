using TMPro;
using UnityEngine;
using UnityEngine.UI;

[ExecuteAlways]
public class BciLslPanel : MonoBehaviour
{
    private static readonly Color DarkNavyBackground = new Color(0.01568628f, 0.03137255f, 0.07843138f, 1.0f);

    [Header("UI")]
    public TMP_Text statusText;
    public TMP_Text leftProbaText;
    public TMP_Text rightProbaText;
    public TMP_Text cueText;
    public Button startCalibrationButton;
    public Button shutdownButton;
    public Button startRealtimeButton;
    public Button stopRealtimeButton;
    public Button saveBundleButton;

    [Header("Screens")]
    public RectTransform mainMenuPanel;
    public RectTransform calibrationPanel;
    public RectTransform realtimePanel;
    public Button menuCalibrationButton;
    public Button menuRealtimeButton;
    public Button backToMenuFromCalibrationButton;
    public Button backToMenuFromRealtimeButton;
    public TMP_InputField subjectIdInput;

    [Header("Calibration Progress")]
    public RectTransform calibrationProgressTrack;
    public RectTransform calibrationProgressFill;
    public TMP_Text calibrationProgressText;
    public TMP_Text warningText;
    public Color warningColor = new Color(1.0f, 0.25f, 0.25f);

    [Header("Cue Display")]
    public Color leftCueColor = new Color(0.35f, 0.85f, 1.0f);
    public Color rightCueColor = new Color(1.0f, 0.55f, 0.35f);
    public Color restCueColor = new Color(0.8f, 0.8f, 0.8f);
    public Color readyCueColor = Color.white;
    public Color realtimeCueColor = new Color(0.45f, 1.0f, 0.55f);

    [Header("Realtime Display")]
    public RectTransform probaIndicatorTrack;
    public RectTransform probaIndicatorThumb;
    public TMP_Text realtimeHintText;

    private BciLslStreams streams;
    private BciScreenNavigator screenNavigator;
    private BciCalibrationController calibrationController;
    private BciRealtimeController realtimeController;

    private void OnEnable()
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            QueueEditorCalibrationPreview();
        }
#endif
    }

    private void OnValidate()
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            QueueEditorCalibrationPreview();
        }
#endif
    }

    private void Start()
    {
        if (!Application.isPlaying)
        {
            BuildEditorCalibrationPreview();
            return;
        }

        ResolveSceneReferences();
        EnsurePanelBackgrounds();

        streams = EnsureComponent<BciLslStreams>();
        screenNavigator = EnsureComponent<BciScreenNavigator>();
        calibrationController = EnsureComponent<BciCalibrationController>();
        realtimeController = EnsureComponent<BciRealtimeController>();

        ConfigureScreenNavigator();
        ConfigureCalibrationController();
        ConfigureRealtimeController();

        InitializeStatusText();
        WarnAboutMissingSceneReferences();

        streams.StatusReceived += UpdateStatusText;
        streams.Initialize();
        screenNavigator.Initialize();
        calibrationController.Initialize(streams);
        realtimeController.Initialize(streams);
    }

    private void OnDestroy()
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            UnityEditor.EditorApplication.delayCall -= BuildEditorCalibrationPreview;
            return;
        }
#endif
        if (streams != null)
        {
            streams.StatusReceived -= UpdateStatusText;
        }
    }

    private void BuildEditorCalibrationPreview()
    {
#if UNITY_EDITOR
        if (this == null || Application.isPlaying)
        {
            return;
        }

        ResolveSceneReferences();
        EnsurePanelBackgrounds();
        calibrationController = EnsureComponent<BciCalibrationController>();
        ConfigureCalibrationController();
        calibrationController.BuildEditorPreview(calibrationPanel);
#endif
    }

    private void QueueEditorCalibrationPreview()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.delayCall -= BuildEditorCalibrationPreview;
        UnityEditor.EditorApplication.delayCall += BuildEditorCalibrationPreview;
#endif
    }

    private void ConfigureScreenNavigator()
    {
        screenNavigator.mainMenuPanel = mainMenuPanel;
        screenNavigator.calibrationPanel = calibrationPanel;
        screenNavigator.realtimePanel = realtimePanel;
        screenNavigator.menuCalibrationButton = menuCalibrationButton;
        screenNavigator.menuRealtimeButton = menuRealtimeButton;
        screenNavigator.backToMenuFromCalibrationButton = backToMenuFromCalibrationButton;
        screenNavigator.backToMenuFromRealtimeButton = backToMenuFromRealtimeButton;
    }

    private void EnsurePanelBackgrounds()
    {
        SetPanelBackground(mainMenuPanel);
        SetPanelBackground(calibrationPanel);
        SetPanelBackground(realtimePanel);
    }

    private static void SetPanelBackground(RectTransform panel)
    {
        if (panel == null)
        {
            return;
        }

        var image = panel.GetComponent<Image>();
        if (image == null)
        {
            image = panel.gameObject.AddComponent<Image>();
        }

        image.color = DarkNavyBackground;
        image.raycastTarget = false;
    }

    private void ConfigureCalibrationController()
    {
        calibrationController.calibrationPanel = calibrationPanel;
        calibrationController.screenNavigator = screenNavigator;
        calibrationController.cueText = cueText;
        calibrationController.startCalibrationButton = startCalibrationButton;
        calibrationController.shutdownButton = shutdownButton;
        calibrationController.saveBundleButton = saveBundleButton;
        calibrationController.subjectIdInput = subjectIdInput;
        calibrationController.calibrationProgressTrack = calibrationProgressTrack;
        calibrationController.calibrationProgressFill = calibrationProgressFill;
        calibrationController.calibrationProgressText = calibrationProgressText;
        calibrationController.warningText = warningText;
        calibrationController.warningColor = warningColor;
        calibrationController.leftCueColor = leftCueColor;
        calibrationController.rightCueColor = rightCueColor;
        calibrationController.restCueColor = restCueColor;
        calibrationController.readyCueColor = readyCueColor;
        calibrationController.realtimeCueColor = realtimeCueColor;
        // Timing/marker fields live on BciCalibrationController directly.
        // Defaults can be tuned via its Inspector; runtime values are
        // overridden by service_config from the Python service.
    }

    private void ConfigureRealtimeController()
    {
        realtimeController.leftProbaText = leftProbaText;
        realtimeController.rightProbaText = rightProbaText;
        realtimeController.cueText = cueText;
        realtimeController.startRealtimeButton = startRealtimeButton;
        realtimeController.stopRealtimeButton = stopRealtimeButton;
        realtimeController.probaIndicatorTrack = probaIndicatorTrack;
        realtimeController.probaIndicatorThumb = probaIndicatorThumb;
        realtimeController.realtimeHintText = realtimeHintText;
        realtimeController.restCueColor = restCueColor;
        realtimeController.realtimeCueColor = realtimeCueColor;
    }

    private void InitializeStatusText()
    {
        if (statusText == null)
        {
            return;
        }

        statusText.text = "state: ---";
        statusText.color = Color.white;
    }

    private void UpdateStatusText(string message)
    {
        if (statusText == null)
        {
            return;
        }

        statusText.text = message;
        statusText.color = message.StartsWith("error:") ? Color.red : Color.white;
    }

    private void ResolveSceneReferences()
    {
        var canvas = GetComponentInParent<Canvas>();
        if (canvas == null)
        {
            Debug.LogWarning("BciLslPanel should be placed under a Canvas.", this);
            return;
        }

        AssignIfMissing(ref statusText, FindComponentByName<TMP_Text>(canvas.transform, "StatusText"));
        AssignIfMissing(ref leftProbaText, FindComponentByName<TMP_Text>(canvas.transform, "LeftProbaText"));
        AssignIfMissing(ref rightProbaText, FindComponentByName<TMP_Text>(canvas.transform, "RightProbaText"));
        AssignIfMissing(ref cueText, FindComponentByName<TMP_Text>(canvas.transform, "CueText"));

        AssignIfMissing(ref mainMenuPanel, FindComponentByName<RectTransform>(canvas.transform, "MainMenuPanel"));
        AssignIfMissing(ref calibrationPanel, FindComponentByName<RectTransform>(canvas.transform, "CalibrationPanel"));
        AssignIfMissing(ref realtimePanel, FindComponentByName<RectTransform>(canvas.transform, "RealtimePanel"));

        AssignIfMissing(ref menuCalibrationButton, FindComponentByName<Button>(canvas.transform, "MenuCalibrationButton"));
        AssignIfMissing(ref menuRealtimeButton, FindComponentByName<Button>(canvas.transform, "MenuRealtimeButton"));
        AssignIfMissing(ref backToMenuFromCalibrationButton, FindComponentByName<Button>(canvas.transform, "BackToMenuFromCalibrationButton"));
        AssignIfMissing(ref backToMenuFromRealtimeButton, FindComponentByName<Button>(canvas.transform, "BackToMenuFromRealtimeButton"));

        AssignIfMissing(ref startCalibrationButton, FindComponentByName<Button>(canvas.transform, "StartCalibrationButton"));
        AssignIfMissing(ref shutdownButton, FindComponentByName<Button>(canvas.transform, "ShutdownButton"));
        AssignIfMissing(ref startRealtimeButton, FindComponentByName<Button>(canvas.transform, "StartRealtimeButton"));
        AssignIfMissing(ref stopRealtimeButton, FindComponentByName<Button>(canvas.transform, "StopRealtimeButton"));
        AssignIfMissing(ref saveBundleButton, FindComponentByName<Button>(canvas.transform, "SaveBundleButton"));
        AssignIfMissing(ref subjectIdInput, FindComponentByName<TMP_InputField>(canvas.transform, "SubjectIdInput"));

        AssignIfMissing(ref calibrationProgressTrack, FindComponentByName<RectTransform>(canvas.transform, "CalibrationProgressTrack"));
        AssignIfMissing(ref calibrationProgressFill, FindComponentByName<RectTransform>(canvas.transform, "CalibrationProgressFill"));
        AssignIfMissing(ref calibrationProgressText, FindComponentByName<TMP_Text>(canvas.transform, "CalibrationProgressText"));
        AssignIfMissing(ref warningText, FindComponentByName<TMP_Text>(canvas.transform, "WarningText"));

        AssignIfMissing(ref probaIndicatorTrack, FindComponentByName<RectTransform>(canvas.transform, "ProbaIndicatorTrack"));
        AssignIfMissing(ref probaIndicatorThumb, FindComponentByName<RectTransform>(canvas.transform, "ProbaIndicatorThumb"));
        AssignIfMissing(ref realtimeHintText, FindComponentByName<TMP_Text>(canvas.transform, "RealtimeHintText"));
    }

    private void WarnAboutMissingSceneReferences()
    {
        WarnIfMissing(statusText, nameof(statusText));
        WarnIfMissing(mainMenuPanel, nameof(mainMenuPanel));
        WarnIfMissing(calibrationPanel, nameof(calibrationPanel));
        WarnIfMissing(realtimePanel, nameof(realtimePanel));
        WarnIfMissing(menuCalibrationButton, nameof(menuCalibrationButton));
        WarnIfMissing(menuRealtimeButton, nameof(menuRealtimeButton));
        WarnIfMissing(shutdownButton, nameof(shutdownButton));
        WarnIfMissing(startRealtimeButton, nameof(startRealtimeButton));
        WarnIfMissing(stopRealtimeButton, nameof(stopRealtimeButton));
    }

    private void WarnIfMissing(UnityEngine.Object value, string fieldName)
    {
        if (value == null)
        {
            Debug.LogWarning($"BciLslPanel missing Scene reference: {fieldName}", this);
        }
    }

    private T EnsureComponent<T>() where T : Component
    {
        if (TryGetComponent(out T component))
        {
            return component;
        }

        return gameObject.AddComponent<T>();
    }

    private static void AssignIfMissing<T>(ref T field, T value) where T : UnityEngine.Object
    {
        if (field == null)
        {
            field = value;
        }
    }

    private T FindComponentByName<T>(Transform root, string objectName) where T : Component
    {
        Transform target = FindTransformByName(root, objectName);
        if (target != null && target.TryGetComponent(out T component))
        {
            return component;
        }

        return null;
    }

    private Transform FindTransformByName(Transform root, string objectName)
    {
        if (root.name == objectName)
        {
            return root;
        }

        for (int i = 0; i < root.childCount; i++)
        {
            Transform found = FindTransformByName(root.GetChild(i), objectName);
            if (found != null)
            {
                return found;
            }
        }

        return null;
    }
}
