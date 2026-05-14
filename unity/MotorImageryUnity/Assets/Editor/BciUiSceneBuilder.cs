using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

public static class BciUiSceneBuilder
{
    private const string MainMenuScenePath = "Assets/MainMenu.unity";
    private static readonly Color DarkNavyBackground = new Color(0.01568628f, 0.03137255f, 0.07843138f, 1.0f);

    [MenuItem("Motor Imagery/Rebuild Editable Main Menu UI")]
    public static void RebuildEditableMainMenuUi()
    {
        var scene = EditorSceneManager.OpenScene(MainMenuScenePath, OpenSceneMode.Single);
        var camera = Camera.main != null ? Camera.main : Object.FindAnyObjectByType<Camera>();
        if (camera != null)
        {
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = DarkNavyBackground;
        }

        var canvas = Object.FindAnyObjectByType<Canvas>();
        if (canvas == null)
        {
            var canvasGo = new GameObject("Canvas", typeof(RectTransform));
            canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1440.0f, 810.0f);
            scaler.matchWidthOrHeight = 0.5f;
            canvasGo.AddComponent<GraphicRaycaster>();
        }
        else
        {
            var scaler = canvas.GetComponent<CanvasScaler>();
            if (scaler != null)
            {
                scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
                scaler.referenceResolution = new Vector2(1440.0f, 810.0f);
                scaler.matchWidthOrHeight = 0.5f;
            }
        }

        var panel = canvas.GetComponent<BciLslPanel>();
        if (panel == null)
        {
            panel = canvas.gameObject.AddComponent<BciLslPanel>();
        }

        DeleteIfExists(canvas.transform, "MainMenuPanel");
        DeleteIfExists(canvas.transform, "CalibrationPanel");
        DeleteIfExists(canvas.transform, "RealtimePanel");
        DeleteIfExists(canvas.transform, "StatusText");
        DeleteIfExists(canvas.transform, "LeftProbaText");
        DeleteIfExists(canvas.transform, "RightProbaText");
        DeleteIfExists(canvas.transform, "StartCalibrationButton");
        DeleteIfExists(canvas.transform, "ShutdownButton");

        var mainMenuPanel = CreatePanel(canvas.transform, "MainMenuPanel", true);
        var calibrationPanel = CreatePanel(canvas.transform, "CalibrationPanel", false);
        var realtimePanel = CreatePanel(canvas.transform, "RealtimePanel", false);

        var statusText = CreateText(
            canvas.transform,
            "StatusText",
            "state: ---",
            22.0f,
            new Vector2(0.0f, 350.0f),
            new Vector2(760.0f, 34.0f)
        );

        CreateText(
            mainMenuPanel,
            "MainMenuTitle",
            "Motor Imagery BCI",
            42.0f,
            new Vector2(0.0f, 160.0f),
            new Vector2(620.0f, 70.0f)
        );
        var menuCalibrationButton = CreateButton(
            mainMenuPanel,
            "MenuCalibrationButton",
            "Calibration",
            new Vector2(0.0f, 55.0f)
        );
        var menuRealtimeButton = CreateButton(
            mainMenuPanel,
            "MenuRealtimeButton",
            "Realtime",
            new Vector2(0.0f, -15.0f)
        );
        var shutdownButton = CreateButton(
            mainMenuPanel,
            "ShutdownButton",
            "Shutdown",
            new Vector2(0.0f, -110.0f)
        );

        TMP_InputField subjectIdInput = null;
        Button startCalibrationButton = null;
        Button saveBundleButton = null;
        Button backCalibrationButton = null;
        RectTransform calibrationProgressTrack = null;
        RectTransform calibrationProgressFill = null;
        TMP_Text calibrationProgressText = null;
        TMP_Text warningText = null;

        var leftProbaText = CreateText(
            realtimePanel,
            "LeftProbaText",
            "p_left: ---",
            36.0f,
            new Vector2(-190.0f, 145.0f),
            new Vector2(340.0f, 58.0f)
        );
        var rightProbaText = CreateText(
            realtimePanel,
            "RightProbaText",
            "p_right: ---",
            36.0f,
            new Vector2(190.0f, 145.0f),
            new Vector2(340.0f, 58.0f)
        );
        var realtimeHintText = CreateText(
            realtimePanel,
            "RealtimeHintText",
            "Realtime idle",
            30.0f,
            new Vector2(0.0f, -175.0f),
            new Vector2(620.0f, 52.0f)
        );
        var cueText = CreateText(
            realtimePanel,
            "RealtimeCueText",
            "",
            64.0f,
            new Vector2(0.0f, 230.0f),
            new Vector2(860.0f, 104.0f)
        );
        cueText.fontStyle = FontStyles.Bold;
        var probaIndicatorTrack = CreateBarTrack(
            realtimePanel,
            "ProbaIndicatorTrack",
            new Vector2(560.0f, 26.0f),
            new Vector2(0.0f, -225.0f)
        );
        var probaIndicatorThumb = CreateBarThumb(
            probaIndicatorTrack,
            "ProbaIndicatorThumb",
            new Color(0.45f, 1.0f, 0.55f)
        );
        var startRealtimeButton = CreateButton(
            realtimePanel,
            "StartRealtimeButton",
            "Start Realtime",
            new Vector2(-155.0f, -275.0f)
        );
        var stopRealtimeButton = CreateButton(
            realtimePanel,
            "StopRealtimeButton",
            "Stop Realtime",
            new Vector2(155.0f, -275.0f)
        );
        var backRealtimeButton = CreateButton(
            realtimePanel,
            "BackToMenuFromRealtimeButton",
            "Back",
            new Vector2(0.0f, -325.0f)
        );

        panel.statusText = statusText;
        panel.leftProbaText = leftProbaText;
        panel.rightProbaText = rightProbaText;
        panel.cueText = cueText;
        panel.startCalibrationButton = startCalibrationButton;
        panel.shutdownButton = shutdownButton;
        panel.startRealtimeButton = startRealtimeButton;
        panel.stopRealtimeButton = stopRealtimeButton;
        panel.saveBundleButton = saveBundleButton;
        panel.mainMenuPanel = mainMenuPanel;
        panel.calibrationPanel = calibrationPanel;
        panel.realtimePanel = realtimePanel;
        panel.menuCalibrationButton = menuCalibrationButton;
        panel.menuRealtimeButton = menuRealtimeButton;
        panel.backToMenuFromCalibrationButton = backCalibrationButton;
        panel.backToMenuFromRealtimeButton = backRealtimeButton;
        panel.subjectIdInput = subjectIdInput;
        panel.calibrationProgressTrack = calibrationProgressTrack;
        panel.calibrationProgressFill = calibrationProgressFill;
        panel.calibrationProgressText = calibrationProgressText;
        panel.warningText = warningText;
        panel.probaIndicatorTrack = probaIndicatorTrack;
        panel.probaIndicatorThumb = probaIndicatorThumb;
        panel.realtimeHintText = realtimeHintText;

        var calibrationController = canvas.GetComponent<BciCalibrationController>();
        if (calibrationController == null)
        {
            calibrationController = canvas.gameObject.AddComponent<BciCalibrationController>();
        }
        calibrationController.calibrationPanel = calibrationPanel;
        calibrationController.warningColor = panel.warningColor;
        calibrationController.leftCueColor = panel.leftCueColor;
        calibrationController.rightCueColor = panel.rightCueColor;
        calibrationController.restCueColor = panel.restCueColor;
        calibrationController.readyCueColor = panel.readyCueColor;
        calibrationController.realtimeCueColor = panel.realtimeCueColor;

        EditorUtility.SetDirty(panel);
        EditorUtility.SetDirty(calibrationController);
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Rebuilt editable Motor Imagery UI in MainMenu.unity");
    }

    private static RectTransform CreatePanel(Transform parent, string objectName, bool active)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.offsetMin = Vector2.zero;
        rect.offsetMax = Vector2.zero;
        var image = go.AddComponent<Image>();
        image.color = DarkNavyBackground;
        image.raycastTarget = false;
        go.SetActive(active);
        return rect;
    }

    private static TMP_Text CreateText(
        Transform parent,
        string objectName,
        string text,
        float fontSize,
        Vector2 anchoredPosition,
        Vector2 size
    )
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

    private static Button CreateButton(
        Transform parent,
        string objectName,
        string label,
        Vector2 anchoredPosition
    )
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = new Vector2(230.0f, 60.0f);
        rect.anchoredPosition = anchoredPosition;
        var image = go.AddComponent<Image>();
        image.color = new Color(0.12f, 0.12f, 0.12f, 0.92f);
        var button = go.AddComponent<Button>();
        button.targetGraphic = image;
        CreateText(go.transform, $"{objectName}Text", label, 22.0f, Vector2.zero, rect.sizeDelta);
        return button;
    }

    private static TMP_InputField CreateInputField(
        Transform parent,
        string objectName,
        string text,
        Vector2 anchoredPosition,
        Vector2 size
    )
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

        var textComponent = CreateText(go.transform, "Text", text, 20.0f, Vector2.zero, size);
        textComponent.alignment = TextAlignmentOptions.MidlineLeft;
        textComponent.rectTransform.offsetMin = new Vector2(12.0f, 4.0f);
        textComponent.rectTransform.offsetMax = new Vector2(-12.0f, -4.0f);
        var placeholder = CreateText(go.transform, "Placeholder", "S001", 20.0f, Vector2.zero, size);
        placeholder.alignment = TextAlignmentOptions.MidlineLeft;
        placeholder.color = new Color(1.0f, 1.0f, 1.0f, 0.45f);
        placeholder.rectTransform.offsetMin = new Vector2(12.0f, 4.0f);
        placeholder.rectTransform.offsetMax = new Vector2(-12.0f, -4.0f);
        input.textComponent = textComponent;
        input.placeholder = placeholder;
        input.text = text;
        return input;
    }

    private static RectTransform CreateBarTrack(
        Transform parent,
        string objectName,
        Vector2 size,
        Vector2 anchoredPosition
    )
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

    private static RectTransform CreateBarFill(
        RectTransform parent,
        string objectName,
        Color color,
        Vector2 size
    )
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

    private static RectTransform CreateBarThumb(RectTransform parent, string objectName, Color color)
    {
        var go = new GameObject(objectName, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = new Vector2(24.0f, 36.0f);
        rect.anchoredPosition = Vector2.zero;
        var image = go.AddComponent<Image>();
        image.color = color;
        image.raycastTarget = false;
        return rect;
    }

    private static void DeleteIfExists(Transform parent, string objectName)
    {
        var child = parent.Find(objectName);
        if (child != null)
        {
            Object.DestroyImmediate(child.gameObject);
        }
    }
}
