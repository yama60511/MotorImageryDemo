using System.Collections.Generic;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class RopeChoiceMotorImagerySceneSetup
{
    private const string MainMenuScenePath = "Assets/MainMenu.unity";
    private const string GameScenePath = "Assets/Scenes/Game_Scene.unity";
    private static readonly Color DarkNavyBackground = new Color(0.01568628f, 0.03137255f, 0.07843138f, 1f);

    [MenuItem("Motor Imagery/Rope Choice/Setup Game Scene")]
    public static void SetupGameScene()
    {
        var scene = EditorSceneManager.OpenScene(GameScenePath, OpenSceneMode.Single);

        var streams = FindOrCreateComponent<BciLslStreams>("BCI LSL Streams");
        var input = FindOrCreateComponent<RopeChoiceInputAggregator>("Rope Choice Input");
        var manager = FindOrCreateComponent<RopeChoiceGameManager>("Rope Choice Game Manager");
        var camera = Camera.main != null ? Camera.main : Object.FindAnyObjectByType<Camera>();
        var canvas = FindOrCreateCanvas("Rope Choice Canvas");
        if (camera != null)
        {
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = DarkNavyBackground;
        }

        EnsureEventSystem();

        var hudPanel = CreatePanel(canvas.transform, "HUD Panel", new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-210f, -96f), new Vector2(380f, 150f), new Color(0.02f, 0.04f, 0.08f, 0.88f));
        CreatePanel(hudPanel.transform, "HUD Accent", new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -3f), new Vector2(0f, 6f), new Color(1f, 0.88f, 0.16f, 1f));
        var roundText = CreateText(hudPanel.transform, "Round Text", "Stage 1 / 8", 28, FontStyle.Bold, TextAnchor.MiddleCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -34f), new Vector2(-24f, 38f), Color.white);
        var smoothText = CreateText(hudPanel.transform, "Smooth Choices Text", "Smooth 0 / 8", 22, FontStyle.Bold, TextAnchor.MiddleCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -78f), new Vector2(-24f, 32f), new Color(1f, 0.88f, 0.16f));
        var statusText = CreateText(hudPanel.transform, "Status Text", "Waiting", 21, FontStyle.Normal, TextAnchor.MiddleCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -114f), new Vector2(-24f, 30f), Color.white);
        var backButton = CreateButton(canvas.transform, "Back Button", "BACK", new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(0f, 54f), new Vector2(220f, 58f));
        var backLoader = backButton.GetComponent<LoadSceneButton>();
        if (backLoader == null)
        {
            backLoader = backButton.gameObject.AddComponent<LoadSceneButton>();
        }

        SetSerializedString(backLoader, "sceneName", "MainMenu");

        var startPanel = CreatePanel(canvas.transform, "Start Panel", new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), Vector2.zero, new Vector2(620f, 460f), new Color(0.02f, 0.04f, 0.08f, 0.92f));
        CreatePanel(startPanel.transform, "Top Accent", new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -4f), new Vector2(0f, 8f), new Color(1f, 0.88f, 0.16f, 1f));
        CreateText(startPanel.transform, "Title Text", "Rope Choice", 48, FontStyle.Bold, TextAnchor.MiddleCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -72f), new Vector2(-36f, 70f), Color.white);
        CreateText(startPanel.transform, "Subtitle Text", "Motor Imagery Route Challenge", 22, FontStyle.Bold, TextAnchor.MiddleCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -126f), new Vector2(-36f, 38f), new Color(1f, 0.88f, 0.16f));
        var circleSprite = AssetDatabase.LoadAssetAtPath<Sprite>("Assets/CharacterAssets/circle.svg");
        var carSprite = AssetDatabase.LoadAssetAtPath<Sprite>("Assets/CharacterAssets/car.svg");
        var flowerSprite = AssetDatabase.LoadAssetAtPath<Sprite>("Assets/CharacterAssets/flower.svg");
        var characterPreviewImage = CreateImage(startPanel.transform, "Character Preview", circleSprite, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(0f, 64f), new Vector2(108f, 108f));
        CreateText(startPanel.transform, "Character Label", "Character", 20, FontStyle.Bold, TextAnchor.MiddleCenter, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(-160f, -22f), new Vector2(150f, 34f), Color.white);
        var characterDropdown = CreateDropdown(startPanel.transform, "Character Dropdown", new List<string> { "Circle", "Car", "Flower" }, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(65f, -22f), new Vector2(260f, 46f));
        var startButton = CreateButton(startPanel.transform, "Start Button", "Start", new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(0f, -178f), new Vector2(260f, 66f));

        var resultPanel = CreatePanel(canvas.transform, "Result Panel", new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), Vector2.zero, new Vector2(780f, 610f), new Color(1f, 0.78f, 0.18f, 0.98f));
        var resultInnerPanel = CreatePanel(resultPanel.transform, "Result Inner Panel", Vector2.zero, Vector2.one, Vector2.zero, new Vector2(-34f, -34f), new Color(0.02f, 0.04f, 0.08f, 0.96f));
        CreatePanel(resultInnerPanel.transform, "Top Accent", new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -4f), new Vector2(0f, 8f), new Color(1f, 0.88f, 0.16f, 1f));
        var resultTitleText = CreateText(resultInnerPanel.transform, "Result Title Text", "おめでとう！", 50, FontStyle.Bold, TextAnchor.MiddleCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -56f), new Vector2(-36f, 64f), Color.white);
        var resultTimeText = CreateText(resultInnerPanel.transform, "Result Time Text", "Time: 0.00 sec", 28, FontStyle.Bold, TextAnchor.MiddleCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -122f), new Vector2(-36f, 40f), Color.white);
        var resultSmoothText = CreateText(resultInnerPanel.transform, "Result Smooth Text", "Smooth choices: 0 / 8", 23, FontStyle.Normal, TextAnchor.MiddleCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -166f), new Vector2(-36f, 34f), Color.white);
        var currentRankText = CreateText(resultInnerPanel.transform, "Current Rank Text", "Your Rank: -", 58, FontStyle.Bold, TextAnchor.MiddleCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -248f), new Vector2(-36f, 72f), new Color(1f, 0.88f, 0.16f));
        CreateText(resultInnerPanel.transform, "Ranking Title Text", "Ranking Top 5", 34, FontStyle.Bold, TextAnchor.MiddleCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -324f), new Vector2(-36f, 44f), Color.white);
        var rankingText = CreateText(resultInnerPanel.transform, "Ranking Text", "No records yet", 28, FontStyle.Bold, TextAnchor.UpperCenter, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -430f), new Vector2(-36f, 130f), Color.yellow);
        var playAgainButton = CreateButton(resultInnerPanel.transform, "Play Again Button", "Play Again", new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(0f, 48f), new Vector2(270f, 62f));
        var homeButton = CreateButton(resultInnerPanel.transform, "Home Button", "Home", Vector2.one, Vector2.one, new Vector2(-132f, 48f), new Vector2(210f, 58f));
        var homeLoader = homeButton.GetComponent<LoadSceneButton>();
        if (homeLoader == null)
        {
            homeLoader = homeButton.gameObject.AddComponent<LoadSceneButton>();
        }

        SetSerializedString(homeLoader, "sceneName", "MainMenu");

        SetSerializedObject(input, "streams", streams);
        SetSerializedObject(manager, "input", input);
        SetSerializedObject(manager, "mainCamera", camera);
        SetSerializedColor(manager, "backgroundColor", DarkNavyBackground);
        SetSerializedFloat(manager, "routeSpread", 2.6f);
        SetSerializedFloat(manager, "longRouteJagAmplitude", 0.42f);
        SetSerializedInt(manager, "shortRoutePointCount", 128);
        SetSerializedInt(manager, "longRoutePointCount", 192);
        SetSerializedObject(manager, "circleSprite", circleSprite);
        SetSerializedObject(manager, "carSprite", carSprite);
        SetSerializedObject(manager, "flowerSprite", flowerSprite);
        SetSerializedObject(manager, "hudPanel", hudPanel);
        SetSerializedObject(manager, "roundText", roundText);
        SetSerializedObject(manager, "smoothChoicesText", smoothText);
        SetSerializedObject(manager, "statusText", statusText);
        SetSerializedObject(manager, "startPanel", startPanel);
        SetSerializedObject(manager, "startButton", startButton);
        SetSerializedObject(manager, "characterDropdown", characterDropdown);
        SetSerializedObject(manager, "characterPreviewImage", characterPreviewImage);
        SetSerializedObject(manager, "resultPanel", resultPanel);
        SetSerializedObject(manager, "resultTitleText", resultTitleText);
        SetSerializedObject(manager, "resultTimeText", resultTimeText);
        SetSerializedObject(manager, "resultSmoothChoicesText", resultSmoothText);
        SetSerializedObject(manager, "currentRankText", currentRankText);
        SetSerializedObject(manager, "rankingText", rankingText);
        SetSerializedObject(manager, "playAgainButton", playAgainButton);

        AddScenesToBuildSettings();

        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Rope Choice Game_Scene setup complete.");
    }

    [MenuItem("Motor Imagery/Rope Choice/Add Game Start Button")]
    public static void AddGameStartButtonToMainMenu()
    {
        var scene = EditorSceneManager.OpenScene(MainMenuScenePath, OpenSceneMode.Single);
        var mainMenuPanel = GameObject.Find("MainMenuPanel");
        if (mainMenuPanel == null)
        {
            Debug.LogWarning("MainMenuPanel was not found in MainMenu.");
            return;
        }

        SetMainMenuButton(mainMenuPanel.transform, "MenuCalibrationButton", "Calibration", new Vector2(0f, 55f));
        SetMainMenuButton(mainMenuPanel.transform, "MenuRealtimeButton", "RealTime", new Vector2(0f, -15f));
        SetMainMenuButton(mainMenuPanel.transform, "ShutdownButton", "Shutdown", new Vector2(0f, -155f));

        var button = CreateTmpButton(mainMenuPanel.transform, "GameStartButton", "GAME", new Vector2(0f, -85f), new Vector2(190f, 52f));
        var loader = button.GetComponent<LoadSceneButton>();
        if (loader == null)
        {
            loader = button.gameObject.AddComponent<LoadSceneButton>();
        }

        SetSerializedString(loader, "sceneName", "Game_Scene");
        AddScenesToBuildSettings();

        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Updated MainMenu button layout and GAME button.");
    }

    [MenuItem("Motor Imagery/Rope Choice/Setup All")]
    public static void SetupAll()
    {
        SetupGameScene();
        AddGameStartButtonToMainMenu();
    }

    private static T FindOrCreateComponent<T>(string objectName) where T : Component
    {
        var component = Object.FindAnyObjectByType<T>();
        if (component != null)
        {
            return component;
        }

        var go = new GameObject(objectName);
        return go.AddComponent<T>();
    }

    private static Canvas FindOrCreateCanvas(string objectName)
    {
        var canvas = Object.FindAnyObjectByType<Canvas>();
        if (canvas == null)
        {
            var go = new GameObject(objectName);
            canvas = go.AddComponent<Canvas>();
            go.AddComponent<CanvasScaler>();
            go.AddComponent<GraphicRaycaster>();
        }

        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        var scaler = canvas.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920f, 1080f);
        scaler.matchWidthOrHeight = 0.5f;
        return canvas;
    }

    private static void EnsureEventSystem()
    {
        var eventSystem = Object.FindAnyObjectByType<EventSystem>();
        if (eventSystem == null)
        {
            var eventSystemObject = new GameObject("EventSystem");
            eventSystem = eventSystemObject.AddComponent<EventSystem>();
        }

        var standalone = eventSystem.GetComponent<StandaloneInputModule>();
        if (standalone != null)
        {
            Object.DestroyImmediate(standalone);
        }

        if (eventSystem.GetComponent<InputSystemUIInputModule>() == null)
        {
            eventSystem.gameObject.AddComponent<InputSystemUIInputModule>();
        }
    }

    private static GameObject CreatePanel(Transform parent, string objectName, Vector2 anchorMin, Vector2 anchorMax, Vector2 anchoredPosition, Vector2 sizeDelta, Color color)
    {
        var go = FindOrCreateChild(parent, objectName);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = sizeDelta;

        var image = go.GetComponent<Image>();
        if (image == null)
        {
            image = go.AddComponent<Image>();
        }

        image.color = color;
        return go;
    }

    private static Text CreateText(Transform parent, string objectName, string text, int fontSize, FontStyle fontStyle, TextAnchor alignment, Vector2 anchorMin, Vector2 anchorMax, Vector2 anchoredPosition, Vector2 sizeDelta, Color color)
    {
        var go = FindOrCreateChild(parent, objectName);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = sizeDelta;

        var label = go.GetComponent<Text>();
        if (label == null)
        {
            label = go.AddComponent<Text>();
        }

        label.text = text;
        label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        label.fontSize = fontSize;
        label.fontStyle = fontStyle;
        label.alignment = alignment;
        label.color = color;
        label.raycastTarget = false;
        return label;
    }

    private static Image CreateImage(Transform parent, string objectName, Sprite sprite, Vector2 anchorMin, Vector2 anchorMax, Vector2 anchoredPosition, Vector2 sizeDelta)
    {
        var go = FindOrCreateChild(parent, objectName);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = sizeDelta;

        var image = go.GetComponent<Image>();
        if (image == null)
        {
            image = go.AddComponent<Image>();
        }

        image.sprite = sprite;
        image.color = Color.white;
        image.preserveAspect = true;
        image.raycastTarget = false;
        return image;
    }

    private static Button CreateButton(Transform parent, string objectName, string labelText, Vector2 anchorMin, Vector2 anchorMax, Vector2 anchoredPosition, Vector2 sizeDelta)
    {
        var go = FindOrCreateChild(parent, objectName);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = sizeDelta;

        var image = go.GetComponent<Image>();
        if (image == null)
        {
            image = go.AddComponent<Image>();
        }

        image.color = new Color(1f, 0.88f, 0.16f);

        var button = go.GetComponent<Button>();
        if (button == null)
        {
            button = go.AddComponent<Button>();
        }

        CreateText(go.transform, "Text", labelText, 28, FontStyle.Bold, TextAnchor.MiddleCenter, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero, new Color(0.04f, 0.05f, 0.08f));
        return button;
    }

    private static Dropdown CreateDropdown(Transform parent, string objectName, List<string> options, Vector2 anchorMin, Vector2 anchorMax, Vector2 anchoredPosition, Vector2 sizeDelta)
    {
        var go = FindOrCreateChild(parent, objectName);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = sizeDelta;

        var existingButton = go.GetComponent<Button>();
        if (existingButton != null)
        {
            Object.DestroyImmediate(existingButton);
        }

        var image = go.GetComponent<Image>();
        if (image == null)
        {
            image = go.AddComponent<Image>();
        }

        image.color = new Color(1f, 0.88f, 0.16f);

        var dropdown = go.GetComponent<Dropdown>();
        if (dropdown == null)
        {
            dropdown = go.AddComponent<Dropdown>();
        }

        var caption = CreateText(go.transform, "Label", options[0], 22, FontStyle.Bold, TextAnchor.MiddleLeft, Vector2.zero, Vector2.one, new Vector2(18f, 0f), new Vector2(-58f, 0f), new Color(0.04f, 0.05f, 0.08f));
        CreateText(go.transform, "Arrow", "v", 20, FontStyle.Bold, TextAnchor.MiddleCenter, new Vector2(1f, 0f), Vector2.one, new Vector2(-24f, 0f), new Vector2(36f, 0f), new Color(0.04f, 0.05f, 0.08f));

        dropdown.captionText = caption;
        dropdown.targetGraphic = image;
        dropdown.options.Clear();
        dropdown.AddOptions(options);
        dropdown.RefreshShownValue();
        return dropdown;
    }

    private static Button CreateTmpButton(Transform parent, string objectName, string labelText, Vector2 anchoredPosition, Vector2 sizeDelta)
    {
        var go = FindOrCreateChild(parent, objectName);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = sizeDelta;

        var image = go.GetComponent<Image>();
        if (image == null)
        {
            image = go.AddComponent<Image>();
        }

        image.color = new Color(1.0f, 0.88f, 0.16f);

        var button = go.GetComponent<Button>();
        if (button == null)
        {
            button = go.AddComponent<Button>();
        }

        var labelObject = FindOrCreateChild(go.transform, "Text");
        var labelRect = labelObject.GetComponent<RectTransform>();
        labelRect.anchorMin = Vector2.zero;
        labelRect.anchorMax = Vector2.one;
        labelRect.anchoredPosition = Vector2.zero;
        labelRect.sizeDelta = Vector2.zero;

        var label = labelObject.GetComponent<TextMeshProUGUI>();
        if (label == null)
        {
            label = labelObject.AddComponent<TextMeshProUGUI>();
        }

        label.text = labelText;
        label.fontSize = 22f;
        label.fontStyle = FontStyles.Bold;
        label.alignment = TextAlignmentOptions.Center;
        label.color = new Color(0.04f, 0.05f, 0.08f);
        label.raycastTarget = false;
        return button;
    }

    private static void SetMainMenuButton(Transform parent, string objectName, string labelText, Vector2 anchoredPosition)
    {
        var child = parent.Find(objectName);
        if (child == null)
        {
            return;
        }

        var rect = child.GetComponent<RectTransform>();
        if (rect != null)
        {
            rect.anchoredPosition = anchoredPosition;
            rect.sizeDelta = new Vector2(190f, 52f);
        }

        var label = child.GetComponentInChildren<TMP_Text>();
        if (label != null)
        {
            label.text = labelText;
        }
    }

    private static GameObject FindOrCreateChild(Transform parent, string objectName)
    {
        Transform existing = parent.Find(objectName);
        if (existing != null)
        {
            return existing.gameObject;
        }

        var go = new GameObject(objectName);
        go.transform.SetParent(parent, false);
        go.AddComponent<RectTransform>();
        return go;
    }

    private static void SetSerializedObject(Object target, string propertyName, Object value)
    {
        var serialized = new SerializedObject(target);
        serialized.FindProperty(propertyName).objectReferenceValue = value;
        serialized.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(target);
    }

    private static void SetSerializedString(Object target, string propertyName, string value)
    {
        var serialized = new SerializedObject(target);
        serialized.FindProperty(propertyName).stringValue = value;
        serialized.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(target);
    }

    private static void SetSerializedFloat(Object target, string propertyName, float value)
    {
        var serialized = new SerializedObject(target);
        serialized.FindProperty(propertyName).floatValue = value;
        serialized.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(target);
    }

    private static void SetSerializedColor(Object target, string propertyName, Color value)
    {
        var serialized = new SerializedObject(target);
        serialized.FindProperty(propertyName).colorValue = value;
        serialized.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(target);
    }

    private static void SetSerializedInt(Object target, string propertyName, int value)
    {
        var serialized = new SerializedObject(target);
        serialized.FindProperty(propertyName).intValue = value;
        serialized.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(target);
    }

    private static void AddScenesToBuildSettings()
    {
        var scenes = new List<EditorBuildSettingsScene>();
        if (System.IO.File.Exists(MainMenuScenePath))
        {
            scenes.Add(new EditorBuildSettingsScene(MainMenuScenePath, true));
        }

        scenes.Add(new EditorBuildSettingsScene(GameScenePath, true));
        EditorBuildSettings.scenes = scenes.ToArray();
    }
}
