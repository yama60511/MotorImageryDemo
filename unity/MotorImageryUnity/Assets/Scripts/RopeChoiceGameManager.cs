using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class RopeChoiceGameManager : MonoBehaviour
{
    private enum GameState
    {
        WaitingToStart,
        AwaitingChoice,
        Moving,
        Finished
    }

    private enum RopeSide
    {
        Left,
        Right
    }

    private enum CharacterKind
    {
        Circle,
        Car,
        Flower
    }

    private struct StagePlan
    {
        public Vector3 Start;
        public Vector3 Goal;
        public RopeSide SmoothSide;
        public RopeSide SelectedSide;
        public bool HasSelection;
        public Vector3[] LeftPath;
        public Vector3[] RightPath;
        public LineRenderer LeftRenderer;
        public LineRenderer RightRenderer;
    }

    private const string RankingPrefsKey = "RopeChoiceGame.RankingTimes";

    [Header("Input")]
    [SerializeField] private RopeChoiceInputAggregator input;
    [SerializeField] private float decisionDelaySeconds = 0.25f;

    [Header("Game")]
    [SerializeField, Min(2)] private int totalStages = 8;
    [SerializeField] private bool keepShortRoutesBalanced = true;
    [SerializeField] private float moveSpeed = 4.2f;
    [SerializeField] private Vector3 startPosition = new Vector3(0.0f, -3.35f, 0.0f);
    [SerializeField] private float stageHeight = 10.0f;
    [SerializeField] private float routeSpread = 2.6f;
    [SerializeField] private float cameraFollowSpeed = 5.0f;
    [SerializeField, Range(0.0f, 1.2f)] private float longRouteJagAmplitude = 0.42f;
    [SerializeField, Range(4.0f, 32.0f)] private float longRouteJagFrequency = 18.0f;
    [SerializeField, Min(8)] private int shortRoutePointCount = 128;
    [SerializeField, Min(8)] private int longRoutePointCount = 192;

    [Header("Look")]
    [SerializeField] private Color backgroundColor = new Color(0.01568628f, 0.03137255f, 0.07843138f);
    [SerializeField] private Color ropeColor = new Color(0.94f, 0.96f, 1.0f);
    [SerializeField] private Color characterColor = new Color(0.20f, 0.72f, 1.0f);
    [SerializeField] private Color smoothTargetColor = new Color(1.0f, 0.88f, 0.16f);
    [SerializeField] private Color completedRopeColor = new Color(0.36f, 0.54f, 0.70f, 0.72f);
    [SerializeField] private Color futureRopeColor = new Color(0.94f, 0.96f, 1.0f, 0.34f);
    [SerializeField] private Color wrongRouteColor = new Color(1.0f, 0.55f, 0.50f);
    [SerializeField] private float ropeWidth = 0.08f;
    [SerializeField] private CharacterKind selectedCharacter = CharacterKind.Circle;
    [SerializeField] private Sprite circleSprite;
    [SerializeField] private Sprite carSprite;
    [SerializeField] private Sprite flowerSprite;
    [SerializeField] private float characterSpriteScale = 0.95f;

    [Header("Scene Objects")]
    [SerializeField] private Camera mainCamera;
    [SerializeField] private Transform generatedRoot;

    [Header("UI")]
    [SerializeField] private GameObject hudPanel;
    [SerializeField] private Text roundText;
    [SerializeField] private Text smoothChoicesText;
    [SerializeField] private Text statusText;
    [SerializeField] private GameObject startPanel;
    [SerializeField] private Button startButton;
    [SerializeField] private Dropdown characterDropdown;
    [SerializeField] private Image characterPreviewImage;
    [SerializeField] private GameObject resultPanel;
    [SerializeField] private Text resultTitleText;
    [SerializeField] private Text resultTimeText;
    [SerializeField] private Text resultSmoothChoicesText;
    [SerializeField] private Text currentRankText;
    [SerializeField] private Text rankingText;
    [SerializeField] private Button playAgainButton;

    private readonly List<float> rankingTimes = new List<float>();
    private StagePlan[] stagePlans;
    private GameState state = GameState.WaitingToStart;
    private GameObject character;
    private GameObject flag;
    private Vector3[] currentMovePath;
    private int currentPathIndex;
    private int roundIndex;
    private int smoothChoices;
    private int currentRank;
    private float runStartTime;
    private float finishTimeSeconds;
    private float nextAcceptedInputTime;
    private RopeSide selectedSide;
    private Vector3 currentStageGoal;

    private int StageCount => Mathf.Max(2, totalStages);

    private void Awake()
    {
        ResolveReferences();
        ConfigureCamera();
        EnsureGeneratedRoot();
        EnsureCharacter();
        LoadRankingTimes();
        BeginWaitingState();
    }

    private void OnEnable()
    {
        if (input != null)
        {
            input.OnChoice.AddListener(HandleChoice);
        }

        if (startButton != null)
        {
            startButton.onClick.AddListener(StartRun);
        }

        if (playAgainButton != null)
        {
            playAgainButton.onClick.AddListener(BeginWaitingState);
        }

        if (characterDropdown != null)
        {
            ConfigureCharacterDropdown();
            UpdateCharacterPreview();
            characterDropdown.onValueChanged.AddListener(HandleCharacterChanged);
        }
    }

    private void OnDisable()
    {
        if (input != null)
        {
            input.OnChoice.RemoveListener(HandleChoice);
        }

        if (startButton != null)
        {
            startButton.onClick.RemoveListener(StartRun);
        }

        if (playAgainButton != null)
        {
            playAgainButton.onClick.RemoveListener(BeginWaitingState);
        }

        if (characterDropdown != null)
        {
            characterDropdown.onValueChanged.RemoveListener(HandleCharacterChanged);
        }
    }

    private void Update()
    {
        if (state == GameState.Moving)
        {
            UpdateMovement();
        }

        UpdateCameraFollow();
        UpdateUi();
    }

    public void HandleChoice(int choice)
    {
        TryChooseSide(choice == 0 ? RopeSide.Left : RopeSide.Right);
    }

    private void TryChooseSide(RopeSide side)
    {
        if (state != GameState.AwaitingChoice || Time.time < nextAcceptedInputTime)
        {
            return;
        }

        StagePlan plan = stagePlans[roundIndex];
        selectedSide = side;
        plan.SelectedSide = side;
        plan.HasSelection = true;
        stagePlans[roundIndex] = plan;

        if (side == plan.SmoothSide)
        {
            smoothChoices++;
        }

        currentMovePath = side == RopeSide.Left ? plan.LeftPath : plan.RightPath;
        currentPathIndex = 1;
        state = GameState.Moving;
        ApplyStageColors();
    }

    private void StartRun()
    {
        roundIndex = 0;
        smoothChoices = 0;
        finishTimeSeconds = 0.0f;
        currentRank = 0;
        character.transform.position = startPosition;
        character.SetActive(true);
        ClearGeneratedRopes();
        SetCameraY(startPosition.y + 2.1f);
        HideFlag();
        runStartTime = Time.time;
        BuildAllStages();
        GenerateRound();
    }

    private void BeginWaitingState()
    {
        state = GameState.WaitingToStart;
        roundIndex = 0;
        smoothChoices = 0;
        finishTimeSeconds = 0.0f;
        currentStageGoal = startPosition + Vector3.up * stageHeight;
        ClearGeneratedRopes();
        HideFlag();
        EnsureCharacter();
        character.transform.position = startPosition;
        character.SetActive(false);
        SetCameraY(startPosition.y + 2.1f);
        UpdateUi();
    }

    private void GenerateRound()
    {
        state = GameState.AwaitingChoice;
        nextAcceptedInputTime = Time.time + decisionDelaySeconds;

        StagePlan plan = stagePlans[roundIndex];
        currentStageGoal = plan.Goal;
        ApplyStageColors();

        if (roundIndex == StageCount - 1)
        {
            ShowFlag(currentStageGoal);
        }
    }

    private void UpdateMovement()
    {
        if (currentMovePath == null || currentMovePath.Length == 0)
        {
            state = GameState.AwaitingChoice;
            return;
        }

        Vector3 target = currentMovePath[Mathf.Clamp(currentPathIndex, 0, currentMovePath.Length - 1)];
        character.transform.position = Vector3.MoveTowards(character.transform.position, target, moveSpeed * Time.deltaTime);

        if ((character.transform.position - target).sqrMagnitude > 0.0001f)
        {
            return;
        }

        currentPathIndex++;
        if (currentPathIndex < currentMovePath.Length)
        {
            return;
        }

        roundIndex++;
        if (roundIndex >= StageCount)
        {
            FinishRun();
        }
        else
        {
            GenerateRound();
        }
    }

    private void FinishRun()
    {
        state = GameState.Finished;
        finishTimeSeconds = Time.time - runStartTime;
        rankingTimes.Add(finishTimeSeconds);
        rankingTimes.Sort();
        currentRank = rankingTimes.IndexOf(finishTimeSeconds) + 1;
        SaveRankingTimes();
        ShowFlag(currentStageGoal);
        ApplyStageColors();
    }

    private void BuildAllStages()
    {
        ClearGeneratedRopes();
        stagePlans = new StagePlan[StageCount];

        int leftShortCount = 0;
        int rightShortCount = 0;
        for (int i = 0; i < StageCount; i++)
        {
            Vector3 start = startPosition + Vector3.up * stageHeight * i;
            Vector3 goal = startPosition + Vector3.up * stageHeight * (i + 1);
            RopeSide smoothSide = ChooseSmoothSide(i, leftShortCount, rightShortCount);
            if (smoothSide == RopeSide.Left)
            {
                leftShortCount++;
            }
            else
            {
                rightShortCount++;
            }

            var leftPath = BuildPath(start, goal, RopeSide.Left, smoothSide == RopeSide.Left);
            var rightPath = BuildPath(start, goal, RopeSide.Right, smoothSide == RopeSide.Right);
            var leftRenderer = CreateRopeRenderer($"Stage {i + 1} Left Rope");
            var rightRenderer = CreateRopeRenderer($"Stage {i + 1} Right Rope");

            stagePlans[i] = new StagePlan
            {
                Start = start,
                Goal = goal,
                SmoothSide = smoothSide,
                LeftPath = leftPath,
                RightPath = rightPath,
                LeftRenderer = leftRenderer,
                RightRenderer = rightRenderer
            };

            ApplyPath(leftRenderer, leftPath, futureRopeColor, 0.7f);
            ApplyPath(rightRenderer, rightPath, futureRopeColor, 0.7f);
        }
    }

    private RopeSide ChooseSmoothSide(int stageIndex, int leftShortCount, int rightShortCount)
    {
        if (!keepShortRoutesBalanced)
        {
            return UnityEngine.Random.value < 0.5f ? RopeSide.Left : RopeSide.Right;
        }

        int remaining = StageCount - stageIndex;
        int targetPerSide = StageCount / 2;
        if (leftShortCount >= targetPerSide)
        {
            return RopeSide.Right;
        }

        if (rightShortCount >= targetPerSide)
        {
            return RopeSide.Left;
        }

        if (remaining <= Mathf.Abs(leftShortCount - rightShortCount))
        {
            return leftShortCount <= rightShortCount ? RopeSide.Left : RopeSide.Right;
        }

        return UnityEngine.Random.value < 0.5f ? RopeSide.Left : RopeSide.Right;
    }

    private Vector3[] BuildPath(Vector3 start, Vector3 goal, RopeSide side, bool isSmooth)
    {
        int count = isSmooth ? shortRoutePointCount : longRoutePointCount;
        Vector3[] points = new Vector3[count];
        float sideSign = side == RopeSide.Left ? -1.0f : 1.0f;

        for (int i = 0; i < count; i++)
        {
            float t = i / (float)(count - 1);
            float y = Mathf.Lerp(start.y, goal.y, t);
            float arch = Mathf.Sin(t * Mathf.PI);
            float x = sideSign * routeSpread * arch;

            if (!isSmooth)
            {
                x += sideSign * Mathf.Sin(t * Mathf.PI * longRouteJagFrequency) * longRouteJagAmplitude * arch;
                y += Mathf.Cos(t * Mathf.PI * longRouteJagFrequency * 0.55f) * longRouteJagAmplitude * 0.45f * arch;
            }

            points[i] = new Vector3(x, y, 0.0f);
        }

        points[0] = start;
        points[count - 1] = goal;
        return points;
    }

    private LineRenderer CreateRopeRenderer(string objectName)
    {
        var rope = new GameObject(objectName);
        rope.transform.SetParent(generatedRoot, false);
        var renderer = rope.AddComponent<LineRenderer>();
        renderer.useWorldSpace = true;
        renderer.material = new Material(Shader.Find("Sprites/Default"));
        renderer.numCornerVertices = 12;
        renderer.numCapVertices = 12;
        renderer.alignment = LineAlignment.View;
        renderer.textureMode = LineTextureMode.Stretch;
        return renderer;
    }

    private void ApplyPath(LineRenderer renderer, Vector3[] points, Color color, float widthMultiplier)
    {
        renderer.enabled = true;
        renderer.positionCount = points.Length;
        renderer.SetPositions(points);
        renderer.startColor = color;
        renderer.endColor = color;
        renderer.startWidth = ropeWidth * widthMultiplier;
        renderer.endWidth = ropeWidth * widthMultiplier;
    }

    private void ApplyStageColors()
    {
        if (stagePlans == null)
        {
            return;
        }

        for (int i = 0; i < stagePlans.Length; i++)
        {
            StagePlan plan = stagePlans[i];
            if (i < roundIndex)
            {
                Color leftColor = plan.HasSelection && plan.SelectedSide == RopeSide.Left
                    ? (plan.SelectedSide == plan.SmoothSide ? smoothTargetColor : wrongRouteColor)
                    : completedRopeColor;
                Color rightColor = plan.HasSelection && plan.SelectedSide == RopeSide.Right
                    ? (plan.SelectedSide == plan.SmoothSide ? smoothTargetColor : wrongRouteColor)
                    : completedRopeColor;
                ApplyPath(plan.LeftRenderer, plan.LeftPath, leftColor, 0.82f);
                ApplyPath(plan.RightRenderer, plan.RightPath, rightColor, 0.82f);
            }
            else if (i == roundIndex)
            {
                ApplyPath(plan.LeftRenderer, plan.LeftPath, ropeColor, plan.SmoothSide == RopeSide.Left ? 1.45f : 1.0f);
                ApplyPath(plan.RightRenderer, plan.RightPath, ropeColor, plan.SmoothSide == RopeSide.Right ? 1.45f : 1.0f);
            }
            else
            {
                ApplyPath(plan.LeftRenderer, plan.LeftPath, futureRopeColor, 0.68f);
                ApplyPath(plan.RightRenderer, plan.RightPath, futureRopeColor, 0.68f);
            }
        }
    }

    private void ResolveReferences()
    {
        if (mainCamera == null)
        {
            mainCamera = Camera.main;
        }

        if (input == null)
        {
            input = FindObjectOfType<RopeChoiceInputAggregator>();
        }
    }

    private void ConfigureCamera()
    {
        if (mainCamera == null)
        {
            return;
        }

        mainCamera.orthographic = true;
        mainCamera.orthographicSize = 5.0f;
        mainCamera.clearFlags = CameraClearFlags.SolidColor;
        mainCamera.backgroundColor = backgroundColor;
        SetCameraY(startPosition.y + 2.1f);
    }

    private void EnsureGeneratedRoot()
    {
        if (generatedRoot != null)
        {
            return;
        }

        var root = new GameObject("Generated Rope Choice Objects");
        generatedRoot = root.transform;
    }

    private void EnsureCharacter()
    {
        if (character != null)
        {
            UpdateCharacterVisual();
            return;
        }

        character = new GameObject("Player Character");
        character.transform.SetParent(generatedRoot, false);
        UpdateCharacterVisual();
    }

    private void UpdateCharacterVisual()
    {
        foreach (Transform child in character.transform)
        {
            Destroy(child.gameObject);
        }

        Sprite sprite = GetSelectedCharacterSprite();
        if (sprite != null)
        {
            CreateSpritePart(sprite);
            return;
        }

        switch (selectedCharacter)
        {
            case CharacterKind.Car:
                CreateBoxPart("Body", new Vector2(0.7f, 0.32f), Vector3.zero, characterColor);
                CreateBoxPart("Cabin", new Vector2(0.36f, 0.24f), new Vector3(0.06f, 0.25f, 0.0f), characterColor * 1.12f);
                CreateCirclePart("Wheel L", 0.13f, new Vector3(-0.24f, -0.22f, 0.0f), Color.white);
                CreateCirclePart("Wheel R", 0.13f, new Vector3(0.24f, -0.22f, 0.0f), Color.white);
                break;
            case CharacterKind.Flower:
                for (int i = 0; i < 6; i++)
                {
                    float angle = i * Mathf.PI * 2.0f / 6.0f;
                    CreateCirclePart($"Petal {i + 1}", 0.18f, new Vector3(Mathf.Cos(angle) * 0.24f, Mathf.Sin(angle) * 0.24f, 0.0f), new Color(1.0f, 0.78f, 0.30f));
                }
                CreateCirclePart("Center", 0.18f, Vector3.zero, characterColor);
                break;
            default:
                CreateCirclePart("Circle", 0.34f, Vector3.zero, characterColor);
                break;
        }
    }

    private Sprite GetSelectedCharacterSprite()
    {
        switch (selectedCharacter)
        {
            case CharacterKind.Car:
                return carSprite;
            case CharacterKind.Flower:
                return flowerSprite;
            default:
                return circleSprite;
        }
    }

    private void CreateSpritePart(Sprite sprite)
    {
        var part = new GameObject("Character Sprite");
        part.transform.SetParent(character.transform, false);
        part.transform.localScale = Vector3.one * characterSpriteScale;
        var renderer = part.AddComponent<SpriteRenderer>();
        renderer.sprite = sprite;
        renderer.sortingOrder = 20;
    }

    private void CreateCirclePart(string objectName, float radius, Vector3 localPosition, Color color)
    {
        var part = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        part.name = objectName;
        part.transform.SetParent(character.transform, false);
        part.transform.localPosition = localPosition;
        part.transform.localScale = new Vector3(radius, radius, 0.04f);
        var renderer = part.GetComponent<Renderer>();
        renderer.material = CreateCharacterMaterial();
        renderer.material.color = color;
    }

    private void CreateBoxPart(string objectName, Vector2 size, Vector3 localPosition, Color color)
    {
        var part = GameObject.CreatePrimitive(PrimitiveType.Cube);
        part.name = objectName;
        part.transform.SetParent(character.transform, false);
        part.transform.localPosition = localPosition;
        part.transform.localScale = new Vector3(size.x, size.y, 0.08f);
        var renderer = part.GetComponent<Renderer>();
        renderer.material = CreateCharacterMaterial();
        renderer.material.color = color;
    }

    private Material CreateCharacterMaterial()
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null)
        {
            shader = Shader.Find("Sprites/Default");
        }

        return new Material(shader);
    }

    private void ShowFlag(Vector3 goal)
    {
        if (flag == null)
        {
            flag = new GameObject("Goal Flag");
            flag.transform.SetParent(generatedRoot, false);
            var pole = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pole.name = "Pole";
            pole.transform.SetParent(flag.transform, false);
            pole.transform.localScale = new Vector3(0.06f, 0.95f, 0.05f);
            pole.transform.localPosition = new Vector3(0.0f, -0.45f, 0.0f);
            pole.GetComponent<Renderer>().material.color = Color.white;

            var cloth = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cloth.name = "Flag";
            cloth.transform.SetParent(flag.transform, false);
            cloth.transform.localScale = new Vector3(0.58f, 0.34f, 0.05f);
            cloth.transform.localPosition = new Vector3(0.29f, -0.12f, 0.0f);
            cloth.GetComponent<Renderer>().material.color = smoothTargetColor;
        }

        flag.transform.position = goal + new Vector3(0.35f, 0.85f, 0.0f);
        flag.SetActive(true);
    }

    private void HideFlag()
    {
        if (flag != null)
        {
            flag.SetActive(false);
        }
    }

    private void ClearGeneratedRopes()
    {
        if (generatedRoot == null)
        {
            return;
        }

        for (int i = generatedRoot.childCount - 1; i >= 0; i--)
        {
            Transform child = generatedRoot.GetChild(i);
            if (child.gameObject == character || child.gameObject == flag)
            {
                continue;
            }

            Destroy(child.gameObject);
        }
    }

    private void UpdateCameraFollow()
    {
        if (mainCamera == null || character == null || !character.activeSelf)
        {
            return;
        }

        float targetY = Mathf.Max(startPosition.y + 2.1f, character.transform.position.y + 2.1f);
        float y = Mathf.Lerp(mainCamera.transform.position.y, targetY, Time.deltaTime * cameraFollowSpeed);
        SetCameraY(y);
    }

    private void SetCameraY(float y)
    {
        if (mainCamera != null)
        {
            mainCamera.transform.position = new Vector3(0.0f, y, -10.0f);
        }
    }

    private void UpdateUi()
    {
        SetActive(hudPanel, state == GameState.AwaitingChoice || state == GameState.Moving);
        SetActive(startPanel, state == GameState.WaitingToStart);
        SetActive(resultPanel, state == GameState.Finished);

        SetText(roundText, $"Stage {Mathf.Min(roundIndex + 1, StageCount)} / {StageCount}");
        SetText(smoothChoicesText, $"Smooth {smoothChoices} / {StageCount}");
        SetText(statusText, BuildStatusText());
        SetText(resultTitleText, "おめでとう！");
        SetText(resultTimeText, $"Time: {finishTimeSeconds:F2} sec");
        SetText(resultSmoothChoicesText, $"Smooth choices: {smoothChoices} / {StageCount}");
        SetText(currentRankText, currentRank > 0 ? $"Your Rank: {currentRank}" : "Your Rank: -");
        SetText(rankingText, BuildRankingText());
    }

    private string BuildStatusText()
    {
        if (state == GameState.AwaitingChoice)
        {
            return "Imagine LEFT or RIGHT";
        }

        if (state == GameState.Moving)
        {
            return selectedSide == RopeSide.Left ? "LEFT" : "RIGHT";
        }

        return "Waiting";
    }

    private void ConfigureCharacterDropdown()
    {
        EnsureDropdownTemplate(characterDropdown);
        characterDropdown.ClearOptions();
        characterDropdown.AddOptions(new List<string> { "Circle", "Car", "Flower" });
        characterDropdown.value = (int)selectedCharacter;
        characterDropdown.RefreshShownValue();
    }

    private void HandleCharacterChanged(int value)
    {
        selectedCharacter = (CharacterKind)Mathf.Clamp(value, 0, 2);
        EnsureCharacter();
        UpdateCharacterPreview();
    }

    private void UpdateCharacterPreview()
    {
        if (characterPreviewImage == null && startPanel != null)
        {
            Transform existingPreview = startPanel.transform.Find("Character Preview");
            characterPreviewImage = existingPreview != null
                ? existingPreview.GetComponent<Image>()
                : CreateCharacterPreviewImage(startPanel.transform);
        }

        if (characterPreviewImage == null)
        {
            return;
        }

        characterPreviewImage.sprite = GetSelectedCharacterSprite();
        characterPreviewImage.preserveAspect = true;
        characterPreviewImage.color = Color.white;
    }

    private static void EnsureDropdownTemplate(Dropdown dropdown)
    {
        if (dropdown == null || (dropdown.template != null && dropdown.itemText != null))
        {
            return;
        }

        RectTransform root = dropdown.GetComponent<RectTransform>();
        RectTransform template = FindOrCreateRect(dropdown.transform, "Template");
        ConfigureRect(template, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -6f), new Vector2(0f, 126f), new Vector2(0.5f, 1f));
        var templateImage = GetOrAddComponent<Image>(template.gameObject);
        templateImage.color = new Color(0.04f, 0.05f, 0.08f, 0.96f);
        var scrollRect = GetOrAddComponent<ScrollRect>(template.gameObject);

        RectTransform viewport = FindOrCreateRect(template, "Viewport");
        ConfigureRect(viewport, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero, new Vector2(0.5f, 0.5f));
        var viewportImage = GetOrAddComponent<Image>(viewport.gameObject);
        viewportImage.color = new Color(1f, 1f, 1f, 0.04f);
        var mask = GetOrAddComponent<Mask>(viewport.gameObject);
        mask.showMaskGraphic = false;

        RectTransform content = FindOrCreateRect(viewport, "Content");
        ConfigureRect(content, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(0.5f, 1f));

        RectTransform item = FindOrCreateRect(content, "Item");
        ConfigureRect(item, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -18f), new Vector2(0f, 36f), new Vector2(0.5f, 1f));
        var itemImage = GetOrAddComponent<Image>(item.gameObject);
        itemImage.color = new Color(1f, 0.88f, 0.16f, 0.24f);
        var toggle = GetOrAddComponent<Toggle>(item.gameObject);
        toggle.targetGraphic = itemImage;

        RectTransform itemLabel = FindOrCreateRect(item, "Item Label");
        ConfigureRect(itemLabel, Vector2.zero, Vector2.one, new Vector2(18f, 0f), new Vector2(-36f, 0f), new Vector2(0.5f, 0.5f));
        var itemText = GetOrAddComponent<Text>(itemLabel.gameObject);
        itemText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        itemText.fontSize = 22;
        itemText.fontStyle = FontStyle.Bold;
        itemText.alignment = TextAnchor.MiddleLeft;
        itemText.color = Color.white;
        itemText.raycastTarget = false;

        scrollRect.content = content;
        scrollRect.viewport = viewport;
        scrollRect.horizontal = false;
        scrollRect.vertical = true;

        dropdown.template = template;
        dropdown.itemText = itemText;
        if (dropdown.captionText == null)
        {
            dropdown.captionText = dropdown.GetComponentInChildren<Text>();
        }

        if (root != null && root.sizeDelta.y > 0f)
        {
            template.sizeDelta = new Vector2(template.sizeDelta.x, root.sizeDelta.y * 3f);
        }

        template.gameObject.SetActive(false);
    }

    private static Image CreateCharacterPreviewImage(Transform parent)
    {
        var go = new GameObject("Character Preview", typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        ConfigureRect(rect, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(0.0f, 64.0f), new Vector2(108.0f, 108.0f), new Vector2(0.5f, 0.5f));
        var image = go.AddComponent<Image>();
        image.color = Color.white;
        image.preserveAspect = true;
        image.raycastTarget = false;
        return image;
    }

    private static RectTransform FindOrCreateRect(Transform parent, string objectName)
    {
        Transform existing = parent.Find(objectName);
        if (existing != null)
        {
            return existing.GetComponent<RectTransform>() ?? existing.gameObject.AddComponent<RectTransform>();
        }

        var go = new GameObject(objectName);
        go.transform.SetParent(parent, false);
        return go.AddComponent<RectTransform>();
    }

    private static void ConfigureRect(RectTransform rect, Vector2 anchorMin, Vector2 anchorMax, Vector2 anchoredPosition, Vector2 sizeDelta, Vector2 pivot)
    {
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = sizeDelta;
        rect.pivot = pivot;
    }

    private static T GetOrAddComponent<T>(GameObject target) where T : Component
    {
        return target.TryGetComponent(out T component) ? component : target.AddComponent<T>();
    }

    private void LoadRankingTimes()
    {
        rankingTimes.Clear();
        string raw = PlayerPrefs.GetString(RankingPrefsKey, string.Empty);
        if (string.IsNullOrEmpty(raw))
        {
            return;
        }

        rankingTimes.AddRange(raw.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(value => float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) ? parsed : -1.0f)
            .Where(value => value >= 0.0f));
        rankingTimes.Sort();
    }

    private void SaveRankingTimes()
    {
        string raw = string.Join(";", rankingTimes.Take(20).Select(value => value.ToString("R", CultureInfo.InvariantCulture)));
        PlayerPrefs.SetString(RankingPrefsKey, raw);
        PlayerPrefs.Save();
    }

    private string BuildRankingText()
    {
        if (rankingTimes.Count == 0)
        {
            return "No records yet";
        }

        return string.Join("\n", rankingTimes.Take(5).Select((time, index) => $"{index + 1}. {time:F2} sec"));
    }

    private static void SetText(Text target, string value)
    {
        if (target != null)
        {
            target.text = value;
        }
    }

    private static void SetActive(GameObject target, bool isActive)
    {
        if (target != null && target.activeSelf != isActive)
        {
            target.SetActive(isActive);
        }
    }
}
