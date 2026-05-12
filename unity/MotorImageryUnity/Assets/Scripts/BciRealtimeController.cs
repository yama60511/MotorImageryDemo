using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class BciRealtimeController : MonoBehaviour
{
    public TMP_Text leftProbaText;
    public TMP_Text rightProbaText;
    public TMP_Text cueText;
    public Button startRealtimeButton;
    public Button stopRealtimeButton;
    public RectTransform probaIndicatorTrack;
    public RectTransform probaIndicatorThumb;
    public TMP_Text realtimeHintText;
    public Color restCueColor = new Color(0.8f, 0.8f, 0.8f);
    public Color realtimeCueColor = new Color(0.45f, 1.0f, 0.55f);

    private BciLslStreams streams;
    private string serviceState = "DISCONNECTED";
    private string selectedSessionId = "";
    private string selectedDecoderName = "csp_lda";
    private bool pendingStartAfterLoad;
    private bool initialized;

    private const float FallbackProbaTrackWidth = 360.0f;

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
            streams.ProbaReceived += HandleProba;
        }

        if (startRealtimeButton != null)
        {
            startRealtimeButton.onClick.AddListener(StartRealtime);
        }

        if (stopRealtimeButton != null)
        {
            stopRealtimeButton.onClick.AddListener(StopRealtime);
        }

        if (leftProbaText != null)
        {
            leftProbaText.text = "p_left: ---";
        }

        if (rightProbaText != null)
        {
            rightProbaText.text = "p_right: ---";
        }

        initialized = true;
        UpdateRealtimeIndicator(0.5f, 0.5f);
        UpdateControlInteractivity();
    }

    private void StartRealtime()
    {
        if (serviceState == "IDLE" && !string.IsNullOrEmpty(selectedSessionId))
        {
            pendingStartAfterLoad = true;
            streams?.PushCommand(
                $"load_bundle:session={selectedSessionId},decoder={selectedDecoderName}"
            );
            UpdateControlInteractivity();
            SetCueText("LOADING BUNDLE", realtimeCueColor);
            if (realtimeHintText != null)
            {
                realtimeHintText.text = $"Loading {selectedSessionId} / {selectedDecoderName}";
            }
            return;
        }

        if (serviceState != "READY")
        {
            Debug.LogWarning($"Start Realtime ignored while service state is {serviceState}");
            UpdateControlInteractivity();
            return;
        }

        streams?.PushCommand("start_realtime");
        pendingStartAfterLoad = false;
        serviceState = "RUNNING";
        UpdateControlInteractivity();
        SetCueText("REALTIME STARTING", realtimeCueColor);
        if (realtimeHintText != null)
        {
            realtimeHintText.text = "Realtime starting";
        }
    }

    private void StopRealtime()
    {
        if (serviceState != "RUNNING")
        {
            Debug.LogWarning($"Stop Realtime ignored while service state is {serviceState}");
            UpdateControlInteractivity();
            return;
        }

        streams?.PushCommand("stop_realtime");
        serviceState = "READY";
        UpdateControlInteractivity();
        SetCueText("REALTIME STOPPED", restCueColor);
        if (realtimeHintText != null)
        {
            realtimeHintText.text = "Realtime stopped";
        }
    }

    private void HandleStatusMessage(string message)
    {
        if (message.StartsWith("state:"))
        {
            serviceState = message.Substring("state:".Length).Trim();
            if (serviceState == "DISCONNECTED")
            {
                pendingStartAfterLoad = false;
                SetCueText("DISCONNECTED", restCueColor);
                if (realtimeHintText != null)
                {
                    realtimeHintText.text = "Python service disconnected";
                }
            }
            UpdateControlInteractivity();
            if (pendingStartAfterLoad && serviceState == "READY")
            {
                StartRealtime();
            }
        }
        else if (message.StartsWith("training_done:"))
        {
            if (TryGetStatusValue(message, "session", out string session))
            {
                selectedSessionId = session;
            }
            if (TryGetStatusValue(message, "decoder", out string decoder))
            {
                selectedDecoderName = decoder;
            }
            if (realtimeHintText != null)
            {
                realtimeHintText.text = $"Bundle ready: {selectedSessionId} / {selectedDecoderName}";
            }
            UpdateControlInteractivity();
        }
        else if (message.StartsWith("bundle_loaded:"))
        {
            if (TryGetStatusValue(message, "session", out string session))
            {
                selectedSessionId = session;
            }
            if (TryGetStatusValue(message, "decoder", out string decoder))
            {
                selectedDecoderName = decoder;
            }
            if (realtimeHintText != null)
            {
                realtimeHintText.text = $"Loaded {selectedSessionId} / {selectedDecoderName}";
            }
            UpdateControlInteractivity();
        }
        else if (message.StartsWith("now_running:"))
        {
            string session = selectedSessionId;
            string decoder = selectedDecoderName;
            if (TryGetStatusValue(message, "session", out string parsedSession))
            {
                session = parsedSession;
                selectedSessionId = parsedSession;
            }
            if (TryGetStatusValue(message, "decoder", out string parsedDecoder))
            {
                decoder = parsedDecoder;
                selectedDecoderName = parsedDecoder;
            }
            string accText = TryGetStatusValue(message, "acc", out string acc)
                ? $" acc={acc}"
                : "";
            if (realtimeHintText != null)
            {
                realtimeHintText.text = $"Running {session} / {decoder}{accText}";
            }
        }
        else if (message.StartsWith("error:"))
        {
            pendingStartAfterLoad = false;
            UpdateControlInteractivity();
        }
    }

    private void HandleProba(float leftProba, float rightProba)
    {
        if (leftProbaText != null)
        {
            leftProbaText.text = $"p_left: {leftProba:0.000}";
        }

        if (rightProbaText != null)
        {
            rightProbaText.text = $"p_right: {rightProba:0.000}";
        }

        UpdateRealtimeIndicator(leftProba, rightProba);
    }

    private void UpdateControlInteractivity()
    {
        bool isReady = serviceState == "READY";
        bool isRunning = serviceState == "RUNNING";
        bool canLoad = serviceState == "IDLE"
            && !string.IsNullOrEmpty(selectedSessionId)
            && !pendingStartAfterLoad;

        if (startRealtimeButton != null)
        {
            startRealtimeButton.interactable = isReady || canLoad;
        }

        if (stopRealtimeButton != null)
        {
            stopRealtimeButton.interactable = isRunning;
        }
    }

    private void UpdateRealtimeIndicator(float leftProba, float rightProba)
    {
        if (probaIndicatorThumb != null)
        {
            float clampedLeft = Mathf.Clamp01(leftProba);
            float clampedRight = Mathf.Clamp01(rightProba);
            float balance = clampedRight - clampedLeft;
            float trackWidth = GetRectWidth(probaIndicatorTrack, FallbackProbaTrackWidth);
            float x = balance * (trackWidth * 0.5f);
            probaIndicatorThumb.anchoredPosition = new Vector2(
                x,
                probaIndicatorThumb.anchoredPosition.y
            );
        }

        if (realtimeHintText != null)
        {
            string side = leftProba >= rightProba ? "LEFT" : "RIGHT";
            realtimeHintText.text = $"{side} {Mathf.Max(leftProba, rightProba):0.000}";
        }
    }

    private void SetCueText(string text, Color color)
    {
        if (cueText == null)
        {
            return;
        }

        cueText.text = text;
        cueText.color = color;
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

    private void OnDestroy()
    {
        if (streams != null)
        {
            streams.StatusReceived -= HandleStatusMessage;
            streams.ProbaReceived -= HandleProba;
        }

        if (startRealtimeButton != null)
        {
            startRealtimeButton.onClick.RemoveListener(StartRealtime);
        }

        if (stopRealtimeButton != null)
        {
            stopRealtimeButton.onClick.RemoveListener(StopRealtime);
        }
    }
}
