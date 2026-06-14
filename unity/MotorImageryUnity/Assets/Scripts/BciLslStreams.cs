using System;
using UnityEngine;
using LSL;

public class BciLslStreams : MonoBehaviour
{
    public event Action<string> StatusReceived;
    public event Action<float, float> ProbaReceived;

    private StreamInlet statusInlet;
    private StreamInlet probaInlet;
    private StreamOutlet commandOutlet;
    private StreamOutlet markerOutlet;
    private ContinuousResolver statusResolver;
    private ContinuousResolver probaResolver;

    private readonly string[] statusSample = new string[1];
    private readonly float[] probaSample = new float[2];
    private readonly string[] commandSample = new string[1];
    private readonly int[] markerSample = new int[1];
    private string lastStatusMessage = "";
    private float nextResolveAt;
    private bool initialized;
    // True once we've sent query_config for the currently open status inlet.
    // Reset when the inlet drops so we re-query on reconnect.
    private bool queriedConfigForCurrentInlet;

    private const float ResolveIntervalSeconds = 1.0f;

    public void Initialize()
    {
        if (initialized)
        {
            return;
        }

        statusResolver = new ContinuousResolver("type", "Status");
        probaResolver = new ContinuousResolver("type", "BCI_Proba");
        commandOutlet = CreateCommandOutlet();
        markerOutlet = CreateMarkerOutlet();
        initialized = true;
        TryResolveInlets();
    }

    public void PushCommand(string command)
    {
        if (commandOutlet == null)
        {
            Debug.LogWarning($"Command ignored before Commands stream is ready: {command}");
            return;
        }

        commandSample[0] = command;
        commandOutlet.push_sample(commandSample, LSL.LSL.local_clock());
        Debug.Log($"Sent command: {command}");
    }

    public void PushMarker(int marker)
    {
        if (markerOutlet == null)
        {
            Debug.LogWarning($"Marker ignored before Markers stream is ready: {marker}");
            return;
        }

        markerSample[0] = marker;
        markerOutlet.push_sample(markerSample, LSL.LSL.local_clock());
    }

    private void Update()
    {
        if (!initialized)
        {
            return;
        }

        if (Time.time >= nextResolveAt)
        {
            TryResolveInlets();
            nextResolveAt = Time.time + ResolveIntervalSeconds;
        }

        // Ask the service for its config the moment we can both receive
        // its replies (status inlet) and send commands (outlet). This
        // gives Unity a fresh snapshot at connect time without manual
        // sync. On reconnect (statusInlet goes null after a LostException
        // and is later re-resolved), we re-query.
        if (!queriedConfigForCurrentInlet && statusInlet != null && commandOutlet != null)
        {
            PushCommand("query_config");
            queriedConfigForCurrentInlet = true;
        }

        PullStatus();
        PullProba();
    }

    private StreamOutlet CreateCommandOutlet()
    {
        var info = new StreamInfo(
            "UnityCommands",
            "Commands",
            1,
            LSL.LSL.IRREGULAR_RATE,
            channel_format_t.cf_string,
            "unity_commands"
        );
        return new StreamOutlet(info);
    }

    private StreamOutlet CreateMarkerOutlet()
    {
        var info = new StreamInfo(
            "UnityMarkers",
            "Markers",
            1,
            LSL.LSL.IRREGULAR_RATE,
            channel_format_t.cf_int32,
            "unity_markers"
        );
        return new StreamOutlet(info);
    }

    private void TryResolveInlets()
    {
        if (statusInlet == null)
        {
            statusInlet = TryOpenInlet(statusResolver, "Status");
        }

        if (probaInlet == null)
        {
            probaInlet = TryOpenInlet(probaResolver, "BCI_Proba");
        }
    }

    private StreamInlet TryOpenInlet(ContinuousResolver resolver, string streamType)
    {
        if (resolver == null)
        {
            return null;
        }

        var streams = resolver.results();
        if (streams.Length == 0)
        {
            return null;
        }

        try
        {
            var inlet = new StreamInlet(streams[0]);
            inlet.open_stream(1.0);
            Debug.Log($"Connected to {streamType} stream: {streams[0].name()}");
            return inlet;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"Could not open {streamType} stream yet: {ex.Message}");
            return null;
        }
    }

    private void PullStatus()
    {
        if (statusInlet == null)
        {
            return;
        }

        try
        {
            double timestamp;
            while ((timestamp = statusInlet.pull_sample(statusSample, 0.0)) != 0.0)
            {
                string message = statusSample[0];
                StatusReceived?.Invoke(message);
                if (message != lastStatusMessage)
                {
                    Debug.Log($"Status stream sample: {message}");
                    lastStatusMessage = message;
                }
            }
        }
        catch (LostException)
        {
            Debug.LogWarning("Status stream lost; will try to reconnect.");
            statusInlet.close_stream();
            statusInlet = null;
            queriedConfigForCurrentInlet = false;
            StatusReceived?.Invoke("state:DISCONNECTED");
        }
    }

    private void PullProba()
    {
        if (probaInlet == null)
        {
            return;
        }

        try
        {
            double timestamp;
            while ((timestamp = probaInlet.pull_sample(probaSample, 0.0)) != 0.0)
            {
                ProbaReceived?.Invoke(probaSample[0], probaSample[1]);
            }
        }
        catch (LostException)
        {
            Debug.LogWarning("BCI_Proba stream lost; will try to reconnect.");
            probaInlet.close_stream();
            probaInlet = null;
        }
    }

    private void OnDestroy()
    {
        statusInlet?.close_stream();
        probaInlet?.close_stream();
        commandOutlet?.Dispose();
        markerOutlet?.Dispose();
        statusResolver?.Dispose();
        probaResolver?.Dispose();
    }
}
