using UnityEngine;
using System.IO;
using System.Collections.Generic;

/// <summary>
/// Logs per-episode metrics during inference for the 50-trial evaluation.
/// Attach to the same GameObject as BipedAgent OR drag the agent in via inspector.
/// </summary>
public class MetricsLogger : MonoBehaviour
{
    [Header("References")]
    public BipedAgent agent;
    public Transform pelvis;

    [Header("Settings")]
    public int totalTrials = 50;
    public string outputPath = "evaluation/data/eval_results.csv";

    private int episodeCount = 0;
    private Vector3 episodeStartPos;
    private float episodeStartTime;
    private float uprightTime = 0f;
    private float lastEpisodeReward = 0f;
    private List<string> rows = new List<string>();
    private bool finished = false;

    void Start()
    {
        rows.Add("episode,distance_m,duration_s,upright_time_s,energy,fell,reward");
        ResetEpisodeTracking();
    }

    void ResetEpisodeTracking()
    {
        if (pelvis != null) episodeStartPos = pelvis.position;
        episodeStartTime = Time.time;
        uprightTime = 0f;
    }

    void FixedUpdate()
    {
        if (finished || pelvis == null) return;

        // Track upright time
        if (Vector3.Dot(pelvis.up, Vector3.up) > 0.7f)
            uprightTime += Time.fixedDeltaTime;

        // Detect episode end via agent's StepCount reset
        if (agent != null && agent.StepCount == 0 && Time.time > episodeStartTime + 0.1f)
        {
            LogEpisode();
            ResetEpisodeTracking();
        }
    }

    void LogEpisode()
    {
        if (finished) return;

        episodeCount++;
        float distance = agent != null ? agent.episodeDistance : 0f;
        float duration = Time.time - episodeStartTime;
        float energy = agent != null ? agent.episodeEnergy : 0f;
        bool fell = agent != null && agent.episodeFell;
        float reward = agent != null ? agent.GetCumulativeReward() : 0f;

        string row = $"{episodeCount},{distance:F3},{duration:F3},{uprightTime:F3},{energy:F3},{(fell ? 1 : 0)},{reward:F3}";
        rows.Add(row);
        Debug.Log($"[MetricsLogger] Episode {episodeCount}: dist={distance:F2}m, fell={fell}, reward={reward:F2}");

        if (episodeCount >= totalTrials)
        {
            SaveResults();
            finished = true;
        }
    }

    void SaveResults()
    {
        try
        {
            string dir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            File.WriteAllLines(outputPath, rows);
            Debug.Log($"✅ Saved {episodeCount} trials to {outputPath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to save metrics: {e.Message}");
        }

        #if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
        #endif
    }
}