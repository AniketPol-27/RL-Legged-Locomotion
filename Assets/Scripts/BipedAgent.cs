using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

/// <summary>
/// PPO-based RL Agent for Biped Locomotion.
/// Observation: joint angles, angular velocities, body orientation, linear velocity
/// Action: continuous torques per joint actuator
/// Reward: forward progression + upright stability - energy usage + alive bonus
/// </summary>
public class BipedAgent : Agent
{
    [Header("Body References")]
    [Tooltip("Pelvis / root body of the biped")]
    public Transform pelvis;
    
    [Tooltip("All articulated joints (~12 DOF total)")]
    public List<ArticulationBody> joints;
    
    [Tooltip("Foot transforms for ground contact detection")]
    public Transform[] feet;

    [Header("Reward Weights")]
    public float forwardRewardWeight = 1.0f;
    public float uprightRewardWeight = 0.5f;
    public float energyPenaltyWeight = 0.005f;
    public float aliveBonus = 0.1f;
    public float fallPenalty = -1.0f;

    [Header("Termination Thresholds")]
    public float minPelvisHeightRatio = 0.6f;
    public float maxTiltAngleDeg = 60f;

    [Header("Action Settings")]
    public float maxTorque = 100f;

    // Internal state
    private Vector3 startPosition;
    private Quaternion startRotation;
    private float initialPelvisHeight;
    private ArticulationBody pelvisBody;

    // Metrics tracking (for MetricsLogger)
    [HideInInspector] public float episodeDistance = 0f;
    [HideInInspector] public float episodeEnergy = 0f;
    [HideInInspector] public bool episodeFell = false;

    public override void Initialize()
    {
        startPosition = pelvis.position;
        startRotation = pelvis.rotation;
        initialPelvisHeight = pelvis.position.y;
        pelvisBody = pelvis.GetComponent<ArticulationBody>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset pose
        if (pelvisBody != null)
        {
            pelvisBody.TeleportRoot(startPosition, startRotation);
        }
        else
        {
            pelvis.position = startPosition;
            pelvis.rotation = startRotation;
        }

        // Reset all joints
        foreach (var j in joints)
        {
            if (j == null) continue;
            j.velocity = Vector3.zero;
            j.angularVelocity = Vector3.zero;
            j.jointPosition = new ArticulationReducedSpace(0f, 0f, 0f);
            j.jointVelocity = new ArticulationReducedSpace(0f, 0f, 0f);
        }

        // Reset metrics
        episodeDistance = 0f;
        episodeEnergy = 0f;
        episodeFell = false;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. Pelvis up vector (3 floats) — body orientation
        sensor.AddObservation(pelvis.up);

        // 2. Pelvis linear velocity (3 floats)
        Vector3 vel = pelvisBody != null ? pelvisBody.velocity : Vector3.zero;
        sensor.AddObservation(vel);

        // 3. Pelvis angular velocity (3 floats)
        Vector3 angVel = pelvisBody != null ? pelvisBody.angularVelocity : Vector3.zero;
        sensor.AddObservation(angVel);

        // 4. Per-joint observations
        foreach (var j in joints)
        {
            if (j == null) 
            { 
                // Pad with zeros if missing
                sensor.AddObservation(Quaternion.identity);
                sensor.AddObservation(Vector3.zero);
                continue;
            }
            sensor.AddObservation(j.transform.localRotation); // 4 floats
            sensor.AddObservation(j.angularVelocity);         // 3 floats
        }

        // 5. Foot ground contact (1 float per foot)
        foreach (var foot in feet)
        {
            if (foot == null) { sensor.AddObservation(0f); continue; }
            sensor.AddObservation(foot.position.y < 0.15f ? 1f : 0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var continuousActions = actions.ContinuousActions;
        float energyThisStep = 0f;

        // Apply torque to each joint
        for (int i = 0; i < joints.Count && i < continuousActions.Length; i++)
        {
            if (joints[i] == null) continue;

            float action = Mathf.Clamp(continuousActions[i], -1f, 1f);
            float torque = action * maxTorque;

            var drive = joints[i].xDrive;
            drive.target = torque;
            joints[i].xDrive = drive;

            energyThisStep += action * action;  // proxy for energy
        }
        episodeEnergy += energyThisStep;

        // ===== REWARD COMPUTATION =====
        Vector3 velocity = pelvisBody != null ? pelvisBody.velocity : Vector3.zero;

        // R1: forward progression (along +Z axis)
        float forwardReward = velocity.z * forwardRewardWeight;

        // R2: upright stability (dot product of pelvis-up with world-up)
        float uprightness = Vector3.Dot(pelvis.up, Vector3.up);
        float uprightReward = Mathf.Max(0f, uprightness) * uprightRewardWeight;

        // R3: energy penalty
        float energyPenalty = -energyThisStep * energyPenaltyWeight;

        // R4: alive bonus
        float alive = aliveBonus;

        AddReward(forwardReward + uprightReward + energyPenalty + alive);

        // Track distance
        episodeDistance = Vector3.Distance(startPosition, pelvis.position);

        // ===== TERMINATION =====
        bool fallen = pelvis.position.y < initialPelvisHeight * minPelvisHeightRatio;
        bool tilted = Vector3.Angle(pelvis.up, Vector3.up) > maxTiltAngleDeg;

        if (fallen || tilted)
        {
            episodeFell = true;
            AddReward(fallPenalty);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Manual control mode — all zeros (used for testing without policy)
        var continuousActions = actionsOut.ContinuousActions;
        for (int i = 0; i < continuousActions.Length; i++)
            continuousActions[i] = 0f;
    }
}