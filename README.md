# RL-Based Legged Locomotion 🤖

**ME 228 Course Project**

A biped robot trained from scratch in Unity3D to learn stable forward walking using Proximal Policy Optimization (PPO) — without imitation learning or predefined gait schedules. The objective is to study the emergence of locomotion purely through reward-driven interaction with a physics-based simulation environment.

---

## 👥 Team

| Name | Roll Number |
|------|-------------|
| Sahil Kadale | 24B2146 |
| Atreem Das | 24B2252 |
| Aniket Pol | 24B2258 |
| Anmol Raj | 24B2260 |

---

## 🎯 Project Overview

### Problem Statement
Legged locomotion is a nonlinear, contact-rich control problem involving dynamic balance, coordinated multi-joint actuation, and intermittent ground contact. Traditional robotic control methods rely on predefined gait trajectories and precise dynamic modeling. This project investigates whether a biped robot trained from scratch in Unity3D can learn stable forward walking using reinforcement learning without imitation learning or predefined gait schedules.

### Key Challenges
- Balance stabilization
- Joint coordination
- Contact timing
- Preventing collapse during early training phases

---

## 🛠️ Tech Stack

- **Unity 2022.3 LTS** — Physics simulation environment
- **Unity ML-Agents Toolkit (v0.30)** — RL framework
- **Python 3.9+** with PyTorch — Training backend
- **PPO (Proximal Policy Optimization)** — RL algorithm
- **TensorBoard** — Training visualization

---

## 📁 Repository Structure

```
RL-Legged-Locomotion/
├── Assets/                    # Unity project assets
│   ├── Scripts/               # C# agent scripts
│   │   ├── BipedAgent.cs      # Main RL agent logic
│   │   └── MetricsLogger.cs   # Evaluation logger
│   ├── Prefabs/               # Biped prefab
│   ├── Scenes/                # Training & inference scenes
│   └── Models/                # Trained .onnx models
├── config/                    # PPO YAML configurations
│   └── biped_walker.yaml      # PPO hyperparameters
├── results/                   # Training outputs
│   ├── checkpoints/           # Model checkpoints
│   └── logs/                  # TensorBoard logs
├── evaluation/                # Evaluation pipeline
│   ├── scripts/               # Python eval scripts
│   │   ├── run_evaluation.py  # Compute success criteria
│   │   └── plot_results.py    # Generate plots
│   ├── data/                  # CSV outputs
│   └── plots/                 # Generated figures
├── docs/                      # Documentation
│   ├── figures/               # Diagrams
│   └── report/                # Final report
├── demo/                      # Demo videos
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```
git clone https://github.com/AniketPol-27/RL-Legged-Locomotion.git
cd RL-Legged-Locomotion
```

### 2. Setup Python Environment

**Windows:**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Open Unity Project

1. Install **Unity Hub** and **Unity 2022.3 LTS**
2. Open Unity Hub → **Add project** → select this folder
3. Install ML-Agents package via **Window → Package Manager**

### 4. Train the Agent

```
mlagents-learn config/biped_walker.yaml --run-id=biped_v1
```

Then press **Play** in Unity.

### 5. Monitor Training

```
tensorboard --logdir results
```

Open browser at `http://localhost:6006`

### 6. Run Evaluation

After training, switch the agent to **Inference Only** mode and load the trained `.onnx` model:

1. Open `Assets/Scenes/BipedInference.unity`
2. Drag trained `.onnx` from `Assets/Models/` to the **Behavior Parameters** component
3. Press **Play** (runs 50 trials automatically)
4. Analyze results:

```
python evaluation/scripts/run_evaluation.py
python evaluation/scripts/plot_results.py
```

---

## 🧠 Methodology

### Biped Model
- **~12 Degrees of Freedom**:
  - 2× Hip joints (3 DOF each)
  - 2× Knee joints (1 DOF each)
  - 2× Ankle joints (2 DOF each)
- Built using Unity's **ArticulationBody** components

### Observation Space (~40 floats)
- Pelvis up vector (3)
- Pelvis linear velocity (3)
- Pelvis angular velocity (3)
- Per-joint local rotation quaternion (4 × N)
- Per-joint angular velocity (3 × N)
- Foot ground contact booleans (1 per foot)

### Action Space
- **12 continuous torque outputs** ∈ [-1, 1]
- Scaled by `maxTorque = 100`
- Applied to joint actuators

### Reward Function

```
r_t = w1 * v_forward + w2 * upright - w3 * energy + w4 * alive
```

| Term | Weight | Purpose |
|------|--------|---------|
| Forward velocity | w1 = 1.0 | Encourage forward progression |
| Upright bonus | w2 = 0.5 | Maintain vertical posture |
| Energy penalty | w3 = 0.005 | Reward efficient movement |
| Alive bonus | w4 = 0.1 | Discourage early termination |
| Fall penalty | -1.0 | Penalize falling |

### Network Architecture
- **MLP** with 3 hidden layers × 256 units each
- PPO hyperparameters: gamma=0.995, lambda=0.95, learning rate 3e-4
- Batch size: 2048, Buffer size: 20480

### Episode Termination
- Pelvis height < 60% of initial height (fall)
- Tilt angle > 60° from vertical
- Maximum 1000 steps per episode

---

## 🎯 Success Criteria

| # | Criterion | Target |
|---|-----------|--------|
| 1 | Forward walking distance | ≥ 5 m without falling |
| 2 | Upright stability | Maintained throughout episode |
| 3 | Fall rate | < 20% over 50 trials |
| 4 | Gait emergence | Periodic joint coordination pattern |

### Performance Metrics
- Episodic reward (mean ± std)
- Distance traveled (m)
- Stability duration (s)
- Energy usage per meter

---

## 📊 Results

Training curves and evaluation plots are stored in:
- `results/` — TensorBoard logs and checkpoints
- `evaluation/plots/` — Distance per trial, reward distribution, stability analysis, success rate
- `docs/report/` — Full project report

---

## 🔗 Base Work & References

The project builds upon:

1. **Unity ML-Agents Toolkit** — Unity Technologies  
   https://github.com/Unity-Technologies/ml-agents

2. **Proximal Policy Optimization** — Schulman et al., 2017  
   https://arxiv.org/abs/1707.06347

3. **OpenAI Gym Locomotion Benchmarks** — Ant, HalfCheetah  
   https://www.gymlibrary.dev

> **Note**: No imitation learning or hand-crafted gait scheduling is used. The biped learns walking purely through reward-driven RL.

---

## 📂 Dataset

**No static dataset is required.** Training data is generated dynamically via interaction within the Unity simulation environment.

- **Observation space**: joint angles, angular velocities, body orientation, linear velocity
- **Action space**: continuous control signals applied to each joint actuator

---

## 🧪 Reproducing Results

```
# 1. Activate environment
source venv/bin/activate    # or venv\Scripts\activate on Windows

# 2. Train (3M steps, ~12 hours on CPU with 16 parallel agents)
mlagents-learn config/biped_walker.yaml --run-id=biped_v1

# 3. Press Play in Unity to start training

# 4. Resume if interrupted
mlagents-learn config/biped_walker.yaml --run-id=biped_v1 --resume

# 5. Run evaluation (50 trials in Unity inference mode)
python evaluation/scripts/run_evaluation.py
python evaluation/scripts/plot_results.py
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `mlagents-learn: command not found` | Activate virtual environment first |
| Unity crashes during training | Reduce parallel agents from 16 → 4 |
| Training diverges | Lower learning rate, normalize observations |
| Robot exploits reward (dives forward) | Increase upright weight, add action smoothness penalty |
| `.onnx` model won't load | Ensure Behavior Name matches YAML (`BipedWalker`) |

---

## 📝 License

MIT License — Academic use only.

---

## 🙏 Acknowledgments

- **Course**: ME 228 (IIT Bombay)
- **Unity Technologies** for the ML-Agents Toolkit
- **OpenAI** for PPO and Gym benchmarks

---

**Project Status**: 🟢 In Progress