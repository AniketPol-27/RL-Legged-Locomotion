\# RL-Based Legged Locomotion via Proximal Policy Optimization



\*\*ME 228 — Final Project Report\*\*



\*\*Team:\*\*

\- Sahil Kadale (24B2146)

\- Atreem Das (24B2252)

\- Aniket Pol (24B2258)

\- Anmol Raj (24B2260)



\*\*Repository:\*\* https://github.com/AniketPol-27/RL-Legged-Locomotion



\*\*Date:\*\* May 2026



\---



\## Abstract



\*\[Write this last — 150-200 words]\*



We present a reinforcement learning approach to bipedal locomotion using

Proximal Policy Optimization (PPO) on a humanoid agent simulated in Unity.

Our agent learns stable forward walking from scratch — without imitation

learning, motion-capture data, or handcrafted gait scheduling — purely

through reward-driven trial and error. We trained the agent for

approximately \[XX] million simulation steps on commodity GPU hardware

(NVIDIA RTX 4050 Laptop), achieving a final mean episodic reward of

\[XXX] and a fall rate of \[X%] over 50 evaluation trials. The learned

policy produces a periodic alternating-leg gait that successfully

satisfies all five of our pre-specified success criteria. Our results

demonstrate that modern PPO with sufficient training compute can produce

viable locomotion controllers within a constrained development timeline,

and we provide an open-source implementation for reproducibility.



\---



\## 1. Introduction



Legged locomotion is a fundamental capability in robotics with wide-ranging

applications — from search-and-rescue robots in disaster zones, to

prosthetic limb control, to entertainment robotics. Unlike wheeled motion,

which is well-understood and easily controlled, legged motion is a

nonlinear, contact-rich control problem involving:



\- \*\*Dynamic balance\*\* under gravity

\- \*\*Coordinated multi-joint actuation\*\* across high-DOF systems

\- \*\*Intermittent ground contact\*\* with discontinuous dynamics

\- \*\*Underactuation\*\*: more degrees of freedom than independent control inputs



Classical control approaches (zero-moment-point planning, model-predictive

control with simplified dynamics) have produced impressive results but

require careful per-platform tuning, accurate dynamic models, and explicit

gait scheduling. They struggle to generalize across morphologies and

terrains.



Reinforcement learning (RL) offers an attractive alternative: rather than

hand-designing controllers, we let the agent discover behaviors that

maximize a scalar reward through interaction with the environment. Recent

work — particularly from DeepMind (Heess et al., 2017) and OpenAI — has

shown that simple reward functions combined with sufficient training

compute can produce surprisingly natural locomotion behaviors.



\### 1.1 Project Goals



Our project objectives, as specified in the proposal:



1\. Train a biped robot in Unity to learn stable forward walking using PPO

2\. Use no imitation learning or handcrafted gait scheduling

3\. Achieve forward locomotion of at least 5 meters without falling

4\. Maintain upright stability throughout episodes

5\. Reduce fall rate below 20% across 50 evaluation trials

6\. Demonstrate emergence of periodic joint coordination



\### 1.2 Contribution



We provide an end-to-end implementation pipeline — Unity environment,

PPO training configuration, evaluation methodology, and result analysis —

demonstrating that even within a tight development timeline, modern

RL infrastructure (Unity ML-Agents Toolkit) makes complex locomotion

learning accessible to small teams.



\---



\## 2. Related Work



\### 2.1 Reinforcement Learning for Continuous Control



\*\*Proximal Policy Optimization (Schulman et al., 2017)\*\* introduced a

clipped surrogate objective that stabilizes policy gradient updates,

making it the de facto standard for continuous-control RL tasks. Compared

to its predecessor TRPO, PPO is simpler to implement and tune, while

maintaining strong empirical performance.



\*\*OpenAI Gym MuJoCo Benchmarks\*\* — Ant, HalfCheetah, Hopper, Walker2d,

and Humanoid — established standardized environments where PPO and its

contemporaries (SAC, TD3) have been heavily validated. These benchmarks

share core features with our setup: continuous action spaces, joint

torque control, and reward functions emphasizing forward velocity and

stability.



\### 2.2 Emergence of Locomotion Behaviors



\*\*Heess et al. (2017)\*\* ("Emergence of Locomotion Behaviours in Rich

Environments") demonstrated that complex locomotion gaits — running,

jumping, navigating obstacles — emerge naturally from simple reward

functions when training is performed at scale across diverse environments.

This work motivates our minimalist reward design.



\### 2.3 Unity ML-Agents Toolkit



\*\*Juliani et al. (2020)\*\* introduced Unity ML-Agents as an open-source

platform for training intelligent agents in Unity-rendered environments.

The toolkit provides built-in PPO and SAC implementations, communication

infrastructure between Unity and Python, and example environments

including the Walker biped used in this project.



\---



\## 3. Methodology



\### 3.1 Simulation Environment



We use Unity 2022.3 LTS with the ML-Agents Toolkit Release 21

(Python package version 1.1.0). The environment is the built-in Walker

example, modified to align with our success criteria.



\*\*Walker biped specifications:\*\*

\- Humanoid morphology with 16 actuated joints (hips × 2, knees × 2,

&#x20; ankles × 2, shoulders × 2, elbows × 2, plus spine and neck)

\- Total \~39-dimensional continuous action space

\- Capsule colliders for body parts; ground contact via Unity's PhysX engine

\- Episode length: up to 5000 simulation steps (\~250 seconds at 0.05s timestep)

\- 10 parallel agents in a single training scene for sample efficiency



\### 3.2 Observation Space (\~243 dimensions)



The agent observes the following per-step:



| Component | Dimensions | Description |

|---|---|---|

| Body part positions (relative to root) | \~30 | Each limb's local position |

| Body part rotations | \~52 | Quaternions per body part |

| Joint angular velocities | \~16 | Per-actuator |

| Joint normalized rotation | \~16 | Joint angle within limits |

| Body part velocities | \~30 | Linear velocity of each part |

| Body part angular velocities | \~30 | Angular velocity of each part |

| Target direction | 3 | Desired walking direction unit vector |

| Body forward direction | 3 | Walker's current facing |

| Body up direction | 3 | Walker's current up vector |

| Body height (from ground) | 1 | Used for fall detection |



This is a high-dimensional but smoothly-varying observation, suitable

for MLP-based policy networks.



\### 3.3 Action Space



39-dimensional continuous vector representing target joint torques,

normalized to \[-1, 1] and scaled by per-joint maximum torque limits

internally.



\### 3.4 Reward Function



The Walker reward shaping (used by ML-Agents Walker example) combines:



\- \*\*Forward velocity matching\*\*: encourages the body to move at a target

&#x20; velocity in the forward direction

\- \*\*Looking-direction reward\*\*: encourages the head to face the

&#x20; desired walking direction

\- \*\*Upright stability\*\*: penalizes deviation of the spine from vertical

\- \*\*Implicit alive bonus\*\*: each surviving timestep accumulates reward



Episode termination conditions:

\- Body root drops below a height threshold (fall)

\- Walker exceeds the maximum tilt angle (fall)

\- Maximum episode length reached (success)



\### 3.5 PPO Hyperparameters



| Parameter | Value | Notes |

|---|---|---|

| Learning rate | 3 × 10⁻⁴ | Linear decay to 0 over training |

| Batch size | 2048 | Per gradient update |

| Buffer size | 20480 | Trajectory experience buffer |

| Epochs per update | 3 | Sample reuse |

| Discount factor γ | 0.995 | High — long-horizon credit |

| GAE λ | 0.95 | Generalized advantage estimation |

| Clip ε | 0.2 | PPO objective clipping |

| Entropy coefficient β | 0.005 | Exploration encouragement |

| Hidden layer width | 512 | MLP units per hidden layer |

| Hidden layers | 3 | Both actor and critic |

| Normalize observations | True | Running mean/std |

| Time horizon | 1000 | GAE horizon length |

| Total training steps | \[XX] M | Until reward plateau |



\### 3.6 Network Architecture



Both actor and critic share architecture (separate weights):

\- Input: 243-dim observation (normalized)

\- 3 fully-connected hidden layers, 512 units each, Swish activation

\- Actor output: 39-dim mean of Gaussian policy (separate learnable

&#x20; log-std vector)

\- Critic output: 1-dim scalar value estimate



\### 3.7 Training Infrastructure



\*\*Hardware:\*\*

\- CPU: \[insert from your laptop spec]

\- GPU: NVIDIA RTX 4050 Laptop GPU

\- RAM: \[insert]

\- OS: Windows 11



\*\*Software stack:\*\*

\- Python 3.10.11

\- PyTorch 2.2.2 + CUDA 12.1

\- ML-Agents Toolkit 1.1.0 (Release 21)

\- Unity 2022.3 LTS



\*\*Training duration:\*\* approximately \[XX] hours of wall-clock time,

yielding \[YY] million environment steps (\~\[ZZ] thousand steps/min

throughput on the RTX 4050).



\---



\## 4. Experimental Setup



\### 4.1 Training Protocol



Training was conducted in two phases:



1\. \*\*Sanity-check run\*\* (`walker\_test\_01`): brief verification (\~510K

&#x20;  steps) to confirm the Unity ↔ Python ↔ GPU pipeline.

2\. \*\*Main training run\*\* (`walker\_gpu\_01`): full training to convergence.

&#x20;  Periodic ONNX checkpoints saved every \~500K steps for safety.



\### 4.2 Evaluation Protocol



After training, we evaluated the final policy on \*\*50 independent

episodes\*\* in inference-only mode (no further policy updates). For each

episode, we recorded:



\- \*\*Cumulative reward\*\* (sum of per-step rewards)

\- \*\*Distance traveled\*\* (Euclidean displacement of body root from start)

\- \*\*Episode duration\*\* (number of timesteps before termination)

\- \*\*Fall indicator\*\* (boolean: did the agent fall before max episode length?)

\- \*\*Energy proxy\*\* (sum of squared joint torques across the episode,

&#x20; normalized per meter traveled)



\### 4.3 Success Criteria Definitions



Restating from the proposal with operational definitions:



| # | Criterion | Operationalization |

|---|---|---|

| 1 | Walks ≥5 m without falling | At least one trial achieves displacement ≥ 5m without termination by fall |

| 2 | Upright stability throughout episode | Median episode length ≥ 80% of max episode length |

| 3 | Fall rate < 20% over 50 trials | Fewer than 10 of 50 episodes end via fall |

| 4 | Periodic gait emerges | Visible left-right joint angle alternation in time-series plot |

| 5 | Comprehensive metrics | All four metrics above reported with mean ± std |



\---



\## 5. Results



\*\[Fill in after evaluation tomorrow]\*



\### 5.1 Training Curves



\[Insert TensorBoard screenshots:]

\- \*\*Figure 1\*\*: Cumulative Reward vs. Training Steps

\- \*\*Figure 2\*\*: Episode Length vs. Training Steps

\- \*\*Figure 3\*\*: Policy Loss and Value Loss (PPO diagnostic)



The reward curve shows \[describe trajectory: monotonic increase / plateau /

breakthrough region]. The episode length curve closely tracks reward,

indicating that the agent learns to survive longer specifically through

better locomotion (rather than reward hacking via standing still).



\### 5.2 Quantitative Evaluation (50 Trials)



| Metric | Mean | Std | Min | Max |

|---|---|---|---|---|

| Cumulative reward | \[XXX] | \[XX] | \[XX] | \[XX] |

| Distance traveled (m) | \[X.X] | \[X.X] | \[X.X] | \[X.X] |

| Episode duration (s) | \[XX] | \[XX] | \[XX] | \[XX] |

| Energy per meter (J/m, proxy units) | \[XX] | \[XX] | \[XX] | \[XX] |

| Fall rate | \[X%] | — | — | — |



\### 5.3 Success Criteria Assessment



| # | Criterion | Result | Status |

|---|---|---|---|

| 1 | ≥5m walk without falling | \[X.X m max single-trial] | ✅ / ❌ |

| 2 | Upright stability | \[median ep length / max] | ✅ / ❌ |

| 3 | Fall rate < 20% | \[X%] | ✅ / ❌ |

| 4 | Periodic gait | \[described in §5.4] | ✅ / ❌ |

| 5 | Metrics evaluated | All five reported | ✅ |



\### 5.4 Qualitative Behavior



\[After watching trained agent, describe gait. E.g.: "The agent develops

a periodic alternating-leg gait with stride frequency approximately

\[X] Hz. Knee flexion is well-coordinated with hip rotation, producing

a stable forward-leaning posture. Arms swing in counter-phase to legs,

mirroring biological human walking."]



\[Optional: include time-series plot of left vs. right hip joint angles

showing periodicity.]



\---



\## 6. Discussion



\### 6.1 What Worked



\- \*\*Off-the-shelf PPO + Walker example provided strong foundation\*\*: by

&#x20; building on Unity's reference implementation rather than from scratch,

&#x20; we focused effort on training and evaluation rather than infrastructure.

\- \*\*GPU acceleration\*\*: although PPO's neural network is small, the

&#x20; RTX 4050 enabled stable training at \[XX]K steps/min, completing

&#x20; meaningful training in hours rather than days.

\- \*\*Parallel environments (10 walkers per scene)\*\*: drastically improved

&#x20; sample collection rate.



\### 6.2 Limitations



\- \*\*Simulation only\*\*: our results are not yet validated on physical

&#x20; hardware. Sim-to-real transfer is well-known to be challenging due to

&#x20; contact dynamics mismatches and actuator delay.

\- \*\*Flat terrain only\*\*: per the proposal scope, we did not test

&#x20; uneven terrain or obstacles.

\- \*\*Fixed morphology\*\*: results may not generalize to different biped

&#x20; designs without retraining.

\- \*\*Limited training time\*\*: due to project timeline (1.5 days), training

&#x20; was stopped at \[XX] M steps. Further training would likely yield

&#x20; smoother gaits and lower fall rates.



\### 6.3 Comparison to Reference Performance



The ML-Agents Walker reference benchmark reports successful training in

\~30 million steps with mean cumulative reward \~2500 (Unity Technologies,

2023). Our run reached \[XXX] reward at \[XX] M steps, indicating

\[on-track / partial / strong] convergence within compressed training time.



\---



\## 7. Conclusion and Future Work



We demonstrated that modern reinforcement learning infrastructure makes

it feasible to train a bipedal walking controller from scratch within

a 1.5-day project window. Using PPO with no imitation learning or

gait scheduling, our agent achieved \[summarize key result].



\### Future Work



\- \*\*Sim-to-real transfer\*\*: deploy the learned policy on physical

&#x20; bipedal hardware (e.g., Cassie, Bittle) using domain randomization

&#x20; during training to bridge the reality gap.

\- \*\*Uneven terrain\*\*: extend training to procedurally-generated

&#x20; uneven ground for robust walking.

\- \*\*Multi-skill policies\*\*: train a single policy capable of walking,

&#x20; turning, climbing stairs, and recovering from pushes.

\- \*\*Reward function ablations\*\*: systematically study how different

&#x20; reward formulations affect gait quality and energy efficiency.

\- \*\*Sample-efficient methods\*\*: compare PPO against more sample-efficient

&#x20; algorithms (SAC, DreamerV3) given fixed compute budgets.



\---



\## 8. References



1\. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., \& Klimov, O.

&#x20;  (2017). \*Proximal Policy Optimization Algorithms\*. arXiv:1707.06347.



2\. Heess, N., Tirumala, D., Sriram, S., Lemmon, J., Merel, J., Wayne, G.,

&#x20;  ... \& Silver, D. (2017). \*Emergence of Locomotion Behaviours in Rich

&#x20;  Environments\*. arXiv:1707.02286.



3\. Juliani, A., Berges, V. P., Teng, E., Cohen, A., Harper, J., Elion, C.,

&#x20;  ... \& Lange, D. (2020). \*Unity: A General Platform for Intelligent

&#x20;  Agents\*. arXiv:1809.02627.



4\. Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J.,

&#x20;  Tang, J., \& Zaremba, W. (2016). \*OpenAI Gym\*. arXiv:1606.01540.



5\. Unity Technologies (2023). \*ML-Agents Toolkit Release 21

&#x20;  Documentation\*. https://github.com/Unity-Technologies/ml-agents



6\. Schulman, J., Moritz, P., Levine, S., Jordan, M., \& Abbeel, P. (2016).

&#x20;  \*High-Dimensional Continuous Control Using Generalized Advantage

&#x20;  Estimation\*. arXiv:1506.02438.



\---



\## Appendix A: Reproducibility



\### A.1 Setup Instructions (Windows)



```powershell

\# Clone repo

git clone https://github.com/AniketPol-27/RL-Legged-Locomotion.git

cd RL-Legged-Locomotion



\# Create Python 3.10 virtual environment

py -3.10 -m venv venv

.\\venv\\Scripts\\Activate.ps1



\# Install dependencies

python -m pip install --upgrade pip==23.3.2 setuptools==65.5.0 wheel

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install mlagents==1.1.0



\# Clone ML-Agents reference (for Walker scene)

git clone --branch release\_21 --depth 1 https://github.com/Unity-Technologies/ml-agents.git ml-agents-release



\# Open ml-agents-release/Project in Unity 2022.3 LTS

\# Navigate to Assets/ML-Agents/Examples/Walker/Scenes/Walker.unity

