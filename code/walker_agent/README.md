\# Walker Agent Source Code (Reference Copy)



These files are \*\*unmodified copies\*\* of the default ML-Agents Walker example,

included here so the reward function and physics-reset behaviour referenced in

`docs/report.md` (§4.3, §4.4) can be verified without cloning the full

`ml-agents-release` repository.



\## Files



| File | Source path in ML-Agents | Purpose |

|---|---|---|

| `WalkerAgent.cs` | `Project/Assets/ML-Agents/Examples/Walker/Scripts/WalkerAgent.cs` | Defines reward function (geometric: `matchSpeedReward \* lookAtTargetReward`) and observation space |

| `JointDriveController.cs` | `Project/Assets/ML-Agents/Examples/SharedAssets/Scripts/JointDriveController.cs` | Defines `Reset(BodyPart)` — the per-episode physics reset that zeroes all linear and angular velocities |



\## Origin

\- ML-Agents Release: "3.0.0-exp.1"

\- Repository: https://github.com/Unity-Technologies/ml-agents

\- License: Apache 2.0 (Unity Technologies)



\## Modifications

None. Our training run used these files verbatim. The decision to keep the

default reward function is documented in `docs/report.md` §4.3.

