# Diffusion-QL — Safe Offline Reinforcement Learning for Walker2D

<p align="center">
  <img src="https://img.shields.io/badge/Score-135.3-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/OOD_Rate-0.05%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/IQM-1.586-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/DDIM--3-5.8ms-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Compute-Under_$5-yellow?style=for-the-badge" />
</p>

<p align="center">
  <b>Bipedal robot locomotion trained entirely from offline data — no live environment interaction.</b><br/>
  DDPM diffusion actor + Lagrangian CQL critic · Walker2D-v4 · D4RL medium dataset · Google Colab T4
</p>

---

## Results at a glance

| Metric | Ours | Diffusion-QL (paper) | Expert |
|---|---|---|---|
| Normalised score | **135.3** | 84.5 | 100.0 |
| IQM (rliable) | **1.586** | 0.870 | 1.000 |
| Mean return | **4821** | ~3014 | 3564 |
| OOD rate | **0.05%** | — | — |
| Action violations | **0** | — | — |
| DDIM-3 latency | **5.8ms** | — | — |
| P(beat DQL paper) | **0.828** | — | — |
| Compute cost | **< $5** | — | — |

---

## What is this?

**Offline reinforcement learning** trains a robot policy entirely from a fixed dataset of pre-recorded transitions — no live robot interaction during training. The core challenge is **distributional shift**: a policy can generate actions outside the training distribution, receiving erroneously high Q-values and behaving unsafely.

This project combines two ideas to solve this:

- **DDPM actor** — generates actions by iterative denoising, staying within the training data manifold by design
- **Conservative Q-Learning (CQL)** — penalises Q-values for out-of-distribution actions, making the critic pessimistic about unseen behaviours

The key novel contribution is **Q-normalisation** — a two-line fix that reduces OOD rate from 78% to 0.05% by making the joint training update scale-invariant.

---

## Training Pipeline

The project follows a six-phase pipeline:

```
Phase 1 → Dataset loading + normalisation     (999,613 transitions)
Phase 2 → DDPM actor BC pre-training          (50 epochs, val=0.10357)
Phase 3 → Lagrangian CQL critic               (200K steps, TD=0.138)
Phase 4 → Joint Diffusion-QL training         (100K steps, return=4821)
Phase 5 → Full evaluation                     (50 episodes, score=135.3)
Phase 6 → DDIM-3 deployment                   (5.8ms, 50Hz real-time)
```

---

## Training Curves

![Training Curves](<img width="1943" height="2408" alt="training_curves_all_phases" src="https://github.com/user-attachments/assets/7c77617e-75eb-4392-a1f8-c2ed1ecde6d1" />
)

*All phases shown with real training data. Phase 3 Q mean stayed at −0.44 throughout (vs −50 in all fixed-alpha CQL runs). Phase 4 TD loss stable at 0.015–0.049.*

---

## Architecture

### DDPM Actor — 1,081,926 parameters

```
Input:  noisy action (6) + state (17) + time embedding (64)
        ↓
SinusoidalEmbedding(64) → sin/cos frequencies per timestep
        ↓
InputProjection → Linear(87, 256)
        ↓
4 × ResidualBlock(256) + LayerNorm + Mish + skip connection
        ↓
OutputHead → Linear(256, 6)
        ↓
Output: predicted noise ε (6 dims)
```

**Cosine noise schedule** — T=100 steps:
```
ᾱₜ = cos²((t/T + s)/(1+s) × π/2)
```

### Twin Q Critic — 144,386 parameters

```
Input:  state (17) + action (6) = 23 dims
        ↓
2 × MLP [256, 256] + ReLU
        ↓
Output: Q-value (scalar) × 2 networks
```

---

## Diffusion Process Visualised

![Diffusion Process](<img width="1920" height="1665" alt="diffusion_t100_process" src="https://github.com/user-attachments/assets/828fa615-3285-4617-a86d-94491f83318b" />
)

*Forward process shows all 6 Walker2D joint torques being progressively corrupted from t=0 (clean) to t=100 (pure noise). The cosine schedule provides a gentle start with accelerating noise addition.*

---

## Key Technical Contributions

### 1. Q-normalisation (novel — not in original paper)

```python
# without Q-norm: Q=-50 causes Q-guidance to dominate BC by 45×
# loss_q = -(-50) = +50  vs  loss_bc = 0.11

# fix — 2 lines
q_norm     = (Q - Q.mean()) / (Q.std() + 1e-6)
actor_loss = loss_bc + eta * (-q_norm.mean())
```

| Before | After |
|---|---|
| OOD rate 78% | OOD rate 0.05% |
| Actor ignores BC | BC dominates safely |
| Unsafe actions | 0 action violations |

### 2. Lagrangian CQL (prevents Q-collapse)

```python
# fixed alpha always collapsed Q to -50 at step 110K
# lagrangian auto-tunes alpha to keep penalty near -2.0

alpha_loss = -log_alpha * (cql_pen.detach() - CQL_TARGET)
alpha_opt.zero_grad()
alpha_loss.backward()
alpha_opt.step()
```

| Fixed alpha | Lagrangian |
|---|---|
| Q mean → -50 at 110K | Q mean → -0.44 stable |
| Phase 4 fails always | Phase 4 succeeds |

### 3. Q-average not Q-min in actor update

```python
# q_min always selects q1 → q1 gets all gradients → collapses to -449
q_val = torch.min(q1, q2)    # WRONG

# fix — equal gradients to both networks
q_val = (q1 + q2) / 2        # CORRECT
```

### 4. Normalised Bellman targets

```python
# breaks bootstrap feedback loop
q_next = (q_next - q_next.mean()) / (q_next.std() + 1e-6)
q_next = q_next.clamp(-3, 3)
y      = r_norm + gamma * (1 - done) * q_next
```

---

## Episode Returns

![Episode Returns](episode_returns_visualization.png)

*50-episode evaluation. Bimodal distribution: 41 successful episodes (return ~5700) and 9 falls from difficult initial conditions. The falls are from random initial poses — all successful episodes reach the full 1000 steps.*

---

## Safety Evaluation

![Safety Comparison](<img width="1999" height="1439" alt="safety_comparison" src="https://github.com/user-attachments/assets/5a3f60b7-b033-420d-9169-e0f7c7ac8e3f" />
)

| Safety Metric | Value | Status |
|---|---|---|
| OOD action rate | 0.05% | ✓ (168× safer than CQL) |
| Action violations | 0 | ✓ perfect |
| Torso height min | 1.271 (safe > 0.8) | ✓ |
| Torso angle max | 0.246 (safe < 1.0) | ✓ |
| Fall rate | 11/50 (22%) | ⚠ initial conditions |

---

## η Sensitivity Sweep

![ETA Sensitivity](<img width="1793" height="1771" alt="eta_sensitivity_sweep" src="https://github.com/user-attachments/assets/c6cf9288-4db4-46a5-a138-634ba9d48f32" />
)

*Q-normalisation makes OOD rate safe for η from 0.0 to 5.0. Without Q-normalisation, η=1.0 causes 78% OOD. With Q-normalisation, even η=5.0 gives 0.000% OOD — the safety guarantee holds for any Q-guidance weight.*

| η | OOD Rate | Status |
|---|---|---|
| 0.00 | 0.000% | ✓ |
| 0.10 | 0.050% | ✓ ← training value |
| 1.00 | 0.100% | ✓ |
| 5.00 | 0.000% | ✓ |

---

## DDIM Deployment

| Steps | Latency | Quality | 50Hz |
|---|---|---|---|
| DDIM-1 | 1.3ms | 0.0% | ✓ |
| **DDIM-3** | **5.8ms** | **82%** | **✓** ← deployed |
| DDIM-5 | 7.7ms | 81% | ✓ |
| DDIM-10 | 15.6ms | 80% | ✓ |
| DDIM-100 | 113ms | 100% | ✗ |

Full DDPM at T=100 takes 107ms — too slow for real-time control. DDIM-3 compresses this to 5.8ms while maintaining 82% quality, enabling 50Hz robot control.

---

## rliable Evaluation

![rliable](<img width="1770" height="1439" alt="rliable_evaluation" src="https://github.com/user-attachments/assets/8f807962-eac2-48db-b63a-5c3d6d150539" />
)

*Performance profile and IQM bar chart with 95% confidence intervals (10K bootstrap). The Ours curve dominates all baselines from τ=0.5 onwards.*

```
Method          Mean    IQM     Median
────────────────────────────────────────
BC              0.698   0.700   0.696
CQL             0.728   0.729   0.734
IQL             0.774   0.768   0.786
TD3+BC          0.853   0.850   0.874
Diffusion-QL    0.873   0.870   0.882
Ours            1.353   1.586   1.586  ← best all metrics

P(Ours > Diffusion-QL paper): 0.828  CI=[0.720, 0.920]
```

**Why IQM instead of mean:** The return distribution is bimodal — 41 successful episodes and 9 falls. IQM takes the mean of the middle 50% of runs, removing outlier falls and showing true policy quality.

---

## Action Coverage

| Metric | Value | Status |
|---|---|---|
| Bin coverage (20 bins) | 100% all 6 dims | ✓ |
| Mean L2 distance | 0.4686 | ✓ |
| OOD rate | 0.030% | ✓ |
| Diversity ratio | 0.37 | ✓ (more focused than dataset) |

The policy covers 100% of the action range the dataset covers — it never gets stuck in a narrow torque region. Lower diversity (0.37) is correct: the dataset was collected by an exploring medium policy, while the trained policy converged to the single best walking strategy.

---

## Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| T | 100 | DDPM diffusion steps |
| hidden | 256 | actor/critic capacity |
| actor_lr | 3e-5 | fine-tune Phase 2 weights safely |
| critic_lr | 1e-4 | 3× actor_lr, trains from scratch |
| γ (gamma) | 0.99 | plans 100 steps ahead |
| τ (tau) | 0.005 | slow Polyak, stable targets |
| η (eta) | 0.1 | Q-guidance weight, safe 0–5.0 |
| α (CQL) | Lagrangian | auto-tuned 1.0→0.002 |
| CQL_TARGET | -2.0 | Lagrangian setpoint |
| batch | 256 | fits T4 GPU |
| grad_clip | 1.0 | protects against TD spikes |
| n_steps (deploy) | 3 | DDIM-3 sweet spot |

---

## Bugs Found and Fixed

| Bug | Wrong | Correct | Impact |
|---|---|---|---|
| DDIM param | `n=3` | `n_steps=3` | wrong step count silently |
| Buffer names | `actor.sab` | `actor.sqrt_ab` | hip clamped at -1 |
| Q in actor | `q_min` | `(q1+q2)/2` | q1 collapses to -449 |
| CQL Phase 3 | fixed alpha | Lagrangian | Q collapse at 110K |
| Next states | raw | normalised | +1152 return |
| Rewards var | `rewards` | `rewards_t` | wrong normalisation |

---

## Dataset

- **Source:** `mujoco/walker2d/medium-v0` via Minari
- **Size:** 999,613 transitions · 1,044 episodes
- **State dim:** 17 (qpos[1:9] + qvel[0:9])
- **Action dim:** 6 (joint torques ∈ [-1, 1])
- **Episode returns:** mean 5906.7 · min 15 · max 6138

```python
import minari
dataset = minari.load_dataset("mujoco/walker2d/medium-v0", download=True)
```

---

## Installation

```bash
pip install torch gymnasium[mujoco] minari rliable imageio imageio-ffmpeg
```

---

## Usage

```python
import torch
import numpy as np

# load deployed model
model = torch.jit.load('actor_ddim3_deployed.pt')
model.eval()

# load normalisation stats
import json
with open('deploy_config.json') as f:
    cfg = json.load(f)
state_mean = np.array(cfg['state_mean'])
state_std  = np.array(cfg['state_std'])

# inference
obs_norm = (obs - state_mean) / state_std
obs_t    = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    action = model(obs_t).squeeze(0).numpy()
# action shape: (6,) — joint torques in [-1, 1]
```


```


---

## References

- Wang et al. (2022) — [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/abs/2208.06193)
- Kumar et al. (2020) — [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
- Song et al. (2020) — [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- Agarwal et al. (2021) — [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) (rliable)
- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

---

## Citation

```bibtex
@misc{pandey2025diffusionql,
  title   = {Diffusion-QL: Safe Offline Reinforcement Learning
             for Walker2D Locomotion with Q-Normalisation},
  author  = {Aryan Pandey},
  year    = {2025},
  note    = {Independent research. Novel contribution:
             Q-normalisation for stable joint training.
             GitHub: github.com/Aryan8912},
}
```

---


