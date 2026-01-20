# NOVA_TrafficAI

Copy and Paste Information into a chatbot along with all of the files before making any chnages.

🚦 NOVA 2.0 — CONTINUATION & DESIGN CONSTRAINTS README
0. PURPOSE OF THIS DOCUMENT (MANDATORY)

This document exists to lock the system semantics.

If you violate the constraints below, you are no longer working on NOVA 2.0, but on a different system that is not defensible relative to our report, presentation, or design intent.

This is not a greenfield project.

1. WHAT THIS SYSTEM IS (NON-NEGOTIABLE)
Core formulation

This project implements:

Parameter-Shared Multi-Agent Reinforcement Learning (PS-MARL)
with Shared PPO, centralized training, and decentralized execution.

Formally:

Environment: One SUMO traffic network

Agents: N traffic lights (TLS), currently up to 24

Policy: One shared PPO actor–critic

Observations: One observation vector per intersection

Actions: One action per intersection per step

Rewards: Per-intersection local reward + shared global reward

This is not:

❌ Single-agent PPO

❌ One action controlling all lights

❌ Independent PPO per intersection

❌ Simulator-privileged RL

2. CURRENT IMPLEMENTATION STATUS (WHERE TO PICK UP)
✅ What is already correctly implemented

You are starting from a working architectural baseline:

Environment

MultiIntersectionEnv

Controls N intersections simultaneously

Returns flattened (N × obs_dim) observation

Uses *MultiDiscrete([A_max]N) action space

Enforces safe phase switching:

Minimum green time

Phase index clamping

No flickering

Observation pipeline (LOCKED)

Detector-only state

Canonical movement abstraction:

12 buckets = N/S/E/W × L/T/R

Per-movement features:

Stop-line occupancy proxy

EWMA queue proxy

Trend

Starvation timer

Control context:

Movement service mask

Time in phase

Eligible-to-change flag

Final obs dim per TLS = 62

⚠️ DO NOT introduce lane waiting time, queue length, or speed from SUMO internals.

Action semantics (LOCKED)

PPO outputs integer a_i ∈ [0, A_max-1] per TLS

Each TLS has its own valid phase count

Actions are:

Clamped per TLS

Gated by min-green

Safely ignored if invalid

Rewards (CURRENT)

Local (per TLS):

Queue proxy penalty

Starvation penalty

Global:

Mean network queue proxy

Combined:

r_i = α * r_local_i + (1-α) * r_global


Environment returns mean(r_i)

This is acceptable for PPO compatibility.

3. HARD CONSTRAINTS (DO NOT CHANGE)
🚫 Architectural constraints

You must not:

Collapse observations into a single intersection

Output a single action for all intersections

Replace shared PPO with:

Independent PPOs

Centralized monolithic agent

Use simulator-only privileged signals:

Lane waiting time

Queue length

Mean speed

Teleport info

Remove min-green safety logic

Add yellow-phase logic incorrectly (must be explicit, not implicit)

If any of the above happens, stop and revert.

4. WHAT IS INTENTIONALLY “IMPERFECT” (AND OK FOR NOW)

These are known approximations, not bugs:

Movement L/T/R inference via geometry heuristics

Service inference via detector deltas

Flattened observation for SB3 (instead of native MARL API)

Scalar env reward instead of vector reward

These are acceptable tradeoffs for:

Training stability

Academic defensibility

Real-world deployability

5. IMMEDIATE NEXT STEPS (SAFE TO DO)
🔹 Step 1 — Verify training loop stability

Before changing anything else:

Run short training (5–10k steps)

Confirm:

No TraCI disconnects

No SUMO flag conflicts

No NaNs in observations

PPO updates occur

🔹 Step 2 — Reward shaping refinement (SAFE)

You may:

Adjust α schedule (anneal toward global)

Normalize reward magnitudes

Add small phase-change penalty (optional)

Do not add new state inputs.

🔹 Step 3 — Logging & diagnostics

Recommended additions:

Per-TLS reward logging

Mean queue proxy over time

Phase switch frequency per TLS

These are read-only diagnostics, not control inputs.

🔹 Step 4 — Evaluation harness

Build a separate evaluation script that:

Loads trained model

Runs fixed seeds

Compares against:

Fixed-time baseline

Actuated baseline

Reports:

Delay proxy

Starvation reduction

Stability metrics

6. FUTURE WORK (DO NOT START YET)

These are explicitly postponed:

Yellow phase insertion logic

Curriculum learning

Transfer learning to new networks

True multi-policy MARL frameworks (PettingZoo, RLlib)

Emissions modeling

Real-world controller deployment

7. HOW TO ANSWER “IS THIS REALLY MULTI-AGENT?”

If asked:

“Isn’t this just one agent?”

The correct answer is:

This is parameter-shared multi-agent RL.
Each intersection is an agent with its own observation, action, and local reward.
A single shared policy is used for scalability and coordination.

This matches:

Decentralized execution

Centralized training

Modern MARL literature

FHWA-style system thinking

8. FINAL WARNING TO FUTURE AI

If you feel tempted to:

“Simplify” the environment

“Just try” a single action

“Temporarily” use simulator queue length

“Refactor” into one big agent

Stop. You are breaking the project.

Ask instead:

“How do I preserve PS-MARL semantics while improving X?”

9. TL;DR

Training works → do not refactor

Detector-only → never violate

One action per TLS per step

Shared PPO ≠ single agent

Safety logic stays
