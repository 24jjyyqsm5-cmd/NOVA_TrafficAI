from __future__ import annotations
import os
import time
import numpy as np
import traci
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from utils.config import load_config
from env.multi_env import MultiIntersectionEnv


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    print("Step 1: Loading config...")
    cfg = load_config()

    log_root = cfg.get("logging", {}).get("log_dir", "./runs")
    ckpt_root = cfg.get("logging", {}).get("ckpt_dir", "./checkpoints")
    ensure_dir(log_root)
    ensure_dir(ckpt_root)

    print("Step 2: Creating multi-agent SUMO environment (evaluation mode)...")
    env = MultiIntersectionEnv(cfg)
    env.start()

    print("Step 3: Loading trained PPO models for each intersection...")
    agents = {}
    for tls_id, tls_env in env.intersections.items():
        model_path = os.path.join(ckpt_root, f"latest_{tls_id}.zip")
        if os.path.exists(model_path):
            agents[tls_id] = PPO.load(model_path, env=tls_env)
            print(f"[INFO] Loaded PPO model for {tls_id}")
        else:
            print(f"[WARN] No trained model found for {tls_id}, using random policy.")
            agents[tls_id] = None

    print("Step 4: Running evaluation (policy-only, no training)...")

    writer = SummaryWriter(log_dir=os.path.join(log_root, "evaluation"))

    total_rewards = {tls: 0.0 for tls in agents.keys()}
    step_counter = 0
    tntt = 0.0
    throughput, wait_times = [], []

    # Initialize states
    states = {tls: env.intersections[tls].get_state() for tls in agents.keys()}

    try:
        while True:
            actions = {}
            for tls_id, model in agents.items():
                if model is not None:
                    obs = np.array(states[tls_id], dtype=np.float32)
                    action, _ = model.predict(obs, deterministic=True)
                    actions[tls_id] = int(action)
                else:
                    actions[tls_id] = np.random.randint(env.intersections[tls_id].action_space.n)

            next_states, rewards, dones = env.step(actions)
            step_counter += 1

            # Accumulate rewards
            for tls_id in rewards.keys():
                total_rewards[tls_id] += float(rewards[tls_id])
                states[tls_id] = next_states[tls_id]

            # --- Telemetry ---
            try:
                total_departed, total_wait = 0.0, 0.0
                for tls_id in env.intersections.keys():
                    lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
                    outgoing_edges = set(traci.lane.getEdgeID(l) for l in lanes)
                    departed = sum(traci.edge.getLastStepVehicleNumber(e) for e in outgoing_edges)
                    total_departed += departed
                    total_wait += np.mean([traci.lane.getWaitingTime(l) for l in lanes]) if lanes else 0.0

                avg_throughput = total_departed / max(len(env.intersections), 1)
                avg_wait = total_wait / max(len(env.intersections), 1)
                throughput.append(avg_throughput)
                wait_times.append(avg_wait)

                # TNTT (Total Network Travel Time)
                active_vehicles = traci.vehicle.getIDList()
                total_travel_time_step = sum(
                    traci.vehicle.getAccumulatedWaitingTime(v) for v in active_vehicles
                )
                tntt += total_travel_time_step

                # TensorBoard logging
                writer.add_scalar("eval/avg_throughput", avg_throughput, env.step_count)
                writer.add_scalar("eval/avg_wait_time", avg_wait, env.step_count)
                writer.add_scalar("eval/total_network_travel_time", tntt, env.step_count)
            except Exception:
                pass

            if env.step_count % 50 == 0:
                mean_r = sum(total_rewards.values()) / len(total_rewards)
                print(
                    f"Step {env.step_count:5d}: mean reward = {mean_r:.3f}, "
                    f"throughput = {avg_throughput:.2f}, avg wait = {avg_wait:.2f}"
                )

            # Stop when SUMO ends
            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        # -----------------------------------------------------
        # Evaluation Summary
        # -----------------------------------------------------
        mean_reward = sum(total_rewards.values()) / len(total_rewards)
        mean_throughput = np.mean(throughput)
        mean_wait = np.mean(wait_times)

        print("\n=== Evaluation Summary ===")
        print(f"Total Steps: {step_counter}")
        print(f"Mean Reward: {mean_reward:.3f}")
        print(f"Average Throughput: {mean_throughput:.2f} veh/step")
        print(f"Average Wait Time: {mean_wait:.2f} s")
        print(f"Total Network Travel Time (TNTT): {tntt:.2f} s")
        print("==========================")

        writer.add_scalar("eval_summary/mean_reward", mean_reward, 0)
        writer.add_scalar("eval_summary/avg_throughput", mean_throughput, 0)
        writer.add_scalar("eval_summary/avg_wait_time", mean_wait, 0)
        writer.add_scalar("eval_summary/total_network_travel_time", tntt, 0)

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation manually stopped.")
    finally:
        try:
            env.close()
        except Exception:
            pass
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
