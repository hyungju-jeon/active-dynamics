import argparse
import yaml
import gymnasium as gym
import os
import torch

from actdyn.metrics import Agent
from actdyn.model import EnsembleModel
from actdyn.policy import MPCPolicy
from actdyn.experiment import Experiment
from actdyn.utils.save_load import save_config
from actdyn.metrics.base import COST_FUNCTIONS

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config(args.config)

log_config_path = os.path.join("logs", os.path.basename(args.config))
os.makedirs("logs", exist_ok=True)
with open(log_config_path, "w") as f:
    yaml.safe_dump(config, f)

env = gym.make(config["env_name"], render_mode="rgb_array")
model = EnsembleModel(num_models=config["ensemble_size"])

# Handle optional goal for goal-based cost functions
goal = torch.tensor(config.get("goal", [0.0, 0.0]), dtype=torch.float32)
cost_fn = lambda traj: (
    COST_FUNCTIONS[config["mpc_cost"]](traj)
    if config["mpc_cost"] != "final_state_distance"
    else COST_FUNCTIONS["final_state_distance"](traj, goal)
)

policy = MPCPolicy(
    model,
    env.action_space,
    horizon=config["rollout_horizon"],
    num_candidates=100,
    cost_fn=cost_fn,
)
agent = Agent(env, model, policy)

experiment = Experiment(agent, config, resume=args.resume)
experiment.run()
