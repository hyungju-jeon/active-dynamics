import torch


def reward_sum(trajectory):
    # Assumes trajectory['reward'] is a tensor of shape (T,)
    return trajectory["reward"].sum()


def final_state_distance(trajectory, goal):
    # Negative distance to goal (minimize distance)
    final_state = trajectory["state"][-1]
    return -torch.norm(final_state - goal)


def discounted_cost_to_go(trajectory, gamma=0.99):
    rewards = trajectory["reward"]
    T = len(rewards)
    discounts = gamma ** torch.arange(T, device=rewards.device, dtype=rewards.dtype)
    return torch.sum(rewards * discounts)


def non_markovian_penalty(trajectory):
    # Penalize if the action sequence is not smooth (non-Markovian)
    actions = trajectory["action"]
    deltas = actions[1:] - actions[:-1]
    smoothness_penalty = (deltas**2).sum()
    return trajectory["reward"].sum() - 0.1 * smoothness_penalty


COST_FUNCTIONS = {
    "reward_sum": reward_sum,
    "final_state_distance": final_state_distance,
    "discounted_cost": discounted_cost_to_go,
    "smoothness_penalty": non_markovian_penalty,
}
