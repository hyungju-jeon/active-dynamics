import numpy as np
import torch
import random
from actdyn.config import ExperimentConfig
from actdyn.utils import setup_experiment, hydra_experiment
from actdyn.utils import save_load
from actdyn.utils.rollout import Rollout


# Use Hydra to manage config and multirun; one (lr, batch) combo per run, save JSON.
@hydra_experiment(
    config_path="conf",
    config_name="config",
)
def main(exp_config: ExperimentConfig):
    exp_config.training.learning_rate = exp_config.training.offline_lr
    # Deterministic seeding
    seed = int(exp_config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and exp_config.device == "cuda":
        torch.set_default_device(exp_config.device)

    # Online
    experiment, agent, env, model_env = setup_experiment(exp_config)
    model = model_env.model
    # dec_net = getattr(model.decoder.mapping, "network", None)
    # obs_net = getattr(env.obs_model, "network", None)
    # if dec_net is not None and obs_net is not None:
    #     if (
    #         hasattr(dec_net, "weight")
    #         and hasattr(obs_net, "weight")
    #         and dec_net.weight.shape == obs_net.weight.shape
    #     ):
    #         dec_net.weight.data.copy_(obs_net.weight.data)
    #         if (
    #             hasattr(dec_net, "bias")
    #             and hasattr(obs_net, "bias")
    #             and dec_net.bias is not None
    #             and obs_net.bias is not None
    #         ):
    #             dec_net.bias.data.copy_(obs_net.bias.data)
    #         if hasattr(dec_net, "weight") and dec_net.weight is not None:
    #             dec_net.weight.requires_grad = False
    #         if hasattr(dec_net, "bias") and dec_net.bias is not None:
    #             dec_net.bias.requires_grad = False
    experiment.run()
    print(f"Online experiment completed. Results directory: {exp_config.results_dir}")

    # Offline
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set up experiment (env/model/policy)
    experiment, agent, env, model_env = setup_experiment(exp_config)
    model = model_env.model

    experiment.offline_learning()
    # Load and merge pre-generated rollouts from the repo-local results folder

    # rollouts_root = os.path.join(os.path.dirname(__file__), "results", "rollouts")
    # if not os.path.isdir(rollouts_root):
    #     raise FileNotFoundError(
    #         f"Rollouts directory not found: {rollouts_root}. Generate rollouts first."
    #     )

    # rollout_files = [f for f in os.listdir(rollouts_root) if f.endswith(".pkl")]
    # if not rollout_files:
    #     raise FileNotFoundError(f"No .pkl rollouts found under {rollouts_root}")
    # try:
    #     rollout_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    # except Exception:
    #     rollout_files.sort()

    # rollout = Rollout()
    # for rf in rollout_files:
    #     rp = os.path.join(rollouts_root, rf)
    #     _r = save_load.load_rollout(rp)
    #     rollout.add(**_r._data)

    # # Offline training configuration (overridable via Hydra)
    # offline_cfg = exp_config.training.get_offline_optim_cfg()

    # print(
    #     f"Starting offline learning with lr={offline_cfg['lr']}, batch={offline_cfg['batch_size']},"
    #     f" epochs={offline_cfg['n_epochs']}"
    # )

    # training_loss = agent.model_env.train_model(rollout, **offline_cfg)

    # # Save results to JSON in Hydra output directory
    # os.makedirs(exp_config.results_dir, exist_ok=True)
    # # training_loss is a list of tensors [loss, log_like, kl] per epoch
    # elbo_list, loglike_list, kl_list = [], [], []
    # for t in training_loss:
    #     elbo_list.append(-float(t[0]))
    #     loglike_list.append(-float(t[1]))
    #     kl_list.append(float(t[2]))

    # result = {
    #     "seed": int(exp_config.seed),
    #     "lr": float(offline_cfg["lr"]),
    #     "batch_size": int(offline_cfg["batch_size"]),
    #     "n_epochs": int(offline_cfg["n_epochs"]),
    #     "elbo": elbo_list,
    #     "log_like": loglike_list,
    #     "kl": kl_list,
    # }
    # fname = f"offline_lr_{result['lr']}_batch_{result['batch_size']}.json"
    # with open(os.path.join(exp_config.results_dir, fname), "w", encoding="utf-8") as f:
    #     json.dump(result, f, indent=2)

    # print(f"Saved JSON to {os.path.join(exp_config.results_dir, fname)}")

    # Clean up var and free gpu memory
    del experiment
    del agent
    del env
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
