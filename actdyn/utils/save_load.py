import os
import json
import pickle


def save_config(config, path="logs/config.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def save_buffer(buffer, path="checkpoints/buffer.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(buffer, f)


def load_buffer(path="checkpoints/buffer.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_logger(logger, path="logs/log.json"):
    logger.save(filename=os.path.basename(path))


def load_logger(path="logs/log.json"):
    from utils.logger import Logger

    logger = Logger()
    with open(path, "r") as f:
        logger.metrics = json.load(f)
    return logger


def save_model(model, path="checkpoints/model"):
    os.makedirs(path, exist_ok=True)
    for i, m in enumerate(model.models):
        with open(f"{path}/model_{i}.pkl", "wb") as f:
            pickle.dump(m, f)


def load_model(model, path="checkpoints/model"):
    model.models.clear()
    for i in range(len(os.listdir(path))):
        with open(f"{path}/model_{i}.pkl", "rb") as f:
            model.models.append(pickle.load(f))
