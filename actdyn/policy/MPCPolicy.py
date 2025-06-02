import numpy as np


class MPCPolicy:
    def __init__(self, model_ensemble, action_space, horizon, num_candidates, score_fn):
        self.models = model_ensemble
        self.action_space = action_space
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.score_fn = score_fn

    def __call__(self, state):
        candidates = np.array(
            [
                [self.action_space.sample() for _ in range(self.horizon)]
                for _ in range(self.num_candidates)
            ]
        )
        scores = []
        for seq in candidates:
            preds = self.predict_trajectories(state, seq)
            scores.append(self.score_fn(preds))
        best_seq = candidates[np.argmax(scores)]
        return best_seq[0]

    def predict_trajectories(self, state, action_seq):
        return [
            self._predict_with_model(model, state, action_seq)
            for model in self.models.models
        ]

    def _predict_with_model(self, model, state, action_seq):
        s = state
        traj = [s]
        for a in action_seq:
            s = model(s, a)
            traj.append(s)
        return traj
