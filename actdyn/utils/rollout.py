# %%
from typing import Iterable
import torch
from tensordict.tensordict import TensorDict
from collections import deque
import random


class Rollout:
    """
    A class to store a single rollout of an agent in an environment.
    """

    allowed_fields = {
        "obs",
        "next_obs",
        "action",
        "env_action",
        "reward",
        "cost",
        "env_state",
        "next_env_state",
        "model_state",
        "next_model_state",
    }

    def __init__(self, device="cpu"):
        self._data = {}
        self.length = 0
        self.device = device
        self.finalized = False

    def add(self, transitions: Iterable = None, **kwargs):
        """
        Add a transition to the rollout. Either provide a list of transitions or keyword arguments.
        """
        if self.finalized:
            raise RuntimeError("Cannot add to a finalized rollout.")

        if transitions is not None and kwargs:
            raise ValueError(
                "Cannot provide both transitions and kwargs. Use one or the other."
            )
        if transitions is not None:
            for transition in transitions:
                assert isinstance(transition, dict), "Each transition must be a dict"
                self.add(**transition)
        if kwargs:
            for key, value in kwargs.items():
                if key not in self.allowed_fields:
                    raise KeyError(
                        f"Key {key} is not allowed. Allowed keys are: {self.allowed_fields}"
                    )
                tensor_value = torch.as_tensor(value, device=self.device)
                if tensor_value.requires_grad:
                    tensor_value = tensor_value.detach()
                if key not in self._data:
                    self._data[key] = [tensor_value]
                else:
                    self._data[key].append(tensor_value)
            self.length += 1

    def finalize(self):
        """
        Finalize the rollout by converting all lists to tensors."""
        if self.finalized:
            return
        for key in self._data:
            self._data[key] = torch.stack(self._data[key], dim=1)
        self.finalized = True

    def as_dict(self):
        return self._data

    def to(self, device):
        for key in self._data:
            if isinstance(self._data[key], torch.Tensor):
                self._data[key] = self._data[key].to(device)
        self.device = device
        return self

    def to_tensordict(self, batch_size=None):
        """
        Convert the rollout to a TensorDict.
        """
        if not self.finalized:
            self.finalize()
        return TensorDict(
            self._data, batch_size=[self.length] if batch_size is None else batch_size
        ).to(self.device)

    def get(self, key, default=None):
        if key not in self._data:
            return default
        return self.__getitem__(key)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._data:
                raise KeyError(f"No {key} in rollout")
            return self._data[key]

        elif isinstance(key, int):
            idx = key
            if idx < 0:
                idx += self.length
            if not (0 <= idx < self.length):
                raise IndexError(f"Index {key} out of bounds for length {self.length}")

            return [{k: v[idx] for k, v in self._data.items()}]

        elif isinstance(key, slice):
            s_start, s_stop, s_step = key.start, key.stop, key.step

            current_start = s_start
            if current_start is not None and current_start < 0:
                current_start += self.length
            current_stop = s_stop
            if current_stop is not None and current_stop < 0:
                current_stop += self.length
            adj_slice = slice(current_start, current_stop, s_step)

            if adj_slice.start is not None and adj_slice.start >= self.length:
                raise IndexError(
                    f"Start index {adj_slice.start} out of bounds for length {self.length}"
                )
            if adj_slice.stop is not None and adj_slice.stop > self.length:
                raise IndexError(
                    f"Stop index {adj_slice.stop} out of bounds for length {self.length}"
                )

            sliced_data = {k: v[adj_slice] for k, v in self._data.items()}
            slice_len = 0
            if sliced_data:
                first_key = next(iter(sliced_data))
                slice_len = len(sliced_data[first_key])

            return [
                {k_item: v_item[i] for k_item, v_item in sliced_data.items()}
                for i in range(slice_len)
            ]

        else:
            raise TypeError(f"Key must be str, int, or slice, not {type(key)}")

    def __len__(self):
        return self.length


class RolloutBuffer:
    def __init__(self, max_size=None):
        self.buffer = deque(maxlen=max_size) if max_size else []

    def add(self, rollout_item: Rollout):
        if not rollout_item.finalized:
            rollout_item.finalize()
        self.buffer.append(rollout_item)

    def get_all(self):
        return list(self.buffer)

    def sample(self, n=1):
        return random.sample(self.buffer, min(len(self.buffer), n))

    def sample_transitions(self, n=1):
        flat = self.flat()
        total = flat[list(flat.keys())[0]].shape[0]
        indices = torch.randint(0, total, (n,))
        return {k: v[indices] for k, v in flat.items()}

    def clear(self):
        self.buffer.clear()

    @property
    def flat(self):
        merged = {}
        for rollout_data in self.buffer:
            for key, val in rollout_data.as_dict().items():
                if key not in merged:
                    merged[key] = []
                merged[key].append(val)
        for key in merged:
            merged[key] = torch.cat(merged[key], dim=0)
        return merged

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self.buffer)[index]
        elif isinstance(index, int):
            return self.buffer[index]
        else:
            raise TypeError("Index must be int or slice")

    def as_batch(self, batch_size=1, shuffle=False):
        """
        Yield batches as flattened dictionaries of tensors, for batch processing (like a DataLoader).
        If shuffle is True, the rollouts are shuffled before batching.
        Each batch is a dict of tensors, where each tensor is of shape [batch_size, ...].
        The last batch may be smaller if the total number is not divisible by batch_size.
        """
        flat = self.flat
        total = next(iter(flat.values())).shape[0] if flat else 0
        indices = list(range(total))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, total, batch_size):
            batch_indices = indices[i : i + batch_size]
            yield {k: v[batch_indices] for k, v in flat.items()}

    def to(self, device):
        for rollout in self.buffer:
            rollout.to(device)
        return self


class RecentRollout(Rollout):
    def __init__(self, max_len, device="cuda"):
        super().__init__(device=device)
        self.max_len = max_len

    def add(self, **kwargs):
        # Override to ensure detachment for RecentRollout as well
        for key, value in kwargs.items():
            if key not in self.allowed_fields:
                raise KeyError(
                    f"Key {key} is not allowed. Allowed keys are: {self.allowed_fields}"
                )
            tensor_value = torch.as_tensor(value, device=self.device)
            if tensor_value.requires_grad:
                tensor_value = tensor_value.detach()
            if key not in self._data:
                self._data[key] = [tensor_value]
            else:
                self._data[key].append(tensor_value)
        self.length += 1
        # If we exceed max_len, truncate oldest entries
        if len(self) > self.max_len:
            for key in self._data:
                self._data[key] = self._data[key][-self.max_len :]
            self.length = self.max_len
            self._finalized = True  # Always finalized after trimming
        else:
            self._finalized = True  # Ensure it's usable immediately

    def as_batch(self):
        return self.as_dict()  # Used directly for model training


# %%
if __name__ == "__main__":
    # Example Rollout usage
    rollout = Rollout()
    rollout.add(obs=[1, 1], action=[0.5], reward=1.0)
    rollout.add(obs=[2, 2], action=[0.6], reward=1.5)
    rollout.finalize()

    rollout2 = Rollout()
    rollout2.add(obs=[1, 2], action=[0.5], reward=1.0)
    rollout2.add(obs=[2, 3], action=[0.6], reward=1.5)

    print(rollout["obs"])  # Should print the tensor of observations
    print(rollout.length)  # Should print 2

    # Example RolloutBuffer usage
    buffer = RolloutBuffer(max_size=5)
    buffer.add(rollout)
    buffer.add(rollout2)
    print(len(buffer))  # Should print 1
    buffer.add(rollout)
    print(len(buffer))  # Should print 2

    flattened = buffer.flat  # Should flatten the buffer into a single tensor dict
    print(flattened)  # Print the flattened tensor dict
    buffer.clear()
    print(len(buffer))  # Should print 0
