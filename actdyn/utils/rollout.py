# %%
from typing import List, Union
import torch
from tensordict.tensordict import TensorDict
from collections import deque
import random
import einops
from torch.utils.data import Dataset


def collate_dict_batch(batch):
    """
    Custom collate function for DataLoader that handles dict batches efficiently.
    This is more memory-efficient than the default collate function.

    Args:
        batch: List of dicts from Dataset.__getitem__

    Returns:
        Dict with same keys, but values are batched tensors
    """
    if not batch:
        return {}

    # Get all keys from first item
    keys = batch[0].keys()
    result = {}

    for key in keys:
        values = [item[key] for item in batch]

        # Check if all values are tensors
        if all(isinstance(v, torch.Tensor) for v in values):
            # Stack tensors efficiently
            result[key] = torch.stack(values, dim=0)
        else:
            # Keep as list for non-tensor data
            result[key] = values

    return result


def collate_timestep_batch(batch):
    """
    Custom collate function for batching time steps from a single rollout.

    Args:
        batch: List of dicts, each representing one time step

    Returns:
        Dict with same keys, but values have shape (1, batch_size, dim)
    """
    if not batch:
        return {}

    # Get all keys from first item
    keys = batch[0].keys()
    result = {}

    for key in keys:
        values = [item[key] for item in batch]

        # Check if all values are tensors
        if all(isinstance(v, torch.Tensor) for v in values):
            # Stack tensors along new batch dimension and add singleton first dim
            # Input: list of (dim,) tensors -> Output: (1, batch_size, dim)
            stacked = torch.stack(values, dim=0)  # (batch_size, dim)
            result[key] = stacked.unsqueeze(0)  # (1, batch_size, dim)
        else:
            # Keep as list for non-tensor data
            result[key] = values

    return result


class Rollout:
    """
    A class to store a single rollout of an agent in an environment.
    """

    allowed_fields = {
        "obs",
        "next_obs",
        "action",
        "env_action",  # encoded action
        "reward",
        "cost",
        "env_state",  # state of the environment
        "next_env_state",  # next state of the environment
        "model_state",  # belief about the state of the model
        "next_model_state",  # belief about the next state of the model
        "model_action",  # action of the model
    }

    def __init__(self, device="cpu"):
        self._data = {}
        self.length = 0
        self.device = torch.device(device)
        self.finalized = False

    def __del__(self):
        """Destructor to ensure proper cleanup of GPU tensors"""
        try:
            if hasattr(self, "_data") and self._data:
                self.clear()
        except Exception:
            pass  # Ignore errors during cleanup

    @property
    def shape(self):
        return self.length

    def downsample(self, n=1):
        """
        Downsample the rollout by keeping every n-th transition.
        """
        if n <= 1:
            return

        if not self.finalized:
            self.finalize()

        for key, value in self._data.items():
            if key == "action":
                new_length = (value.shape[1]) // n
                temp = torch.zeros(
                    (value.shape[0], new_length, value.shape[-1]),
                    device=self.device,
                )
                for i in range(new_length):
                    temp[:, i, :] = value[:, i * n : (i + 1) * n, :].sum(dim=1)
                self._data[key] = temp
            else:
                self._data[key] = self._data[key][:, ::n]

        self.length = max([v.shape[1] for v in self._data.values()]) if self._data else 0

    def add_dict(self, **kwargs):
        """
        Add a dictionary of values to the rollout.
        """
        for key, value in kwargs.items():
            if key not in self.allowed_fields:
                raise KeyError(f"Key {key} is not allowed. Allowed keys are: {self.allowed_fields}")
            value = torch.as_tensor(value) if not isinstance(value, torch.Tensor) else value
            if isinstance(value, torch.Tensor):
                if value.requires_grad:
                    value = value.detach()
            if value.ndim < 2:
                tensor_values = value.view(-1, 1)
            elif value.ndim == 3 and value.shape[0] == 1:
                tensor_values = value.squeeze(0)
            elif value.ndim == 2:
                tensor_values = value
            else:
                raise ValueError(f"Expected value shape is (time, dim), but got {value.shape}")

            if key not in self._data:
                self._data[key] = [value.unsqueeze(0) for value in tensor_values]
            else:
                self._data[key].extend([value.unsqueeze(0) for value in tensor_values])

    def add(self, transitions=None, **kwargs):
        """
        Add a transition to the rollout. Either provide a list of transitions or keyword arguments.
        """
        if self.finalized:
            raise RuntimeError("Cannot add to a finalized rollout.")
        if transitions is not None and kwargs:
            raise ValueError("Cannot provide both transitions and kwargs. Use one or the other.")
        if transitions is not None:
            for transition in transitions:
                assert isinstance(transition, dict), "Each transition must be a dict"
                self.add(**transition)
        if kwargs:
            self.add_dict(**kwargs)
        self.length = max([len(v) for v in self._data.values()]) if self._data else 0

    def finalize(self):
        """
        Finalize the rollout by converting all lists to tensors with proper (1, time, dim) shape.
        Each rollout represents a single trajectory, so batch dimension is 1.
        """
        if self.finalized:
            return
        for key in self._data:
            if not isinstance(self._data[key], torch.Tensor):
                if len(self._data[key]) == 0:
                    # Handle empty data gracefully
                    self._data[key] = torch.empty((1, 0), device=self.device)
                else:
                    # Stack along time dimension (dim=0), then add batch dimension
                    self._data[key] = einops.rearrange(self._data[key], "t () d -> () t d")

        self.finalized = True

    def as_dict(self):
        return self._data

    def to(self, device):
        for key in self._data:
            if isinstance(self._data[key], torch.Tensor):
                self._data[key] = self._data[key].to(device)
        self.device = torch.device(device)
        return self

    def get(self, key, default=None):
        if key not in self._data:
            return default
        return (
            torch.stack(self._data[key], dim=0)
            if isinstance(self._data[key], list)
            else self._data[key]
        )

    def copy(self):
        """
        Create a memory-efficient copy of the rollout.
        """
        new_rollout = Rollout(device=str(self.device))
        new_rollout._data = {}

        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                # For finalized rollouts, clone and detach to ensure independence
                new_rollout._data[k] = v.clone().detach()
            elif isinstance(v, list):
                # For unfinalized rollouts, deep copy the list but clone tensors efficiently
                new_rollout._data[k] = []
                for tensor in v:
                    if isinstance(tensor, torch.Tensor):
                        new_rollout._data[k].append(tensor.clone().detach())
                    else:
                        new_rollout._data[k].append(tensor)
            else:
                # For any other type, copy the reference
                new_rollout._data[k] = v

        new_rollout.length = self.length
        new_rollout.finalized = self.finalized
        return new_rollout

    def __getitem__(self, key) -> Union[List[torch.Tensor], List[dict], torch.Tensor]:
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

            if self.finalized:
                # Return a dict of sliced tensors
                return {k: v[:, adj_slice] for k, v in self._data.items()}
            else:
                # Return a list of dicts (as before)
                sliced_data = {k: v[:, adj_slice] for k, v in self._data.items()}
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

    def clear(self, keep_last=0):
        """
        Clear old rollout data to manage memory, keeping only the last `keep_last` transitions.
        """
        if not self._data:
            return

        if not self.finalized:
            # If not finalized, work with lists
            for key in list(
                self._data.keys()
            ):  # Create list copy to avoid dict change during iteration
                if keep_last == 0:
                    # Clear list and delete tensor references
                    for item in self._data[key]:
                        if isinstance(item, torch.Tensor):
                            del item
                    self._data[key].clear()
                else:
                    if len(self._data[key]) > keep_last:
                        # Delete old tensor references explicitly
                        old_items = self._data[key][:-keep_last]
                        for item in old_items:
                            if isinstance(item, torch.Tensor):
                                del item
                        # Keep only the last keep_last items
                        self._data[key] = self._data[key][-keep_last:]
        else:
            # If finalized, work with tensors
            for key in list(self._data.keys()):
                if keep_last == 0:
                    # Explicitly delete the tensor and create empty list
                    if isinstance(self._data[key], torch.Tensor):
                        del self._data[key]
                    self._data[key] = []
                elif (
                    isinstance(self._data[key], torch.Tensor)
                    and self._data[key].shape[0] > keep_last
                ):
                    # Keep only the last keep_last elements
                    old_tensor = self._data[key]
                    self._data[key] = self._data[key][-keep_last:].clone()
                    del old_tensor  # Explicitly delete old tensor

        self.finalized = False  # Reset finalized state to allow re-adding data

        self.length = min(self.length, keep_last)

        # Force garbage collection of GPU memory if using CUDA
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def get_dataloader(
        self,
        batch_size: int = 32,
        chunk_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        **dataloader_kwargs,
    ):
        """Create a DataLoader over temporal chunks of this rollout.

        Input tensor shapes inside the rollout are (1, time, dim). We slice the time
        dimension into fixed-length chunks of length ``chunk_size``. The DataLoader
        then batches multiple chunks together to produce batches of shape:

            (batch_size, chunk_size, dim)

        Args:
            batch_size: Number of temporal chunks per batch returned by the DataLoader.
            chunk_size: Number of consecutive timesteps per chunk. If None, defaults to ``batch_size`` (backward compatibility with previous API where a single argument acted as chunk size).
            shuffle: Whether to shuffle the order of temporal chunks.
            drop_last: If True, drop the last (possibly shorter) chunk instead of padding.
            num_workers: DataLoader workers.
            **dataloader_kwargs: Passed through to ``torch.utils.data.DataLoader``.

        Returns:
            DataLoader yielding dictionaries whose tensor values have shape
            (batch_size, chunk_size, dim) (or (batch_size, chunk_size) for 1D features).
        """
        from torch.utils.data import DataLoader, TensorDataset

        if not self.finalized:
            self.finalize()

        # Backward compatibility: if chunk_size not provided, use batch_size
        if chunk_size is None:
            chunk_size = batch_size

        chunk_size = min(chunk_size, len(self))  # Ensure chunk_size at least length of rollout

        # Collect tensor data (ignore empty tensors)
        tensor_dict: dict[str, torch.Tensor] = {
            k: v for k, v in self._data.items() if isinstance(v, torch.Tensor) and v.numel() > 0
        }

        if not tensor_dict:
            empty_dataset = TensorDataset(torch.empty(0))
            return DataLoader(empty_dataset, batch_size=batch_size, **dataloader_kwargs)

        class TimeChunkDataset(Dataset):
            def __init__(
                self, rollout_tensors: dict[str, torch.Tensor], csize: int, drop_last: bool
            ):
                # Remove singleton batch dim -> (time, ...)
                self.data = {k: t.squeeze(0) for k, t in rollout_tensors.items()}
                self.keys = list(self.data.keys())
                self.chunk_size = csize
                self.drop_last = drop_last
                first = self.data[self.keys[0]]
                self.total_time = first.shape[0]
                if drop_last:
                    self.num_chunks = self.total_time // self.chunk_size
                else:
                    self.num_chunks = (self.total_time + self.chunk_size - 1) // self.chunk_size

            def __len__(self):
                return self.num_chunks

            def __getitem__(self, idx: int):
                if idx < 0 or idx >= self.num_chunks:
                    raise IndexError(idx)
                start = idx * self.chunk_size
                end = start + self.chunk_size
                if end > self.total_time:
                    if self.drop_last:
                        raise IndexError("Index out of range due to drop_last")
                    end = self.total_time
                chunk_dict = {}
                for k in self.keys:
                    t = self.data[k]  # (time, dim?) or (time,)
                    slice_t = t[start:end]
                    length = slice_t.shape[0]
                    if length < self.chunk_size and not self.drop_last:
                        # Pad to fixed size
                        pad_shape = (self.chunk_size - length,) + slice_t.shape[1:]
                        pad = torch.zeros(pad_shape, dtype=slice_t.dtype, device=slice_t.device)
                        slice_t = torch.cat([slice_t, pad], dim=0)
                    chunk_dict[k] = slice_t  # (chunk_size, dim) or (chunk_size,)
                return chunk_dict

        dataset = TimeChunkDataset(tensor_dict, chunk_size, drop_last)

        # Use loose typing to avoid static type checker complaints when values are non-tensors
        def default_collate(batch):
            if not batch:
                return {}
            out = {}
            keys = batch[0].keys()
            for k in keys:
                elems = [item[k] for item in batch]
                if all(isinstance(e, torch.Tensor) for e in elems):
                    out[k] = torch.stack(elems, dim=0)  # (batch_size, chunk_size, ...)
                else:
                    out[k] = elems
            return out

        if "collate_fn" not in dataloader_kwargs:
            dataloader_kwargs["collate_fn"] = default_collate

        # Only create generator for deterministic CPU shuffling (avoid CUDA RNG differences)
        generator = None
        if shuffle and num_workers == 0:
            generator = torch.Generator()

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
            **dataloader_kwargs,
        )


class RolloutBuffer:
    def __init__(self, max_size=None, device="cpu"):
        self.device = device
        self.buffer = deque(maxlen=max_size) if max_size else []
        self._cached_flat = None
        self._cache_dirty = False

    def add(self, item: Union[Rollout, dict, List[dict], List[Rollout]]):
        if isinstance(item, Rollout):
            self.add_rollout(item)
        elif isinstance(item, list):
            for sub_item in item:
                self.add(sub_item)
        elif isinstance(item, dict):
            self.add_dict(item)
        else:
            raise ValueError(f"Unsupported item type: {type(item)}")

    def copy(self):
        new_buffer = RolloutBuffer(
            max_size=len(self.buffer) if isinstance(self.buffer, deque) else None,
            device=str(self.device),
        )
        for rollout in self.buffer:
            new_buffer.add_rollout(rollout.copy())
        return new_buffer

    def add_rollout(self, rollout_item: Rollout):
        if not rollout_item.finalized:
            rollout_item.finalize()
        self.buffer.append(rollout_item)
        self._cache_dirty = True

    def _invalidate_cache(self):
        """Invalidate and clean up cached flat data"""
        if self._cached_flat is not None:
            for key, tensor in self._cached_flat.items():
                if isinstance(tensor, torch.Tensor):
                    del tensor
            self._cached_flat = None
        self._cache_dirty = True

    def add_dict(self, data: dict):
        """
        Efficiently populate buffer from dictionary data.
        """
        if not data:
            return
        # Get the shape information from the first data field
        first_key = next(iter(data.keys()))
        first_tensor = data[first_key]
        if not isinstance(first_tensor, torch.Tensor):
            first_tensor = torch.as_tensor(first_tensor, device=self.device)

        if first_tensor.ndim < 2:
            raise ValueError(
                f"Expected at least 2D data (buffer_num, time, ...), got {first_tensor.ndim}D"
            )
        elif first_tensor.ndim == 2:
            first_tensor = first_tensor.unsqueeze(0)

        buffer_num, time_steps = first_tensor.shape[0], first_tensor.shape[1]

        if buffer_num == 0 or time_steps == 0:
            return

        # Pre-allocate buffer for efficiency
        self.buffer = [Rollout(device=self.device) for _ in range(buffer_num)]

        for key, values in data.items():
            # Convert to tensor if not already, ensuring proper device placement
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values, device=self.device)
            else:
                values = values.to(self.device)

            # Validate shape consistency
            if values.shape[0] != buffer_num or values.shape[1] != time_steps:
                raise ValueError(
                    f"Inconsistent shapes: {key} has shape {values.shape}, "
                    f"expected ({buffer_num}, {time_steps}, ...)"
                )

            # Add data to each rollout: extract (time, ...) slices for each buffer index
            for buffer_idx in range(buffer_num):
                trajectory = values[buffer_idx]  # Shape: (time, ...)
                self.buffer[buffer_idx].add(**{key: trajectory})

            del values

        for rollout in self.buffer:
            rollout.finalize()

        self._invalidate_cache()

    def get_all(self):
        return list(self.buffer)

    def sample(self, n=1):
        return random.sample(self.buffer, min(len(self.buffer), n))

    def sample_transitions(self, n=1):
        flat = self.flat
        total = flat[list(flat.keys())[0]].shape[0]
        indices = torch.randint(0, total, (n,))
        return {k: v[indices] for k, v in flat.items()}

    def clear(self):
        """
        Clear buffer with proper memory cleanup.
        """
        for rollout in self.buffer:
            rollout.clear()
        self.buffer.clear()

        self._invalidate_cache()

        # Force garbage collection of GPU memory if using CUDA
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def downsample(self, n=1):
        for rollout in self.buffer:
            rollout.downsample(n=n)
        self._invalidate_cache()

    @property
    def is_empty(self):
        return len(self.buffer) == 0

    @property
    def shape(self):
        return (len(self), len(self.buffer[0])) if self.buffer else (0, 0)

    @property
    def flat(self) -> dict[str, torch.Tensor]:
        """
        Stack all rollouts into a single tensor dictionary. (buffer, time, dim)
        """
        if self._cached_flat is not None and not self._cache_dirty:
            return self._cached_flat

        self._invalidate_cache()

        if not self.buffer:
            self._cached_flat = {}
            self._cache_dirty = False
            return self._cached_flat

        merged = {}

        try:
            buffer_size = len(self.buffer)
            if buffer_size == 0:
                self._cached_flat = {}
                self._cache_dirty = False
                return self._cached_flat

            # Get structure from first rollout and determine max time length
            first_rollout = self.buffer[0]
            if not first_rollout.finalized:
                first_rollout.finalize()
            first_dict = first_rollout.as_dict()

            max_time_length = max(len(rollout) for rollout in self.buffer)

            for key, tensor in first_dict.items():
                if isinstance(tensor, torch.Tensor):
                    if tensor.ndim == 2:
                        full_shape = (buffer_size, max_time_length)
                    elif tensor.ndim == 3:
                        full_shape = (buffer_size, max_time_length, tensor.shape[2])
                    else:
                        full_shape = (buffer_size, max_time_length) + tensor.shape[2:]
                    merged[key] = torch.zeros(full_shape, dtype=tensor.dtype, device=tensor.device)
                else:
                    merged[key] = []

            for buffer_idx, rollout_data in enumerate(self.buffer):
                if not rollout_data.finalized:
                    rollout_data.finalize()

                rollout_dict = rollout_data.as_dict()
                rollout_length = len(rollout_data)

                for key, val in rollout_dict.items():
                    if isinstance(val, torch.Tensor):
                        if val.ndim >= 2:
                            val_squeezed = val.squeeze(0)
                            merged[key][buffer_idx, :rollout_length] = val_squeezed.detach()
                        else:
                            merged[key][buffer_idx, :rollout_length] = val.detach()
                    else:
                        if buffer_idx == 0:
                            merged[key] = [val]
                        else:
                            merged[key].append(val)

            self._cached_flat = merged
            self._cache_dirty = False

        except Exception as e:
            if merged:
                for key, tensor in merged.items():
                    if isinstance(tensor, torch.Tensor):
                        del tensor
                merged.clear()

            self._cached_flat = None
            self._cache_dirty = True
            raise e

        return self._cached_flat

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index) -> Union[Rollout, List[Rollout], torch.Tensor]:
        if isinstance(index, slice):
            return list(self.buffer)[index]
        elif isinstance(index, int):
            return self.buffer[index]
        elif isinstance(index, str):
            return self.flat[index]
        else:
            raise TypeError("Index must be int or slice")

    def to(self, device):
        for rollout in self.buffer:
            rollout.to(device)
        self.device = device
        return self

    def to_dataset(self):
        """
        Convert RolloutBuffer to a PyTorch Dataset for use with DataLoader.
        """
        return BufferDataset(self)

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=0, **dataloader_kwargs):
        """
        Convenience method to create a DataLoader directly from RolloutBuffer.
        """
        from torch.utils.data import DataLoader

        dataset = self.to_dataset()

        # Use custom collate function for better memory efficiency
        if "collate_fn" not in dataloader_kwargs:
            dataloader_kwargs["collate_fn"] = collate_dict_batch

        # Create appropriate generator for device compatibility
        generator = None
        if shuffle and self.device != "cpu" and "cuda" in str(self.device):
            import torch

            generator = torch.Generator(device=self.device)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
        )


class RecentRollout(Rollout):
    def __init__(self, max_len, device="cuda"):
        super().__init__(device=device)
        self.max_len = max_len

    def add(self, **kwargs):
        if self.finalized:
            for key, value in kwargs.items():
                if key not in self.allowed_fields:
                    raise KeyError(
                        f"Key {key} is not allowed. Allowed keys are: {self.allowed_fields}"
                    )
                tensor_value = torch.as_tensor(value, device=self.device)

                if tensor_value.requires_grad:
                    tensor_value = tensor_value.detach()

                # Use roll instead of clone to avoid memory allocation
                # The data is in (1, time, dim) format, so roll along the time dimension
                self._data[key] = torch.roll(self._data[key], shifts=-1, dims=1)

                # Handle different tensor shapes appropriately for (1, time, dim) format
                if tensor_value.ndim == 0:
                    # Scalar value: place in (1, time) tensor
                    self._data[key][0, -1] = tensor_value
                elif tensor_value.ndim == 1:
                    # 1D vector: place in (1, time, dim) tensor
                    self._data[key][0, -1] = tensor_value
                elif tensor_value.ndim == 2 and tensor_value.shape[0] == 1:
                    # (1, dim) -> place in (1, time, dim) tensor
                    self._data[key][0, -1] = tensor_value.squeeze(0)
                else:
                    # Other shapes: try to fit into the pre-allocated structure
                    if tensor_value.shape == self._data[key][0, -1].shape:
                        self._data[key][0, -1] = tensor_value
                    else:
                        # Reshape if needed to match pre-allocated shape
                        self._data[key][0, -1] = tensor_value.reshape(self._data[key][0, -1].shape)
        else:
            super().add(**kwargs)
            if len(self) >= self.max_len:
                for key in self._data:
                    # Keep only the last max_len timesteps
                    if isinstance(self._data[key], list):
                        self._data[key] = self._data[key][-self.max_len :]
                self.length = self.max_len
                self.finalize()

    # def as_batch(self):
    #     """Return data as a batch with batch dimension added. Uses views where possible to avoid copying.
    #     Handles the (1, time, dim) format properly by adding an additional batch dimension for training.
    #     """
    #     batch_data = {}
    #     for k, v in self._data.items():
    #         if isinstance(v, torch.Tensor):
    #             # v is already (1, time, dim) from finalize()
    #             # For batch processing, we can keep this as is since batch_size=1
    #             batch_data[k] = v.detach()
    #         else:
    #             batch_data[k] = v
    #     return batch_data


class RolloutDataset(Dataset):
    """
    PyTorch Dataset wrapper for individual time steps from a single Rollout.
    This makes it possible to batch individual time steps for training.

    For a rollout with data shape (1, time, dim), this dataset returns
    individual time steps that can be batched into (1, batch_size, dim).
    """

    def __init__(self, rollout):
        """
        Args:
            rollout: Rollout object with finalized data
        """
        self.rollout = rollout
        if not rollout.finalized:
            rollout.finalize()
        self.length = rollout.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Get a single time step from the rollout.

        Args:
            idx: Time step index

        Returns:
            Dict with tensors of shape (dim,) for each key
        """
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for rollout length {self.length}")

        # Extract single time step from (1, time, dim) data
        timestep_data = {}
        rollout_dict = self.rollout.as_dict()

        for key, tensor in rollout_dict.items():
            if isinstance(tensor, torch.Tensor):
                # Extract time step: (1, time, dim) -> (dim,)
                timestep_data[key] = tensor[0, idx].detach()
            else:
                # Handle non-tensor data
                if isinstance(tensor, list) and len(tensor) > idx:
                    timestep_data[key] = tensor[idx]
                else:
                    timestep_data[key] = tensor

        return timestep_data


class BufferDataset(Dataset):
    """
    PyTorch Dataset wrapper for RolloutBuffer.
    This makes RolloutBuffer compatible with DataLoader for efficient, memory-safe training.
    """

    def __init__(self, rollout_buffer):
        """
        Args:
            rollout_buffer: RolloutBuffer object
            use_flat_data: If True, flattens all rollouts into individual transitions.
                          If False, treats each rollout as a single item (for sequence models).
        """
        self.rollout_buffer = rollout_buffer

        self._setup_sequence_data()

    def _setup_sequence_data(self):
        """Set up dataset to return full rollout sequences"""
        self.data = {}  # Empty dict instead of None
        self.keys = []  # Empty list instead of None
        self.length = len(self.rollout_buffer.buffer)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return full rollout sequence
        if idx >= len(self.rollout_buffer.buffer):
            raise IndexError(
                f"Index {idx} out of range for buffer size {len(self.rollout_buffer.buffer)}"
            )

        rollout = self.rollout_buffer.buffer[idx]
        if not rollout.finalized:
            rollout.finalize()

        # Return rollout data with detached tensors
        # For sequence mode, squeeze out singleton batch dimension to convert (1, time, dim) -> (time, dim)
        # This allows collate_dict_batch to properly stack to (batch_size, time, dim)
        rollout_data = {}
        for key, tensor in rollout.as_dict().items():
            if isinstance(tensor, torch.Tensor):
                # Squeeze out singleton batch dimension for proper stacking in DataLoader
                rollout_data[key] = tensor.squeeze(0).detach()  # (1, time, dim) -> (time, dim)
            else:
                rollout_data[key] = tensor
        return rollout_data

    def refresh(self):
        """Call this if the underlying rollout_buffer has changed"""
        self._setup_sequence_data()
