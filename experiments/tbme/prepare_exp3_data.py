#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import requests
from pynwb import NWBHDF5IO


REPO_ROOT = Path(__file__).resolve().parents[2]
DANDISET_ID = "000129"
DANDI_VERSION = "0.241017.1444"


@dataclass(frozen=True)
class DandiAsset:
    name: str
    asset_id: str

    @property
    def api_download_url(self) -> str:
        return f"https://api.dandiarchive.org/api/assets/{self.asset_id}/download/"


TRAIN_NWB = DandiAsset(
    name="sub-Indy_desc-train_behavior+ecephys.nwb",
    asset_id="2ae6bf3c-788b-4ece-8c01-4b4a5680b25b",
)
TEST_NWB = DandiAsset(
    name="sub-Indy_desc-test_ecephys.nwb",
    asset_id="648a7418-98e8-4413-ba97-3772dd325ecc",
)
MC_RTT_ASSETS = (TRAIN_NWB, TEST_NWB)


def _as_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def download_asset(
    asset: DandiAsset,
    *,
    raw_dir: Path,
    overwrite: bool = False,
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / asset.name
    if output_path.exists() and not overwrite:
        print(f"Using existing asset: {output_path}")
        return output_path

    print(f"Downloading {asset.name} from DANDI {DANDISET_ID}@{DANDI_VERSION}")
    with requests.get(asset.api_download_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    print(f"Saved {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def download_mcrtt_assets(
    *,
    raw_dir: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    return {
        asset.name: download_asset(asset, raw_dir=raw_dir, overwrite=overwrite)
        for asset in MC_RTT_ASSETS
    }


def _behavior_matrix_from_nwb(
    nwb_path: Path,
    *,
    behavior_fields: Sequence[str],
) -> tuple[np.ndarray, float]:
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        behavior_module = nwb.processing["behavior"]
        arrays: list[np.ndarray] = []
        sample_rate_hz: float | None = None
        num_samples: int | None = None
        for field in behavior_fields:
            series = behavior_module.data_interfaces[field]
            field_rate = float(series.rate)
            data = np.asarray(series.data[:], dtype=np.float32)
            if data.ndim != 2:
                raise ValueError(f"Behavior field {field!r} must be rank-2, got {data.shape}")
            if sample_rate_hz is None:
                sample_rate_hz = field_rate
            elif not np.isclose(sample_rate_hz, field_rate):
                raise ValueError(
                    f"Behavior fields must share one sample rate, got {sample_rate_hz} and {field_rate}"
                )
            if num_samples is None:
                num_samples = int(data.shape[0])
            elif num_samples != int(data.shape[0]):
                raise ValueError(
                    f"Behavior fields must share one time axis, got {num_samples} and {data.shape[0]}"
                )
            arrays.append(data)

    if sample_rate_hz is None or num_samples is None:
        raise ValueError(f"No behavior fields found in {nwb_path}")
    return np.concatenate(arrays, axis=1), float(sample_rate_hz)


def fill_nan_timeseries(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected rank-2 timeseries array, got {values.shape}")
    if not np.isnan(values).any():
        return values

    repaired = values.copy()
    row_index = np.arange(repaired.shape[0], dtype=np.float64)
    for col in range(repaired.shape[1]):
        column = repaired[:, col]
        missing = np.isnan(column)
        if not missing.any():
            continue
        valid = ~missing
        if not valid.any():
            raise ValueError(f"Column {col} is entirely NaN and cannot be repaired.")
        repaired[missing, col] = np.interp(row_index[missing], row_index[valid], column[valid]).astype(
            np.float32
        )
    return repaired


def bin_regular_timeseries(
    data: np.ndarray,
    *,
    sample_rate_hz: float,
    bin_ms: float,
) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected rank-2 timeseries array, got {values.shape}")
    samples_per_bin = int(round(float(sample_rate_hz) * float(bin_ms) / 1000.0))
    if samples_per_bin <= 0:
        raise ValueError(f"Invalid samples_per_bin={samples_per_bin} for bin_ms={bin_ms}")
    n_full = values.shape[0] // samples_per_bin
    if n_full < 2:
        raise ValueError("Timeseries is too short after binning")
    trimmed = values[: n_full * samples_per_bin]
    reshaped = trimmed.reshape(n_full, samples_per_bin, values.shape[1])
    return reshaped.mean(axis=1, dtype=np.float32)


def _selected_unit_spike_times(
    nwb_path: Path,
    *,
    include_heldout: bool,
) -> list[np.ndarray]:
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
    if not include_heldout and "heldout" in units.columns:
        units = units[~units["heldout"].astype(bool)]
    return [np.asarray(times, dtype=np.float64) for times in units["spike_times"].tolist()]


def bin_spike_trains(
    spike_trains: Iterable[np.ndarray],
    *,
    num_bins: int,
    dt_sec: float,
) -> np.ndarray:
    counts: list[np.ndarray] = []
    for spike_times in spike_trains:
        times = np.asarray(spike_times, dtype=np.float64)
        if times.size == 0:
            counts.append(np.zeros((num_bins,), dtype=np.float32))
            continue
        idx = np.floor(times / float(dt_sec)).astype(np.int64)
        valid = idx[(idx >= 0) & (idx < int(num_bins))]
        counts.append(np.bincount(valid, minlength=int(num_bins)).astype(np.float32))
    if not counts:
        raise ValueError("No spike trains available for replay conversion")
    return np.stack(counts, axis=1)


def prepare_mcrtt_replay_npz(
    *,
    train_nwb_path: Path,
    output_path: Path,
    bin_ms: float = 20.0,
    behavior_fields: Sequence[str] = ("cursor_pos", "finger_vel"),
    include_heldout: bool = False,
    overwrite: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite-output to replace it.")

    behavior_raw, sample_rate_hz = _behavior_matrix_from_nwb(
        train_nwb_path,
        behavior_fields=behavior_fields,
    )
    behavior = bin_regular_timeseries(
        fill_nan_timeseries(behavior_raw),
        sample_rate_hz=sample_rate_hz,
        bin_ms=bin_ms,
    )
    dt_sec = float(bin_ms) / 1000.0
    spike_trains = _selected_unit_spike_times(train_nwb_path, include_heldout=include_heldout)
    spikes = bin_spike_trains(spike_trains, num_bins=int(behavior.shape[0]), dt_sec=dt_sec)

    np.savez_compressed(
        output_path,
        behavior=behavior.astype(np.float32, copy=False),
        spikes=spikes.astype(np.float32, copy=False),
        dt=np.asarray([dt_sec], dtype=np.float32),
        behavior_fields=np.asarray([str(field) for field in behavior_fields]),
        source_dandiset=np.asarray([DANDISET_ID]),
        source_version=np.asarray([DANDI_VERSION]),
        include_heldout=np.asarray([bool(include_heldout)]),
    )
    print(
        f"Wrote {output_path} with behavior shape {behavior.shape}, "
        f"spikes shape {spikes.shape}, dt={dt_sec:.3f}s"
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TBME Experiment 3 MC_RTT replay data")
    parser.add_argument("--raw-dir", type=str, default="data/mcrtt/raw")
    parser.add_argument("--output", type=str, default="data/mcrtt/mcrtt_replay.npz")
    parser.add_argument(
        "--behavior-fields",
        type=str,
        default="cursor_pos,finger_vel",
        help="Comma-separated behavior fields from the NWB behavior module.",
    )
    parser.add_argument("--bin-ms", type=float, default=20.0)
    parser.add_argument("--include-heldout", action="store_true")
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    raw_dir = _as_repo_path(args.raw_dir)
    output_path = _as_repo_path(args.output)
    behavior_fields = [item.strip() for item in str(args.behavior_fields).split(",") if item.strip()]
    if not behavior_fields:
        parser.error("At least one behavior field must be specified.")

    if args.skip_download:
        asset_paths = {asset.name: raw_dir / asset.name for asset in MC_RTT_ASSETS}
    else:
        asset_paths = download_mcrtt_assets(raw_dir=raw_dir, overwrite=bool(args.overwrite_downloads))

    train_path = asset_paths[TRAIN_NWB.name]
    prepare_mcrtt_replay_npz(
        train_nwb_path=train_path,
        output_path=output_path,
        bin_ms=float(args.bin_ms),
        behavior_fields=behavior_fields,
        include_heldout=bool(args.include_heldout),
        overwrite=bool(args.overwrite_output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
