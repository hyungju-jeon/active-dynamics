# TBME Presentation Animation Production Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible renderer that outputs 36 stitchable animation files (`clean` + `symbol`) covering 18 approved chunks (V1/V2/V3, 30s each).

**Architecture:** Implement a chunk registry plus deterministic rendering pipeline on top of existing Matplotlib animation tooling (`FuncAnimation`, `FFMpegWriter`). Separate concerns into: scene registry, shared visual primitives/styles, and a CLI renderer that emits fixed-duration chunk files with strict naming and frame alignment.

**Tech Stack:** Python 3, NumPy, Matplotlib animation, existing `actdyn` utilities, pytest.

---

### Task 1: Create chunk specification and naming contract

**Files:**
- Create: `actdyn/presentation/chunks.py`
- Create: `actdyn/presentation/__init__.py`
- Test: `tests/test_tbme_chunk_registry.py`

**Step 1: Write the failing test**

```python
from actdyn.presentation.chunks import list_chunks


def test_registry_has_18_unique_chunks():
    chunks = list_chunks()
    ids = [c.chunk_id for c in chunks]
    assert len(chunks) == 18
    assert len(set(ids)) == 18


def test_chunk_duration_lock():
    chunks = list_chunks()
    assert all(abs(c.duration_sec - 30.0) < 1e-9 for c in chunks)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tbme_chunk_registry.py -v`  
Expected: FAIL with import/module error for `actdyn.presentation.chunks`.

**Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    video_id: str
    duration_sec: float = 30.0


def list_chunks() -> list[ChunkSpec]:
    out: list[ChunkSpec] = []
    for v in ("V1", "V2", "V3"):
        for i in range(1, 7):
            out.append(ChunkSpec(chunk_id=f"{v}_C{i:02d}", video_id=v))
    return out
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tbme_chunk_registry.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add actdyn/presentation/__init__.py actdyn/presentation/chunks.py tests/test_tbme_chunk_registry.py
git commit -m "feat: add TBME chunk registry with duration lock"
```

### Task 2: Add visual style and semantic color contract

**Files:**
- Create: `actdyn/presentation/style.py`
- Test: `tests/test_tbme_chunk_registry.py`

**Step 1: Write the failing test**

```python
from actdyn.presentation.style import SEMANTIC_COLORS


def test_semantic_colors_have_required_keys():
    required = {"latent", "param", "obs", "control", "info", "louis_subtract"}
    assert required.issubset(set(SEMANTIC_COLORS))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tbme_chunk_registry.py -v`  
Expected: FAIL with missing `style` module or keys.

**Step 3: Write minimal implementation**

```python
SEMANTIC_COLORS = {
    "latent": "#3A7DFF",
    "param": "#F39C12",
    "obs": "#9E9E9E",
    "control": "#E74C3C",
    "info": "#F1C40F",
    "louis_subtract": "#D64541",
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tbme_chunk_registry.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add actdyn/presentation/style.py tests/test_tbme_chunk_registry.py
git commit -m "feat: define fixed semantic color palette for TBME animations"
```

### Task 3: Build render context and deterministic frame budget helper

**Files:**
- Create: `actdyn/presentation/render_core.py`
- Test: `tests/test_tbme_render_plan.py`

**Step 1: Write the failing test**

```python
from actdyn.presentation.render_core import frame_count_for_chunk


def test_frame_budget_is_fixed():
    assert frame_count_for_chunk(duration_sec=30.0, fps=30) == 900
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tbme_render_plan.py -v`  
Expected: FAIL with missing module/function.

**Step 3: Write minimal implementation**

```python
def frame_count_for_chunk(duration_sec: float, fps: int) -> int:
    return int(round(duration_sec * fps))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tbme_render_plan.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add actdyn/presentation/render_core.py tests/test_tbme_render_plan.py
git commit -m "feat: add deterministic frame budget helper"
```

### Task 4: Implement symbol-overlay toggle with strict clean/symbol parity

**Files:**
- Create: `actdyn/presentation/overlays.py`
- Modify: `actdyn/presentation/render_core.py`
- Test: `tests/test_tbme_render_plan.py`

**Step 1: Write the failing test**

```python
from actdyn.presentation.overlays import overlays_for_chunk


def test_clean_mode_has_no_overlays():
    assert overlays_for_chunk("V2_C03", variant="clean") == []


def test_symbol_mode_has_compact_tokens_only():
    labels = overlays_for_chunk("V2_C03", variant="symbol")
    assert labels
    assert all(len(lbl.split()) == 1 for lbl in labels)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tbme_render_plan.py -v`  
Expected: FAIL with missing overlay logic.

**Step 3: Write minimal implementation**

```python
TOKENS = {
    "V2_C03": ["s_z", "I_z", "P_z"],
    "V2_C05": ["G", "s_theta", "I_theta"],
}


def overlays_for_chunk(chunk_id: str, variant: str) -> list[str]:
    if variant == "clean":
        return []
    return TOKENS.get(chunk_id, [])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tbme_render_plan.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add actdyn/presentation/overlays.py actdyn/presentation/render_core.py tests/test_tbme_render_plan.py
git commit -m "feat: add clean/symbol overlay toggle with token-only symbols"
```

### Task 5: Implement scene registry stubs for all 18 chunks

**Files:**
- Create: `actdyn/presentation/scenes.py`
- Test: `tests/test_tbme_render_plan.py`

**Step 1: Write the failing test**

```python
from actdyn.presentation.scenes import get_scene_renderer
from actdyn.presentation.chunks import list_chunks


def test_every_chunk_has_scene_renderer():
    for spec in list_chunks():
        fn = get_scene_renderer(spec.chunk_id)
        assert callable(fn)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tbme_render_plan.py -v`  
Expected: FAIL for missing scene registry.

**Step 3: Write minimal implementation**

```python
def _stub_scene(*_args, **_kwargs):
    # Placeholder animation hook; replaced chunk-by-chunk in later tasks.
    return None


def get_scene_renderer(chunk_id: str):
    registry = {f"V{v}_C{i:02d}": _stub_scene for v in (1, 2, 3) for i in range(1, 7)}
    return registry[chunk_id]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tbme_render_plan.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add actdyn/presentation/scenes.py tests/test_tbme_render_plan.py
git commit -m "feat: add scene registry covering all approved chunks"
```

### Task 6: Implement CLI renderer for chunk-wise export

**Files:**
- Create: `scripts/render_tbme_presentation_videos.py`
- Modify: `actdyn/presentation/__init__.py`
- Test: `tests/test_cli_smoke.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

from scripts.render_tbme_presentation_videos import build_parser


def test_renderer_cli_accepts_variant_and_chunk():
    parser = build_parser()
    ns = parser.parse_args(["--chunk", "V1_C01", "--variant", "clean", "--dry-run"])
    assert ns.chunk == "V1_C01"
    assert ns.variant == "clean"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_smoke.py::test_renderer_cli_accepts_variant_and_chunk -v`  
Expected: FAIL with missing script/parser.

**Step 3: Write minimal implementation**

```python
import argparse


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", required=True)
    p.add_argument("--variant", choices=["clean", "symbol"], required=True)
    p.add_argument("--outdir", default="results/presentation_videos")
    p.add_argument("--dry-run", action="store_true")
    return p
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_smoke.py::test_renderer_cli_accepts_variant_and_chunk -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/render_tbme_presentation_videos.py actdyn/presentation/__init__.py tests/test_cli_smoke.py
git commit -m "feat: add TBME presentation renderer CLI"
```

### Task 7: Implement full output naming and parity checks in CLI dry-run

**Files:**
- Modify: `scripts/render_tbme_presentation_videos.py`
- Modify: `actdyn/presentation/chunks.py`
- Test: `tests/test_tbme_render_plan.py`

**Step 1: Write the failing test**

```python
from actdyn.presentation.chunks import expected_output_filename


def test_output_filename_contract():
    assert expected_output_filename("V3_C06", "symbol") == "V3_C06_symbol.mp4"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tbme_render_plan.py::test_output_filename_contract -v`  
Expected: FAIL with missing helper.

**Step 3: Write minimal implementation**

```python
def expected_output_filename(chunk_id: str, variant: str) -> str:
    return f"{chunk_id}_{variant}.mp4"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tbme_render_plan.py::test_output_filename_contract -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add actdyn/presentation/chunks.py scripts/render_tbme_presentation_videos.py tests/test_tbme_render_plan.py
git commit -m "feat: enforce chunk output naming contract"
```

### Task 8: Implement chunk-level scene content for Video 1 (concept)

**Files:**
- Modify: `actdyn/presentation/scenes.py`
- Test: `tests/test_tbme_render_plan.py`

**Step 1: Write the failing test**

```python
from actdyn.presentation.scenes import scene_metadata


def test_video1_scene_metadata_exists():
    for cid in [f"V1_C{i:02d}" for i in range(1, 7)]:
        meta = scene_metadata(cid)
        assert meta["family"] == "concept"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tbme_render_plan.py::test_video1_scene_metadata_exists -v`  
Expected: FAIL with missing metadata.

**Step 3: Write minimal implementation**

```python
def scene_metadata(chunk_id: str) -> dict:
    if chunk_id.startswith("V1_"):
        return {"family": "concept", "duration_sec": 30.0}
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tbme_render_plan.py::test_video1_scene_metadata_exists -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add actdyn/presentation/scenes.py tests/test_tbme_render_plan.py
git commit -m "feat: add concept scene metadata and hooks for V1 chunks"
```

### Task 9: Implement chunk-level scene content for Video 2 and Video 3

**Files:**
- Modify: `actdyn/presentation/scenes.py`
- Test: `tests/test_tbme_render_plan.py`

**Step 1: Write the failing test**

```python
from actdyn.presentation.scenes import scene_metadata


def test_video2_and_video3_scene_families():
    assert scene_metadata("V2_C01")["family"] == "method"
    assert scene_metadata("V3_C01")["family"] == "planning_louis"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tbme_render_plan.py::test_video2_and_video3_scene_families -v`  
Expected: FAIL until mappings are added.

**Step 3: Write minimal implementation**

```python
if chunk_id.startswith("V2_"):
    return {"family": "method", "duration_sec": 30.0}
if chunk_id.startswith("V3_"):
    return {"family": "planning_louis", "duration_sec": 30.0}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tbme_render_plan.py::test_video2_and_video3_scene_families -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add actdyn/presentation/scenes.py tests/test_tbme_render_plan.py
git commit -m "feat: add scene metadata and hooks for V2/V3 chunks"
```

### Task 10: Add operator documentation and end-to-end dry-run verification

**Files:**
- Create: `docs/presentation/tbme-animation-rendering.md`
- Modify: `README.md`
- Test: `tests/test_cli_smoke.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_tbme_rendering_doc_exists():
    assert Path("docs/presentation/tbme-animation-rendering.md").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_smoke.py::test_tbme_rendering_doc_exists -v`  
Expected: FAIL with missing document.

**Step 3: Write minimal implementation**

```markdown
# TBME Animation Rendering

## Render one chunk
python scripts/render_tbme_presentation_videos.py --chunk V1_C01 --variant clean

## Render all chunks (both variants)
python scripts/render_tbme_presentation_videos.py --all --variants clean symbol
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_smoke.py::test_tbme_rendering_doc_exists -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/presentation/tbme-animation-rendering.md README.md tests/test_cli_smoke.py
git commit -m "docs: add TBME animation rendering workflow"
```

## Verification Gate (Before Claiming Completion)

Run:

```bash
pytest tests/test_tbme_chunk_registry.py tests/test_tbme_render_plan.py tests/test_cli_smoke.py -v
python scripts/render_tbme_presentation_videos.py --all --variants clean symbol --dry-run
```

Expected:

1. All tests pass.
2. Dry-run prints exactly 36 expected output filenames.
3. No naming/timing parity violations reported.

## Output Directory Contract

1. Default root: `results/presentation_videos/`.
2. Each run writes:
   - `results/presentation_videos/V1_C01_clean.mp4`
   - ...
   - `results/presentation_videos/V3_C06_symbol.mp4`

## Risks and Controls

1. FFmpeg codec mismatch:
   - Control: provide fallback codec switch (`libx264`) in CLI.
2. Timing drift across variants:
   - Control: shared frame-budget function used by both variants.
3. Symbol overlay clutter:
   - Control: token-only overlays and no sentence captions.

