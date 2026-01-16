# Change Notes

## 2026-01-16 - Bug Fixes and Naming Consistency (Branch: copilot)

### Summary
Fixed import errors, implementation bugs, parameter naming inconsistencies, and runtime errors across the codebase.

### Test Progress
- **Initial**: Import failed with `ImportError: cannot import name 'BaseDynamicsEnv'`
- **After Phase 1**: 150 tests passing, 142 failing
- **After Phase 2**: 166 tests passing, 162 failing

### Major Bug Fixes

#### Import Errors
- **`actdyn/environment/__init__.py`**: Removed non-existent `BaseDynamicsEnv` import; changed type hint to use `gym.Env`
- **`actdyn/models/model.py`**: 
  - Removed spurious `from calendar import c` import
  - Added missing `Belief` import from `actdyn.utils.helper`
- **`actdyn/core/experiment.py`**: Fixed trailing comma syntax error on line 14

#### Implementation Fixes
- **`actdyn/models/base.py`**: Fixed `BaseModel.update()` method which referenced undefined variables (`observation`, `z`, `action`). Now properly extracts from `recent` Rollout parameter.
- **`actdyn/models/dynamics.py`**: 
  - Fixed `LinearDynamics.__init__()` to properly pass `dt` and `is_residual` to parent class via kwargs
  - Fixed `RBFDynamicsEnsemble` and `MLPDynamicsEnsemble` to pass `state_dim` in `dynamics_config` instead of as separate kwarg
- **`actdyn/environment/vectorfield.py`**: Fixed `VectorFieldEnv.set_params()` to handle `None` parameter gracefully
- **`actdyn/config.py`**: Fixed `ExperimentConfig.to_dict()` to use `asdict()` for proper nested dataclass serialization
- **`actdyn/environment/base.py`**: Added `observe()` method to `BaseObservation` class (alias for `forward()`)
- **`actdyn/environment/env_wrapper.py`**: 
  - Fixed `_to_tensor()` dtype from `float16` to `float32` to avoid tensor type mismatch
  - Fixed `observation_space` dtype from `np.float16` to `np.float32`
- **`actdyn/core/experiment.py`**: 
  - Added `self.pbar = None` in `__init__()` to prevent `AttributeError` in `__del__`
  - Made `_finalize_experiment()` more defensive with `hasattr()` checks

### Naming Consistency Fixes

#### Parameter Naming Convention
The codebase uses two different naming conventions:
- **Environment module** (`actdyn.environment`): `d_action`, `d_latent`, `d_obs`, `R`
- **Models module** (`actdyn.models`): `action_dim`, `latent_dim`, `obs_dim`

#### Fixes Applied
- **`actdyn/models/base.py`**: `BaseModel.__init__()` now supports both naming conventions:
  ```python
  self.action_dim = getattr(self.action_encoder, "action_dim", None) or getattr(self.action_encoder, "d_action", 0)
  ```

### Test File Fixes

#### Removed Tests for Non-Existent Classes
- `tests/test_environment.py`: Removed `TestBaseDynamicsEnv` class (class never existed)
- `tests/test_utils_helpers.py`: Removed `TestParseSubconfig` class (function never existed)
- `tests/test_utils_logger.py.skipped`: Skipped (tests non-existent `Logger` class)
- `tests/test_utils_save_load.py.skipped`: Skipped (tests non-existent `save_buffer`, `load_buffer` functions)

#### Parameter Naming Fixes in Tests
- `tests/test_acrobot.py`: Fixed `MLPEncoder` and `LinearActionEncoder` parameters
- `tests/test_environment.py`: Fixed `BaseAction` and `BaseObservation` parameters to use `d_action`, `d_latent`, `d_obs`
- `tests/test_core_agent.py`: Fixed parameter names:
  - `SimpleObsModel`: `obs_dim=10` → `d_obs=10`
  - `SimpleActModel`: `action_dim=2, latent_dim=2` → `d_action=2, d_latent=2`
  - `RNNEncoder`: `d_action=2` → `action_dim=2`
  - `LinearMapping`, `GaussianNoise`: `d_obs=10` → `obs_dim=10`
- `tests/test_core_experiment.py`: 
  - Fixed `ExperimentConfig` parameters: `d_action` → `action_dim`, `d_latent` → `latent_dim`
  - Fixed `SimpleObsModel`, `SimpleActModel` parameters to use environment naming convention
- `tests/test_config.py`: Fixed `video_path` → `video_filename`
- `tests/test_models_model.py`: Fixed `SimpleActModel` parameters
- `tests/test_policy.py`: Added `IdentityActionEncoder` import and fixed `basic_mpc` fixture to include action_encoder
- `tests/old/test_model.py`: Fixed `LinearMapping` duplicate parameter
- `tests/old/test_policy.py`: Fixed `LinearMapping` duplicate parameter

### Files Modified
- `actdyn/config.py`
- `actdyn/core/experiment.py`
- `actdyn/environment/__init__.py`
- `actdyn/environment/base.py`
- `actdyn/environment/env_wrapper.py`
- `actdyn/environment/vectorfield.py`
- `actdyn/models/base.py`
- `actdyn/models/dynamics.py`
- `actdyn/models/model.py`
- `tests/test_acrobot.py`
- `tests/test_config.py`
- `tests/test_core_agent.py`
- `tests/test_core_experiment.py`
- `tests/test_environment.py`
- `tests/test_models_model.py`
- `tests/test_policy.py`
- `tests/test_utils_helpers.py`
- `tests/old/test_model.py`
- `tests/old/test_policy.py`
