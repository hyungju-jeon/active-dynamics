# Change Notes

## 2026-01-16 - Bug Fixes and Naming Consistency (Branch: copilot)

### Summary
Fixed import errors, implementation bugs, and parameter naming inconsistencies across the codebase.

### Major Bug Fixes

#### Import Errors
- **`actdyn/environment/__init__.py`**: Removed non-existent `BaseDynamicsEnv` import; changed type hint to use `gym.Env`
- **`actdyn/models/model.py`**: 
  - Removed spurious `from calendar import c` import
  - Added missing `Belief` import from `actdyn.utils.helper`
- **`actdyn/core/experiment.py`**: Fixed trailing comma syntax error on line 14

#### Implementation Fixes
- **`actdyn/models/base.py`**: Fixed `BaseModel.update()` method which referenced undefined variables (`observation`, `z`, `action`). Now properly extracts from `recent` Rollout parameter.
- **`actdyn/models/dynamics.py`**: Fixed `LinearDynamics.__init__()` to properly pass `dt` and `is_residual` to parent class via kwargs
- **`actdyn/environment/vectorfield.py`**: Fixed `VectorFieldEnv.set_params()` to handle `None` parameter gracefully
- **`actdyn/config.py`**: Fixed `ExperimentConfig.to_dict()` to use `asdict()` for proper nested dataclass serialization

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
- `tests/test_core_agent.py`: Fixed `SimpleObsModel` and `SimpleActModel` parameters
- `tests/test_config.py`: Fixed `video_path` → `video_filename`
- `tests/test_models_model.py`: Fixed `SimpleActModel` parameters
- `tests/old/test_model.py`: Fixed `LinearMapping` duplicate parameter
- `tests/old/test_policy.py`: Fixed `LinearMapping` duplicate parameter

### Test Results
- **Before**: Import failed with `ImportError: cannot import name 'BaseDynamicsEnv'`
- **After**: `import actdyn` succeeds; 150 tests passing, 142 failing (mostly due to additional test parameter issues)

### Files Modified
- `actdyn/config.py`
- `actdyn/core/experiment.py`
- `actdyn/environment/__init__.py`
- `actdyn/environment/vectorfield.py`
- `actdyn/models/base.py`
- `actdyn/models/dynamics.py`
- `actdyn/models/model.py`
- `tests/test_acrobot.py`
- `tests/test_config.py`
- `tests/test_core_agent.py`
- `tests/test_environment.py`
- `tests/test_models_model.py`
- `tests/test_utils_helpers.py`
- `tests/old/test_model.py`
- `tests/old/test_policy.py`
