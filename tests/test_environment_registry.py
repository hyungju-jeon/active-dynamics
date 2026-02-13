import pytest

from actdyn.environment import environment_from_str


def test_environment_registry_supports_vectorfield():
    env_cls = environment_from_str("vectorfield")
    assert env_cls.__name__ == "VectorFieldEnv"


def test_environment_registry_rejects_removed_key():
    with pytest.raises(ImportError, match="continuous_cartpole"):
        environment_from_str("continuous_cartpole")
