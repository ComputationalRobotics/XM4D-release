"""Accessor for the parameters in config.json at the repo root (loaded in ``utils.definitions``).

Resolution order for every parameter (first hit wins):
    1. environment variable (manual override)
    2. config.json (definitions.active_config)
    3. the hard-coded ``default`` passed at the call site
"""

import os

import utils.definitions

_CFG = utils.definitions.active_config
CONFIG_NAME = utils.definitions.active_config_name


def _resolve(config_key, default, env):
    if env is not None and os.environ.get(env) not in (None, ""):
        return os.environ[env]
    if config_key in _CFG:
        return _CFG[config_key]
    return default


def get_float(config_key, default, env=None):
    return float(_resolve(config_key, default, env))


def get_int(config_key, default, env=None):
    return int(_resolve(config_key, default, env))


def get_str(config_key, default, env=None):
    return str(_resolve(config_key, default, env)).strip()


def get_bool(config_key, default, env=None):
    value = _resolve(config_key, default, env)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
