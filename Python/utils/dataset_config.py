"""Accessor for the parameters in config.json at the repo root (loaded in ``utils.definitions``): the value in
config.json, else the hard-coded ``default`` passed at the call site."""

import utils.definitions

_CFG = utils.definitions.active_config


def _resolve(config_key, default):
    return _CFG[config_key] if config_key in _CFG else default


def get_float(config_key, default):
    return float(_resolve(config_key, default))


def get_int(config_key, default):
    return int(_resolve(config_key, default))


def get_str(config_key, default):
    return str(_resolve(config_key, default)).strip()


def get_bool(config_key, default):
    return bool(_resolve(config_key, default))
