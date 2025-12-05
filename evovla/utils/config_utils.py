"""Configuration utilities for EvoVLA."""

import yaml
from pathlib import Path
from omegaconf import OmegaConf
from typing import Any, Dict


def load_config(config_path: str) -> Any:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def save_config(config: Any, save_path: str):
    """Save configuration to file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(config), f, default_flow_style=False)

