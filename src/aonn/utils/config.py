# src/aonn/utils/config.py
"""
配置加载工具
"""
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典
    """
    merged = {}
    for config in configs:
        merged.update(config)
    return merged

