# src/aonn/utils/registry.py
"""
Aspect/Object 注册表
"""
from typing import Dict, Type, Any
from aonn.core.aspect_base import AspectBase
from aonn.core.object import ObjectNode


class AspectRegistry:
    """
    Aspect 注册表：管理不同类型的 Aspect 类
    """
    _registry: Dict[str, Type[AspectBase]] = {}

    @classmethod
    def register(cls, name: str):
        """
        注册装饰器
        """
        def decorator(aspect_class: Type[AspectBase]):
            cls._registry[name] = aspect_class
            return aspect_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[AspectBase]:
        """
        获取 Aspect 类
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown aspect: {name}")
        return cls._registry[name]

    @classmethod
    def list_all(cls):
        """
        列出所有已注册的 Aspect
        """
        return list(cls._registry.keys())


class ObjectRegistry:
    """
    Object 注册表：管理不同类型的 Object 配置
    """
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, **kwargs):
        """
        注册 Object 配置
        """
        cls._registry[name] = kwargs

    @classmethod
    def get(cls, name: str) -> Dict[str, Any]:
        """
        获取 Object 配置
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown object: {name}")
        return cls._registry[name]

