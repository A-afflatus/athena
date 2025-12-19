"""
应用配置管理模块

采用单例模式管理配置，支持多环境配置合并和嵌套键访问。
"""

from __future__ import annotations

import os
from functools import reduce
from pathlib import Path
from typing import Any, TypeVar, overload

import yaml

from athena.utils.collections import deep_merge

T = TypeVar("T")


class Settings:
    """
    应用设置管理器
    
    支持通过点号分隔的路径访问嵌套配置，例如：
    - settings.get("log.level") -> "INFO"
    - settings.get("app.name", "default") -> "Athena"
    - settings.get("items[0].name") -> 访问列表元素
    
    Attributes:
        _config: 配置字典
        _profile: 当前环境标识
    """
    
    _instance: Settings | None = None
    _initialized: bool = False
    
    def __new__(cls) -> Settings:
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        # 避免重复初始化
        if Settings._initialized:
            return
        self._config: dict[str, Any] = {}
        self._profile: str = ""
    
    def load(self, profile: str = "dev", config_dir: Path | None = None) -> None:
        """
        加载配置文件
        
        Args:
            profile: 环境标识，如 dev, prod, test
            config_dir: 配置文件目录，默认为项目根目录
        """
        if config_dir is None:
            # 默认配置文件目录：项目根目录下的 configs/
            # athena/config/settings.py -> athena/config -> athena -> 项目根目录
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "configs"
        
        self._profile = profile
        
        # 加载基础配置
        base_config = self._load_yaml(config_dir / "config.yaml")
        
        # 加载环境配置（可选）
        profile_config_path = config_dir / f"config-{profile}.yaml"
        profile_config = self._load_yaml(profile_config_path, required=False)
        
        # 深度合并配置
        self._config = deep_merge(base_config, profile_config)
        
        # 处理环境变量覆盖（格式：ATHENA_LOG_LEVEL -> log.level）
        self._apply_env_overrides()
        
        Settings._initialized = True
    
    def _load_yaml(self, path: Path, required: bool = True) -> dict[str, Any]:
        """
        安全加载 YAML 文件
        
        Args:
            path: YAML 文件路径
            required: 是否必须存在，为 False 时文件不存在返回空字典
            
        Returns:
            解析后的配置字典
            
        Raises:
            FileNotFoundError: required=True 且文件不存在时抛出
        """
        if not path.exists():
            if required:
                raise FileNotFoundError(f"配置文件不存在: {path}")
            return {}
        
        with open(path, encoding="utf-8") as f:
            # 使用 safe_load 避免安全隐患
            return yaml.safe_load(f) or {}
    
    def _apply_env_overrides(self) -> None:
        """
        应用环境变量覆盖配置
        
        环境变量格式: ATHENA_<PATH>，其中 PATH 用下划线分隔
        例如: ATHENA_LOG_LEVEL=DEBUG 会覆盖 log.level
        """
        prefix = "ATHENA_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix):].lower().replace("_", ".")
                self._set_nested(config_path, self._parse_env_value(value))
    
    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值为适当的类型"""
        # 布尔值
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        # 整数
        try:
            return int(value)
        except ValueError:
            pass
        # 浮点数
        try:
            return float(value)
        except ValueError:
            pass
        # 字符串
        return value
    
    def _set_nested(self, path: str, value: Any) -> None:
        """设置嵌套配置值"""
        keys = path.split(".")
        target = self._config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    
    @overload
    def get(self, key: str) -> Any: ...
    
    @overload
    def get(self, key: str, default: T) -> T: ...
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套路径
        
        Args:
            key: 配置键，如 "log.level" 或 "items[0].name"
            default: 默认值
            
        Returns:
            配置值或默认值
            
        Examples:
            >>> settings.get("log.level")
            'INFO'
            >>> settings.get("not.exist", "default")
            'default'
        """
        try:
            return self._resolve_path(key)
        except (KeyError, IndexError, TypeError):
            return default
    
    def _resolve_path(self, key: str) -> Any:
        """
        解析配置路径
        
        支持两种语法：
        - 点号分隔: log.level
        - 数组索引: items[0].name, items[-1]
        """
        # 解析路径组件
        parts = self._parse_key(key)
        
        # 使用 reduce 遍历路径
        return reduce(self._get_part, parts, self._config)
    
    def _parse_key(self, key: str) -> list[str | int]:
        """将键路径解析为组件列表"""
        parts: list[str | int] = []
        current = ""
        i = 0
        
        while i < len(key):
            char = key[i]
            
            if char == ".":
                if current:
                    parts.append(current)
                    current = ""
            elif char == "[":
                if current:
                    parts.append(current)
                    current = ""
                # 提取索引
                end = key.index("]", i)
                index_str = key[i + 1:end]
                parts.append(int(index_str))
                i = end
            else:
                current += char
            
            i += 1
        
        if current:
            parts.append(current)
        
        return parts
    
    def _get_part(self, obj: Any, part: str | int) -> Any:
        """获取对象的某个部分"""
        if isinstance(part, int):
            return obj[part]
        return obj[part]
    
    @property
    def profile(self) -> str:
        """当前环境标识"""
        return self._profile
    
    @property
    def config(self) -> dict[str, Any]:
        """获取完整配置字典（只读）"""
        return self._config.copy()
    
    def __repr__(self) -> str:
        return f"Settings(profile={self._profile!r}, keys={list(self._config.keys())})"


# 便捷访问的全局实例
settings = Settings()


def get_settings() -> Settings:
    return settings
