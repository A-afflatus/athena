"""
应用配置
"""

import os
from pathlib import Path
from typing import Dict, Any


class Settings:
    """应用设置"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.base_dir / "config"
        self.data_dir = self.base_dir / "data"
        self.log_dir = self.base_dir / "logs"
        
        # 创建必要的目录
        self.data_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # 默认配置
        self.default_config = {
            "assistant": {
                "name": "Athena",
                "version": "0.1.0",
            },
            "chat": {
                "max_history": 100,
                "enable_context": True,
            },
            "tasks": {
                "timeout": 30,
            },
            "plugins": {
                "auto_load": True,
                "plugin_dir": "plugins",
            },
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split(".")
        value = self.default_config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def load_from_env(self):
        """从环境变量加载配置"""
        # TODO: 实现环境变量配置加载
        pass

