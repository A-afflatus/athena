"""
应用配置
"""

from pathlib import Path
from typing import Any
import yaml

from utils.collections import deep_merge

class Settings:
    """应用设置"""

    def __init__(self, config: dict):
        self.default_config = config

    def get(self, key: str) -> Any:
        # 解析键路径，支持数组索引语法
        parts = []
        current_part = ""
        i = 0
        while i < len(key):
            if key[i] == ".":
                if current_part:
                    parts.append(current_part)
                    current_part = ""
            elif key[i] == "[":
                # 找到数组索引的开始
                if current_part:
                    parts.append(current_part)
                    current_part = ""
                # 提取索引内容，支持负数
                bracket_content = ""
                i += 1
                while i < len(key) and key[i] != "]":
                    bracket_content += key[i]
                    i += 1
                if i < len(key) and key[i] == "]":
                    parts.append(f"[{bracket_content}]")
                    i += 1
                    continue
            else:
                current_part += key[i]
            i += 1

        if current_part:
            parts.append(current_part)

        # 遍历路径获取值
        value = self.default_config
        for part in parts:
            if value is None:
                return None

            # 处理数组索引
            if part.startswith("[") and part.endswith("]"):
                index_str = part[1:-1]
                if not isinstance(value, (list, tuple)):
                    return None
                try:
                    index = int(index_str)
                    if abs(index) >= len(value):
                        return None
                    value = value[index]
                except (ValueError, IndexError):
                    return None
            else:
                # 处理字典键
                if not isinstance(value, dict):
                    return None
                value = value.get(part)

        return value


# 全局配置
settings: Settings | None = None


# 挂载配置
def mount_config(profile: str) -> Settings:
    # 加载默认配置
    config_file = Path(__file__).parent.parent / "config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config: dict = yaml.load(f, Loader=yaml.FullLoader)
    # 加载环境配置
    profile_config_file = Path(__file__).parent.parent / f"config-{profile}.yaml"
    with open(profile_config_file, "r", encoding="utf-8") as f:
        profile_config: dict = yaml.load(f, Loader=yaml.FullLoader)
    # 深度合并配置
    config = deep_merge(config, profile_config)
    # 创建设置实例
    global settings
    settings = Settings(config)
    return settings
