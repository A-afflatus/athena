"""
集合工具函数

提供字典、列表等数据结构的常用操作。
"""

from __future__ import annotations

from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    深度合并两个字典
    
    递归地合并嵌套字典，override 中的值会覆盖 base 中的值。
    对于嵌套的字典会递归合并，而非直接覆盖。
    
    Args:
        base: 基础字典
        override: 覆盖字典，其值优先级更高
        
    Returns:
        合并后的新字典（不修改原始字典）
        
    Examples:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"c": 10, "e": 5}}
        >>> deep_merge(base, override)
        {'a': 1, 'b': {'c': 10, 'd': 3, 'e': 5}}
    """
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            # 递归合并嵌套字典
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(
    data: dict[str, Any],
    separator: str = ".",
    prefix: str = "",
) -> dict[str, Any]:
    """
    将嵌套字典展平为单层字典
    
    Args:
        data: 要展平的字典
        separator: 键之间的分隔符
        prefix: 键的前缀
        
    Returns:
        展平后的字典
        
    Examples:
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {'a.b': 1, 'a.c': 2}
    """
    result: dict[str, Any] = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_dict(value, separator, new_key))
        else:
            result[new_key] = value
    
    return result
