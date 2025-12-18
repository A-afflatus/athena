# 深度合并两个字典
def deep_merge(base: dict, override: dict) -> dict:
    """
    深度合并两个字典
    递归地合并嵌套字典，而不是覆盖整个键
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
