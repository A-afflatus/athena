"""
日志工具
"""

import logging


class TraceFormatter(logging.Formatter):
    """自定义格式化器，支持traceid占位符"""
    
    def format(self, record):
        # 如果没有traceid，使用占位符
        if not hasattr(record, 'traceid'):
            record.traceid = '________________'
        return super().format(record)

# 日志级别
_log_level = "INFO"
# 日志文件
_log_file = "logs/athena.log"
# 机器id
_machine_id = "000000"
# 控制台输出
_console_output = True

def setup_logger(log_level: str = "INFO", log_file: str = "logs/athena.log", machine_id: str = "000000", console_output: bool = True):
    global _log_level, _log_file, _machine_id, _console_output
    _log_level = log_level
    _log_file = log_file
    _machine_id = machine_id
    _console_output = console_output


def get_logger(name: str = "athena") -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(_log_level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 格式化
    formatter = TraceFormatter(
        # '%(asctime)s [%(levelname)s] [%(processName)s] [%(threadName)s] [%(thread)d] [%(traceid)-16s] >>> %(message)s'
        '%(asctime)s[%(levelname)s][%(threadName)s][{machine_id}.%(thread)d.%(traceid)-16s] >>> %(message)s'.format(machine_id=_machine_id)
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(_log_file)
    file_handler.setLevel(_log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器
    if _console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

