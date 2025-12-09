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


def setup_logger(name: str = "athena", log_file: str = "athena.log") -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 机器id
    machine_id = "000000"
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = TraceFormatter(
        # '%(asctime)s [%(levelname)s] [%(processName)s] [%(threadName)s] [%(thread)d] [%(traceid)-16s] >>> %(message)s'
        '%(asctime)s[%(levelname)s][%(threadName)s][{machine_id}.%(thread)d.%(traceid)-16s] >>> %(message)s'.format(machine_id=machine_id)
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

