"""日志管理工具"""
import logging
import os
from pathlib import Path
from ..config import LOG_DIR, LOG_FORMAT, LOG_LEVEL


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件名（可选）

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler
    if log_file:
        log_path = Path(LOG_DIR) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
