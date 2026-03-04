"""
my_etf - ETF量化交易系统

提供数据获取、技术分析、机器学习预测和策略回测的完整功能。
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    RECOMMENDATIONS_DIR,
    DB_PATH,
    ETF_NAMES_FILE,
    get_etf_codes,
    FEATURE_COLS,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_DIR,
    MODEL_PARAMS,
)

__all__ = [
    "__version__",
    "__author__",
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "RECOMMENDATIONS_DIR",
    "DB_PATH",
    "ETF_NAMES_FILE",
    "get_etf_codes",
    "FEATURE_COLS",
    "LOG_FORMAT",
    "LOG_LEVEL",
    "LOG_DIR",
    "MODEL_PARAMS",
]
