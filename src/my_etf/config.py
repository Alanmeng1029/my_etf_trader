"""项目统一配置管理"""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_DIR = str(PROJECT_ROOT)

# 路径配置
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
REPORTS_DIR = os.path.join(SCRIPT_DIR, 'reports')
RECOMMENDATIONS_DIR = os.path.join(SCRIPT_DIR, 'recommendations')
DB_PATH = os.path.join(DATA_DIR, 'etf_data.db')
ETF_NAMES_FILE = os.path.join(REPORTS_DIR, 'etf_names.json')
CYCLICAL_ETF_FILE = os.path.join(REPORTS_DIR, 'cyclical_etf_candidates.json')


def get_etf_codes() -> List[str]:
    """从环境变量获取ETF代码列表"""
    etf_list = os.getenv('ETF_LIST', '')
    if etf_list:
        # 支持逗号、空格、换行分隔
        import re
        codes = re.split(r'[,，\s\n]+', etf_list.strip())
        return [code for code in codes if code]

    # 如果没有ETF_LIST，尝试CORE_ETF_LIST
    etf_list = os.getenv('CORE_ETF_LIST', '')
    if etf_list:
        import re
        codes = re.split(r'[,，\s\n]+', etf_list.strip())
        return [code for code in codes if code]

    # 默认ETF列表
    return ["510050", "159915", "510300", "510500", "159901"]


def get_cyclical_etf_codes() -> List[str]:
    """
    从 cyclical_etf_candidates.json 加载所有周期性ETF代码

    Returns:
        List[str]: 所有周期性ETF的代码列表（从所有tier中提取）

    Note:
        如果文件不存在或格式错误，返回空列表
        该函数是可选的，主ETF列表仍由环境变量ETF_LIST控制
    """
    import json
    from pathlib import Path
    from my_etf.utils.logger import setup_logger

    try:
        cyclical_file = Path(CYCLICAL_ETF_FILE)
        if not cyclical_file.exists():
            logger = setup_logger("config", "config.log")
            logger.warning(f"周期性ETF配置文件不存在: {CYCLICAL_ETF_FILE}")
            return []

        with open(cyclical_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 从所有tier中提取ETF代码
        codes = []
        for tier_key in ['tier_1', 'tier_2', 'tier_3']:
            if tier_key in data and 'etfs' in data[tier_key]:
                codes.extend(data[tier_key]['etfs'].keys())

        logger = setup_logger("config", "config.log")
        logger.info(f"从周期性ETF配置文件加载 {len(codes)} 个ETF代码")
        return codes

    except json.JSONDecodeError as e:
        logger = setup_logger("config", "config.log")
        logger.error(f"周期性ETF配置文件格式错误: {e}")
        return []
    except Exception as e:
        logger = setup_logger("config", "config.log")
        logger.error(f"加载周期性ETF配置失败: {e}")
        return []


# 特征列（与训练时保持一致）
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'amount',
    'MA5', 'MA10', 'MA20', 'MA60',
    'dif', 'dea', 'macd',
    'k', 'd', 'j',
    'rsi',
    'boll_upper', 'boll_middle', 'boll_lower',
]

# 基础特征列（21个）
BASIC_FEATURE_COLS = FEATURE_COLS

# 高级特征列（27个）
ADVANCED_FEATURE_COLS = [
    # 市场波动率因子 (3个)
    'volatility_20d', 'volatility_60d', 'atr',
    # 价格动量因子 (6个)
    'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
    'acceleration', 'roc_5d', 'roc_20d',
    # 成交量因子 (4个)
    'volume_ratio', 'volume_surge', 'obv', 'ad',
    # 相对强弱因子 (3个)
    'rsi_sma', 'price_vs_ma20', 'price_vs_ma60',
    # 布林带位置因子 (2个)
    'boll_position', 'boll_width_pct',
    # 时间特征 (5个)
    'day_of_week', 'month_of_year', 'quarter', 'is_month_end', 'is_quarter_end',
    # 市场环境因子 (2个)
    'is_bullish', 'market_regime',
    # 风险指标 (2个)
    'drawdown_20d', 'drawdown_from_high',
]

# 全部特征列（48个）
ALL_FEATURE_COLS = BASIC_FEATURE_COLS + ADVANCED_FEATURE_COLS

# 基准ETF（用于计算超额收益）
BENCHMARK_ETF = "510300"  # 沪深300 ETF

# 目标变量配置
TARGET_CONFIG = {
    'type': 'excess',  # 'absolute' 或 'excess'
    'absolute_col': 'week_return',
    'excess_col': 'excess_return',
    'horizon': 5,  # 预测未来5天
}

# 模型架构配置
MODEL_ARCHITECTURE = 'two-stage'  # 'separate', 'unified', 'two-stage'

# 数据库配置
DB_MIN_DATE = '2015-01-01'

# 日志配置
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR = DATA_DIR

# 模型配置
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse', 'mae'],
    'random_state': 42,
}

# 回测配置
BACKTEST_CONFIG = {
    'INITIAL_CAPITAL': 100000,
    'REBALANCE_DAYS': 5,
    'TRANSACTION_RATE': 0.0003,
    'BACKTEST_START_DATE': '2025-01-01',
    'MAX_STALENESS_DAYS': 10,
    'MIN_HISTORY_DAYS': 252,
    'COMMISSION_RATE': 0.0003,
    'SLIPPAGE_BPS': 5,
    'MIN_TRADE_AMOUNT': 1000,
    'MAX_POSITION_WEIGHT': 0.30,
    'CLASSIFICATION_PROB_THRESHOLD': 0.10,
    'TRADE_EXECUTION': 'next_open',
    'WALK_FORWARD_TRAIN_DAYS': 756,
    'WALK_FORWARD_VALIDATION_DAYS': 63,
    'WALK_FORWARD_STEP_DAYS': 21,
    'EMBARGO_DAYS': 5,
}

# 策略配置（选择不同数量的ETF）
STRATEGY_CONFIGS = [
    {'name': 'TOP2', 'top_n': 2},
    {'name': 'TOP3', 'top_n': 3},
    {'name': 'TOP4', 'top_n': 4},
    {'name': 'TOP5', 'top_n': 5},
]

# 分类目标配置
CLASSIFICATION_TARGET = 'week_return_class'
CLASSIFICATION_CONFIG = {
    'bins': [float('-inf'), -5, 0, 5, float('inf')],
    'labels': [0, 1, 2, 3],  # XGBoost expects classes starting from 0
    'label_names': {
        0: '< -5% (大幅下跌)',
        1: '-5% ~ 0% (小幅下跌)',
        2: '0% ~ 5% (小幅上涨)',
        3: '> 5% (大幅上涨)'
    }
}

# 分类模型参数
CLASSIFICATION_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'eval_metric': ['mlogloss', 'merror'],
    'num_class': 4,
    'random_state': 42,
}
