"""常量定义（从config导入）"""
from ..config import FEATURE_COLS

# 指标定义
ALL_INDICATORS = ["ma", "macd", "kdj", "rsi", "boll", "advanced"]

# 指标到列名的映射
INDICATOR_COLUMNS = {
    "ma": ["MA5", "MA10", "MA20", "MA60"],
    "macd": ["dif", "dea", "macd"],
    "kdj": ["k", "d", "j"],
    "rsi": ["rsi"],
    "boll": ["boll_upper", "boll_middle", "boll_lower"],
    "advanced": [
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
}

# 基础特征列（21个）
BASIC_FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'amount',
    'MA5', 'MA10', 'MA20', 'MA60',
    'dif', 'dea', 'macd',
    'k', 'd', 'j',
    'rsi',
    'boll_upper', 'boll_middle', 'boll_lower',
]

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

# 指标周期配置
MA_PERIODS = [5, 10, 20, 60]
MACD_SHORT = 12
MACD_LONG = 26
MACD_MID = 9
KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3
RSI_PERIOD = 14
BOLL_PERIOD = 20
BOLL_NUM_STD = 2.0

# 基准ETF（用于计算超额收益）
BENCHMARK_ETF = "510300"  # 沪深300 ETF

# 目标变量
TARGET_ABSOLUTE_RETURN = 'week_return'  # 绝对收益率
TARGET_EXCESS_RETURN = 'excess_return'   # 超额收益率（相对于基准）
TARGET_DIRECTION = 'direction'           # 方向分类（1=涨, 0=跌）
