# my_etf 开发指南

## 项目架构

```
my_etf/
├── src/my_etf/              # 项目包
│   ├── __init__.py         # 包初始化
│   ├── config.py           # 统一配置管理
│   ├── utils/              # 公共工具模块
│   │   ├── database.py     # 数据库操作
│   │   ├── logger.py       # 日志管理
│   │   └── constants.py    # 常量定义
│   ├── fetch/              # 数据获取模块
│   │   └── fetcher.py     # 数据获取和更新
│   ├── indicators/         # 技术指标模块
│   │   └── calculator.py  # 指标计算
│   ├── models/             # 模型训练模块
│   │   ├── train.py       # 模型训练
│   │   └── predict.py     # 模型预测
│   ├── backtest/           # 回测模块
│   │   └── strategy.py    # 策略回测
│   ├── reports/            # 报告生成模块
│   │   └── generator.py   # 报告生成
│   └── workflows/          # 工作流模块
│       ├── full_workflow.py         # 全流程
│       └── recommend_workflow.py    # 推荐流程
├── scripts/               # 便捷脚本（CLI入口）
│   ├── fetch_data.py
│   ├── indicators.py
│   ├── train_models.py
│   ├── run_backtest.py
│   ├── full_workflow.py
│   └── recommend.py
├── tests/                 # 测试目录
├── data/                  # 数据目录
├── models/                # 模型目录
├── reports/               # 报告目录
└── recommendations/        # 推荐目录
```

## 开发环境搭建

### 1. 克隆项目
```bash
git clone <repository-url>
cd my_etf
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，配置ETF列表等参数
```

## 模块说明

### config.py
统一配置管理，包括：
- 路径配置（数据目录、模型目录等）
- ETF配置（ETF代码列表）
- 特征列定义
- 模型参数
- 回测配置

### utils/database.py
数据库操作工具：
- `get_connection()`: 获取数据库连接
- `read_etf_data(code, min_date)`: 读取ETF数据
- `get_all_etf_tables()`: 获取所有ETF表
- `add_indicator_columns()`: 添加指标列

### utils/logger.py
日志管理工具：
- `setup_logger(name, log_file)`: 设置日志记录器

### fetch/fetcher.py
数据获取模块：
- `fetch_etf_history(code, days, end_date)`: 获取历史数据
- `save_etf_data(df, code)`: 保存数据到数据库
- `update_etf_data(code)`: 更新ETF数据
- `fetch_all_etfs(etf_list, update_only)`: 批量获取

### indicators/calculator.py
技术指标计算模块：
- `calculate_ma(df)`: 移动平均线
- `calculate_macd(df)`: MACD指标
- `calculate_kdj(df)`: KDJ指标
- `calculate_rsi(df)`: RSI指标
- `calculate_boll(df)`: 布林带
- `calculate_all_indicators(df)`: 计算所有指标

### models/train.py
模型训练模块：
- `create_target(df)`: 创建预测目标
- `clean_data(df)`: 清洗数据
- `split_data(df)`: 划分数据集
- `train_and_save_model(code, train_df)`: 训练并保存模型

### models/predict.py
模型预测模块：
- `load_latest_model_and_scaler(code)`: 加载最新模型
- `generate_daily_predictions()`: 生成每日预测

### backtest/strategy.py
回测模块：
- `load_latest_predictions()`: 加载预测数据
- `run_backtest(...)`: 运行回测策略
- `calculate_metrics(results_df, initial_capital)`: 计算性能指标

## 代码规范

### 命名规范
- 模块名：小写字母，下划线分隔
- 类名：大驼峰命名法 (CamelCase)
- 函数名：小写字母，下划线分隔
- 常量：大写字母，下划线分隔
- 私有函数：前缀下划线

### 文档字符串
所有公共函数都应包含文档字符串，格式如下：
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    函数简短描述

    Args:
        param1: 参数1的说明
        param2: 参数2的说明

    Returns:
        返回值的说明

    Raises:
        Exception: 异常说明
    """
```

### 类型注解
尽量使用类型注解提高代码可读性：
```python
from typing import List, Optional, Dict

def read_etf_data(code: str, min_date: Optional[str] = None) -> pd.DataFrame:
    ...
```

## 测试指南

### 运行测试
```bash
pytest tests/
```

### 运行特定测试文件
```bash
pytest tests/test_config.py
```

### 查看测试覆盖率
```bash
pytest --cov=src/my_etf tests/
```

## 如何添加新ETF

1. 编辑 `.env` 文件，在 `ETF_LIST` 中添加新的ETF代码：
```
ETF_LIST=512010 512800 159326 新ETF代码
```

2. 运行数据获取：
```bash
python scripts/fetch_data.py --update --list
```

3. 计算指标：
```bash
python scripts/indicators.py --code 新ETF代码
```

4. 训练模型：
```bash
python scripts/train_models.py
```

## 如何添加新特征

1. 在 `config.py` 的 `FEATURE_COLS` 中添加新特征列名

2. 在数据获取或指标计算逻辑中生成新特征

3. 更新数据库表结构，添加新列

4. 重新训练模型

## 常见问题

### Q: 如何修改模型参数？
A: 在 `config.py` 中修改 `MODEL_PARAMS` 字典

### Q: 如何调整回测参数？
A: 在 `config.py` 中修改 `BACKTEST_CONFIG` 字典

### Q: 如何更改日志级别？
A: 在 `.env` 文件中设置 `LOG_LEVEL=DEBUG/INFO/WARNING/ERROR`

### Q: 数据库文件太大怎么办？
A: 可以定期清理历史数据，或使用数据库压缩工具
