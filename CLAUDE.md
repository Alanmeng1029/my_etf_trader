# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

my_etf 是一个 ETF 量化交易预测系统，使用机器学习（XGBoost）预测 ETF 未来收益并提供交易建议。系统支持完整的数据处理流程：数据获取、技术指标计算（基础+高级）、模型训练、策略回测和推荐生成。

## 开发命令

### 环境配置

```bash
# 从示例文件创建 .env
cp .env.example .env

# 编辑 .env 文件配置 ETF 列表
# CORE_ETF_LIST=核心ETF列表（逗号/空格分隔）
# ETF_LIST=完整ETF列表
```

### 数据获取和指标计算

```bash
# 从项目根目录运行
cd my_etf

# 更新所有 ETF 数据
python -m my_etf.fetch.fetcher --update --list

# 获取单个 ETF 数据
python -m my_etf.fetch.fetcher --fetch --code 510050

# 计算基础技术指标
python -m my_etf.indicators.calculator --list

# 计算高级技术指标
python -m my_etf.indicators.advanced_calculator --list

# 强制重新计算所有指标
python -m my_etf.indicators.calculator --list --force
python -m my_etf.indicators.advanced_calculator --list --force

# 生成 ETF 汇总报告
python -m my_etf.summary.generator
```

### 模型训练和预测

```bash
# 训练模型（默认 separate 模式）
python -m my_etf.models.train

# 训练指定模型类型（separate/unified/two-stage）
python -m my_etf.models.train --model-type separate

# 训练时不使用高级特征
python -m my_etf.models.train --no-advanced

# 生成回测预测数据
python -m my_etf.models.generate_backtest_predictions --model-type separate

# 使用现有模型生成今日预测
python -m my_etf.models.predict
```

### 回测和推荐

```bash
# 回测指定模型类型
python -m my_etf.backtest.strategy --model-type separate
python -m my_etf.backtest.strategy --model-type both  # 回测 separate 和 unified

# 回测分类模型（标准）
python -m my_etf.backtest.strategy --classification

# 回测分类模型（排除类别0 - 大幅下跌）
python -m my_etf.backtest.strategy --classification  # 使用排除类别0的逻辑（已内置）

# 运行完整工作流（数据→指标→训练→回测）
python -m my_etf.workflows.full_workflow
python -m my_etf.workflows.full_workflow --model-type both

# 运行推荐工作流（数据→指标→预测）
python -m my_etf.workflows.recommend_workflow
python -m my_etf.workflows.recommend_workflow --classification

# 使用 UTF-8 编码运行（避免中文乱码）
python -X utf8 -m my_etf.workflows.recommend_workflow --classification
```

### 测试

```bash
# 运行测试（如果存在）
pytest tests/
```

## 核心架构

### 数据流

```
akshare (外部)
    ↓
fetch.fetcher (数据获取)
    ↓
SQLite (etf_data.db, 每个ETF一张表: etf_{code})
    ↓
indicators.calculator (基础指标: MA, MACD, KDJ, RSI, BOLL)
indicators.advanced_calculator (高级指标: 波动率、动量、成交量、市场环境等)
    ↓
models.train (XGBoost训练) → 模型保存在 data/{code}/
    ↓
models.predict / backtest.strategy (预测或回测)
    ↓
reports/ 或 recommendations/ (输出结果)
    ↓
summary.generator (生成ETF汇总HTML报告 → index_summary/)
```

### 模型架构

系统支持多种模型架构：
1. **separate** (回归模型): 每个 ETF 独立训练一个模型，预测绝对收益
   - 目标: `week_return` (T+5日收益率)
   - 特征: 48 个（21 基础 + 27 高级）
   - 输出: 每个ETF一个 `{code}_separate.joblib` 模型

2. **unified** (回归模型): 统一模型，所有ETF共用，预测超额收益
   - 目标: `excess_return` (相对于沪深300的超额收益)
   - 输出: `unified.joblib`

3. **classification** (分类模型): 每个 ETF 独立训练一个模型，预测未来收益类别
   - 目标: `week_return_class` (4类: 0=大幅下跌<-5%, 1=小幅下跌-5%~0%, 2=小幅上涨0%~5%, 3=大幅上涨>5%)
   - 特征: 48 个（21 基础 + 27 高级）
   - 输出: 每个ETF一个 `{code}_classification.pkl` 模型
   - 预测输出: 类别ID (0-3), 各类概率分布, 大幅上涨概率

### 特征工程

**基础特征 (21个)** - `FEATURE_COLS`:
- OHLCV: open, high, low, close, volume, amount
- MA: MA5, MA10, MA20, MA60
- MACD: dif, dea, macd
- KDJ: k, d, j
- RSI: rsi
- BOLL: boll_upper, boll_middle, boll_lower

**高级特征 (27个)** - `ADVANCED_FEATURE_COLS`:
- 波动率: volatility_20d, volatility_60d, atr
- 动量: momentum_5d/10d/20d/60d, acceleration, roc_5d/20d
- 成交量: volume_ratio, volume_surge, obv, ad
- 相对强弱: rsi_sma, price_vs_ma20, price_vs_ma60
- 布林带位置: boll_position, boll_width_pct
- 时间特征: day_of_week, month_of_year, quarter, is_month_end, is_quarter_end
- 市场环境: is_bullish, market_regime
- 风险: drawdown_20d, drawdown_from_high

### 工作流模块

**full_workflow.py** - 完整流程自动化
1. `run_data_update()`: 更新所有 ETF 数据
2. `run_indicators_calc()`: 计算基础+高级指标
3. `run_prediction()`: 训练 separate 模型
4. `run_backtest()`: 回测 separate 模型

**recommend_workflow.py** - 快速推荐流程
1. `run_data_update()`: 更新数据
2. `run_indicators_calc()`: 计算基础+高级指标
3. `generate_daily_predictions()`: 使用现有模型预测（回归模型）
4. `generate_classification_predictions()`: 使用现有分类模型预测（实时预测）
   - **重要修改**: 已从读取旧预测文件改为实时预测（直接调用 `models.predict.generate_daily_predictions(model_type='classification')`）
   - 确保预测日期是最新的，避免使用过期的预测数据
5. 输出: 推荐报告 (TXT/CSV/HTML)
   - 支持 `--classification` 参数使用分类模型
   - 支持 `--threshold` 参数设置概率阈值（默认 10%）
   - 使用 UTF-8 编码运行避免中文乱码：`python -X utf8 -m my_etf.workflows.recommend_workflow --classification`

**summary.generator.py** - ETF 汇总报告生成
- 从数据库读取所有 ETF 数据
- 生成包含代码、名称、起始日期、终止日期、最新价格、最近一周涨幅的 HTML 报告
- 文件保存在 `index_summary/` 目录，文件名格式: `etf_summary_YYYYMMDD.html`
- 支持保留历史记录，每次运行生成新文件

### 数据库结构

**etf_data.db** (SQLite):
- 每个ETF一个表: `etf_{code}`
- 列: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额 + 指标列
- 指标列在首次计算时动态添加

### 模型文件结构

```
data/
├── {etf_code}/           # 每个 ETF 一个目录
│   ├── {code}_{code}_{timestamp}_separate.joblib  # XGBoost 模型
│   ├── {code}_{code}_{timestamp}_separate_scaler.pkl  # StandardScaler
│   └── {code}_{code}_{timestamp}_separate.csv       # 预测数据
├── etf_data.db
├── reports/
│   ├── backtest_report_separate.html
│   ├── backtest_results_separate.csv
│   └── backtest_comparison.csv
├── recommendations/
│   └── recommendations_{timestamp}.{txt,csv,html}
└── index_summary/         # ETF 汇总报告
    └── etf_summary_YYYYMMDD.html
```

## 关键配置

### config.py 配置项

```python
# 特征列 - 注意：FEATURE_COLS 只包含基础特征
FEATURE_COLS = [...]  # 21个基础特征
ALL_FEATURE_COLS = FEATURE_COLS + ADVANCED_FEATURE_COLS  # 48个全部特征

# 模型参数
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    ...
}

# 回测参数
BACKTEST_CONFIG = {
    'INITIAL_CAPITAL': 100000,
    'REBALANCE_DAYS': 5,
    'TRANSACTION_RATE': 0.0003,
    'BACKTEST_START_DATE': '2025-01-01',
}

# 策略配置
STRATEGY_CONFIGS = [
    {'name': 'TOP2', 'top_n': 2},
    {'name': 'TOP3', 'top_n': 3},
    {'name': 'TOP4', 'top_n': 4},
    {'name': 'TOP5', 'top_n': 5},
]
```

### .env 环境变量

```bash
# ETF 列表（优先级：ETF_LIST > CORE_ETF_LIST > 默认列表）
ETF_LIST=512480 512760 512400 515980 588000 515050 159326 ...
CORE_ETF_LIST=510050 159915 510300 510500 159901

# 日志级别
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## 重要实现细节

### 特征列选择

- **训练时**: 使用 `ALL_FEATURE_COLS` (48个) - 需要 `use_advanced=True`
- **预测时**: 必须与训练一致 - `predict.py` 已修复为使用 `ALL_FEATURE_COLS`
- **旧代码**: 某些地方仍使用 `FEATURE_COLS` (21个)，注意检查

### predict.py 使用注意事项

`models/predict.py` 的 `generate_daily_predictions()` 函数:
- 必须使用 `read_etf_data(..., use_advanced=True)` 加载 48 个特征
- 使用 `ALL_FEATURE_COLS` 而非 `FEATURE_COLS` 选择列
- scaler 期望的特征数量必须与训练时一致

### 指标计算器

**calculator.py** (基础指标):
- 支持 `--indicators` 参数选择指标类型
- `--all` 计算所有基础指标
- 注意: `ALL_INDICATORS` 包含 "advanced"，但 calculator.py 不处理它（会跳过）

**advanced_calculator.py** (高级指标):
- 独立计算器，需要单独调用
- 计算 28 个高级指标列
- 支持相同的 `--all`, `--code`, `--list` 参数

### ETF 汇总报告

**summary.generator.py** (ETF 汇总):
- 从数据库读取所有 ETF 的 OHLCV 数据
- 计算最近一周（5个交易日）的涨跌幅
- 生成美观的 HTML 报告，包含：
  - ETF 代码
  - ETF 名称（内置映射表）
  - 数据起始日期
  - 数据终止日期
  - 最新收盘价
  - 最近一周涨幅（红涨绿跌）
- 自动创建 `index_summary/` 目录
- 文件名包含日期（如 `etf_summary_20260228.html`），保留历史记录
- 支持自定义 ETF 名称映射

### 策略回测

**回归模型回测逻辑** (`backtest/strategy.py`):
- 从预测文件加载数据 (`load_latest_predictions`)
- 选择预测收益最高的 N 个 ETF
- 等权分配资金
- 每调仓周期日调仓一次
- 考虑交易费用（默认 0.03%）
- 支持 `--model-type` 参数: separate, unified, both

**分类模型回测逻辑** (`backtest/strategy.py`):
- 支持分类目标（4类）：0=大幅下跌(<-5%), 1=小幅下跌(-5%~0%), 2=小幅上涨(0%~5%), 3=大幅上涨(>5%)
- **标准分类策略** (CLASSIFICATION_TOP5):
  - 按大幅上涨概率降序排序
  - 筛选概率 >= 10% 的 ETF
  - 最多选择 5 个 ETF
  - 如果满足条件的 ETF < 5 个，选择全部满足条件的
  - 如果没有满足条件的 ETF，保持空仓
- **排除类别0策略** (CLASSIFICATION_TOP5_NO_CLASS_0):
  - **首先排除预测类别为 0（大幅下跌）的 ETF**
  - 按大幅上涨概率降序排序
  - 筛选概率 >= 10% 的 ETF
  - 最多选择 5 个 ETF
  - 如果满足条件的 ETF < 5 个，选择全部满足条件的
  - 如果没有满足条件的 ETF，保持空仓
- **现金对冲策略** (CLASSIFICATION_TOP5_CASH_HEDGE):
  - 检查中证500 ETF 的预测类别
  - 如果预测为类别 0（大幅下跌），清仓所有持仓保持空仓
  - 否则使用标准分类策略

**分类模型回测参数**:
- `--classification`: 使用分类模型进行回测
- `--with-cash-hedge`: 使用分类模型进行带现金对冲的回测（需配合 --classification）
- `--exclude-class-0`: 分类模型回测时排除预测类别为 0（大幅下跌）的 ETF（新逻辑已内置）

**full_workflow.py 参数**:
- `--model-type`: 回归模型类型 (separate/unified/both)
- `--classification`: 使用分类模型进行回测
- `--with-cash-hedge`: 使用分类模型进行带现金对冲的回测（需配合 --classification）
- `--exclude-class-0`: 排除预测类别为 0 的 ETF（已内置到标准分类策略中）
- `--skip-data-update`: 跳过数据更新步骤
- `--skip-indicators`: 跳过指标计算步骤
- `--skip-training`: 跳过模型训练步骤（使用现有模型）

### 基准 ETF

默认基准: `510300` (沪深300)
- 用于计算超额收益 (unified 模型)
- 用于计算累计超额收益曲线 (回测报告)

## 常见开发场景

### 添加新的技术指标

1. 在 `indicators/calculator.py` 或 `indicators/advanced_calculator.py` 添加计算函数
2. 在 `utils/constants.py` 的 `INDICATOR_COLUMNS` 添加列名
3. 更新 `config.py` 中的 `FEATURE_COLS` 或 `ADVANCED_FEATURE_COLS`

### 修改模型参数

编辑 `config.py` 中的 `MODEL_PARAMS` 字典。

### 更改回测起始日期

编辑 `config.py` 中的 `BACKTEST_CONFIG['BACKTEST_START_DATE']`。

### 修复特征数量不匹配错误

如果遇到 "StandardScaler is expecting X features but got Y features":
1. 确认训练和预测都使用 `use_advanced=True`
2. 确认都使用 `ALL_FEATURE_COLS` 选择列
3. 删除旧模型文件重新训练

### 数据库列不存在错误

首次运行时，指标计算会自动添加列到数据库。如果遇到列不存在错误:
1. 运行 `python -m my_etf.indicators.calculator --list --force`
2. 运行 `python -m my_etf.indicators.advanced_calculator --list --force`

## 已知限制

1. **unified 模型未被 workflow 使用**: 当前 workflow 硬编码使用 separate 模式
2. **advanced 指标在基础计算器中被忽略**: calculator.py 的 ALL_INDICATORS 包含 "advanced" 但不处理
3. **模型版本管理**: 没有版本回滚或对比机制，新的训练会覆盖旧模型
4. **Windows 编码问题**: 控制台输出中文时可能出现编码警告（不影响功能）
5. **分类模型预测文件过期问题**:
   - `generate_backtest_predictions.py` 生成的预测文件只包含测试集数据（通常不是最新数据）
   - **已修复**: `recommend_workflow.py` 的 `generate_classification_predictions()` 已改为实时预测（直接调用 `models.predict.generate_daily_predictions(model_type='classification')`）
   - 这确保推荐结果始终使用最新的数据和模型
   - 回测时仍使用预测文件，但回测数据范围有限制
6. **分类模型策略优化**:
   - 标准分类策略可能选择预测类别为 0（大幅下跌）的 ETF
   - 已优化: 新增 `CLASSIFICATION_TOP5_NO_CLASS_0` 策略，在筛选前先排除类别 0 的 ETF
   - 逻辑: 排除类别0 → 按概率排序 → 筛选概率>=10% → 最多选5只

## 模块导出

各模块的 `__init__.py` 定义了公共 API，例如:

```python
from my_etf.utils.database import read_etf_data, get_connection
from my_etf.utils.logger import setup_logger
from my_etf.models.predict import generate_daily_predictions
from my_etf.indicators.calculator import calculate_all_indicators
from my_etf.indicators.advanced_calculator import calculate_all_advanced_indicators
from my_etf.summary.generator import generate_html, get_etf_summary
```
