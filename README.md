# my_etf - ETF量化交易系统

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

完整功能的ETF量化交易系统，涵盖数据获取、技术分析、机器学习预测和策略回测。

## 功能特性

### 数据获取
- 支持从akshare获取ETF历史数据
- 智能增量更新
- 多数据源支持（东方财富、新浪）
- 批量处理ETF列表

### 技术分析
- MA（移动平均线）：MA5, MA10, MA20, MA60
- MACD（指数平滑异同移动平均线）
- KDJ（随机指标）
- RSI（相对强弱指标）
- BOLL（布林带）

### 机器学习预测
- XGBoost模型训练
- T+5日收益率预测
- 模型版本管理
- 特征标准化

### 策略回测
- 多策略对比（TOP2-TOP5）
- 完整性能指标（收益率、回撤、夏普比率等）
- 可视化报告（HTML/CSV）

### 工作流自动化
- 全流程一键执行（数据更新→指标计算→模型训练→回测）
- 推荐工作流（快速生成今日推荐）

## 项目结构

```
my_etf/
├── src/my_etf/              # 项目包
│   ├── __init__.py          # 包初始化
│   ├── config.py            # 统一配置管理
│   ├── utils/               # 公共工具模块
│   │   ├── database.py     # 数据库操作
│   │   ├── logger.py       # 日志管理
│   │   └── constants.py    # 常量定义
│   ├── fetch/               # 数据获取模块
│   │   └── fetcher.py     # 数据获取和更新
│   ├── indicators/          # 技术指标模块
│   │   └── calculator.py  # 指标计算
│   ├── models/              # 模型训练模块
│   │   ├── train.py       # 模型训练
│   │   └── predict.py     # 模型预测
│   ├── backtest/            # 回测模块
│   │   └── strategy.py    # 策略回测
│   └── workflows/           # 工作流模块
│       ├── full_workflow.py         # 全流程
│       └── recommend_workflow.py    # 推荐流程
├── scripts/                 # 便捷脚本（CLI入口）
│   ├── fetch_data.py
│   ├── indicators.py
│   ├── train_models.py
│   ├── run_backtest.py
│   ├── full_workflow.py
│   └── recommend.py
├── tests/                  # 测试目录
├── data/                   # 数据目录
├── models/                 # 模型目录
├── reports/                # 报告目录
├── recommendations/         # 推荐目录
├── requirements.txt         # 依赖列表
├── pyproject.toml          # 项目配置
├── .gitignore              # Git忽略文件
├── .env.example            # 环境变量示例
├── README.md               # 项目说明
├── DEVELOPMENT.md          # 开发指南
└── CHANGELOG.md            # 变更日志
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑.env文件，配置ETF列表
```

### 3. 获取数据

```bash
# 更新所有ETF数据
python scripts/fetch_data.py --update --list

# 计算技术指标
python scripts/indicators.py --all
```

### 4. 训练模型

```bash
python scripts/train_models.py
```

### 5. 运行回测

```bash
python scripts/run_backtest.py
```

### 6. 生成推荐

```bash
python scripts/recommend.py
```

## 使用示例

### 数据获取

```bash
# 获取单个ETF历史数据
python scripts/fetch_data.py --fetch --code 510050

# 更新所有ETF数据
python scripts/fetch_data.py --update --list
```

### 指标计算

```bash
# 计算所有ETF的所有指标
python scripts/indicators.py --all

# 计算指定ETF的特定指标
python scripts/indicators.py --code 510050 --indicators ma macd
```

### 模型训练

```bash
# 训练所有ETF的模型
python scripts/train_models.py
```

### 回测

```bash
# 运行回测并生成报告
python scripts/run_backtest.py
```

### 推荐系统

```bash
# 生成今日推荐
python scripts/recommend.py
```

### 全流程自动化

```bash
# 一键执行：数据更新 → 指标计算 → 模型训练 → 回测
python scripts/full_workflow.py
```

## 配置说明

### ETF列表配置

在 `.env` 文件中配置ETF列表：

```env
# 核心ETF列表
CORE_ETF_LIST=510050 159915 510300 510500 159901

# 完整ETF列表
ETF_LIST=512480 512760 512400 515980 588000 515050 ...
```

### 模型参数

在 `src/my_etf/config.py` 中配置模型参数：

```python
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    ...
}
```

### 回测参数

在 `src/my_etf/config.py` 中配置回测参数：

```python
BACKTEST_CONFIG = {
    'INITIAL_CAPITAL': 100000,
    'REBALANCE_DAYS': 5,
    'TRANSACTION_RATE': 0.0003,
    ...
}
```

## 开发指南

详细的开发指南请参考 [DEVELOPMENT.md](DEVELOPMENT.md)

### 运行测试

```bash
pytest tests/
```

### 代码规范

- 遵循PEP 8代码规范
- 使用类型注解
- 编写文档字符串
- 添加单元测试

## 常见问题

### Q: 如何添加新的ETF？

A: 在 `.env` 文件中的 `ETF_LIST` 添加ETF代码，然后运行数据获取。

### Q: 如何修改模型参数？

A: 在 `src/my_etf/config.py` 中修改 `MODEL_PARAMS` 字典。

### Q: 如何调整回测参数？

A: 在 `src/my_etf/config.py` 中修改 `BACKTEST_CONFIG` 字典。

### Q: 数据库文件太大怎么办？

A: 可以定期清理历史数据，或使用数据库压缩工具。

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。使用本项目产生的任何投资损失，项目作者不承担任何责任。

## 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]

## 致谢

- [akshare](https://github.com/akfamily/akshare) - 提供金融数据接口
- [XGBoost](https://xgboost.readthedocs.io/) - 机器学习框架
- [pandas](https://pandas.pydata.org/) - 数据分析库
