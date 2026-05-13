# my_etf Quick Start

本文档记录当前项目的最新实操流程。默认在项目根目录执行：

```powershell
cd "D:\大学\claude尝试\ETF 尝试\my_etf"
```

## 1. Python 解释器

优先使用：

```powershell
& "D:\Anaconda3\python.exe" --version
```

项目要求 Python `>=3.9`。不要使用旧 Anaconda Python 3.6 环境。

## 2. 安装依赖

```powershell
& "D:\Anaconda3\python.exe" -m pip install -r requirements.txt
```

## 3. 配置 ETF 列表

配置文件：`.env`

当前默认 ETF 列表已经移除 `512780`，因为该 ETF 已退市，数据库最新只能到 `2022-10-17`，不应进入实盘 universe。

仍在观察但会被数据健康检查暂时剔除的 ETF：

- `159189`：历史样本不足 252 天。
- `516230`：历史样本不足 252 天。

## 4. 更新最新数据

联网更新全部 ETF：

```powershell
& "D:\Anaconda3\python.exe" -m my_etf.fetch.fetcher --update --list
```

如果本地沙箱或防火墙拦截网络，需允许访问东方财富/新浪等数据源后重跑。

更新单只 ETF：

```powershell
& "D:\Anaconda3\python.exe" -m my_etf.fetch.fetcher --fetch --code 510300
```

## 5. 计算指标

请按顺序执行，不要并行运行。高级指标依赖部分基础指标列。

```powershell
& "D:\Anaconda3\python.exe" -m my_etf.indicators.calculator --list --force
& "D:\Anaconda3\python.exe" -m my_etf.indicators.advanced_calculator --list --force
```

说明：基础指标命令目前会打印 `advanced` 未知指标以及部分统计阶段警告，但只要日志里出现“数据库更新已提交”，基础指标已写入。随后再运行高级指标即可。

## 6. 数据健康检查

生成 universe 快照并查看剔除原因：

```powershell
& "D:\Anaconda3\python.exe" -c "from my_etf.utils.data_health import collect_data_health, write_universe_snapshot; from my_etf.config import DATA_DIR; df=collect_data_health(); print(write_universe_snapshot(df,'manual_health_check',DATA_DIR)); print(df[['code','rows','max_date','staleness_days','is_eligible','exclude_reason']].sort_values(['is_eligible','code']).to_string(index=False))"
```

本轮检查结果：

- 合格 universe：40 / 45 张 ETF 表。
- `512780`：退市/过期，自动剔除。
- `512780_test`、`512780_test2`：测试表，自动剔除。
- `159189`、`516230`：样本不足，自动剔除。

快照示例：

```text
data/universe_snapshot_manual_health_check_20260513_111240.json
```

## 7. 生成回测预测

使用现有 separate 模型，基于最新数据库重新生成回测预测文件：

```powershell
& "D:\Anaconda3\python.exe" -m my_etf.models.generate_backtest_predictions --separate --all-features
```

注意：新增周期 ETF 如果还没有训练好的模型，会在预测阶段被跳过。需要纳入回测前，先重新训练对应模型。

## 8. 运行回测

标准 separate 回测：

```powershell
& "D:\Anaconda3\python.exe" -m my_etf.backtest.strategy --model-type separate
```

分类回测：

```powershell
& "D:\Anaconda3\python.exe" -m my_etf.backtest.strategy --classification
```

分类现金对冲回测：

```powershell
& "D:\Anaconda3\python.exe" -m my_etf.backtest.strategy --classification --with-cash-hedge
```

## 9. 最新回测报告

本轮已生成：

```text
reports/backtest_results_TOP5_separate.csv
reports/backtest_report_TOP5_separate.html
reports/backtest_comparison.csv
data/universe_snapshot_backtest_separate_20260513_111422.json
```

本轮 TOP5 separate 结果：

| 指标 | 数值 |
| --- | ---: |
| 总收益率 | 44.79% |
| 最终净值 | 144,790.68 |
| 最大回撤 | -11.43% |
| 年化收益率 | 33.12% |
| Sharpe | 1.72 |
| Sortino | 2.21 |
| Calmar | 2.90 |
| 换手率合计 | 44.1821 |
| 交易成本 | 4,357.01 |
| 成本/盈利 | 9.73% |
| 平均持仓数 | 4.98 |
| 空仓天数 | 1 |

## 10. 生成推荐

回归推荐：

```powershell
& "D:\Anaconda3\python.exe" -m my_etf.workflows.recommend_workflow --skip-data-update --skip-indicators
```

分类推荐：

```powershell
& "D:\Anaconda3\python.exe" -X utf8 -m my_etf.workflows.recommend_workflow --classification --skip-data-update --skip-indicators
```

分类推荐会输出 `class_0_prob` 到 `class_3_prob` 的完整概率，并先排除预测类别 `0`。

## 11. 验证

运行测试：

```powershell
& "D:\Anaconda3\python.exe" -m pytest tests
```

编译检查：

```powershell
& "D:\Anaconda3\python.exe" -m compileall src\my_etf
```

## 12. 常见问题

### 数据更新显示 0 成功

通常是网络权限或代理问题。确认已允许访问外部行情源后重跑：

```powershell
& "D:\Anaconda3\python.exe" -m my_etf.fetch.fetcher --update --list
```

### `512780` 仍显示 stale

这是预期行为。该 ETF 已退市，不能补齐到最新交易日，系统会自动剔除。

### 新增 ETF 未进入回测

常见原因：

- 历史数据少于 `BACKTEST_CONFIG['MIN_HISTORY_DAYS']`。
- 没有对应模型文件。
- 预测文件最早日期晚于回测起始日期。

先训练模型，再重新生成预测：

```powershell
& "D:\Anaconda3\python.exe" -m my_etf.models.train --model-type separate
& "D:\Anaconda3\python.exe" -m my_etf.models.generate_backtest_predictions --separate --all-features
```
