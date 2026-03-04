# 变更日志

## [1.0.0] - 2026-02-27

### 新增功能
- 实现标准Python项目结构重构
- 统一配置管理 (config.py)
- 公共工具模块 (utils/)
- 数据获取模块 (fetch/)
- 技术指标计算模块 (indicators/)
- 模型训练和预测模块 (models/)
- 策略回测模块 (backtest/)
- 工作流模块 (workflows/)
- 便捷CLI脚本 (scripts/)
- 单元测试 (tests/)
- 项目文档 (DEVELOPMENT.md, CHANGELOG.md)

### 改进
- 消除代码重复，提取公共函数到utils模块
- 统一日志管理
- 改进路径配置，支持跨平台
- 添加pyproject.toml支持现代Python项目配置
- 添加requirements.txt明确依赖

### 技术细节
- 使用python-dotenv管理环境变量
- 标准化数据库操作接口
- 统一模型参数配置
- 改进错误处理和日志记录

### 文档
- 新增DEVELOPMENT.md开发指南
- 更新项目架构说明
- 添加测试用例

### 待优化
- [ ] 完善单元测试覆盖率
- [ ] 添加CI/CD配置
- [ ] 添加更多技术指标
- [ ] 优化模型性能
- [ ] 支持更多数据源

## [0.1.0] - 初始版本

### 核心功能
- ETF数据获取 (etf_data_fetcher.py)
- 技术指标计算 (indicators.py)
- 模型训练和预测 (predict.py)
- 策略回测 (backtest_strategy.py)
- 全流程自动化 (full_workflow.py)
- 推荐系统 (recommend_workflow.py)
