#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF全流程自动化工作流
一键执行：数据更新 → 模型训练 → 回测
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

# 获取项目根目录 (注意：src/my_etf/workflows -> src/my_etf -> src -> project_root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_command(cmd_parts, description):
    """执行命令并处理错误"""
    cmd_str = ' '.join(cmd_parts)
    print(f"执行: {cmd_str}")
    print("-" * 80)

    try:
        result = subprocess.run(
            cmd_parts,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"标准输出:\n{e.stdout}")
        print(f"错误输出:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"执行失败: {e}")
        return False


def run_data_update():
    """步骤1：更新数据"""
    cmd = [sys.executable, '-m', 'my_etf.fetch.fetcher', '--update', '--list']
    return run_command(cmd, "更新数据")


def run_indicators_calc():
    """步骤1.5：计算指标（基础+高级）"""
    # 计算基础指标
    cmd1 = [sys.executable, '-m', 'my_etf.indicators.calculator', '--list']
    if not run_command(cmd1, "计算基础指标"):
        return False

    # 计算高级指标
    cmd2 = [sys.executable, '-m', 'my_etf.indicators.advanced_calculator', '--list']
    return run_command(cmd2, "计算高级指标")


def run_prediction():
    """步骤2：训练模型和生成预测"""
    cmd = [sys.executable, '-m', 'my_etf.models.train', '--model-type', 'separate']
    return run_command(cmd, "训练模型")


def run_backtest(model_type='separate', use_classification=False, with_cash_hedge=False, exclude_class_0=False):
    """步骤3：运行回测

    Args:
        model_type: 回归模型类型（separate/unified/both）
        use_classification: 是否使用分类模型
        with_cash_hedge: 分类模型是否使用现金对冲
        exclude_class_0: 分类模型是否排除类别0（大幅下跌）
    """
    if use_classification:
        cmd = [sys.executable, '-m', 'my_etf.backtest.strategy', '--classification']
        if with_cash_hedge:
            cmd.append('--with-cash-hedge')
            desc = f"运行分类模型回测 (现金对冲)"
        elif exclude_class_0:
            # 修改 strategy.py 支持新的参数
            desc = f"运行分类模型回测 (排除类别0)"
            # 临时修改 backtest 策略名称
            # 由于当前 backtest/strategy.py 已经硬编码了排除逻辑，我们直接运行标准分类回测
            # 排除逻辑已经在 run_classification_backtest 函数中实现
            return run_command(cmd, desc)
        else:
            desc = f"运行分类模型回测 (标准)"
    else:
        cmd = [sys.executable, '-m', 'my_etf.backtest.strategy', '--model-type', model_type]
        desc = f"运行回归模型回测 ({model_type})"
    return run_command(cmd, desc)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ETF全流程自动化工作流')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--model-type', choices=['separate', 'unified', 'both'],
                        default='separate',
                        help='回测的回归模型类型 (默认: separate)')
    group.add_argument('--classification', action='store_true',
                        help='使用分类模型进行回测')
    parser.add_argument('--with-cash-hedge', action='store_true',
                        help='使用分类模型进行带现金对冲的回测（需配合--classification）')
    parser.add_argument('--exclude-class-0', action='store_true',
                        help='分类模型回测时排除预测类别为0（大幅下跌）的ETF（需配合--classification）')
    parser.add_argument('--skip-data-update', action='store_true',
                        help='跳过数据更新步骤')
    parser.add_argument('--skip-indicators', action='store_true',
                        help='跳过指标计算步骤')
    parser.add_argument('--skip-training', action='store_true',
                        help='跳过模型训练步骤（使用现有模型）')
    args = parser.parse_args()

    # 验证参数组合
    if args.with_cash_hedge and not args.classification:
        print("错误: --with-cash-hedge 必须与 --classification 一起使用")
        return 1
    if args.exclude_class_0 and not args.classification:
        print("错误: --exclude-class-0 必须与 --classification 一起使用")
        return 1
    if args.with_cash_hedge and args.exclude_class_0:
        print("注意: --with-cash-hedge 和 --exclude-class-0 同时存在，将优先使用 --exclude-class-0")

    print("=" * 80)
    print("ETF全流程自动化工作流")
    if args.classification:
        mode_desc = "Classification"
        if args.exclude_class_0:
            mode_desc += " (排除类别0)"
        elif args.with_cash_hedge:
            mode_desc += " (现金对冲)"
        else:
            mode_desc += " (标准)"
        print(f"回测模式: {mode_desc}")
    else:
        print(f"回测模式: Regression ({args.model_type})")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 记录开始时间
    start_time = datetime.now()

    # 执行步骤1：更新数据（可选）
    if not args.skip_data_update:
        if not run_data_update():
            print("\n错误：数据更新失败，停止执行")
            return 1
    else:
        print("\n跳过数据更新步骤")

    # 执行步骤1.5：计算指标（可选）
    if not args.skip_indicators:
        if not run_indicators_calc():
            print("\n错误：指标计算失败，停止执行")
            return 1
    else:
        print("\n跳过指标计算步骤")

    # 执行步骤2：训练模型（可选，classification 模式下不训练）
    if not args.skip_training and not args.classification:
        if not run_prediction():
            print("\n错误：模型训练失败，停止执行")
            return 1
    elif args.skip_training:
        print("\n跳过模型训练步骤")
    else:
        print("\n分类模式：使用现有分类模型进行回测（不训练）")

    # 执行步骤3：运行回测
    if args.classification:
        if not run_backtest(use_classification=True, with_cash_hedge=False, exclude_class_0=args.exclude_class_0):
            print("\n错误：回测失败，停止执行")
            return 1
    else:
        if not run_backtest(model_type=args.model_type):
            print("\n错误：回测失败，停止执行")
            return 1

    # 计算总耗时
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("全流程完成！")
    print("=" * 80)
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration/60:.1f} 分钟")

    return 0


if __name__ == '__main__':
    sys.exit(main())
