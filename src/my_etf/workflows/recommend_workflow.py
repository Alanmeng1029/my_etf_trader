#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF推荐工作流
更新数据并推荐当前最应该买的ETF（使用现有模型，不重新训练）
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import pandas as pd
import json

from ..config import (
    RECOMMENDATIONS_DIR,
    ETF_NAMES_FILE,
    FEATURE_COLS,
    get_etf_codes,
)
from ..utils.logger import setup_logger

logger = setup_logger("etf_recommend", "recommend.log")


# 获取项目根目录 (注意：src/my_etf/workflows -> src/my_etf -> src -> project_root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_command(cmd_parts, description):
    """执行命令并处理错误"""
    cmd_str = ' '.join(cmd_parts)
    print(f"执行: {cmd_str}")
    print("-" * 80)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        result = subprocess.run(
            cmd_parts,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
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


def load_etf_names(names_file: str) -> Dict[str, str]:
    """加载ETF中文名映射"""
    try:
        with open(names_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: 未找到ETF名称文件 {names_file}")
        return {}
    except json.JSONDecodeError:
        print(f"警告: ETF名称文件格式错误 {names_file}")
        return {}


def generate_daily_predictions():
    """
    使用现有模型生成所有ETF的最新预测

    Returns:
        dict: {code: predicted_return}
    """
    from ..models.predict import generate_daily_predictions
    return generate_daily_predictions(save_universe_snapshot=True)


def generate_classification_predictions():
    """
    使用现有分类模型生成所有ETF的最新预测（实时预测）

    Returns:
        dict: {code: {'predicted_class': int, 'class_probs': list, 'class_4_prob': float, 'date': str}}
    """
    try:
        from ..models.predict import generate_daily_predictions

        # 使用实时预测
        raw_predictions = generate_daily_predictions(
            model_type='classification',
            save_universe_snapshot=True,
        )

        if not raw_predictions:
            print("错误: 无法生成分类预测，请确保已训练分类模型")
            return {}

        predictions = {}
        for code, pred in raw_predictions.items():
            # class_id 是 0-3，proba 是 [class_0_prob, class_1_prob, class_2_prob, class_3_prob]
            class_id = pred['class_id']
            proba = list(pred['proba'])

            # 确保有4个概率值（类别0-3）
            while len(proba) < 4:
                proba.append(0.0)

            # 类别3是大幅上涨。保留 class_4_prob 字段名以兼容旧报表/回测CSV。
            class_4_prob = proba[3] if len(proba) > 3 else 0.0

            predictions[code] = {
                'predicted_class': class_id,
                'class_probs': proba,  # [class_0_prob, class_1_prob, class_2_prob, class_3_prob]
                'class_4_prob': class_4_prob,
                'date': pred['date'],
                'model_version': pred.get('model_version'),
            }

        return predictions

    except Exception as e:
        print(f"错误: 生成分类预测失败 - {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_top_n_recommendations(predictions: dict, n=5) -> List[Dict]:
    """
    获取预测收益最高的N只ETF

    Args:
        predictions: {code: {'predicted_return': float, 'date': str, 'model_version': str}}
        n: 返回的ETF数量

    Returns:
        List of dict: {rank, code, name, predicted_return, prediction_date}
    """
    if not predictions:
        print("错误: 没有找到任何预测数据")
        return []

    # 转换为列表并按预测收益率降序排序
    sorted_predictions = sorted(
        predictions.items(),
        key=lambda x: x[1]['predicted_return'],
        reverse=True
    )

    # 加载ETF中文名称
    etf_names = load_etf_names(ETF_NAMES_FILE)

    # 为推荐的ETF添加中文名称
    recommendations = []
    for i, (code, pred) in enumerate(sorted_predictions[:n]):
        recommendations.append({
            'rank': i + 1,
            'code': code,
            'name': etf_names.get(code, code),
            'predicted_return': pred['predicted_return'],
            'prediction_date': pred['date']
        })

    return recommendations


def get_threshold_based_recommendations(predictions: dict, min_prob_threshold=0.1, max_n=5) -> List[Dict]:
    """
    基于概率阈值获取推荐ETF（与回测逻辑一致）

    Args:
        predictions: {code: {'class_4_prob': float, 'date': str, ...}}
        min_prob_threshold: 最小概率阈值（默认10%）
        max_n: 最大推荐数量（默认5）

    Returns:
        List of dict: {rank, code, name, class_4_prob, predicted_class, prediction_date}
    """
    if not predictions:
        print("错误: 没有找到任何预测数据")
        return []

    # 与分类回测保持一致：先排除类别0（大幅下跌），再按大幅上涨概率排序和筛选。
    eligible_predictions = [
        (code, pred) for code, pred in predictions.items()
        if pred.get('predicted_class') != 0
    ]

    # 转换为列表并按class_4_prob降序排序
    sorted_predictions = sorted(
        eligible_predictions,
        key=lambda x: x[1].get('class_4_prob', 0),
        reverse=True
    )

    # 筛选概率>=阈值的ETF
    high_prob_etfs = [
        (code, pred) for code, pred in sorted_predictions
        if pred.get('class_4_prob', 0) >= min_prob_threshold
    ]

    # 选择：如果满足条件的ETF>=max_n个，选前max_n个；否则选全部满足条件的
    if len(high_prob_etfs) >= max_n:
        selected_etfs = high_prob_etfs[:max_n]
    else:
        selected_etfs = high_prob_etfs

    # 加载ETF中文名称
    etf_names = load_etf_names(ETF_NAMES_FILE)

    # 为推荐的ETF添加信息
    recommendations = []
    for i, (code, pred) in enumerate(selected_etfs):
        recommendations.append({
            'rank': i + 1,
            'code': code,
            'name': etf_names.get(code, code),
            'class_4_prob': pred.get('class_4_prob', 0),
            'predicted_class': pred.get('predicted_class', 0),
            'class_0_prob': pred.get('class_probs', [0, 0, 0, 0])[0],
            'class_1_prob': pred.get('class_probs', [0, 0, 0, 0])[1],
            'class_2_prob': pred.get('class_probs', [0, 0, 0, 0])[2],
            'class_3_prob': pred.get('class_probs', [0, 0, 0, 0])[3],
            'prediction_date': pred['date'],
            'model_version': pred.get('model_version'),
        })

    return recommendations


def print_recommendations(recommendations: List[Dict], use_classification=False) -> None:
    """打印推荐结果"""
    print("\n" + "=" * 80)
    if use_classification:
        print(f"ETF推荐结果 (基于概率阈值，共{len(recommendations)}只)")
        print("=" * 80)
        print("\n排名 | 代码    | 名称          | P0/P1/P2/P3                     | 预测类别 | 预测日期")
        print("-" * 80)

        for rec in recommendations:
            print(f"  {rec['rank']:>2}  | {rec['code']:>6}  | {rec['name']:<12} | "
                  f"{rec['class_0_prob']*100:>5.1f}/{rec['class_1_prob']*100:>5.1f}/"
                  f"{rec['class_2_prob']*100:>5.1f}/{rec['class_3_prob']*100:>5.1f}% | "
                  f"{rec['predicted_class']:>8}  | {rec['prediction_date']}")
    else:
        print("ETF推荐结果 (TOP5)")
        print("=" * 80)
        print("\n排名 | 代码    | 名称          | 预测收益率 | 预测日期")
        print("-" * 70)

        for rec in recommendations:
            print(f"  {rec['rank']:>2}  | {rec['code']:>6}  | {rec['name']:<12} | {rec['predicted_return']:>8.2f}%      | {rec['prediction_date']}")

    print("=" * 80)
    print("注：预测结果仅供参考，不构成投资建议")
    print("=" * 80)


def save_recommendations(recommendations: List[Dict], use_classification=False) -> Dict[str, str]:
    """保存推荐结果到文件"""
    os.makedirs(RECOMMENDATIONS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = "_classification" if use_classification else ""

    # 保存为TXT
    txt_path = os.path.join(RECOMMENDATIONS_DIR, f"recommendations{mode_suffix}_{timestamp}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        if use_classification:
            f.write(f"ETF推荐结果 (基于概率阈值，共{len(recommendations)}只)\n")
        else:
            f.write("ETF推荐结果 (TOP5)\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for rec in recommendations:
            if use_classification:
                f.write(
                    f"{rec['rank']}. {rec['code']} ({rec['name']}) - "
                    f"class0: {rec['class_0_prob']*100:.2f}%, "
                    f"class1: {rec['class_1_prob']*100:.2f}%, "
                    f"class2: {rec['class_2_prob']*100:.2f}%, "
                    f"class3: {rec['class_3_prob']*100:.2f}%, "
                    f"预测类别: {rec['predicted_class']}, "
                    f"模型版本: {rec.get('model_version', '')}\n"
                )
            else:
                f.write(f"{rec['rank']}. {rec['code']} ({rec['name']}) - 预测收益率: {rec['predicted_return']:.2f}%\n")

        f.write("\n注：预测结果仅供参考，不构成投资建议\n")
    print(f"推荐结果已保存: {txt_path}")

    # 保存为CSV
    csv_path = os.path.join(RECOMMENDATIONS_DIR, f"recommendations{mode_suffix}_{timestamp}.csv")
    df = pd.DataFrame(recommendations)
    if use_classification:
        df.to_csv(
            csv_path,
            index=False,
            columns=[
                'rank', 'code', 'name', 'predicted_class',
                'class_0_prob', 'class_1_prob', 'class_2_prob', 'class_3_prob',
                'class_4_prob', 'prediction_date', 'model_version'
            ],
        )
    else:
        df.to_csv(csv_path, index=False, columns=['rank', 'code', 'name', 'predicted_return', 'prediction_date'])
    print(f"推荐CSV已保存: {csv_path}")

    # 保存为HTML
    html_path = os.path.join(RECOMMENDATIONS_DIR, f"recommendations{mode_suffix}_{timestamp}.html")
    generate_html_report(recommendations, html_path, use_classification=use_classification)
    print(f"推荐HTML已保存: {html_path}")
    return {
        'txt': txt_path,
        'csv': csv_path,
        'html': html_path,
    }


def generate_html_report(recommendations: List[Dict], output_path: str, use_classification=False) -> None:
    """生成HTML格式的推荐报告"""
    if use_classification:
        subtitle = f"基于概率阈值 (共{len(recommendations)}只)"
        return_label = "大幅上涨概率"
    else:
        subtitle = "基于XGBoost模型的TOP5预测"
        return_label = "预测收益率"

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF推荐报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
        }}
        .content {{
            padding: 40px;
        }}
        .recommendation-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .rank {{
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
            width: 80px;
            text-align: center;
        }}
        .info {{
            flex: 1;
            margin-left: 20px;
        }}
        .code {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .name {{
            font-size: 1.2em;
            color: #666;
            margin-top: 5px;
        }}
        .return {{
            font-size: 2em;
            font-weight: bold;
            color: #2e7d32;
            text-align: right;
            width: 150px;
        }}
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 25px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ETF推荐报告</h1>
            <p>{subtitle}</p>
        </div>
        <div class="content">
"""

    for rec in recommendations:
        if use_classification:
            return_value = f"{rec['class_4_prob']*100:.2f}%"
            probability_line = (
                f"P0 {rec['class_0_prob']*100:.2f}% / "
                f"P1 {rec['class_1_prob']*100:.2f}% / "
                f"P2 {rec['class_2_prob']*100:.2f}% / "
                f"P3 {rec['class_3_prob']*100:.2f}%"
            )
        else:
            return_value = f"{rec['predicted_return']:.2f}%"
            probability_line = ""

        html_content += f"""
            <div class="recommendation-card">
                <div class="rank">#{rec['rank']}</div>
                <div class="info">
                    <div class="code">{rec['code']}</div>
                    <div class="name">{rec['name']}</div>
                    <div class="name">{probability_line}</div>
                </div>
                <div class="return">{return_value}</div>
            </div>
"""

    html_content += f"""
        </div>
        <div class="footer">
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="color: #95a5a6; font-size: 0.9em;">注：预测结果仅供参考，不构成投资建议</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ETF推荐工作流')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--regression', action='store_true', default=True,
                        help='使用回归模型进行推荐（默认）')
    group.add_argument('--classification', action='store_true',
                        help='使用分类模型进行推荐（基于概率阈值）')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='分类模型概率阈值（默认: 0.1 即10%%）')
    parser.add_argument('--skip-data-update', action='store_true',
                        help='跳过数据更新步骤')
    parser.add_argument('--skip-indicators', action='store_true',
                        help='跳过指标计算步骤')
    args = parser.parse_args()

    # 修正 argparse 的行为：--classification 的作用是设置 classification=True
    use_classification = args.classification

    print("=" * 80)
    print("ETF推荐工作流")
    if use_classification:
        print(f"推荐模式: Classification (阈值: {args.threshold*100:.0f}%)")
    else:
        print("推荐模式: Regression")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = datetime.now()

    # 步骤1：更新数据（可选）
    if not args.skip_data_update:
        if not run_data_update():
            print("\n错误：数据更新失败，停止执行")
            return 1
    else:
        print("\n跳过数据更新步骤")

    # 步骤1.5：计算指标（可选）
    if not args.skip_indicators:
        if not run_indicators_calc():
            print("\n错误：指标计算失败，停止执行")
            return 1
    else:
        print("\n跳过指标计算步骤")

    # 步骤2：使用现有模型生成预测（不重新训练）
    if use_classification:
        predictions = generate_classification_predictions()

        if not predictions:
            print("\n错误：无法生成分类预测，请确保已训练分类模型")
            return 1

        # 步骤3：获取推荐（基于概率阈值）
        recommendations = get_threshold_based_recommendations(predictions, min_prob_threshold=args.threshold, max_n=5)

        if not recommendations:
            print("\n提示: 无满足阈值条件的ETF，建议降低阈值或等待更合适的时机")
            return 0  # 不是错误，只是没有推荐
    else:
        predictions = generate_daily_predictions()

        if not predictions:
            print("\n错误：无法生成预测")
            return 1

        # 步骤3：获取推荐（TOP5）
        recommendations = get_top_n_recommendations(predictions, n=5)

        if not recommendations:
            print("\n错误：无法生成推荐")
            return 1

    # 步骤4：显示推荐结果
    print_recommendations(recommendations, use_classification=use_classification)

    # 步骤5：保存推荐结果
    save_recommendations(recommendations, use_classification=use_classification)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("推荐完成！")
    print("=" * 80)
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration/60:.1f} 分钟")

    return 0


if __name__ == '__main__':
    sys.exit(main())
