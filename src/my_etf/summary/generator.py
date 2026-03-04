# -*- coding: utf-8 -*-
"""
生成ETF汇总HTML报告
包含：代码、名称、起始日期、终止日期、最新价格、最近一周涨幅
"""
import os
import sys
from datetime import datetime
import pandas as pd

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from my_etf.utils.database import get_etf_price

# ETF 名称映射（主要ETF）
ETF_NAMES = {
    '510050': '上证50ETF',
    '510300': '沪深300ETF',
    '510500': '中证500ETF',
    '159915': '创业板ETF',
    '159901': '深证100ETF',
    '515050': '5G ETF',
    '515790': '人工智能ETF',
    '512480': '半导体ETF',
    '515980': '科技龙头ETF',
    '588000': '科创50ETF',
    '515060': '央企创新ETF',
    '512010': '医药ETF',
    '512170': '医疗ETF',
    '512400': '有色金属ETF',
    '512760': 'CXO ETF',
    '512660': '军工ETF',
    '512700': '新能源ETF',
    '512880': '证券ETF',
    '515700': '新能源龙头ETF',
    '159326': '半导体设备ETF',
    '159566': '红利低波ETF',
    '159828': '消费电子ETF',
    '159869': '工业母机ETF',
    '159886': '红利指数ETF',
    '159928': '消费ETF',
    '159995': '芯片ETF',
    '516160': '新能源车ETF',
    '516230': '白酒ETF',
    '516880': '光伏ETF',
    '518660': '光伏龙头ETF',
    '518880': '通信ETF',
    '588220': '科创ETF',
}


def calculate_one_week_return(df: pd.DataFrame) -> float:
    """
    计算最近一周涨幅（5个交易日）
    """
    if len(df) < 6:
        return None

    # 获取最新收盘价和5个交易日前的收盘价
    latest_close = df.iloc[-1]['close']
    five_days_ago_close = df.iloc[-6]['close']

    if five_days_ago_close == 0:
        return None

    return ((latest_close - five_days_ago_close) / five_days_ago_close) * 100


def get_etf_summary(etf_code: str) -> dict:
    """
    获取单个ETF的汇总信息
    """
    try:
        df = get_etf_price(etf_code)

        if df.empty:
            return None

        # 确保按日期排序
        df = df.sort_values('date')

        # 获取名称
        name = ETF_NAMES.get(etf_code, f'ETF_{etf_code}')

        # 计算一周涨幅
        week_return = calculate_one_week_return(df)

        return {
            'code': etf_code,
            'name': name,
            'start_date': df.iloc[0]['date'],
            'end_date': df.iloc[-1]['date'],
            'latest_price': round(df.iloc[-1]['close'], 3),
            'week_return': round(week_return, 2) if week_return is not None else 'N/A',
        }
    except Exception as e:
        print(f"获取 {etf_code} 数据失败: {e}")
        return None


def generate_html(etf_summaries: list, output_path: str):
    """
    生成HTML报告
    """
    # 按代码排序
    summaries_sorted = sorted(etf_summaries, key=lambda x: x['code'])

    html_template = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF 汇总报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .update-time {{
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: center;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .positive {{
            color: #e53935;
            font-weight: bold;
        }}
        .negative {{
            color: #43a047;
            font-weight: bold;
        }}
        .neutral {{
            color: #666;
        }}
        .code {{
            font-family: Consolas, monospace;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>ETF 汇总报告</h1>
    <p class="update-time">更新时间: {update_time}</p>

    <table>
        <thead>
            <tr>
                <th>ETF代码</th>
                <th>ETF名称</th>
                <th>起始日期</th>
                <th>终止日期</th>
                <th>最新价格</th>
                <th>最近一周涨幅(%)</th>
            </tr>
        </thead>
        <tbody>
{rows}
        </tbody>
    </table>

    <p style="text-align: center; margin-top: 20px; color: #999;">
        共 {count} 只ETF
    </p>
</body>
</html>
'''

    # 生成表格行
    rows_html = []
    for summary in summaries_sorted:
        week_return = summary['week_return']
        if week_return == 'N/A':
            return_class = 'neutral'
            return_display = 'N/A'
        elif week_return > 0:
            return_class = 'positive'
            return_display = f'+{week_return}%'
        elif week_return < 0:
            return_class = 'negative'
            return_display = f'{week_return}%'
        else:
            return_class = 'neutral'
            return_display = '0.00%'

        row_html = f'''            <tr>
                <td class="code">{summary['code']}</td>
                <td>{summary['name']}</td>
                <td>{summary['start_date']}</td>
                <td>{summary['end_date']}</td>
                <td>{summary['latest_price']}</td>
                <td class="{return_class}">{return_display}</td>
            </tr>'''
        rows_html.append(row_html)

    rows_content = '\n'.join(rows_html)

    # 生成完整HTML
    html_content = html_template.format(
        update_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        rows=rows_content,
        count=len(summaries_sorted)
    )

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML报告已生成: {output_path}")


def main():
    """主函数"""
    # 获取所有ETF代码
    from my_etf.utils.database import get_all_etf_tables

    etf_tables = get_all_etf_tables()
    etf_codes = [table.replace('etf_', '') for table in etf_tables]

    print(f"找到 {len(etf_codes)} 个ETF，开始获取数据...")

    # 获取每个ETF的汇总信息
    summaries = []
    for i, code in enumerate(etf_codes, 1):
        print(f"[{i}/{len(etf_codes)}] 正在获取 {code} 的数据...")
        summary = get_etf_summary(code)
        if summary:
            summaries.append(summary)

    # 确保输出目录存在
    output_dir = 'index_summary'
    os.makedirs(output_dir, exist_ok=True)

    # 生成带日期的文件名
    current_date = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f'etf_summary_{current_date}.html')

    # 生成HTML
    generate_html(summaries, output_path)

    print(f"\n完成！共生成 {len(summaries)} 个ETF的汇总信息。")


if __name__ == '__main__':
    main()
