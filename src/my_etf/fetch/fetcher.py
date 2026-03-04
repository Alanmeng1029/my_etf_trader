# -*- coding: utf-8 -*-
"""
ETF数据获取和更新脚本
独立模块，用于从akshare获取ETF历史数据和增量更新
数据存储到 my_etf/data/etf_data.db
"""
import argparse
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

try:
    import akshare as ak
except Exception:
    ak = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from ..config import DB_PATH
from ..utils.logger import setup_logger

# 默认数据更新天数
DEFAULT_DAYS = 700

# 日志系统
logger = setup_logger("etf_fetcher", "fetch.log")


# akshare配置

def _configure_ak_session():
    """为 akshare 配置浏览器头和重试机制"""
    global ak
    if ak is None:
        return
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://quote.eastmoney.com/",
            "Connection": "keep-alive",
        }
        ak.headers = headers

        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        ak.session = session
        logger.info("akshare会话配置成功")
    except Exception as e:
        logger.warning(f"akshare会话配置失败，使用默认设置: {e}")


# 模块加载时自动配置
_configure_ak_session()


# 数据获取函数

def fetch_etf_history(etf_code: str, days: int = DEFAULT_DAYS, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    获取ETF历史数据

    Args:
        etf_code: ETF代码（如 510050）
        days: 获取最近多少天的数据
        end_date: 结束日期，格式 YYYYMMDD，默认为当前日期

    Returns:
        包含OHLCV数据的DataFrame（日期、开盘、收盘、最高、最低、成交量、成交额）

    Raises:
        RuntimeError: akshare未安装
        ValueError: 未获取到数据
    """
    if ak is None:
        raise RuntimeError("akshare 未安装，请先 pip install akshare")

    _end_dt = datetime.strptime(end_date, '%Y%m%d') if end_date else datetime.now()
    _start_dt = _end_dt - timedelta(days=days)

    df = None

    # 优先东方财富日线
    try:
        if hasattr(ak, 'fund_etf_hist_em'):
            df = ak.fund_etf_hist_em(
                symbol=etf_code,
                period="daily",
                start_date=_start_dt.strftime('%Y%m%d'),
                end_date=_end_dt.strftime('%Y%m%d'),
                adjust="qfq"
            )
            if not df.empty:
                logger.info(f"从东方财富获取 {etf_code} 数据成功，共 {len(df)} 条记录")
    except Exception as e:
        logger.warning(f"东方财富数据源失败: {e}")
        df = None

    # 回退：新浪数据源
    if (df is None or df.empty) and hasattr(ak, 'fund_etf_hist_sina'):
        try:
            def _pref(code: str) -> str:
                c = str(code)
                return ('sh' + c) if c.startswith(('5', '6')) else ('sz' + c)

            df = ak.fund_etf_hist_sina(symbol=_pref(etf_code))

            # 仅保留时间窗内数据
            if df is not None and not df.empty and '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
                df = df[(df['日期'] >= _start_dt) & (df['日期'] <= _end_dt)].copy()
                logger.info(f"从新浪回退获取 {etf_code} 数据成功，共 {len(df)} 条记录")
        except Exception as e:
            logger.warning(f"新浪数据源失败: {e}")
            df = None

    if df is None or df.empty:
        raise ValueError(f"未获取到 {etf_code} 的ETF日线数据")

    # 标准化列名和日期 - 使用英文列名
    if '日期' in df.columns:
        # akshare 可能返回中文列名，转换为英文
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount'
        }
        df = df.rename(columns=column_mapping)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    else:
        raise KeyError("返回数据缺少日期列")

    return df


# 数据存储函数

def save_etf_data(df: pd.DataFrame, etf_code: str, if_exists: str = 'append') -> None:
    """
    保存ETF数据到SQLite数据库

    Args:
        df: 包含ETF数据的DataFrame
        etf_code: ETF代码
        if_exists: 如果表已存在，'append' 追加数据，'replace' 替换表
    """
    # 确保数据目录存在
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # 确保日期列存在并为字符串格式
    if 'date' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # 确保只保留必要的列（使用英文列名）
    required_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
    existing_columns = [col for col in required_columns if col in df.columns]

    if len(existing_columns) < len(required_columns):
        missing = set(required_columns) - set(existing_columns)
        logger.warning(f"数据中缺少必要的列: {missing}")

    df_to_save = df[existing_columns]

    conn = sqlite3.connect(DB_PATH)
    try:
        df_to_save.to_sql(f'etf_{etf_code}', conn, if_exists=if_exists, index=False)
        logger.info(f"已保存 {etf_code} 数据到数据库: {DB_PATH}，模式: {if_exists}，记录数: {len(df_to_save)}")
    except Exception as e:
        logger.error(f"保存 {etf_code} 数据到数据库失败: {e}")
        raise
    finally:
        conn.close()


# 数据更新函数

def update_etf_data(etf_code: str) -> None:
    """
    更新ETF数据：从数据库读取最新日期，获取新数据并合并

    Args:
        etf_code: ETF代码
    """
    # 检查数据库是否存在表
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    table_name = f'etf_{etf_code}'

    # 检查表是否存在
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    table_exists = cursor.fetchone() is not None

    if not table_exists:
        logger.info(f"表 {table_name} 不存在，执行完整数据获取")
        conn.close()
        df = fetch_etf_history(etf_code)
        save_etf_data(df, etf_code, if_exists='replace')
        return

    # 获取最新日期 - 假设数据库使用英文列名
    date_column = 'date'
    cursor.execute(f"SELECT MAX({date_column}) FROM {table_name}")
    result = cursor.fetchone()
    conn.close()

    if result[0] is None:
        logger.info(f"表 {table_name} 为空，执行完整数据获取")
        df = fetch_etf_history(etf_code)
        save_etf_data(df, etf_code, if_exists='replace')
        return

    last_date_str = result[0]
    last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
    today = datetime.now()

    # 计算需要获取的天数
    days_needed = (today - last_date).days + 2  # 多取几天确保覆盖

    if days_needed <= 1:
        logger.info(f"{etf_code} 数据已是最新（最新日期: {last_date_str}），无需更新")
        return

    logger.info(f"{etf_code} 最新日期: {last_date_str}，获取 {days_needed} 天的新数据")

    try:
        # 获取新数据
        df_new = fetch_etf_history(etf_code, days=days_needed)

        # 读取现有数据
        conn = sqlite3.connect(DB_PATH)
        df_existing = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()

        # 标准化列名 - 确保使用英文列名
        chinese_columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额']
        english_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']

        # 检测当前数据使用的列名体系
        has_chinese = any(col in df_existing.columns for col in chinese_columns)
        has_english = any(col in df_existing.columns for col in english_columns)

        if has_english:
            # 数据库使用英文列名 - 标准化处理
            pass
        elif has_chinese:
            # 数据库使用中文列名 - 标准化为英文
            df_existing = df_existing.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                                                       '最高': 'high', '最低': 'low',
                                                       '成交量': 'volume', '成交额': 'amount'})
        else:
            # 没有必要的日期列
            logger.warning(f"数据中缺少日期列")

        # 确保有日期列以便格式化
        if 'date' in df_existing.columns:
            df_existing['date'] = pd.to_datetime(df_existing['date']).dt.strftime('%Y-%m-%d')

        # 合并数据（去重）
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
        df_merged['date'] = pd.to_datetime(df_merged['date'])
        df_merged = df_merged.drop_duplicates(subset=['date'], keep='last')
        df_merged = df_merged.sort_values('date').reset_index(drop=True)

        # 保存合并后的数据
        save_etf_data(df_merged, etf_code, if_exists='replace')
        logger.info(f"{etf_code} 更新完成，原有记录: {len(df_existing)}，新增记录: {len(df_new)}，合并后: {len(df_merged)}")

    except Exception as e:
        logger.error(f"更新 {etf_code} 数据失败: {e}")
        raise


# 批量处理函数

def fetch_all_etfs(etf_list: List[str], update_only: bool = False) -> None:
    """
    批量获取或更新ETF数据

    Args:
        etf_list: ETF代码列表
        update_only: 是否仅更新模式（True=更新，False=获取历史）
    """
    if not etf_list:
        logger.warning("ETF列表为空，未执行任何操作")
        return

    mode = "更新" if update_only else "获取历史"
    logger.info(f"开始{mode}模式，共 {len(etf_list)} 个ETF")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for etf_code in etf_list:
        logger.info(f"{'='*60}")
        logger.info(f"处理 ETF: {etf_code}")

        try:
            start_time = datetime.now()

            if update_only:
                update_etf_data(etf_code)
                # 检查是否因已是最新而跳过
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM etf_{etf_code}")
                row_count = cursor.fetchone()[0]
                conn.close()
                if row_count > 0:
                    success_count += 1
                else:
                    skip_count += 1
            else:
                df = fetch_etf_history(etf_code)
                save_etf_data(df, etf_code, if_exists='replace')
                success_count += 1

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"ETF {etf_code} 处理成功，耗时: {elapsed:.2f}秒")

        except Exception as e:
            fail_count += 1
            logger.error(f"ETF {etf_code} 处理失败: {e}")

    logger.info(f"{'='*60}")
    logger.info(f"批量处理完成: 成功 {success_count}，失败 {fail_count}，跳过 {skip_count}")
    logger.info(f"{'='*60}")


# 辅助函数

def load_etf_list_from_env() -> List[str]:
    """从环境变量或.env文件加载ETF列表"""
    from ..config import get_etf_codes
    return get_etf_codes()


# 命令行接口

def main():
    parser = argparse.ArgumentParser(description="ETF数据获取和更新脚本")
    parser.add_argument('--fetch', action='store_true', help='获取历史数据')
    parser.add_argument('--update', action='store_true', help='更新现有数据')
    parser.add_argument('--code', type=str, help='指定单个ETF代码')
    parser.add_argument('--list', action='store_true', help='使用.env中的ETF列表')
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS, help=f'获取最近多少天的数据（默认: {DEFAULT_DAYS}）')
    parser.add_argument('--end-date', type=str, help='结束日期，格式 YYYYMMDD（默认为当前日期）')

    args = parser.parse_args()

    # 参数验证
    if not args.fetch and not args.update:
        parser.print_help()
        logger.info("请指定 --fetch 或 --update 操作")
        return

    if args.code and args.list:
        logger.error("不能同时指定 --code 和 --list")
        return

    # 确定要处理的ETF列表
    if args.code:
        etf_list = [args.code]
        logger.info(f"处理单个ETF: {args.code}")
    elif args.list:
        etf_list = load_etf_list_from_env()
        logger.info(f"处理ETF列表: {etf_list}")
    else:
        parser.print_help()
        logger.info("请指定 --code 或 --list")
        return

    # 执行操作
    try:
        start_time = datetime.now()
        fetch_all_etfs(etf_list, update_only=args.update)
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"总耗时: {elapsed:.2f}秒")
    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == '__main__':
    main()
