# 增强的新浪回退逻辑
# 这个文件包含改进的代码片段，将合并到主fetcher.py

import time

def fetch_etf_history_with_sina_fallback(etf_code: str, days: int, end_date: Optional[str] = None, max_retries: int = 3) -> pd.DataFrame:
    """
    增强版ETF历史数据获取函数，带有新浪回退的重试机制

    Args:
        etf_code: ETF代码
        days: 获取最近多少天的数据
        end_date: 结束日期
        max_retries: 新浪回退最大重试次数

    Returns:
        包含OHLCV数据的DataFrame
    """
    import akshare as ak
    from datetime import datetime, timedelta
    import pandas as pd

    _end_dt = datetime.strptime(end_date, '%Y%m%d') if end_date else datetime.now()
    _start_dt = _end_dt - timedelta(days=days)

    # 尝试东方财富数据源
    df = None
    em_success = False

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[Attempt {attempt}/{max_retries}] 尝试东方财富数据源获取 {etf_code}")

            if hasattr(ak, 'fund_etf_hist_em'):
                df = ak.fund_etf_hist_em(
                    symbol=etf_code,
                    period="daily",
                    start_date=_start_dt.strftime('%Y%m%d'),
                    end_date=_end_dt.strftime('%Y%m%d'),
                    adjust="qfq"
                )

                if df is not None and not df.empty:
                    em_success = True
                    logger.info(f"从东方财富获取 {etf_code} 数据成功，共 {len(df)} 条记录")
                    break
                else:
                    logger.warning(f"东方财富返回空数据（尝试 {attempt}）")

        except Exception as e:
            error_msg = f"东方财富数据源失败: {str(e)}"

            # 检查是否是网络错误或超时
            error_type = type(e).__name__
            if 'RemoteDisconnected' in str(e) or 'ConnectionError' in str(e) or 'Timeout' in str(e):
                logger.warning(f"网络错误类型: {error_type} - 将尝试新浪回退")

                # 尝试新浪回退（带重试）
                logger.info(f"[Attempt {attempt}/{max_retries}] 东方财富失败，尝试新浪回退数据源...")
                df_sina = None
                sina_success = False

                for sina_attempt in range(1, max_retries + 1):
                    try:
                        def _pref(code: str) -> str:
                            c = str(code)
                            return ('sh' + c) if c.startswith(('5', '6')) else ('sz' + c)

                        df_sina = ak.fund_etf_hist_sina(symbol=_pref(etf_code))

                        # 检查返回数据
                        if df_sina is not None and not df_sina.empty and '日期' in df_sina.columns:
                            df_sina['日期'] = pd.to_datetime(df_sina['日期'], errors='coerce')
                            df_sina = df_sina[(df_sina['日期'] >= _start_dt) & (df_sina['日期'] <= _end_dt)].copy()

                            if not df_sina.empty:
                                sina_success = True
                                logger.info(f"从新浪回退获取 {etf_code} 数据成功，共 {len(df_sina)} 条记录")
                                break
                            else:
                                logger.warning(f"新浪返回空数据（尝试 {attempt}.{sina_attempt}）")

                                # 重试间隔
                                if sina_attempt < max_retries:
                                    time.sleep(RETRY_DELAY)

                    except Exception as e_sina:
                        logger.warning(f"新浪回退失败（尝试 {attempt}.{sina_attempt}）: {str(e_sina)}")

                if sina_success:
                    df = df_sina
                    logger.info(f"成功：使用新浪回退数据源获取 {etf_code}")
                    break

            # 如果东方财富成功，立即退出循环
            if em_success:
                break

    # 最终检查
    if df is None or df.empty:
        logger.error(f"所有数据源均失败，未获取到 {etf_code} 的ETF日线数据")
        raise ValueError(f"未获取到 {etf_code} 的ETF日线数据（东方财富和新浪均失败）")

    # 标准化列名和日期 - 使用英文列名
    if df is not None:
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
        elif '净值日期' in df.columns:
            # 东方财富返回净值日期，转换为标准date列
            df['date'] = pd.to_datetime(df['净值日期'])
        elif 'date' not in df.columns:
            raise KeyError("返回数据缺少日期列")

        df = df.sort_values('date').reset_index(drop=True)

    return df
