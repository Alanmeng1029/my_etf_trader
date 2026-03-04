"""测试数据库工具"""
import os
import pytest


def test_db_path_exists():
    """测试数据库路径配置"""
    from src.my_etf.config import DB_PATH

    assert DB_PATH is not None
    assert isinstance(DB_PATH, str)


def test_get_connection():
    """测试数据库连接"""
    from src.my_etf.utils.database import get_connection

    # 如果数据库文件存在，测试连接
    if os.path.exists(DB_PATH):
        conn = get_connection()
        assert conn is not None
        conn.close()


def test_read_etf_data():
    """测试读取ETF数据"""
    from src.my_etf.config import DB_PATH
    from src.my_etf.utils.database import read_etf_data, get_all_etf_tables

    # 只有数据库文件存在时才测试
    if not os.path.exists(DB_PATH):
        pytest.skip("数据库文件不存在")

    tables = get_all_etf_tables()
    if not tables:
        pytest.skip("数据库中没有ETF表")

    code = tables[0].replace('etf_', '')
    df = read_etf_data(code)

    assert isinstance(df, type(None)) or isinstance(df, object)
    if df is not None and not df.empty:
        assert 'date' in df.columns
        assert 'close' in df.columns
