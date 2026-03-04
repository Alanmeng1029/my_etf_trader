#!/usr/bin/env python
"""指标计算命令行入口"""
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from my_etf.indicators.calculator import main

if __name__ == '__main__':
    main()
