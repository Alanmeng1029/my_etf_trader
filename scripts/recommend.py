#!/usr/bin/env python
"""推荐命令行入口"""
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from my_etf.workflows.recommend_workflow import main

if __name__ == '__main__':
    sys.exit(main())
