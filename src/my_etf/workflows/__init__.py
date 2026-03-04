"""工作流模块"""

from .full_workflow import main as full_main
from .recommend_workflow import main as recommend_main

__all__ = ['full_main', 'recommend_main']
