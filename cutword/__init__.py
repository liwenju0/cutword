#! -*- coding: utf-8 -*-
try:
    from .ner import NER
except ModuleNotFoundError as e:
    print(f"导入ner模块时，发生异常 {e}, 将不能使用ner功能")

from .cutword import Cutter


__version__ = '0.1.0'
