jieba 不维护了，所以有了cutword。

本项目充分借鉴了苏神的bytepiece的代码，在此表示感谢。

https://github.com/bojone/bytepiece


cutword 是一个中文分词库，字典文件根据最新数据统计得到，词频更加合理。

分词速度是jieba的两倍。
可通过 python -m cutword.comparewithjieba 进行测试。

# 安装：
```
pip install cutword
```

# 使用：

## 分词功能

```python
from  cutword import Cutter

cutter = Cutter()
res = cutter.cutword("你好，世界")
print(res)

```
本分词器提供两种词典库，一种是基本的词库，默认加载。一种是升级词库，升级词库总体长度会比基本词库更长一点。

如需要加载升级词库，需要将 want_long_word 设为True
```python
from  cutword import Cutter

cutter = Cutter()
res = cutter.cutword("精诚所至，金石为开")
print(res) # ['精诚', '所', '至', '，', '金石为开']

cutter = Cutter(want_long_word=True)
res = cutter.cutword("精诚所至，金石为开")
print(res) # ['精诚所至', '，', '金石为开']

```

## 命名实体识别

```python
from pprint import pprint
from  cutword import NER

ner = NER()
res = ner.predict("奈雪的茶，新茶饮赛道开创者，创立于2015年，推出“茶饮+软欧包”双品类模式。聚焦以茶为核心的现代生活方式，奈雪已形成“现制茶饮”、“奈雪茗茶”及“RTD瓶装茶”三大业务版块，成功打造“霸气玉油柑”、“鸭屎香宝藏茶”等多款行业经典产品。")
pprint(res) 
'''
[[NERItem(entity='奈雪的茶', begin=0, end=4, ner_type_en='COMMERCIAL', ner_type_zh='商业'),
  NERItem(entity='茶饮', begin=6, end=8, ner_type_en='MANUFACTURE', ner_type_zh='物品'),
  NERItem(entity='2015年', begin=17, end=22, ner_type_en='TIME', ner_type_zh='时间'),
  NERItem(entity='茶饮', begin=26, end=28, ner_type_en='MANUFACTURE', ner_type_zh='物品'),
  NERItem(entity='软欧包', begin=29, end=32, ner_type_en='MANUFACTURE', ner_type_zh='物品'),
  NERItem(entity='茶', begin=42, end=43, ner_type_en='FOOD', ner_type_zh='食品'),
  NERItem(entity='现代', begin=47, end=49, ner_type_en='TIME', ner_type_zh='时间'),
  NERItem(entity='奈雪', begin=54, end=56, ner_type_en='ORG', ner_type_zh='组织'),
  NERItem(entity='茶饮', begin=62, end=64, ner_type_en='MANUFACTURE', ner_type_zh='物品'),
  NERItem(entity='奈雪茗茶', begin=67, end=71, ner_type_en='COMMERCIAL', ner_type_zh='商业'),
  NERItem(entity='RTD瓶装茶', begin=74, end=80, ner_type_en='FOOD', ner_type_zh='食品'),
  NERItem(entity='玉油柑', begin=95, end=98, ner_type_en='FOOD', ner_type_zh='食品'),
  NERItem(entity='鸭屎香宝藏茶', begin=101, end=107, ner_type_en='FOOD', ner_type_zh='食品')]]
'''

```



## Star History

![Star History Chart](https://api.star-history.com/svg?repos=liwenju0/cutword&type=Date)
