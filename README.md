![cutword](https://github.com/liwenju0/cutword/assets/16049564/7236459c-50c0-4031-a83f-b0b54975d7f0)




jieba 不维护了，所以有了cutword。

本项目充分借鉴了苏神的bytepiece的代码，在此表示感谢。

https://github.com/bojone/bytepiece


cutword 是一个中文分词库，字典文件根据最新数据统计得到，词频更加合理。

分词速度是jieba的两倍。
可通过 python -m cutword.comparewithjieba 进行测试。

# 1、安装：
```
pip install cutword
```

# 2、使用：

## 2.1分词功能

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

## 2.2命名实体识别

### 2.2.1 使用方法

```python
from pprint import pprint
from  cutword import NER

ner = NER()
res = ner.predict("奈雪的茶，新茶饮赛道开创者，创立于2015年，推出“茶饮+软欧包”双品类模式。聚焦以茶为核心的现代生活方式，奈雪已形成“现制茶饮”、“奈雪茗茶”及“RTD瓶装茶”三大业务版块，成功打造“霸气玉油柑”、“鸭屎香宝藏茶”等多款行业经典产品。", return_words=False)
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
### 2.2.2 支持的实体类型
| 编号 | 英文类型名 | 中文类型名 |
| --- | --- | --- |
| 1 | FOOD | 食品 |
| 2 | MATTER | 物质 |
| 3 | MANUFACTURE | 物品 |
| 4 | CREATION | 作品 |
| 5 | ORG | 组织 |
| 6 | PC | 计算机 |
| 7 | PROPERTY | 属性 |
| 8 | COMMERCIAL | 商业 |
| 9 | INCIDENT | 事件 |
| 10 | CREATURE | 生物 |
| 11 | BASE | 基础 |
| 12 | AFFAIR | 活动 |
| 13 | TIME | 时间 |
| 14 | LOC | 位置 |
| 15 | PHYSIOLOGY | 组织器官 |
| 16 | PERSON | 人名 |
| 17 | TERMINOLOGY | 领域术语 |




## Star History

![Star History Chart](https://api.star-history.com/svg?repos=liwenju0/cutword&type=Date)
