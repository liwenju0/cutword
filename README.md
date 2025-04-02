```
              _____      _                         _
             / ____|    | |                       | |
            | |    _   _| |___      _____  _ __ __| |
            | |   | | | | __\ \ /\ / / _ \| '__/ _` |
            | |___| |_| | |_ \ V  V / (_) | | | (_| |
             \_____\__,_|\__| \_/\_/ \___/|_|  \__,_|
                        切词 • 分析 • 识别

```
**jieba不维护了，所以有了cutword。**

cutword 是一个中文分词库，字典文件根据截止到2024年1月份的最新数据统计得到，词频更加合理。
基于ac自动机实现的分词算法，分词速度是jieba的两倍。
可通过 python -m cutword.comparewithjieba 进行测试。

Note：本项目并不支持英文实体的识别。如需要英文实体的识别，推荐使用nltk。

# 1、安装：
```
pip install -U cutword
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
初始化Cutter时，支持传入用户自定义的词典，词典格式需要和本项目的dict文件保持一致，词典中的词性一列，暂时没有使用，可随意填写。

## 2.2命名实体识别

### 2.2.1 使用方法

```python
from pprint import pprint
from  cutword import NER

ner = NER()
res = ner.predict(
  "奈雪的茶，新茶饮赛道开创者，创立于2015年，推出“茶饮+软欧包”双品类模式。聚焦以茶为核心的现代生活方式，奈雪已形成“现制茶饮”、“奈雪茗茶”及“RTD瓶装茶”三大业务版块，成功打造“霸气玉油柑”、“鸭屎香宝藏茶”等多款行业经典产品。",
  return_words=False
)
# 如果需要分词结果，可将return_words设为True。返回的是(res, words)
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

### 2.2.3 性能比较
下面比较使用的数据集是我们内部的验证集

cutword：

| Category | Precision | Recall | F1    |
|----------|-----------|--------|-------|
| ACTIVITY | 0.786     | 0.679  | 0.729 |
| ATTR     | 0.772     | 0.703  | 0.736 |
| BASIC    | 0.878     | 0.854  | 0.866 |
| BIOLOGY  | 0.867     | 0.877  | 0.872 |
| BUSINESS | 0.728     | 0.552  | 0.628 |
| COMPUTER | 0.854     | 0.645  | 0.735 |
| EVENT    | 0.786     | 0.716  | 0.75  |
| FOOD     | 0.758     | 0.742  | 0.75  |
| KAFIELD  | 0.753     | 0.706  | 0.728 |
| LIFE     | 0.798     | 0.766  | 0.782 |
| MATTER   | 0.826     | 0.748  | 0.785 |
| PRODUCT  | 0.776     | 0.72   | 0.747 |
| TIME     | 0.916     | 0.897  | 0.906 |
| WORK     | 0.912     | 0.794  | 0.849 |
| ORG      | 0.814     | 0.765  | 0.788 |
| LOC      | 0.83      | 0.81   | 0.82  |
| PERSON   | 0.888     | 0.876  | 0.882 |


LAC:

| Category | Precision | Recall | F1    |
|----------|-----------|--------|-------|
| ORG      | 0.665     | 0.422  | 0.516 |
| LOC      | 0.608     | 0.436  | 0.508 |
| PERSON   | 0.776     | 0.741  | 0.758 |


本项目是[匠数科技](https://www.deepctrl.net)根据多年业务积累开发的NLP基础工具。匠数科技是一家专注于内容安全领域的国家高新技术企业。可以对大模型输出内容、各类网站、媒体进行安全性检测并生成检测报告。

本项目借鉴了苏神的[bytepiece](https://github.com/bojone/bytepiece)的代码，在此表示感谢。


## Star History

![Star History Chart](https://api.star-history.com/svg?repos=liwenju0/cutword&type=Date)
