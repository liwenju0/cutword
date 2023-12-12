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

```python
from  cutword import Cutter

cutter = Cutter()
res = cutter.cutword("你好，世界")
print(res)

```

# TODO
添加命名实体识别功能

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=liwenju0/cutword&type=Date)
