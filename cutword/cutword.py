# -*- coding: utf-8 -*-

import numpy as np
import re
import ahocorasick
import re
import os

re_han = re.compile("([\u4E00-\u9FD5]+)")
re_skip = re.compile("([a-zA-Z0-9]+(?:\.\d+)?%?)")

root_path = os.path.dirname(os.path.realpath(__file__))
class Cutter:
    """Unigram tokenizer with Aho-Corasick automaton
    """
    def __init__(
            self, 
            dict_name="dict.txt", 
            union_name="unionwords.txt",
            custom_dict_path=None
        ):
        '''
        # dict_name:默认的词典的名称
        # need_mid_long  是否加载中等长度的词典，
                        该词典中会包含中等长度的短语，
                        如：一卦千金，谨防疏漏等。
                        也可能包含一些短的词语，
                        但是这些词语不包含在基本的词典中。
                        如明神宗，玉带，采苓等
        # need_long   是否加载长的词典， 该词典中会包含比较长的短语，
                      如：阅卷阶段发现作答内容雷同，进取的人生态度等
        # custom_dict_path  用户自定义词典
        '''
        dict_path = os.path.join(root_path, dict_name)
        self._pieces = {}
        self._load_dict(dict_path)
        union_path = os.path.join(root_path, union_name)
        self._load_dict(union_path)
        if custom_dict_path:
            if not os.path.exists(custom_dict_path):
                raise Exception("custom dict path not exists: %s" % custom_dict_path)
            self._load_dict(custom_dict_path)

       
        # Aho-Corasick automaton
        log_total = np.log(sum([_[0] for _ in self._pieces.values()]))
        self._automaton = ahocorasick.Automaton()
        for k, v in self._pieces.items():
            self._automaton.add_word(k, (len(k), np.log(v[0]) - log_total, v[1]))
        self._automaton.make_automaton()

    def _load_dict(self, file_path):
        print("Loading dictionaries from %s" % file_path)
        for line in open(file_path):
            line = line.strip()
            r = line.split()
            if len(r) >= 2:
                self._pieces[r[0]] = [
                    int(r[1]) + 1e-10, 
                    r[2] if len(r) > 2 else ''
                ]
            else:
                raise ValueError(
                    'invalid dict file format', line
                )


    def _tokenize(self, text):
        inf = -1e10
        scores = [0] + [inf] * len(text)
        routes = list(range(len(text) + 1))
        tokens = []
        for e, (k, v, p) in self._automaton.iter(text):
            s, e = e - k + 1, e + 1
            if scores[s] == inf:
                #word not include in dict
                last = s
                while scores[last] == inf and last > 0:
                    last -= 1
                scores[s] = scores[last] - 10 #punish score 
                routes[s] = last
                   
            score = scores[s] + v
            if score > scores[e]:
                scores[e], routes[e] = score, s

        if e < len(text):
            tokens.append(text[e:])
            text = text[:e]

        while text:
            s = routes[e]
            tokens.append(text[s:e])
            text, e = text[:s], s
        return tokens[::-1]

    def cutword(self, text, return_pos=False):
        res = []
        blocks = re_han.split(text)
        for blk in blocks:
            if re_han.match(blk):
                res.extend(self._tokenize(blk))
            else:
                tmp = re_skip.split(blk)
                tmp = [i for i in tmp if i]
                res.extend(tmp)
        return res
        

if __name__ == "__main__":
    tokenizer = Cutter()
    text = "小明来到中国科学院进修"
    res = tokenizer.cutword(text)
    print(res)


    

    

    