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
    def __init__(self, dict_name="dict.txt"):
        dict_path = os.path.join(root_path, dict_name)
        self._pieces = {}
        for line in open(dict_path):
            line = line.strip()
            word, freq, pos = line.split()
            self._pieces[word] = [int(freq) + 1e-10, pos]
       
        # Aho-Corasick automaton
        log_total = np.log(sum([_[0] for _ in self._pieces.values()]))
        self._automaton = ahocorasick.Automaton()
        for k, v in self._pieces.items():
            self._automaton.add_word(k, (len(k), np.log(v[0]) - log_total, v[1]))
        self._automaton.make_automaton()


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
                scores[s] = scores[last] -10 #punish score 
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

    def cutword(self, text):
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
    text = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"
    res = tokenizer.cutword(text)
    print(res)


    

    

    