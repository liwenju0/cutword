import torch
import re
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
'''
lstm+crf，训练得到的最好macro-f1是0.686。
'''
import random
import json
from model.model_ref import LstmNerModel
import collections
from typing import List
import cutword


@dataclass
class NERInputItem:
    sent: str = ""
    
    #为了兼容cython添加
    __annotations__ = {
        'sent': str,
    }
    
@dataclass
class NERInput:
    input: List[NERInputItem]
    
    #为了兼容cython添加
    __annotations__ = {
        'input': List[NERInputItem]
    }


@dataclass
class NERResultItem:
    sent: str 
    result: List[dict]
    
    
    __annotations__ = {
        'str':str,
        'result': List[dict]
    }
    
@dataclass
class NERResult:
    results: List[NERResultItem]
    __annotations__ = {
        'results': List[NERResultItem]
    }
    


class NER(object):
    def __init__(self, device, model_path=None, preprocess_data_path=None):
        self.model_path = model_path if model_path else 'cutword/cutword/model_params.pth'
        self.preprocess_data_path = preprocess_data_path if preprocess_data_path else 'cutword/cutword/preprocess_data_final.json'
        self.device = device
        
        with open(self.preprocess_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        self.id2char = data['id2char']
        self.char2id = data['char2id']
        self.label2id = data['label2id']
        self.id2label = {k: v for v, k in self.label2id.items()}
        self.cutword = cutword.Cutter()
        self.init_model()
    def init_model(self):
        model = LstmNerModel(embedding_size=256, hidden_size=128, vocab_size=len(self.char2id), num_tags=len(self.label2id))
        checkpoint = torch.load(self.model_path)

        # 初始化模型和优化器


        # 处理模型加载时的参数名称匹配问题（如果模型在训练时使用了数据并行）
        if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
            new_state_dict = {}
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:]  # 去除模块前缀
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        model = model.to(self.device)
        self.model = model
        self.model.eval()
        
    def cut_sentences(self, para, drop_empty_line=True, strip=False, deduplicate=False):
        '''cut_sentences
        :param para: 输入文本
        :param drop_empty_line: 是否丢弃空行
        :param strip: 是否对每一句话做一次strip
        :param deduplicate: 是否对连续标点去重，帮助对连续标点结尾的句子分句
        :return: sentences: list of str
        '''
        if deduplicate:
            para = re.sub(r"([。！？\!\?;；])\1+", r"\1", para)

        para = re.sub(r'([。！？\?!])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub(r'(\.{3,6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub(r'(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub(r'([。！？\?!][”’])([^，。！？\?])', r'\1\n\2', para)
        # 去掉分号断句，因为可能引发语法检测的误报，
        # 比如这句：彭涛说，小时候，自己梦想走出大山；一年支教，让自己又回到了大山；和孩子们相处一年，自己未来多了几分扎根大山的打算。
        # 一年支教，让自己又回到了大山；这句话会引发成分缺失的误报
        # para = re.sub(r'([;；])([^，。！？\?])', r'\1\n\2', para) 
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        sentences = para.split("\n")
        if strip:
            sentences = [sent.strip() for sent in sentences]
        if drop_empty_line:
            sentences = [sent for sent in sentences if len(sent.strip()) > 0]
        return sentences
        
    def digit_alpha_map(self, text):
        digit_and_alpha_map = {
            '１':'1',
            '２':'2',
            '３':'3',
            '４':'4',
            '５':'5',
            '６':'6',
            '７':'7',
            '８':'8',
            '９':'9',
            '０':'0',
            'Ａ':'A',
            'Ｂ':'B',
            'Ｃ':'C',
            'Ｄ':'D',
            'Ｅ':'E',
            'Ｆ':'F',
            'Ｇ':'G',
            'Ｈ':'H',
            'Ｉ':'I',
            'Ｊ':'J',
            'Ｋ':'K',
            'Ｌ':'L',
            'Ｍ':'M',
            'Ｎ':'N',
            'Ｏ':'O',
            'Ｐ':'P',
            'Ｑ':'Q',
            'Ｒ':'R',
            'Ｓ':'S',
            'Ｔ':'T',
            'Ｕ':'U',
            'Ｖ':'V',
            'Ｗ':'W',
            'Ｘ':'X',
            'Ｙ':'Y',
            'Ｚ':'Z',
            'ａ':'a',
            'ｂ':'b',
            'ｃ':'c',
            'ｄ':'d',
            'ｅ':'e',
            'ｆ':'f',
            'ｇ':'g',
            'ｈ':'h',
            'ｉ':'i',
            'ｊ':'j',
            'ｋ':'k',
            'ｌ':'l',
            'ｍ':'m',
            'ｎ':'n',
            'ｏ':'o',
            'ｐ':'p',
            'ｑ':'q',
            'ｒ':'r',
            'ｓ':'s',
            'ｔ':'t',
            'ｕ':'u',
            'ｖ':'v',
            'ｗ':'w',
            'ｘ':'x',
            'ｙ':'y',
            'ｚ':'z',
        }
        new_words = []
        for char in text:
            if char in digit_and_alpha_map:
                new_words.append(digit_and_alpha_map[char])
            else:
                new_words.append(char)
        return ''.join(new_words)
    def is_digit(self, text):
        return text.isdigit()
    
    def is_hanzi(self, text):
        """
        判断序列是否为英文

        :param text: 待判断的序列
        :return: True表示是英文，False表示不是英文
        """
        # 利用正则表达式判断序列是否只包含英文字符
        import re
        pattern = re.compile(r"^[\u4e00-\u9fff]*$")
        if re.match(pattern, text):
            return True
        else:
            return False
        

    def is_english(self, text):
        pattern1 = re.compile(r"^[A-Za-z]+$")
        if re.match(pattern1, text):
            return True
        else:
            return False
    
    def is_special_token(self, text):
        pattern1 = re.compile(r"^[A-Za-z]+$")
        pattern2 = re.compile(r"^[A-Za-z0-9]+$")
        pattern3 = re.compile(r"^[0-9]+$")
        if re.match(pattern1, text):
            return True
        elif not re.match(pattern3, text) and re.match(pattern2, text):
            return True
        else:
            return False
        

    
    def batch_preidct(self, input_list: NERInput):
        results = []
        for item in input_list.input:
            sent = item.sent
            result = self.single_preidct(item)
            results.append(NERResultItem(sent, result))
        ner_result = NERResult(results)
        return ner_result
    
    
    def single_preidct(self, item: NERInputItem):
        text = item.sent
        input_list = self.cut_sentences(text)
        input_tensors, seq_lens, input_lists = self.encode(input_list)
        result = self.predict_tags(input_tensors, seq_lens, input_lists, text)
        return result
    
    
    
    
    def predict_tags(self, input_tensor, seq_lens, input_lists, text):
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output_fc, mask = self.model(input_tensor, seq_lens)
            predict_tags = self.model.crf.decode(output_fc, mask)
            # print_tag = 0
            # target_tag = random.randint(0, len(true_tags)-1)
            results = []

            chars_len = 0
            for pre, chars in zip(predict_tags, input_lists):
                pre = [self.id2label[t] for t in pre]
                pre_result = self.decode_prediction(chars, pre, chars_len, text)
                results.extend(pre_result)
                chars_len += len(chars)
                
            return results
        
    def encode(self, input_str_list: NERInput):
        input_tensors = []
        seq_lens = []
        input_lists = []
        for input_str in input_str_list:
            
            words = self.cutword.cutword(input_str)
            # print(words)
            input_list = []
            for word in words:
                word = word.lower()
                word = self.digit_alpha_map(word)
                if word.strip() == '':
                    for _ in range(len(word.strip())):
                        input_list.append('[SEP]')
                else:

                    for char in word:
                        input_list.append(char)
        
            input_tensor = []
            for char in input_list:
                if char == '[SEP]':
                    continue
                if self.char2id.get(char):
                    input_tensor.append(self.char2id[char])
                else:
                    if self.is_digit(char):
                        input_tensor.append(self.char2id['[NUMBER]'])
                    elif self.is_special_token(char):
                        input_tensor.append(self.char2id['[EWORD]'])
                    elif self.is_hanzi(char):
                        input_tensor.append(self.char2id['[HANZI]'])
                    else:
                        input_tensor.append(self.char2id['[UNK]'])
                
            seq_len = len(input_tensor)

            input_tensors.append(torch.tensor(input_tensor))
            seq_lens.append(seq_len)
            input_lists.append(input_list)
        input_tensors = pad_sequence(input_tensors, batch_first=True, padding_value=0)
        return torch.tensor(input_tensors), torch.tensor(seq_lens), input_lists
                
    def decode_prediction(self, chars, tags, chars_len, text):
        new_chars = []
        for char in chars:
            if char == '[SEP]':
                new_chars[-1] += ' '
            else:
                new_chars.append(char)
        assert len(new_chars) == len(tags), "{}{}".format(new_chars, tags)
        result = []
        temp = {
            'str':'',
            'begin':-1,
            'end':-1,
            'type':'',            
            }
  
        idx = chars_len
        for char, tag in zip(new_chars, tags):
            if tag == "O":
                idx += len(char)
                continue
            char_len = len(char)
            head = tag.split('_')[0]
            label = tag.split('_')[-1]
            if "S" in head:
                
                temp['str'] = char 
                temp['begin'] = idx 
                temp['end'] = idx+1+char_len 
                temp['type'] = label
                while text[temp['begin']:temp['end']] != temp['str'] and temp['end'] < len(text):
                    temp['begin'] += 1
                    temp['end'] += 1
                result.append(temp)

                temp = {
                    'str':'',
                    'begin':-1,
                    'end':-1,
                    'type':''
                    }

            if 'B' in head:
                temp['str'] = char
                temp['begin'] = idx
                temp['type'] = label      
        
                
            elif 'M' in head:
                if not temp['str']:
                    temp = {
                    'str':'',
                    'begin':-1,
                    'end':-1,
                    'type':''
                    }

                    continue
                else:
                    temp['str'] += char
            elif 'E' in head:
                if not temp['str']:
                    temp = {
                    'str':'',
                    'begin':-1,
                    'end':-1,
                    'type':''
                    }
                    continue
                else:
                    temp['str'] += char
                    temp['end'] = idx + char_len 
                    while text[temp['begin']:temp['end']] != temp['str'] and temp['end'] < len(text):
                        temp['begin'] += 1
                        temp['end'] += 1

                    
                    result.append(temp)
                    temp = {
                        'str':'',
                        'begin':-1,
                        'end':-1,
                        'type':''
                    }
            else:
                raise Exception("head error")
            idx += char_len
        return result
    
    
    
if __name__ == '__main__':
    
    
    model_path = '/data/cutword/cutword/model_params.pth'
    processed_data = '/data/cutword/cutword/preprocess_data_final.json'
    device = torch.device('cpu')
    ner_model = NER(model_path=model_path, device=device, preprocess_data_path=processed_data)
    sentence_list = []
    a = '在江苏下辖的13太保中，大型民航运输机场有南京禄口国际机场、无锡硕放机场、 常州奔牛国际机场、\n徐州观音国际机场、南通兴东国际机场、扬州泰州国际机场、连云港花果山国际机场、盐城南洋国际机场、淮安涟水国际机场。涉及到了10个城市，只有苏州、镇江、宿迁三个城市没有自己冠名的大型民航运输机场。镇江、宿迁没有，还情有可原，毕竟这样体量的城市，没有机场的还有一大把，但苏州没有不应该。'
    for _ in range(10):
        sentence_list.append(NERInputItem(sent=a))
    input = NERInput(input=sentence_list)
    result = ner_model.batch_preidct(input)
    print(result)
