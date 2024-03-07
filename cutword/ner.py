# coding=utf-8
import sys
sys.path.append('../')
import torch
import re
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
'''
lstm+crf，训练得到的最好macro-f1是0.686。
'''
import json
try:
    from .model_ner import LstmNerModel
    from .cutword import Cutter
except:
    from model_ner import LstmNerModel
    from cutword import Cutter
from typing import List
import os
import math
from tqdm import tqdm

try:
    from opencc import OpenCC
    s2t_converter = OpenCC('s2t')
    t2s_converter = OpenCC('t2s')

    def s2t(text: str) -> str:
        return s2t_converter.convert(text)

    def t2s(text: str) -> str:
        return t2s_converter.convert(text)
except Exception as e:
    print("opencc 初始化失败！项目将没有繁简转化功能", e)

    def s2t(text: str) -> str:
        return text

    def t2s(text: str) -> str:
        return text

root_path = os.path.dirname(os.path.realpath(__file__))
re_han_split = re.compile(
    r"([\u4E00-\u9Fa5a-zA-Z0-9+#&【】《》<>（）()〔﹝\[﹞〕\]“”\"]+)", re.U)


@dataclass
class NERInputItem:
    sent: str = ""

    # 为了兼容cython添加
    __annotations__ = {
        'sent': str,
    }


@dataclass
class NERItem:
    entity: str = ''
    begin: int = -1
    end: int = -1
    ner_type_en: str = ''
    ner_type_zh: str = ''

    __annotations__ = {
        'entity': str,
        'begin': int,
        'end': int,
        'ner_type_en': str,
        'ner_type_zh': str
    }


@dataclass
class NERInput:
    input: List[NERInputItem]

    # 为了兼容cython添加
    __annotations__ = {
        'input': List[NERInputItem]
    }


@dataclass
class NERResultItem:
    sent: str
    result: List[dict]

    __annotations__ = {
        'str': str,
        'result': List[dict]
    }


@dataclass
class NERResult:
    results: List[NERResultItem]
    __annotations__ = {
        'results': List[NERResultItem]
    }


class NER(object):
    def __init__(self, device=None, model_path=None, preprocess_data_path=None, max_batch_size=128):
        if model_path is None:
            self._model_path = os.path.join(root_path, 'model_params.pth')
        else:
            self._model_path = model_path
        if preprocess_data_path is None:
            self._preprocess_data_path = os.path.join(
                root_path, 'ner_vocab_tags.json')
        else:
            self._preprocess_data_path = preprocess_data_path
        if device is None:
            self._device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device

        with open(self._preprocess_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._id2char = data['id2char']
        self._char2id = data['char2id']
        self._label2id = data['label2id']
        self._id2label = {k: v for v, k in self._label2id.items()}
        self._label_en2zh = data['en2zh']
        self._cutword = Cutter()
        self.__init_model()
        self.max_batch_size = max_batch_size

    def __init_model(self):
        model = LstmNerModel(
            embedding_size=256,
            hidden_size=128,
            vocab_size=len(self._char2id),
            num_tags=len(self._label2id)
        )
        checkpoint = torch.load(
            self._model_path,
            map_location=self._device
        )

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

        model = model.to(self._device)
        self._model = model
        self._model.eval()

    def __split_2_short_text(self, sent, max_len=126):
        if not sent:
            return []
        if len(sent) <= max_len:
            return [sent]
        sents = re_han_split.split(sent)
        res_sents = []
        cur = ""
        for s in sents:
            if len(cur+s) <= max_len:
                cur += s
            else:
                res_sents.append(cur)
                cur = s
        if cur:
            res_sents.append(cur)

        return res_sents

    def __cut_sentences(self, para, drop_empty_line=True, strip=True, deduplicate=False, split_2_short=False, max_len=256, need_clean=False):
        '''cut_sentences
        :param para: 输入文本
        :param drop_empty_line: 是否丢弃空行
        :param strip: 是否对每一句话做一次strip
        :param deduplicate: 是否对连续标点去重，帮助对连续标点结尾的句子分句
        :return: sentences: list of str
        '''
        if para is None or len(para) == 0:
            return []

        if deduplicate:
            para = re.sub(r"([。！？\!\?;；])\1+", r"\1", para)
        para = re.sub(r'(\s+)\1', r'\n\1\n', para)
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

        if not split_2_short:
            return sentences

        assert max_len > 0, f"max_len must be set to a positive integer, got {max_len}"

        short_sents = []
        for sent in sentences:
            s_sents = self.__split_2_short_text(sent, max_len)
            short_sents.extend(s_sents)
        return short_sents

    def __digit_alpha_map(self, text):
        digit_and_alpha_map = {
            '１': '1',
            '２': '2',
            '３': '3',
            '４': '4',
            '５': '5',
            '６': '6',
            '７': '7',
            '８': '8',
            '９': '9',
            '０': '0',
            'Ａ': 'A',
            'Ｂ': 'B',
            'Ｃ': 'C',
            'Ｄ': 'D',
            'Ｅ': 'E',
            'Ｆ': 'F',
            'Ｇ': 'G',
            'Ｈ': 'H',
            'Ｉ': 'I',
            'Ｊ': 'J',
            'Ｋ': 'K',
            'Ｌ': 'L',
            'Ｍ': 'M',
            'Ｎ': 'N',
            'Ｏ': 'O',
            'Ｐ': 'P',
            'Ｑ': 'Q',
            'Ｒ': 'R',
            'Ｓ': 'S',
            'Ｔ': 'T',
            'Ｕ': 'U',
            'Ｖ': 'V',
            'Ｗ': 'W',
            'Ｘ': 'X',
            'Ｙ': 'Y',
            'Ｚ': 'Z',
            'ａ': 'a',
            'ｂ': 'b',
            'ｃ': 'c',
            'ｄ': 'd',
            'ｅ': 'e',
            'ｆ': 'f',
            'ｇ': 'g',
            'ｈ': 'h',
            'ｉ': 'i',
            'ｊ': 'j',
            'ｋ': 'k',
            'ｌ': 'l',
            'ｍ': 'm',
            'ｎ': 'n',
            'ｏ': 'o',
            'ｐ': 'p',
            'ｑ': 'q',
            'ｒ': 'r',
            'ｓ': 's',
            'ｔ': 't',
            'ｕ': 'u',
            'ｖ': 'v',
            'ｗ': 'w',
            'ｘ': 'x',
            'ｙ': 'y',
            'ｚ': 'z',
        }
        new_words = []
        for char in text:
            if char in digit_and_alpha_map:
                new_words.append(digit_and_alpha_map[char])
            else:
                new_words.append(char)
        return ''.join(new_words)

    def _is_digit(self, text):
        return text.isdigit()

    def _is_hanzi(self, text):
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

    def _is_english(self, text):
        pattern1 = re.compile(r"^[A-Za-z]+$")
        if re.match(pattern1, text):
            return True
        else:
            return False

    def __is_special_token(self, text):
        pattern1 = re.compile(r"^[A-Za-z]+$")
        pattern2 = re.compile(r"^[A-Za-z0-9]+$")
        pattern3 = re.compile(r"^[0-9]+$")
        if re.match(pattern1, text):
            return True
        elif not re.match(pattern3, text) and re.match(pattern2, text):
            return True
        else:
            return False

    def __make_input(self, text: str):

        return NERInputItem(sent=text)

    def _get_sent_list(self, texts):
        sentence_ids = []
        sentence_lists = []
        for sentence_id, text in enumerate(texts):
            item = self.__make_input(text)
            text = item.sent

            sentence_list = self.__cut_sentences(text)
            sentence_lists.extend(sentence_list)
            sentence_ids.extend([sentence_id]*len(sentence_list))
        assert len(sentence_ids) == len(sentence_lists), f"sentence_ids length not equal sentence_lists {len(sentence_ids)} {len(sentence_lists)}"
        return sentence_ids, sentence_lists

    
    def _make_batch_encode_predict(self, sentence_list:"list[str]"):
        
        #对句子进行分组
        batched_sentence_lists = []
        if len(sentence_list) <= self.max_batch_size:
            batched_sentence_lists = [sentence_list]
        else:
            num_batch = math.ceil(len(sentence_list) / self.max_batch_size)
            for i in range(num_batch):
                start = i * self.max_batch_size
                end = min((i + 1) * self.max_batch_size, len(sentence_list))
                batched_sentence_lists.append(sentence_list[start:end])
        
        pred_crf_tags_all, tokens_lists_all, word_list_all = [], [], []
        for a_batch_sents in batched_sentence_lists:
            a_batch_word_list = [
                self._cutword.cutword(
                    self.__get_normalize_sent(sent)
                ) 
                for sent in a_batch_sents
            ]

            (
                a_batch_input_tensors, 
                a_batch_seq_lens, 
                a_batch_tokens_list
            ) = self.__encode(a_batch_word_list)
            
            a_batch_pred_crf_tags = self.__get_model_output(
                a_batch_input_tensors,
                a_batch_seq_lens,
            )
            pred_crf_tags_all.extend(a_batch_pred_crf_tags)
            tokens_lists_all.extend(a_batch_tokens_list)
            word_list_all.extend(a_batch_word_list)
        
        assert len(pred_crf_tags_all) == len(
            tokens_lists_all) == len(
            sentence_list
            ), f'predict_tags_all:{len(pred_crf_tags_all)}, \
                tokens_lists: {len(tokens_lists_all)},\
                sentence_lists: {len(sentence_list)} \
                NOT equal'
        
        return pred_crf_tags_all, tokens_lists_all, word_list_all

    def __get_text_output(self, tokens_lists, predict_tags_all, sentence_ids):
        grouped_tokens_list = []
        grouped_predict_tags = []
        temp_token_list = []
        temp_predict_tags = []
        pre = None
        for i in range(len(sentence_ids)):
            if sentence_ids[i] == pre:
                temp_token_list.extend(tokens_lists[i])
                temp_predict_tags.extend(predict_tags_all[i])
            else:
                if pre is None:
                    temp_token_list.extend(tokens_lists[i])
                    temp_predict_tags.extend(predict_tags_all[i])
                    pre = sentence_ids[i]
                else:
                    grouped_predict_tags.append(temp_predict_tags)
                    grouped_tokens_list.append(temp_token_list)
                    temp_token_list = tokens_lists[i]
                    temp_predict_tags = predict_tags_all[i]
                    pre = sentence_ids[i]

        grouped_tokens_list.append(temp_token_list)
        grouped_predict_tags.append(temp_predict_tags)
        return grouped_tokens_list, grouped_predict_tags

    def __get_ori_word_list(self, sentence_list, word_list_all, sentence_id_list):
        new_word_lists = []
        temp_word_list = []
        pre = None
        for i in range(len(sentence_id_list)):
            sentence_id = sentence_id_list[i]
            word_list = word_list_all[i]
            sentence = sentence_list[i]
            idx = 0
            if sentence_id == pre:
                for word in word_list:
                    temp_word_list.append(sentence[idx:idx + len(word)])
                    idx += len(word)
            else:
                if pre is None:
                    for word in word_list:
                        temp_word_list.append(sentence[idx:idx + len(word)])
                        idx += len(word)
                    pre = sentence_id
                else:
                    idx = 0
                    new_word_lists.append(temp_word_list)
                    temp_word_list = []
                    for word in word_list:
                        temp_word_list.append(sentence[idx:idx + len(word)])
                        idx += len(word)
                    pre = sentence_id
        new_word_lists.append(temp_word_list)
        return new_word_lists
    
    def _is_empty(self, texts):
        if not texts:
            return True 
        if isinstance(texts, list) and all(not a for a in texts):
            return True
        return  False
        
    def predict(self, texts: 'str|list', return_words=False):
        if self._is_empty(texts):
            if not return_words:
                return []
            return [], []

        if isinstance(texts, str):
            texts = [texts]

        #记录下空的text，最后将手动补充结果
        not_empty = []
        empty_ids = []
        for i , t in enumerate(texts):
            if not t:
                empty_ids.append(i)
            else:
                not_empty.append(t)
        texts = not_empty
        
        
        sentence_id_list, sentence_list = self._get_sent_list(texts)

        (
            predict_tags_all, 
            tokens_lists, 
            word_list_all
        ) = self._make_batch_encode_predict(sentence_list)

        grouped_tokens_list, grouped_predict_tags = self.__get_text_output(
            tokens_lists, 
            predict_tags_all, 
            sentence_id_list
        )
        res = self.__predict_tags(
            grouped_predict_tags, grouped_tokens_list, texts)
        
        #手动设置空的结果
        for emp_id in empty_ids:
            res.insert(emp_id, [])
            word_list_all.insert(emp_id, [])

        if return_words:
            word_list_all = self.__get_ori_word_list(
                sentence_list, word_list_all, sentence_id_list)
            return res, word_list_all
        else:
            return res

    def __get_model_output(self, input_tensor, seq_lens):
        with torch.no_grad():
            input_tensor = input_tensor.to(self._device)
            seq_lens = seq_lens.to(self._device)
            output_fc, mask = self._model(input_tensor, seq_lens)

            predict_tags = self._model.crf.decode(output_fc, mask)
        return predict_tags

    def __predict_tags(self, grouped_predict_tags, grouped_tokens_list, texts):

        results = []
        for i in range(len(texts)):
            chars_len = 0
            predict_tags = grouped_predict_tags[i]
            tokens_list = grouped_tokens_list[i]
            text = texts[i]
            text_simple = t2s(text)
            text_simple = text_simple.lower()
            text_simple = self.__digit_alpha_map(text_simple)
            predict_tags = [self._id2label[t] for t in predict_tags]
            pre_result = self.__decode_prediction(
                tokens_list, 
                predict_tags, 
                chars_len, 
                text, 
                text_simple
            )

            results.append(pre_result)
        return results

    def __get_normalize_sent(self, sent):
        sent_normalized = sent.lower()
        sent_normalized = self.__digit_alpha_map(sent_normalized)
        sent_normalized = t2s(sent_normalized)
        return sent_normalized

    def __encode(self, a_batch_words_list: "list[list[str]]"):
        input_tensors = []
        seq_lens = []
        input_lists = []
        for a_sent_words in a_batch_words_list:
            a_sent_chars = []
            for word in a_sent_words:
                word = word.lower()
                word = self.__digit_alpha_map(word)
                if word.strip() == '':
                    for _ in range(len(word.strip())):
                        a_sent_chars.append('[SEP]')
                else:
                    if self._char2id.get(word) is not None:
                        a_sent_chars.append(word)
                    else:
                        for char in word:
                            a_sent_chars.append(char)

            a_sent_input_tensor = []
            for char in a_sent_chars:
                if self._char2id.get(char):
                    a_sent_input_tensor.append(self._char2id[char])
                else:
                    if self._is_digit(char):
                        a_sent_input_tensor.append(self._char2id['[NUMBER]'])
                    elif self._is_hanzi(char):
                        a_sent_input_tensor.append(self._char2id['[HANZI]'])
                    else:
                        a_sent_input_tensor.append(self._char2id['[UNK]'])

            a_sent_seq_len = len(a_sent_input_tensor)

            input_tensors.append(torch.tensor(a_sent_input_tensor))
            seq_lens.append(a_sent_seq_len)
            input_lists.append(a_sent_chars)
        
        input_tensors = pad_sequence(
            input_tensors, 
            batch_first=True, 
            padding_value=0
        )
        return torch.tensor(input_tensors), torch.tensor(seq_lens), input_lists

    def __decode_prediction(self, chars, tags, chars_len, text, text_simple):
        new_chars = []
        for char in chars:
            if char == '[SEP]':
                new_chars[-1] += ' '
            else:
                new_chars.append(char)
        assert len(new_chars) == len(tags), "{}{}".format(new_chars, tags)
        result = []
        temp = NERItem()

        idx = chars_len
        for char, tag in zip(new_chars, tags):
            if tag == "O":
                idx += len(char)
                continue
            char_len = len(char)
            head = tag.split('_')[0]
            label = tag.split('_')[-1]
            if "S" in head:

                temp.entity = char
                temp.begin = idx
                temp.end = idx+char_len
                temp.ner_type_en = label
                temp.ner_type_zh = self._label_en2zh[label]
                while text_simple[temp.begin:temp.end] != temp.entity and temp.end < len(text):
                    temp.begin += 1
                    temp.end += 1
                if text_simple[temp.begin:temp.end] != temp.entity:
                    temp = NERItem()
                    continue
                temp.entity = text[temp.begin:temp.end]
                result.append(temp)

                temp = NERItem()

            elif 'B' in head:
                temp.entity = char
                temp.begin = idx
                temp.ner_type_en = label
                temp.ner_type_zh = self._label_en2zh[label]

            elif 'M' in head:
                if not temp.entity:
                    temp = NERItem()
                    continue
                else:
                    temp.entity += char
            elif 'E' in head:
                if not temp.entity:
                    temp = NERItem()
                    continue
                else:
                    temp.entity += char
                    temp.end = idx + char_len
                    while text_simple[temp.begin:temp.end] != temp.entity and temp.end < len(text):
                        temp.begin += 1
                        temp.end += 1
                    if text_simple[temp.begin:temp.end] != temp.entity:
                        temp = NERItem()
                        continue
                    temp.entity = text[temp.begin:temp.end]
                    result.append(temp)
                    temp = NERItem()

            idx += char_len
        return result


if __name__ == '__main__':

    print(root_path)
    # ner_model = NER()
    ner_model = NER(model_path='/Users/milter/Downloads/临时数据/cutword/cutword/model_params.pth',
                    preprocess_data_path='/Users/milter/Downloads/临时数据/cutword/cutword/ner_vocab_tags.json')
    sentence_list = []
    a = '''
    布莱恩科比，是世界文明的篮球巨星。世事无常，令人感到挽惜的是，科比因为飞机事故原因已经离开了人世，一代巨星的陨落，让整个篮球界都感到非常很悲伤。科比一生己经为湖人队效力了20个赛季，在这20个赛季中，科比将曼巴精神演艺到了极致，带着伤病坚持比赛.科比精神不但激励大家，而且鼓舞人心。



他与湖人队签订了一份为期两年价值4850万美元的续约合同，这将使他成为第一位为同一支球队效力达到20年的NBA球员，。 2014年3月12日，湖人队宣布科比2013-14赛季报销。



2014-15赛季，科比复出征战他代表湖人队的弟19个赛季。2014年11月30日，在一场129-122加时赛击败多伦多猛龙队的比赛中，科比得到了生涯第20次三双，31分，12次助攻以及11个篮板。在36岁的年纪，他成为NBA得到30分，10个篮板，10次助攻的最年长球员，这是全世界的创举一个。如果说篮球是一座链接世界的桥，那么科比就是这座桥的其中一个桥礅。


    '''
    sents =  ['布莱恩科比，是世界文明的篮球巨星。', '世事无常，令人感到挽惜的是，科比因为飞机事故原因已经离开了人世，一代巨星的陨落，让整个篮球界都感到非常很悲伤。', '科比一生己经为湖人队效力了20个赛季，在这20个赛季中，科比将曼巴精神演艺到了极致，带着伤病坚持比赛.科比精神不但激励大家，而且鼓舞人心。', '他与湖人队签订了一份为期两年价值4850万美元的续约合同，这将使他成为第一位为同一支球队效力达到20年的NBA球员，。', '2014年3月12日，湖人队宣布科比2013-14赛季报销。', '2014-15赛季，科比复出征战他代表湖人队的弟19个赛季。', '2014年11月30日，在一场 129-122加时赛击败多伦多猛龙队的比赛中，科比得到了生涯第20次三双，31分，12次助攻以及11个篮板。', '在36岁的年纪，他成为NBA得到30分，10个篮板，10次助攻的最年长球员，这是全世界的创举一个。', '如果说篮球是一座链接世界的桥，那么科比就是这座桥的其中一个桥礅。']
    sents =  ['2014年11月30日，在一场 129-122加时赛击败多伦多猛龙队的比赛中，科比得到了生涯第20次三双，31分，12次助攻以及11个篮板。']

    sents = ['首页党的二十大抗击疫情新时代新担当新作为为你喝彩接诉即办脱贫攻坚不忘初心吹哨报到民呼我应向前一步', '*', '要闻速递', '*', '权威解读', '*', '典型经验', '*', '党员榜样', '*', '精品原创特别策划｜十年间，总书记为首都发展指明方向每周政知道（2月18日至2月25日）春节有我！', '温暖不停歇一位京剧团团长的京 剧革新之旅#', '市推进京津冀协同发展领导小组召开会议北京组工', '2024-03-06#', '传达学习全市组织部长会议精神，各区这样做北京组工', '2024-03-06#', '抢险救灾、灾后重建，他们深入一线奋战（上）北京组工', '2024-03-06#', '抢险救灾、灾后重建，他们深入一线奋战（下）北京组工', '2024-03-06查看更多__00:03:55:00#', '团团的初心北京党员教育网', '2021-04-23__00:04:24:00#', '大北京城里的小保安北京党员教育网', '2021-04-23__00:05:12:00#', '忠诚礼赞——2016“北京榜样·最美警察”宣传片：海淀分局', '陈佳伍北京党员教育网', '2021-04-23__00:12:45:00#', '我是“村小二”北京党员教育网', '2021-04-23__00:04:06:00#', '画笔讲述党的故事北京党员教育网', '2021-04-23__00:10:34:00#', '老张的三餐北京党员教育网', '2021-04-23查看更多#', '党建引领持续发力，各区抓这件“关键小事”很走心北京组工', '2020-11-18#', '物业管理焕新意，激活基层治理“红色细胞”北京组工', '2020-11-17#', '探索“红色物业”新模式，社区激发多元共治新活力！', '北京组工', '2020-11-16#', '【经验说】如何激 励干部担当作为，看看各地怎么干？', '北京组工', '2020-10-23#', '亮点频出！', '全国各地各单位这样落实《干部考核条例》北京组工', '2020-10-21#', '党建引领下， 物业管理还有这些打开方式北京组工', '2020-09-22查看更多__00:03:30:00#', '党建公益宣传片——党员·传递北京党员教育网', '2021-05-10__00:04:58:00#', '党性的光辉——顺义榜样之80后义工社党支部北京党员教育网', '2021-05-10__00:26:13:00#', '刑警日记北京党员教育网', '2021-05-10__00:13:16:00#', '平凡中的坚守北京党员教育网', '2021-05-10__00:03:26:00#', '廉洁颂——我身边的好规矩：京剧谭家七代传艺北京党员教育网', '2021-05-10__00:03:09:00#', '廉洁颂——我身边的好规矩：北京交大“知行”校训北京党员教育网', '2021-05-10查看更多__00:01:21:00#', '书记来了：周到书记市委组织部', '2021-12-16__00:01:15:00#', '书记来了：琢磨书记市委组织部', '2021-12-16__00:01:45:00#', '书记来了：谈判书记市委组织部', '2021-12-16__00:01:12:00#', '书记来了：二姑书记市委组织部', '2021-12-16__00:01:07:00#', '书记来了：交代书记市委组织部', '2021-12-13__00:00:56:00#', '书记来了：必须书记市委组织部', '2021-12-13查看更多全媒体平台更多共产党员网人民日报新华社CCTV北京组工网北京日 报北京晚报北京青年报BTV北京时间RBCIPTV新媒体矩阵更多', '*', '微信矩阵', '*', '头条矩阵', '*', '时间矩阵', '*', '', '抖音矩阵精彩联盟北京组工北京长城网首都 人才党员E先锋北京老干部东城党建西城组工朝阳组工红色海淀丰台组工门头沟组工北京组工东城组工西城组工朝阳组工红色海淀丰台组工石景山组工门头沟组工房山组工通州组工顺义组工昌平沟组工北京组工东城组工西城组工朝阳时间红色海淀丰台组工石景山组工门头沟组工房山组工通州组工顺义组工昌平组工织布坊全媒体运用案例', '*', '抗击疫情', '*', '庆祝建党100周年', '*', '为你喝彩', '*', '别样党课北京日报微信公众号极简版！', '北京最新防控优化措施速览北京日报客户端划重点！', '北京10条发布， 防疫政策有调整！', '一图读懂北京日报微信公众号北京哪些场所还需扫码核酸？', '哪些不再查验？', '一图速查北京日报客户端北京：无症状感染者、轻症患者由社区全科 医生进行评估人民日报如何用马克思主义真理力量激活中华文明？', '这篇文章里有答案→《党建研究》百年来党的基层组织建设有哪些经验启示？', '中组部从七方面阐述北京组工点赞！', '他们的默默坚守，让北京更美好北京组工点赞这些优秀党务工作者，汲取榜样力量！', '市委组织部用数据智能驱动未来——赵勇市委组织部报国一甲子铸造最强 盾——钱七虎市委组织部百岁翻译家——许渊冲市委组织部引领中国轨道交通信号新方向——郜春海北京新媒体集团北京市“不忘初心、牢记使命”主题教育先进典型事迹报告会北京新 媒体集团党旗红·冰雪蓝·', '映照我初心——《别样党课》走进冬奥组委北京组工“党旗耀京华——北京市党的建设与组织工作融媒体平台”上线啦！', '北京组工《别样党课》第二 期开讲啦！', '一起聆听榜样的故事北京组工【别样党课】第三期开讲啦！', '走进香山查看全部案例主办：中共北京市委组织部', '技术支持：北京市农林科学院数据科学与 农业经济研究所', '地点：', '北京市通州区运河东大街56号院', '邮编：100743京ICP备05060934号', '党旗耀京华-融媒体平台', '党旗耀京华-融媒体平台', '党旗耀京华- 融媒体平台']
    res, word_list = ner_model.predict(sents, return_words=True)
    print(res)
    print(word_list)
    # for _ in range(10):
    #     sentence_list.append(a)
