import time
import torch
import re
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
'''
lstm+crf，训练得到的最好macro-f1是0.686。
'''
import json
from .model_ner import LstmNerModel
from .cutword import Cutter
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
        para = re.sub(r'\s+', '\n', para)
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
        return sentence_ids, sentence_lists

    def _make_batch_encode_predict(self, sentence_list):
        batched_sentence_lists = []
        if len(sentence_list) <= self.max_batch_size:
            batched_sentence_lists = [sentence_list]
        else:
            num_batch = math.ceil(len(sentence_list) / self.max_batch_size)
            for i in range(num_batch):
                start = i * self.max_batch_size
                end = min((i + 1) * self.max_batch_size, len(sentence_list))
                batched_sentence_lists.append(sentence_list[start:end])
        predict_tags_all, tokens_lists = [], []
        word_list_all = []
        for batched_sentence_list in batched_sentence_lists:
            word_list = [self._cutword.cutword(
                self.__get_normalize_sent(sent)) for sent in batched_sentence_list]

            input_tensors_batched, seq_lens_batched, tokens_list = self.__encode(
                word_list)
            predict_tags = self.__get_model_output(
                input_tensors_batched,
                seq_lens_batched,
            )
            predict_tags_all.append(predict_tags)
            tokens_lists.append(tokens_list)
            word_list_all.extend(word_list)
        assert len(predict_tags_all) == len(tokens_lists) == len(
            sentence_list), f'predict_tags_all:{len(predict_tags_all)}, tokens_lists, sentence_lists not equal'
        return predict_tags_all, tokens_lists, word_list_all

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
                    grouped_predict_tags.temp_token_list(temp_predict_tags)
                    grouped_tokens_list.temp_token_list(temp_token_list)
                    temp_token_list = []
                    temp_predict_tags = []

        grouped_tokens_list.extend(temp_token_list)
        grouped_predict_tags.extend(temp_predict_tags)
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

    def predict(self, texts: 'str|list', return_words=False):
        if not texts:
            return []
        if isinstance(texts, list) and all(not a for a in texts):
            return []

        if isinstance(texts, str):
            texts = [texts]

        sentence_id_list, sentence_list = self._get_sent_list(texts)

        predict_tags_all, tokens_lists, word_list_all = self._make_batch_encode_predict(
            sentence_list)

        grouped_tokens_list, grouped_predict_tags = self.__get_text_output(
            tokens_lists, predict_tags_all, sentence_id_list)
        res = self.__predict_tags(
            grouped_predict_tags, grouped_tokens_list, texts)

        if return_words:
            word_list_all = self.__get_ori_word_list(
                sentence_list, word_list_all, sentence_id_list)
            return res, word_list_all
        else:
            return res

    def __get_model_output(self, input_tensor, seq_lens):
        with torch.no_grad():
            input_tensor = input_tensor.to(self._device)
            output_fc, mask = self._model(input_tensor, seq_lens)

            predict_tags = self._model.crf.decode(output_fc, mask)
        return predict_tags

    def __predict_tags(self, grouped_predict_tags, grouped_tokens_list, texts):

        results = []
        for i in range(len(texts)):
            result = []
            chars_len = 0
            predict_tags = grouped_predict_tags[i]
            tokens_list = grouped_tokens_list[i]
            text = texts[i]
            text_simple = t2s(text)
            text_simple = text_simple.lower()
            text_simple = self.__digit_alpha_map(text_simple)
            predict_tags = [self._id2label[t] for t in predict_tags]
            pre_result = self.__decode_prediction(
                tokens_list, predict_tags, chars_len, text, text_simple)
            result.extend(pre_result)

        results.append(result)
        return results

    def __get_normalize_sent(self, sent):
        sent_normalized = sent.lower()
        sent_normalized = self.__digit_alpha_map(sent_normalized)
        sent_normalized = t2s(sent_normalized)
        return sent_normalized

    def __encode(self, words_list: "list[list[str]]"):
        input_tensors = []
        seq_lens = []
        input_lists = []
        for words in words_list:
            input_list = []
            for word in words:
                word = word.lower()
                word = self.__digit_alpha_map(word)
                if word.strip() == '':
                    for _ in range(len(word.strip())):
                        input_list.append('[SEP]')
                else:
                    if self._char2id.get(word) is not None:
                        input_list.append(word)
                    else:
                        for char in word:
                            input_list.append(char)

            input_tensor = []
            for char in input_list:
                if self._char2id.get(char):
                    input_tensor.append(self._char2id[char])
                else:
                    if self._is_digit(char):
                        input_tensor.append(self._char2id['[NUMBER]'])
                    elif self._is_hanzi(char):
                        input_tensor.append(self._char2id['[HANZI]'])
                    else:
                        input_tensor.append(self._char2id['[UNK]'])

            seq_len = len(input_tensor)

            input_tensors.append(torch.tensor(input_tensor))
            seq_lens.append(seq_len)
            input_lists.append(input_list)
        input_tensors = pad_sequence(
            input_tensors, batch_first=True, padding_value=0)
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
    ner_model = NER(model_path='/data/cutword/cutword/model_params.pth',
                    preprocess_data_path='/data/cutword/cutword/ner_vocab_tags.json')
    sentence_list = []
    a = '蘇恩曦Jack和酒德麻衣rose神色凝重地走向大廳。'
    res, word_list = ner_model.predict(a, return_words=True)
    print(res)
    print(word_list)
    # for _ in range(10):
    #     sentence_list.append(a)
