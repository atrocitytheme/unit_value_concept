"""
    主文件，可在此内进行单位模型抽取
    分为extractor和Analyzer两部分
    以及部分转化的函数工具
    Analyzer负责处理extractor已经处理好的数据，并对新数据进行分类
    extractor负责处理现有数据，产生载入Analyzer的必要文件

    应用extractor所需文件: 
                    全属性文件：attribute_dict_all.xlsx(不去重):
                        用align_tools里的get_attribute处理deal_product_all得到该文件
                    相似词典: attribute_similar_dict_verify.xlsx/
                            attribute_similar_dict_all.xlsx

    Analyzer需要单位或数值出现次数的统计数据，

    计数的excel文件必须以counter_<属性名>.xlsx做名字

    method1 (不进行对齐):
    unit_extractor -> unit_analyzer -> merge ||                -> Value_Analyzer -> output
    value_extractor/unit_value_extractor/advanced_value_extractor -> merge ||
    method2: (进行单位对齐)
    unit_extractor -> unit_analyzer -> unit_value_extractor/advanced_value_extractor ->(save) value_analyzer -> output
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import re, string
from .words_handler import *
from .Feature_align import similard2excel
from .train_produce import *
from .config import *
from typing import Union, Tuple
from .align_tools import *
import os, re
import Levenshtein, string
import warnings
from .Anomaly import Anomaly_Detect
from .IO import *

def _to_save_dict(d: dict) -> dict:
    """
        convert bucket_comp to save_form
        将分桶结果保存为可存储的字典形式
            {
                "attr1":  {
                    "bucket_id;min;max": [<counter>],
                    "bucket_id;min;max": [<counter>],
                    ...
                },
                "attr2": [<same above>],
                "attr3": [<same above>],
                ...
            }
    """
    r = {}
    for attr in d:
        r[attr], t = {}, d[attr]
        for key in sorted(t.keys(), key=lambda x: int(x[0])):
            val = t[key]
            new_key = str(key[0]) + ';' + str(key[1]) + ';' + \
                        str(key[2])
            r[attr][new_key] = [val]
    return r

def _to_bucket_dict(d: dict) -> dict:
    """
        convert save_dict form to bucket_form
    """
    r = {}
    for attr in d:
        r[attr], t = {}, d[attr]
        for key in t:
            val, key = t[key], key.split(';')
            new_key = (key[0], key[1], key[2])
            r[attr][new_key] = val
    return r

def _convert_index(index):
    """
        转换index
    """
    r = ''
    for d in index:
        if d.isdigit() or d == '.':
            r += d
        else:
            break
    return r

def distance(s1, s2):
    return 1 - Levenshtein.distance(s1, s2)/max(len(s1), len(s2))
# anomaly_vals = defaultdict(list)
Similar_corpus = './hood/attribute_similar_dict_verify.xlsx'

def generate_dict(df):
    """
        将attrbute_dict_all的DataFrame处理为attrd
    """
    o = {}
    # display(df)
    for key in df.index:
        # print(key)
        o[key] = [x for x in df.loc[key].tolist() if x]
        # print(df.loc[key])
    # print('s_dict ready!')
    return o

def generate_mapped_dict(filename: str) -> dict:
    """
        generate total_counter of the form: {"a": [1], "b": [10], ...}
        filename should be in excel form
    """

    df = pd.read_excel(filename, header=None).fillna('')
    df = df.set_index(0, drop=True)
    # print(df)
    return generate_dict(df)

def generate_sep_mapped_dict(filenames: list, attrnames: list) -> dict:
    """
            returns form: {"attr1": {"a1": [1]}...} from 
            files in filenames to fit sep_counter
            where attrnames are: ['attr1', ...]
            precondition: len(filenames) == len(attrnames)
            filenames should be in excel form
    """
    r = {}
    for file, attr in zip(filenames, attrnames):
        r[attr] = generate_mapped_dict(file)
    return r

def generate_sep_mapped_dict_advanced(directory: str, attrnames: list) -> dict:
    """通过directory来进行提取sep_mapped_dict
        precondition:
            directory中counter_excel文件的数量与len(attrnames)相同
    """
    excel_finder = re.compile(r'^counter_.*\.xlsx$')
    abs_path = os.path.abspath(directory)
    filenames = os.listdir(abs_path)
    # print(filenames)
    filenames = [x for x in filenames if bool(excel_finder.match(x))]
    # print(filenames)
    new_filenames = [None] * len(filenames)
    for index, attr in enumerate(attrnames):
        for file in filenames:
            if trim_attr(attr) == file[8:-5]:
                new_filenames[index] = os.path.join(abs_path, file)

    return generate_sep_mapped_dict(new_filenames, attrnames)

class AttributeMatchError(Exception):
    pass

class NoUnitCounterError(Exception):
    pass

class NoValueCounterError(Exception):
    pass

class ComponentLoadError(Exception):
    pass

class BucketAssignError(Exception):
    pass

class AnalyzerLockedError(Exception):
    pass



"""Extractors
"""
class unit_extractor(object):
    """
        批量单位抽取机，可从批量中得到合适的单位库
        similar_attrs:
            {"attr1": ["attrx", "attry"...], "ar1": ["arx", "ary"]}, where ..x, ..y is the similar attribute of "ar1" or "attr1"
        similar_corpus:
            {"attr1": [v1, v2, v3 ... ], "attr2": [a1, a2, a3...s]} where v1, v2, v3... are values that are similar
        or similar_corpus is attrd
        *** similar_attrs is used exclusively for method <self.get_similar_attr>
    """
    def __init__(self, similar_attrs: dict, similar_corpus: dict):
        """
            the form of similar_attrs should be: {"attr1": ["attrx", "attry"...], "ar1": ["arx", "ary"]}, where ..x, ..y is the similar attribute of "ar1" or "attr1"
            the form of similar_corpus should be: {"attr1": [v1, v2, v3 ... ], "attr2": [a1, a2, a3...s]} where v1, v2, v3... are values that are similar
            or similar_corpus is attrd
            similar_attrs is used exclusively for method <self.get_similar_attr>
        """
        self._similar_corpus = similar_attrs
        self.s_dict = similar_corpus
        self._unit_counter = defaultdict(lambda: 0)
        self._unit_counter_sep = {}
        self._unit_got = False
        print('unit extractor initiated!')

    def get_similar_attr(self, attrname) -> list:
        """
            get similar attr from the attribute similar corpus
            returns [attrname, attr1, attr2 ...]
        """
        Similar_corpus = self._similar_corpus
        for key in Similar_corpus:
            if key == attrname:
                return [key] + Similar_corpus[key]
        return []

    def get_all_values(self, attrnames) -> dict:
        """
            从语料库中得到attrnames中attrname的所有值和相似属性的值

            attrnames: 
                ["attr1", "attr2" .... ]
            returns: 
                {"attr1": [v1, v2, v3, ...], 
                "attr2": [s1, s2, s3, ...],
                ...}
        """
        n_dict = defaultdict(list)
        s_dict = self.s_dict
        for attr in attrnames:
            all_names = self.get_similar_attr(attr)
            for name in all_names:
                n_dict[attr].extend(s_dict[name])
        return n_dict

    def get_attr_unit(self, attr_name) -> set:
        """
            返回属性名所有单位集合
        """
        value_dict = self.get_all_values([attr_name])
        # print(value_dict)
        value_lst = value_dict[attr_name]
        # Can use Anomaly detect here for value_lst
        units = self._extract_unit(value_lst, attr_name)
        return units

    def get_unit_dict(self, attr_names: list = [], min_ratio: float = None, outfile='') -> dict:
        """
            return a dict form of units of all attr_names
            根据attrname in attr_names得到对应的单位dict
            min_ratio: 最小出现比率
            对最后结果按照总词数进行筛选
            outfile: the path to store
            将结果保存的路径
        """
        if not attr_names:
            attr_names = list(self._similar_corpus.keys())

        r_dict = {}
        # print(attr_names)

        for attr in attr_names:
            # print(attr)
            r_dict[attr] = self.get_attr_unit(attr)

        counter, total = self.get_unit_count()
        # print(counter, total)
        min_ratio = min_ratio if min_ratio is not None else 2/total

        self.delete_repeat(r_dict)
        self.unlock_counter()
        self._threshold(min_ratio, total, r_dict, counter)

        if outfile:
            similard2excel(r_dict, outfile)

        return r_dict

    def get_main_unit_dict(self, r_dict={}) -> dict:
        """
            r_dict: 
                {"attr1": ["u1", "u2", ...] ...}
            returns
                {"attr1": "u1"...}
        """

        if self.is_locked():
            raise NoUnitCounterError("before getting a main_unit_dict, you should unlock the \
                                        extractor or get_unit_dict first")
        r = {}
        unit_dict = r_dict if r_dict else self.get_unit_dict()

        for key in unit_dict:
            r[key] = unit_dict[key][0] if unit_dict[key] else ''



        return r

    def get_unit_count(self) -> Tuple[dict, int]:
        """
            得到抽取的unit计数和总出现次数
            outfile: 以excel形式保存到指定文件
        """
        counter = dict(self._unit_counter)

        return counter, sum(counter.values())

    def units(self, out_directory='', total_outfile='') -> tuple:
        """
            第一个返回值得到单位计数字典，相当于返回get_unit_count的count dict, 但得到的是similard的形式
            e.g: {'a': [1], 'b': [10]}
            用于适配similard2excel
            第二个返回值得到在属性下分别出现的次数，同样用于适配 similard2excel
            e.g: {'attr1': {'a': [1], 'b': [10]}, 
                  'attr2': {'c': [100], 'b': [5]}}
            out_directory: 可将结果输出为多个文件到指定目录
            total_outfile: 将总计数结果保存到指定文件

            >>> extrctor.units()
            >>> ({'a': [1], 'b': [10]}, {'attr1': {'a': [1], 'b': [10]}, 
                'attr2': {'c': [100], 'b': [5]}})
        """
        if not self._unit_got:
            raise NoUnitCounterError("No unit data in this extractor \
                                   You may run self.get_unit_dict() or\
                                   self.unlock_counter()\
                                    to get the unit")

        counter = self.get_unit_count()[0]

        counter_sep = self._unit_counter_sep.copy()

        for key in counter:
            counter[key] = [counter[key]]

        for key in counter_sep:
            counter_sep[key] = dict(counter_sep[key])
            for ele in counter_sep[key]:
                counter_sep[key][ele] = [counter_sep[key][ele]]

        if total_outfile:
            similard2excel(counter, total_outfile)

        if out_directory:
            for attr in counter_sep:
                d = counter_sep[attr]
                similard2excel(d, os.path.join(out_directory, 'counter_{}.xlsx'.format(trim_attr(attr))))
        return counter, counter_sep

    def delete_repeat(self, out_dict):
        """
            为out_dict去重
        """
        for key in out_dict:
            out_dict[key] = list(set([x for x in out_dict[key] if x]))
        # print(out_dict)

    def parse_unit(self, w: str):
        """
            从w中提取单位

            >>> extractor.parse_unit("5匹")
            >>> 匹
        """
        return process_unit(w)

    def load(self, unit_counter, unit_counter_sep):
        """
            加载unit_counter: {"a": 1, "b": 10, ...}
            加载unit_counter_sep: 见self.units()
        """
        self._unit_counter = unit_counter
        self._unit_counter_sep = unit_counter_sep

    def _extract_unit(self, lst: list, attr_name) -> set:
        """
            lst: list
                在一个给定str的list中抽取单位集合
            attr_name:
                所抽取的属性类别名
            returns 
                list of extracted units
        """
        r = []
        for unit in lst:
            r.append(self._filter_unit(unit, attr_name))
        r = set(r)
        # print('extracted units here: ...')
        # print(r)
        # print('\n')
        return r

    def _filter_unit(self, w: str, attr_name, change_unit=True) -> str:
        """
            从w中提取出单位

            w: 
                句子
            attr_name: 
                所抽取的属性类别名
            change_unit: 
                是否修改各类字典
            returns
                单位，并对其出现进行计数
        """
        r = process_unit(w)

        if change_unit:
            self.unit_change(r, attr_name)
        return r

    def unit_change(self, r, attr_name=None):
        """
            对r的出现在特定attr_name中进行一次计数
        """
        sep_counter = self._unit_counter_sep

        if r:
            self._unit_counter[r] += 1

            if attr_name:
                if attr_name in sep_counter:
                    sep_counter[attr_name][r] += 1
                else:
                    sep_counter[attr_name] = defaultdict(lambda: 0)

    def _threshold(self, min_ratio, total, r_dict, counter):
        """
            通过比率排除结果
        """
        for key in r_dict:
            r_dict[key] = [x for x in r_dict[key] if counter[x]/total > min_ratio]

    def clear(self):
        """
            清理get_unit_dict收集产生的数据
        """
        self._unit_got = False
        self._unit_counter = defaultdict(lambda: 0)
        self._unit_counter_sep = {}

    def unlock_counter(self):
        """
            allow retrieving counter
        """
        self._unit_got = True

    def is_locked(self):
        """
            see whether extractor is locked
        """
        return self._unit_got == False

class value_extractor(object):
    """
        the form of similar_attrs should be: {"attr1": ["attrx", "attry"...], "ar1": ["arx", "ary"]}, where ..x, ..y is the similar attribute of "ar1" or "attr1"
        the form of similar_corpus should be: {"attr1": [v1, v2, v3 ... ], "attr2": [a1, a2, a3...s]} where v1, v2, v3... are values that are similar
        or similar_corpus is attrd
        *** similar_attrs is used exclusively for method <self.get_similar_attr>
    """

    def __init__(self, similar_attrs: dict, similar_corpus: dict):
        """
            the form of similar_attrs should be: {"attr1": ["attrx", "attry"...], "ar1": ["arx", "ary"]}, where ..x, ..y is the similar attribute of "ar1" or "attr1"
            the form of similar_corpus should be: {"attr1": [v1, v2, v3 ... ], "attr2": [a1, a2, a3...s]} where v1, v2, v3... are values that are similar
            or similar_corpus is attrd
            *** similar_attrs is used exclusively for method <self.get_similar_attr>
        """
        self._similar_corpus = similar_attrs
        self.s_dict = similar_corpus
        self._value_counter = {}
        self._val_c_got = False
        print('value extractor initiated!')

    def get_similar_attr(self, attrname) -> list:
        """
            get similar attr from the attribute similar corpus
            returns [attrname, attr1, attr2 ...]
        """
        Similar_corpus = self._similar_corpus
        for key in Similar_corpus:
            if key == attrname:
                return [key] + Similar_corpus[key]
        return []

    def get_all_values(self, attrnames) -> dict:
        """
            从语料库中得到attrnames中attrname的所有值和相似属性的值
            attrnames: 
                ["attr1", "attr2" .... ]
            returns 
                    {"attr1": [v1, v2, v3, ...], 
                        "attr2": [s1, s2, s3, ...],
                        ...}
        """
        n_dict = defaultdict(list)
        s_dict = self.s_dict
        for attr in attrnames:
            all_names = self.get_similar_attr(attr)
            for name in all_names:
                n_dict[attr].extend(s_dict[name])
        return n_dict

    def get_attr_value(self, attr_name) -> set:
        """
            返回属性名所有数值集合
        """
        value_dict = self.get_all_values([attr_name])
        value_lst = value_dict[attr_name]
        # Can use Anomaly detect here for value_lst
        units = self._extract_value(value_lst, attr_name)
        return units

    def get_value_dict(self, attr_names: list = [], outfile='') -> dict:
        """
            根据 attrname in attr_names得到对应的单位dict
        """
        if not attr_names:
            attr_names = list(self._similar_corpus.keys())

        r_dict = {}
        for attr in attr_names:
            r_dict[attr] = self.get_attr_value(attr)

        # self.delete_repeat(r_dict)
        self.unlock_counter() # unlock this

        if outfile:
            similard2excel(r_dict, outfile)

        return r_dict

    def delete_repeat(self, out_dict):
        """
            为out_dict去重
        """
        for key in out_dict:
            out_dict[key] = list(set([x for x in out_dict[key] if x]))
        # print(out_dict)

    def _extract_value(self, lst: list, attr_name: str) -> set:
        """
            从给定str的list中提取出数值集合
            attr_name: attr_name
        """
        r = []
        for value in lst:
            r.append(self._filter_value(value, attr_name))
        return r

    def _filter_value(self, w, attrname, change_counter=True) -> str:
        """
            从w中提取出数值, 非raw
            w: 
                句子
            change_counter:
                是否影响counter
            attrname: 
                用于sep_counter来对数值出现次数进行计数
        """
        cur_val = process_value(w)

        puncs = ['~', '"', '-', '/', '.', '*', '}', ',', ';', "'", \
                 '#', '^', '+', ':', '{', '>', '`', '(', '$', '=', \
                '@', '&', '!', '%', ')', '?', '|', '_', '\\', '<']

        for punc in puncs:
            if punc in cur_val:
                cur_val = cur_val.split(punc)
                try:
                    if len(cur_val) == 3 and cur_val[1] == '10':
                        index = _convert_index(cur_val[2])
                        cur_val = float(cur_val[0]) * (10 ** (-int(index)))
                        cur_val = str(cur_val)
                    else:
                        cur_val = str(max(map(float, cur_val)))
                except ValueError: # for some wierd w: 47/44/41;58
                    # print('value error at: ...')
                    # print(w)
                    # print('\n')
                    cur_val = ''

                break

        if change_counter and attrname:
            self.value_change(cur_val, attrname)

        return cur_val

    def parse_raw_value(self, w: str) -> str:
        """
            从w中提取出raw数值
            w: 
                句子
        """
        return process_value(w)

    def parse_value(self, w: str) -> str:
        """
            从w中提取出数值
            w: 
                句子
        """

        return self._filter_value(w, '', False)

    def save(self, out_directory='') -> dict:
        """
            得到值计数表
        """

        if not self._val_c_got:
            raise NoValueCounterError('no value counter, you may run self.get_value_dict() or\
                                        self.unlock_counter() \
                                        to have a counter')
        counter_sep = self._value_counter.copy()

        for key in counter_sep:
            counter_sep[key] = dict(counter_sep[key])
            for ele in counter_sep[key]:
                counter_sep[key][ele] = [counter_sep[key][ele]]

        if out_directory:
            for attr in counter_sep:
                d = counter_sep[attr]
                similard2excel(d, os.path.join(out_directory, 'counter_{}.xlsx'.format(trim_attr(attr))))
        return counter_sep

    def value_change(self, r, attr_name):
        sep_counter = self._value_counter
        if r:
            if attr_name in sep_counter:
                r = float(r)
                sep_counter[attr_name][r] += 1
            else:
                sep_counter[attr_name] = defaultdict(lambda: 0)

    def clear(self):
        """
            清除计数
        """
        self._val_c_got = False
        self._value_counter = {}

    def unlock_counter(self):
        """
            允许调用self.save()
        """
        self._val_c_got = True

class advanced_value_extractor(value_extractor):
    """
        仅改变_filter_values，详见value_extractor
        附加analyzer
        多加了单位转换功能，同时不加载Unit_Analyzer时
        可以像value_extractor一样不进行单位
        转换，
        Unit_Analyzer: 
            用于分析的Unit_Analyzer
        align_dict: 
            用于单位对齐的字段
        main_unit_dict: 
            各属性的主单位dict
        形式: 
            {"attr1": 'unit1', "attr2": 'unit2', ....}
    """
    def __init__(self, similar_attrs: dict, similar_corpus: dict, Unit_Analyzer = None, align_dict={}, \
                main_unit_dict: dict = {}):
        """
            Unit_Analyzer: 用于分析的Unit_Analyzer
            align_dict: 用于单位对齐的字段
            main_unit_dict: 各属性的主单位dict
            形式: {"attr1": 'unit1', "attr2": 'unit2', ....}
            precondition:
                set of keys of align_dict should be the same as 
                set of values of main_unit_dicts
        """
        super().__init__(similar_attrs, similar_corpus)

        if not set(main_unit_dict.values()).issubset(set(align_dict.keys())):
            raise MainUnitMatchError("the values of the main_unit_dict should be contained in \n \
                                      the keys of align_dict, currently: {} isn't a subset of {}".\
                                                        format(set(main_unit_dict.values()), set(align_dict.keys())))
        self._Unit_Analyzer = Unit_Analyzer
        self._align_dict = align_dict
        self._main_unit_dict = main_unit_dict
        self.check_analyzer_available(True)

    def _filter_value(self, w, attrname, change_counter=True) -> str:
        """
           @override
              从w中提取出数值, 非raw
           w: 
               句子
           change_counter: 
                是否影响counter
        """
        cur_val = process_value(w)

        puncs = ['~', '"', '-', '/', '.', '*', '}', ',', ';', "'", \
                 '#', '^', '+', ':', '{', '>', '`', '(', '$', '=', \
                '@', '&', '!', '%', ')', '?', '|', '_', '\\', '<']

        for punc in puncs:
            if punc in cur_val:
                cur_val = cur_val.split(punc)
                try:
                    if len(cur_val) == 3 and cur_val[1] == '10':
                        index = _convert_index(cur_val[2])
                        cur_val = float(cur_val[0]) * (10 ** (-int(index)))
                        cur_val = cur_val
                    else:
                        cur_val = max(map(float, cur_val))
                except ValueError: # for some wierd w: 47/44/41;58
                    cur_val = ''

                break

        try:
            cur_val = float(cur_val)
        except ValueError: # it would be cur_val == '' for most of the times
            return ''
        # get the transformer of unit
        if self.check_analyzer_available(): # check whether it's available
            try:
                analyzer = self._Unit_Analyzer
                main_unit = self._main_unit_dict[attrname]
                t = analyzer.parse_sentence_align(w, attrname, main_unit, \
                                 self._align_dict, strict=False)

                if t:
                    converter = t[0]
                    cur_val = converter(cur_val)

            except KeyError: # usually ''
                pass

        cur_val = str(cur_val)

        if change_counter and attrname:
            self.value_change(cur_val, attrname)
        return cur_val

    def parse_value(self, w: str, attrname: str, change=False):
        """
            @override
            attrname: 
                the attrname to transform
            returns：
                进行单位对齐后转换的数值
            >>> ex.parse_value("1kw")
            >>> 1000
        """
        return self._filter_value(w, attrname, change)

    def load_analyzer(self, Unit_Analyzer):
        """
            载入analyzer
        """
        self._Unit_Analyzer = Unit_Analyzer

    def load_align_dict(self, align_dict: dict):
        self._align_dict = align_dict

    def clear(self):
        """
            @override
            全部清理
        """
        super().clear()
        self._Unit_Analyzer = None
        self._align_dict = {}

    def check_analyzer_available(self, warning=False) -> bool:
        """
            check whether this is available
        """
        if self._Unit_Analyzer:
            if not (self._align_dict and self._main_unit_dict):
                if warning:
                    raise ComponentLoadError("You must load align_dict and main\
                                                _unit_dict after having a Unit_Analyzer")
                else:
                    return False
            return True
        return False

class unit_value_extractor(object):
    """
        单位数值抽取机和批量单位抽取整合，未来可做扩充
        the form of similar_attrs should be: 
            {"attr1": ["attrx", "attry"...], "ar1": ["arx", "ary"]}, where ..x, ..y is the similar attribute of "ar1" or "attr1"
        the form of similar_corpus should be: 
            {"attr1": [v1, v2, v3 ... ], "attr2": [a1, a2, a3...s]} 
            where v1, v2, v3... are values that are similar
            or similar_corpus is attrd
        *** 
        similar_attrs is used exclusively for method <self.get_similar_attr>
        Unit_Analyzer is used when we need the transform of units
    """
    def __init__(self, similar_attrs: dict, similar_corpus: dict, Unit_Analyzer=None, \
                align_dict={}, main_unit_dict={}):
        """
            the form of similar_attrs should be: 
                {"attr1": ["attrx", "attry"...], "ar1": ["arx", "ary"]}, where ..x, ..y is the similar attribute of "ar1" or "attr1"
            the form of similar_corpus should be: 
                {"attr1": [v1, v2, v3 ... ], "attr2": [a1, a2, a3...s]} where v1, v2, v3... are values that are similar
                or similar_corpus is attrd
            *** similar_attrs is used exclusively for method <self.get_similar_attr>
            Unit_Analyzer is used when we need the transform of units
        """
        self.unit_extractor = unit_extractor(similar_attrs, similar_corpus)
        self.value_extractor = advanced_value_extractor(similar_attrs, similar_corpus, \
                                                        Unit_Analyzer, align_dict, main_unit_dict)
        print('extractor initiated!')

    def get_unit_dict(self, attr_names: list = [], min_ratio: float = None, outfile='') -> dict:
        """
            return a dict form of units of all attr_names
            根据attrname in attr_names得到对应的单位dict
            min_ratio: 
                最小出现比率
        """

        return self.unit_extractor.get_unit_dict(attr_names, min_ratio, outfile)

    def get_main_unit_dict(self, r_dict={}):
        return self.unit_extractor.get_main_unit_dict(r_dict)

    def get_value_dict(self, attr_names: list = []) -> dict:
        """
            根据 attrname in attr_names得到对应的单位dict
        """

        return self.value_extractor.get_value_dict(attr_names)

    def units(self, out_directory='', total_outfile='') -> dict:
        """
            得到单位计数字典
            并且可以将该字典保存
        """
        return self.unit_extractor.units(out_directory, total_outfile)

    def parse_unit(self, w: str):
        """
            从w中提取单位
        """
        return self.unit_extractor.parse_unit(w)

    def parse_value(self, w: str, attrname='', change=False):
        """
            从w中提取数值
        """
        return self.value_extractor.parse_value(w, attrname, change)

    def save(self, out_directory='') -> dict:

        return self.value_extractor.save(out_directory)

class Unit_Analyzer(object):
    """
        Analyze unit from the previous results
        @public
            self.extractor: 
                extractor
            self.unit_dict: 
                the unit dict
            self.counter_dict: 
                the counter dict
            self.anomaly_unit: 
                the unit that cannot be analyzed
        @private:
            _transform_dict: 
                the unit transformer


        sep_counter: 
                    {"attr1": {"a": [1], "b": [10]}, "attr2": {"c": [5], "b": [1]}}
        total_counter: 
                    {"a": [1], "b": [10], "c": [100]}
        attrd: 
            {"attr1": ["v1", "v2" ... ], "attr2": ["c1", "c2" ... ]}
        transform_dict: 
            for making unit conversion
        unit_dict: 
                属性与单位的对应字典

        sep_counter和total_counter可通过unit_extractor.units()获取

        通过已有的词库识别单位，并进行转换，并且可进行单位批量粗过滤
    """
    def __init__(self, sep_counter={}, total_counter={}, \
                unit_dict={}, transform_dict={}):
        """
            sep_counter: 
                {"attr1": {"a": [1], "b": [10]}, "attr2": {"c": [5], "b": [1]}}
            total_counter: 
                {"a": [1], "b": [10], "c": [100]}
            attrd: 
                the attribute dict, e.g {"attr1": ["v1", "v2" ... ], "attr2": ["c1", "c2" ... ]}
            transform_dict: 
                for making unit conversion
            unit_dict: 
                每个属性的对应多个单位字典mapped_unit_dict
                e.g:
                {"attr1": ["attr2", "attr3"], ...}

            sep_counter和total_counter可通过unit_extractor.units()获取
        """
        self.sep_counter = sep_counter
        self.total_counter = total_counter
        self.anomaly_unit = []
        self.unit_dict = unit_dict
        self._transform_dict = transform_dict

    def parse_sentence(self, sent: str, attrname=None) -> Tuple[str, str]:
        """
            该方法待扩充，根据unit_dict进行单位对齐，或者判定为异常单位
            返回其需要对齐的单位

            returns:
                aligned unit and is_anomaly
                对齐的单位以及判断是否为异常单位
        """

        r, unit_parsed = '', process_unit(sent)
        is_anomaly = False

        if not unit_parsed:
            print('unit cannot be extracted')
            print('\n')
            return ('', True)

        if attrname and attrname in self.unit_dict:
            target = self.unit_dict[attrname]
            # print(target)
            max_sim, max_importance = 0, 0 # similarity and importance
            for unit in target:
                sim = distance(unit_parsed.lower(), unit.lower())
                imp = self._importance_rank(unit, attrname)
                if sim > 0.4 and sim > max_sim:
                    r = unit
                    max_sim = sim
                    max_importance = imp
                elif sim > 0.4 and sim == max_sim:
                    if imp > max_importance:
                        r = unit
                        max_sim = sim
                        max_importance = imp

        if not r:
            print('cannot align the unit in this attr')
            self.anomaly_unit.append((attrname, unit_parsed))
            is_anomaly = True

        return r, is_anomaly

    def parse_sentence_align(self, sent: str, attr_name: str, main_unit: str, \
                             align_dict: dict, strict=False) -> Tuple['function', 'main_unit']:
        """
            parse the unit of the sentence and align it to the main_unit
            
            attr_name: 
                the attribute name
            main_unit: 
                the dest unit to align
            align_dict: 
                the ways to convert the unit
        """
        unit, anomaly = self.parse_sentence(sent, attr_name)

        if anomaly:
            return ()

        if not main_unit:
            return ()

        converter = self.align(unit, main_unit, align_dict, strict)

        return converter, main_unit

    def verify_unit_dict(self, unit_dict: dict, threshold=1) -> dict:

        """
            通过出现次数和重要性来筛选单位，验证外部的unit_dict
        """
        new_dict = {}
        #
        for key in unit_dict:
            units = unit_dict[key]
            new_dict[key] = []
            if key in self.sep_counter:
                #提取总次数
                total = sum(sum(self.sep_counter[key].values(), [])) if self.sep_counter[key] else 1
                for unit in units:
                    # print(key)
                    # print(unit)
                    # print(self.sep_counter[key])
                    #出现次数
                    times = self.sep_counter[key][unit][0] if unit in self.sep_counter[key] else 1
                    #粗筛选过程
                    if times/total > threshold/total:
                        new_dict[key].append(unit)
            else:
                #提取总次数
                total = sum(sum(self.total_counter.values(), []))
                for unit in units:
                    #出现次数
                    times = self.total_counter[unit][0]
                    #粗筛选过程
                    if times/total > threshold/total:
                        new_dict[key].append(unit)

        for attrname in new_dict:
            new_dict[attrname] = self.sort_unit_lst(attrname, new_dict[attrname])

        self.load_unit_dict(new_dict)

        return new_dict

    def align(self, unit, main_unit, align_dict: dict, strict=False) -> 'function':
        """
            return a function that converts unit to main_unit
        """

        aligner = Unit_Aligner(align_dict)
        return aligner.convert(unit, main_unit, strict)

    def sort_unit_lst(self, attrname, lst2sort):
        """
            sort the unit list according to the importance
        """
        comp = []
        for unit in lst2sort:
            importance = self._importance_rank(unit, attrname)
            comp.append((unit, importance))
        comp = sorted(comp, key= lambda x: x[1], reverse=True)

        return [x[0] for x in comp]

    def align(self, unit, main_unit, align_dict: dict, strict=False) -> 'function':
        """
            return a function that converts unit to main_unit
        """

        aligner = Unit_Aligner(align_dict)
        return aligner.convert(unit, main_unit, strict)

    def load_unit_dict(self, out_dict):
        """
            若开始没有设定unit_dict，可用来加载外部单位单位字典
            该字典装了每个属性出现的单位
        """
        self.unit_dict = out_dict
        
    def get_anomaly(self):
        print('current anomaly: ...')
        print('\n')
        print(self.anomaly_unit)
        return self.anomaly_unit
    
    def _importance_rank(self, unit, attrname):
        """
            count the importance value of the unit
        """
        tf = self.sep_counter[attrname][unit][0] if unit in self.sep_counter[attrname] else 0

        tf = tf/max(sum(self.sep_counter[attrname].values(), []))

        occ = self.total_counter[unit][0] if unit in self.total_counter else 0

        idf = np.log10(sum(sum(self.total_counter.values(), [])) / (1 + occ))/100

        return tf * idf


"""Analyzers
"""

class Value_Analyzer_Basic(object):
    """
        处理数值分桶工作，实行的是人为按照数值大小切割的简单分桶
        @param {dict} value_dict: 
            保存的数值字典
            要保证value_dict, value_count_dict中的所有value均已对齐
        _bucket_comp form: 
                    {
                            "attr1":  {
                                (bucket_id, min, max): <counter>,
                                (bucket_id, min, max): <counter>,
                                ...
                            },
                            "attr2": <same above>,
                            "attr3": <same above>,
                            ...
                        } 
                        
                        where bucket_id is represented by numbers

    """
    def __init__(self, value_dict: dict, value_count_dict: dict = {}, \
                Unit_Analyzer=None, align_dict={}, main_unit_dict={}, \
                    bucket_num=10):
        """     
            @param {dict} value_dict: 
                保存的数值字典
            要保证value_dict, value_count_dict中的所有value均已对齐

            _bucket_comp form: 
                {
                    "attr1":  {
                        (bucket_id, min, max): <counter>,
                        (bucket_id, min, max): <counter>,
                        ...
                    },
                    "attr2": <same above>,
                    "attr3": <same above>,
                    ...
                }

                where bucket_id is represented by numbers

            value_count_dict: 
                {
                    "attr1": {
                        "v1": [<counter>],
                        "v2": [<counter>],
                        "v3": [<counter>],
                        ...
                    },
                    "attr2": {
                        "v1": [<counter>],
                        "v2": [<counter>]
                    }
                }

            Unit_Analyzer: 
                Unit_Analyzer
            align_dict: 
                see advanced_value_extractor
            main_unit_dict: 
                see advanced_value_extractor
        """
        if bucket_num <= 2:
            raise BucketAssignError("bucket_num should be greater than 2")

        self._value_dict = value_dict

        self._align_dict = align_dict

        self._main_unit_dict = main_unit_dict

        self._bucket_num = bucket_num

        self._Unit_Analyzer = Unit_Analyzer

        self._bucket_comp = {}

        self._value_count_dict = value_count_dict

        self._loaded = False

        self._initialized = False

        self._bucket_num_dict = {}

        self._anomaly_detect()

        self._initialize()

    def get_bucket_id(self, num: float, attr_name: str, change=False) -> int:
        """
            将数值归类到桶内
            attrname: 
                要归类的属性名
            change: 
                whether to change the counter
        """
        if not self._loaded:
            raise AnalyzerLockedError("you may run unlock() or load_value() to unlock this")

        target = self._bucket_comp[attr_name]

        return self._get_id(num, target)

    def parse_sentence_align(self, w: str, attrname: str, change=False):
        """
            将单位中的数值抽出并进行分桶
            w: 
                sentece
            attrname: 
                属性名
            change: 
                是否改变counter值

            返回数值桶的id和对齐的主单位
        """

        if not self._initialized:
            raise ComponentLoadError("no value_count_dict or even value_dict in this analyzer \n \
                                        or you have cleared all")

        cur_val = process_value(w)

        puncs = ['~', '"', '-', '/', '.', '*', '}', ',', ';', "'", \
                 '#', '^', '+', ':', '{', '>', '`', '(', '$', '=', \
                '@', '&', '!', '%', ')', '?', '|', '_', '\\', '<']

        for punc in puncs:
            if punc in cur_val:
                cur_val = cur_val.split(punc)
                try:
                    if len(cur_val) == 3 and cur_val[1] == '10':
                        index = _convert_index(cur_val[2])
                        cur_val = float(cur_val[0]) * (10 ** (-int(index)))
                        cur_val = cur_val
                    else:
                        cur_val = max(map(float, cur_val))
                except ValueError: # for some wierd w: 47/44/41;58
                    cur_val = ''

                break

        try:
            cur_val = float(cur_val)
        except ValueError: # it would be cur_val == '' for most of the times
            return None, main_unit
        # get the transformer of unit
        if self.check_analyzer_available(): # check whether it's available
            try:
                analyzer = self._Unit_Analyzer
                main_unit = self._main_unit_dict[attrname]
                t = analyzer.parse_sentence_align(w, attrname, main_unit, \
                                 self._align_dict, strict=False)

                if t:
                    converter = t[0]
                    cur_val = converter(cur_val)

            except KeyError:
                pass

        # dynamic updates the bucket
        # 对桶进行动态更新

        if change and attrname:
            #得到桶id
            comp_dict = self._bucket_comp[attrname]
            print(comp_dict)
            print(self._value_count_dict[attrname])
            b_id = self._get_id(cur_val, comp_dict, attrname, True)
            print('counter added for...{}----{}'.format(b_id, attrname))
            print('\n')
            # after some conditions
            print(comp_dict)
            print(self._value_count_dict[attrname])
            cur_average = sum(comp_dict.values()) / len(comp_dict.values())
            print('limit:...', cur_average*1.7)
            print('cur_num_for_bucket--{}'.format(b_id), comp_dict[self._get_bucket_key(b_id, comp_dict)])
            #若桶数量失衡，则重新分配桶
            if cur_average * 1.7 < comp_dict[self._get_bucket_key(b_id, comp_dict)]:
                print('chaning bucket.....')
                print('\n')
                
                if attrname in self._bucket_num_dict:
                    self._bucket_num_dict[attrname] += 1
                else:
                    self._bucket_num_dict[attrname] = self._bucket_num + 1
                self._initialize(attrname)


        return self.get_bucket_id(cur_val, attrname), main_unit

    def check_analyzer_available(self, warning=False) -> bool:
        """check whether this is available
        """
        if self._Unit_Analyzer:
            if not (self._align_dict and self._main_unit_dict):
                if warning:
                    raise ComponentLoadError("You must load align_dict and main\
                                                _unit_dict after having a Unit_Analyzer")
                else:
                    return False
            return True
        return False

    def counter_change(self, attr_name: str, s_id: int, add_value=1):
        """
            change the counter of the attr
            通过指定特定id来改变attr_name下特定桶的计数
            s_id: 
                该属性下指定的id号
            add_value: 
                the value to add
        """
        t = self._bucket_comp[attr_name]

        t[self._get_bucket_key(s_id, t)] += add_value


        
    def clear(self):
        """
            清理并锁住Analyzer
        """
        self._value_dict = {}
        self._value_count_dict = {}
        self._bucket_comp = {}
        self._loaded = False
        self._initialized = False

    def clear_counter(self):
        """
            清理之前非基本，除了value_dict以外积累的计数
        """
        self._bucket_comp = {}
        self._loaded = False
        self._initialize()

    def get_bucket_dict(self):
        """
            return self._bucket_comp
        """
        return self._bucket_comp


    def load(value_dict: dict, value_count_dict: dict, \
                Unit_Analyzer=None, align_dict={}, main_unit_dict={}):
        """
            load the necessary data for the Value Analyzer
        """
        self._value_dict = value_dict
        self._initialize()
        self._Unit_Analyzer = Unit_Analyzer
        self._align_dict = align_dict
        self._main_unit_dict = main_unit_dict

    def save(self, directory_path='', outpath=''):
        """
            保存分桶结果的字典
            directory_path: self._bucket_comp保存路径
            outpath: self._value_count_dict保存路径
            返回保存的字典形式及值计数字典
        """
        if not self._loaded:
            raise AnalyzerLockedError("you may run unlock() or load_value() to unlock this")

        save_dict = self._to_save_dict(self._bucket_comp)

        if directory_path:
            for attr in save_dict:
                attrd = save_dict[attr]
                similard2excel(attrd, os.path.join(directory_path, "counter_{}.xlsx".format(trim_attr(attr))))

        if outpath:
            similard2excel(self._value_count_dict, outpath)

        return save_dict, self._value_count_dict

    def unlock(self):
        """
            将save等功能解锁
        """
        self._loaded = True

    def check_attrname(self, attrname, \
                        warning_words="attrname not available for this analyzer", \
                        warn=False):
        """
            To check whether this attrname is available for this analyzer
            returns bool
        """

        if not self._initialized:
            return True

        if attrname not in self._value_count_dict or attrname not in self._bucket_comp:
            if warn:
                raise BucketAssignError(warning_words)
            else:
                return False

        return True

    def _initialize(self, bufferd=''):
        """
            初始化，生成各个属性的bucket
        """
        value_dict, bucket_dict = self._value_dict, self._bucket_comp

        # create buckets
        for attr in value_dict:
            lst, counter = value_dict[attr], 2
            #
            max_val, min_val = max(lst), min(lst)

            number_of_buckets = self._bucket_num if attr not in self._bucket_num_dict else self._bucket_num_dict[attr]
            #简单进行分桶
            bucket_dict[attr], sep_bucket = {}, np.linspace(min_val, max_val, number_of_buckets - 1)
            #
            target = bucket_dict[attr]
            #
            key_first, key_last = (1, -float('inf'), min_val), (number_of_buckets, max_val, float('inf'))
            # create
            target[key_first], target[key_last] = 0, 0
            #初始化桶
            for index in range(1, len(sep_bucket)):
                key_created = (counter, sep_bucket[index - 1], sep_bucket[index])
                target[key_created] = 0
                counter += 1

        self.unlock()

        if self._value_count_dict:
            self._add_counter(self._value_count_dict)

        self._initialized = True



    def _anomaly_detect(self):
        """排除各属性异常值
        """
        value_dict, value_count_dict = self._value_dict, self._value_count_dict
        for attr in value_count_dict:
            eles, count_items = [], value_count_dict[attr]
            for value, count in count_items.items():
                counter = count[0]
                eles += [value] * counter
            eles = list(sorted(eles))
            anomaly_values = Anomaly_Detect(eles)
            # print('anomaly_values:...')
            # print(attr)
            # print('\n')
            # print(anomaly_values)
            for item in list(set(anomaly_values)):
                del count_items[item]

            if value_dict:
                value_dict[attr] = list(filter(lambda a: a not in set(anomaly_values), value_dict[attr]))

    def _to_save_dict(self, d: dict) -> dict:
        """
            convert bucket_form to save_form
        """
        return _to_save_dict(d)

    def _to_load_dict(self, d: dict) -> dict:
        """
            convert save_form to bucket_comp form
        """
        return _to_bucket_dict(d)

    def _get_id(self, num, comp_dict, attr_name='' , change=False) -> int:
        """
            get the id according to num from comp_dict
            format of comp_dict: 
                {(x, x, x): <counter>, ...}
            attr_name:
                attrname
            change: 
                whether to change the count of analyzer
        """
        keys = comp_dict.keys()
        for key in keys:
            b_id, min_v, max_v = key
            if min_v <= num and num < max_v:
                if change and attr_name:
                    self.counter_change(attr_name, b_id)
                    value_count_dict = self._value_count_dict[attr_name]

                    if num in value_count_dict:
                        value_count_dict[num][0] += 1
                    else:
                        value_count_dict[num] = [1]

                return b_id
        # print('did not find: ...')
        # print(num)
        # print(comp_dict)
        return -1

    def _add_counter(self, count_dict):
        for attr in count_dict:
            target = count_dict[attr]
            for num, count in target.items():
                s_id = self.get_bucket_id(num, attr)
                self.counter_change(attr, s_id, count[0])

    def _get_bucket_key(self, s_id: int, d: dict):
        for key in d.keys():
            if s_id == key[0]:
                return key
"""value_count_dict无法同bucket_dict一同更新计数
"""
class Value_Analyzer(Value_Analyzer_Basic):

    """
        需重写_initialize, __init__, 和load
        通过数值密度算法进行分桶，如果不传入value_count_dict则作为普通Basic来使用
        @param {dict} attr_buckets: 
            defines the bucket_num for certain attr \n
        format: 
            {"attr1": <bucket_num>, "attr2": <bucket_num>}
        @param {dict} configuration:
             ways to assign the bucket
        @param {dict} value_count_dict: 
            {
                "attr1": {
                    "v1": [<counter>]
                    "v2": [<counter>]
                }
            }
    """
    def __init__(self, value_count_dict: dict = {}, \
                Unit_Analyzer=None, align_dict={}, main_unit_dict={}, \
                    attr_buckets={}, bucket_num=10, value_dict: dict = {}):
        """
            attr_buckets: 
                defines the bucket_num for certain attrs
                if not defined, then no use, its hasn't been implemented yet
            format: 
                {"attr1": <bucket_num>, "attr2": <bucket_num>}
        """
        self._initialized = False
        super().__init__(value_dict, value_count_dict,\
                         Unit_Analyzer, align_dict, \
                         main_unit_dict, bucket_num)
        self._attr_buckets = attr_buckets


    def determine_extra_bucket_id(self, attrname, bucket_num) -> list:
        """
            在分桶状态下，决定哪些桶会多分到一个值
        """
        target_counter = self._value_count_dict[attrname]

        counter_total = sum([x[0] for x in target_counter.values()])
        # number of eles in each bucket
        bucket_c = counter_total // bucket_num

        samples_needed = counter_total - (bucket_c) * bucket_num

        return random_sampling(bucket_num + 1, samples_needed, 2).tolist()

    def assign_bucket(self, attrname: str, bucket_num: int, extra_samples=[]) -> bool:
        """
            assign the values into different buckets,
            returns whether the split is successful

            extra_examples: 特殊的桶id，即多分一个的桶id
            
            该方法为按照密度分层
        """

        self.check_attrname(attrname, "no attrname in the value_counter", True)

        value_counter_dict = self._value_count_dict[attrname]
        self._bucket_comp[attrname] = {}

        bucket_comp, extra_samples = self._bucket_comp[attrname], set(extra_samples)
        total = sum([x[0] for x in value_counter_dict.values()])
        values, split_dist = [], total//bucket_num
        #建立values
        for key in value_counter_dict:
            counter = value_counter_dict[key][0]
            values.extend([key] * counter)
        values = list(sorted(values))

        if not values:
            return False

        if bucket_num == 1:

            self._bucket_comp[attrname][(1, -float('-inf'), values[0])] = 0
            self._bucket_comp[attrname][(2, values[0], values[-1])] = 0
            self._bucket_comp[attrname][(3, values[0], float('inf'))] = 0

            return True

        gen = self._create_bucket_keys(values, split_dist, extra_samples)

        for key in gen:
            if key[1] == key[2]:
                self._bucket_comp[attrname] = {}
                return False
            self._bucket_comp[attrname][key] = 0
        print(self._bucket_comp[attrname])
        return True

    def load(value_count_dict: dict, \
                Unit_Analyzer=None, align_dict={}, main_unit_dict={}, value_dict: dict = {}):
        """
            load the necessary data for the Value Analyzer
        """
        self._value_count_dict = value_count_dict
        self._initialize()
        self._value_dict = value_dict
        self._Unit_Analyzer = Unit_Analyzer
        self._align_dict = align_dict
        self._main_unit_dict = main_unit_dict

    def _initialize(self, specific_bucket=''):
        """
            初始化，生成各个属性的bucket
        """
        value_dict, bucket_dict = self._value_dict, self._bucket_comp
        value_count_dict = self._value_count_dict
        # create buckets
        # requires anomaly detection/outlier analysis
        # requires bucket assignment

        # 如果没有value_count_dict，则进行普通按数值分层

        if not self._value_count_dict:
            
            if not value_dict:
                return

            for attr in value_dict:
                lst, counter = value_dict[attr], 2
                #
                max_val, min_val = max(lst), min(lst)
                number_of_buckets = self._bucket_num if attr not in self._bucket_num_dict else self._bucket_num_dict[attr]
                #简单进行分桶
                bucket_dict[attr], sep_bucket = {}, np.linspace(min_val, max_val, number_of_buckets - 1)
                #
                target = bucket_dict[attr]
                #
                key_first, key_last = (1, min_val, min_val), (number_of_buckets, max_val, max_val)
                # create
                target[key_first], target[key_last] = 0, 0
                #初始化桶
                for index in range(1, len(sep_bucket)):
                    key_created = (counter, sep_bucket[index - 1], sep_bucket[index])
                    target[key_created] = 0
                    counter += 1
        else:
            if not specific_bucket:
                for attr in self._value_count_dict:
                    bucket_num = self._bucket_num if attr not in self._bucket_num_dict else self._bucket_num_dict[attr]
                    self._split_bucket(attr, bucket_num)
            else:
                bucket_num = self._bucket_num if specific_bucket not in self._bucket_num_dict else self._bucket_num_dict[specific_bucket]
                self._split_bucket(specific_bucket, bucket_num)

        self.unlock()

        if self._value_count_dict:
            if not specific_bucket:
                self._add_counter(self._value_count_dict)
            else:
                self._add_single_counter(self._value_count_dict, specific_bucket)

        self._initialized = True

    def _add_single_counter(self, count_dict, specific_bucket=''):
        print('single added!')
        target = count_dict[specific_bucket]
        for num, count in target.items():
            s_id = self.get_bucket_id(num, specific_bucket)
            self.counter_change(specific_bucket, s_id, count[0])

    def _create_bucket_keys(self, values: list, split_dist: int, special_id: set):
        """
            返回bucket_keys的集合, (id, min, max)
            split_dist: 
                当前分割的长度
            special_id: 
                特殊的id，多加一个长度
            values: 
                所有的值
        """
        keys, counter = [], 2
        prev, cur = 0, split_dist

        total = len(values)

        yield (1, -float('inf'), values[prev])

        while cur < total:

            round_add = 0

            if counter in special_id:
                round_add = 1

            if cur + (split_dist + round_add) >= total:
                yield (counter, values[prev], values[-1])
            else:
                yield (counter, values[prev], values[cur])

            counter += 1
            prev = cur
            cur += (split_dist + round_add)

        yield (counter, values[-1], float('inf'))

        print("values match counter:...")
        print("\n")
        print(len(values))



    def _split_bucket(self, attrname, bucket_num, counta=0):
        """
            split the buckets into <bucket_num> bucket
            counta用于进行问题追踪
            将值全部分桶
        """

        self.check_attrname(attrname, "No buckets for this attribute: {}".\
                                    format(attrname), True)

        total_counter = self._value_count_dict[attrname]

        t = sum([x[0] for x in total_counter.values()])

        if t < bucket_num or bucket_num < 1:
            if t < bucket_num:
                warnings.warn("Too many buckets {} for total sum: {} in attrname: {}".format(bucket_num, t, attrname))
                self._split_bucket(attrname, bucket_num - 1, counta)
                return 
            if bucket_num < 1:
                raise BucketAssignError("invalid bucket nums: {} for total sum: {} in {}".format(bucket_num, t, attrname))

        extra_samples = self.determine_extra_bucket_id(attrname, bucket_num)

        brutal_try = self.assign_bucket(attrname, bucket_num, extra_samples)
        #如果不行则再进行一次分桶
        if not brutal_try:
            try:
                self._split_bucket(attrname, bucket_num-1, counta+1)
            except BucketAssignError:
                if counta == 0:
                    raise BucketAssignError("Sorry, cannot assign bucket nums {}, \n \
                                            for this attribute: {}".format(bucket_num, attrname))
                return
