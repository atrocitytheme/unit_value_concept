"""way to extract unit and values
    直接对process_unit或process_value进行修改即可更改提取方法
"""
import re
from collections import defaultdict
from .judge_model import UnitGetter_Model
from .config import *

Chinese_finder = re.compile(r'[\u4e00-\u9fff]')
spec_notation_finder = re.compile(r'[,!?\-×x\\;@+*]')
spec_notation_finder_unit = re.compile(r'[,!?\-×x\\;@+*/<\.]')
bracket_killer = re.compile(r'\(.*\)')
digit_finder = re.compile(r'[a-zA-Z]')

digit_finder = re.compile(r'[0-9]+')
# checked_data = set(config_anomaly_values) # 正则方法查询异常值来源

anomaly_vals = defaultdict(list) # anomaly value dict
anomaly_units = defaultdict(list) # anomaly unit dict

def process_unit(w):
    """对单位的提取方法，可选择judge_model内的模型，也可选择文件内的regexp方法
        w: 待抽取的句子
        returns 单位
    """
    # 去括号测试
    # w = bracket_killer.sub('', w)
    model = UnitGetter_Model(trained_model_path)

    return model.gather_unit(w)

def process_value(w):
    """对单位的提取方法，可选择模型，也可选择文件内的regexp方法
    w: 待抽取句子
    returns 数值
    """
    return reg_exp_value(w)

"""regular expression methods
"""
def reg_exp_unit(w):

    """正则方法抽取单位
    """

    original, w = w, bracket_killer.sub(' ', w)
    w = digit_finder.sub(' ', w)
    w = spec_notation_finder_unit.sub(' ', w).strip().split(' ')
    r = w[-1] if w else ''
    print('final value: ... ')
    print(r)
    return r


def reg_exp_value(w):

    """正则方法抽取值
    """

    r, original = '', w
    w = Chinese_finder.sub(' ', w)
    w = spec_notation_finder.sub(' ', w)
    w = bracket_killer.sub(' ', w)
    print('w:...')
    print(original)

    new_lst = [x for x in w.split(' ') if x]
    print(new_lst)
    if len(new_lst) != 1:
        n = []
        for item in new_lst:
            word = digit_finder.findall(item)
            if word:
                n.append('.'.join(word))
        if len(n) != 2:
            r = '~'.join(n)
        else:
            # try:
            #     r = str((int(n[0]) + int(n[1]))/2)
            # except:
            if len(new_lst) < 4:
                r = '-'.join(n)
            else:
                r = '/'.join(n)
    else:
        # print(new_lst[0])
        if '.' not in new_lst[0]:
            r = digit_finder.findall(new_lst[0])
            r = r[0] if r else ''
        else:
            r = '.'.join(digit_finder.findall(new_lst[0]))
    print(r)
    return r

"""model methods
"""

#get the anomaly values

class Anomaly_Selection_Error(Exception):
    pass