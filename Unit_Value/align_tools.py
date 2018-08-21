"""做单位对齐等工具
"""
import numpy as np
from .IO import load_data_from_excel_with_sheet,build_excel,dataLst2txt,readtxt2Lst,txt2excel
from .Feature_align import similard2l

class MainUnitMatchError(Exception):
    pass

class UnitNotMatchError(Exception):
    pass

class Unit_Aligner(object):
    """
        进行单位对齐，可直接在align_dict中写入
        对齐数字，最后返回的函数以乘以该数字进行单位转换
    """
    def __init__(self, align_dict):
        """
            the format of align_dict should be:
                {
                 "main_unit1": {"unit1": <ways to convert>, 
                                ...
                               }, 
                 "main_unit2": {},
                  ...
                }
        """

        self._align_dict = align_dict

    def convert(self, unit, main_unit, strict=False) -> 'function':
        """
            unit: 
                unit to align
            main_unit: 
                the unit of the main
            strict: 
                whether to match strictly
        """
        if main_unit not in self._align_dict:
            raise MainUnitMatchError('main_unit not included in self._align_dict')

        tar_unit = _is_unit_in(unit, self._align_dict[main_unit].keys())
        
        if tar_unit:
            # target should be numbers or functions
            target = self._align_dict[main_unit][tar_unit]
            if isinstance(target, int) or isinstance(target, float):
                return lambda x: x * target
            return target

        if strict:
            raise UnitNotMatchError('no unit is included in self._align_dict')

        return self._direct_convert()


    def _direct_convert(self):
        """
            直接值相等转换
        """
        return lambda x: x

def random_sampling(total_nums: int, samples_needed: int, start_num=1):
    """
        generate <sample_needed> number of 
        non-repeated integers from start_num to total_nums
        returns 
            an array
            random_sampling(10, 5)
        >>> [7 2 8 1 6]
            random_sampling(10, 5)
        >>> [7 4 9 5 10]
    """
    target = np.arange(start_num, total_nums + 1)
    np.random.shuffle(target)

    return target[:samples_needed]


def _is_unit_in(unit, lst):
    """
        check whether the unit is in the lst
    """
    unit = unit.lower()

    for target_unit in lst:
        if unit == target_unit.lower():
            return target_unit
    return ''

def get_attribute(corpus_file, attrd_out=None):
    data_l = load_data_from_excel_with_sheet(corpus_file,sheet_index=[0])[0]
    attr_l = []
    attr_d = {}
    for i,l in enumerate(data_l):
        #print(l)
        attr_l.extend([t.split(':')[0] for t in l[2:] if t])
        for d in l[2:]:
            if len(d.split(':')) == 1:
                continue
            if d.split(':')[0] in attr_d:
                attr_d[d.split(':')[0]].extend([d.split(':')[1].strip()])
            else:
                attr_d[d.split(':')[0]] = list([d.split(':')[1].strip()])
    if attrd_out:
        attrd_l = similard2l(attr_d)
        build_excel(attrd_out, attrd_l)
    return attr_d

def trim_attr(attr: str):
    """让属性名符合路径
    """
    return attr.replace('(', '').\
                replace(')', '').\
                replace('/', '-').\
                replace('\\', '')
