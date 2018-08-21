"""
    全流程包装，无人为操控环节，自动根据已有文档生成分桶器
    所需文件： deal_product_all, 相似属性字典, 主单位字典，转换字典
    save即可保存分桶结果
    暴露parse_sentence_align接口
    得到unit_analyzer和value_analyzer
"""
from .align_tools import get_attribute
from .extract import unit_extractor, unit_value_extractor,\
                     Unit_Analyzer, Value_Analyzer, \
                     generate_mapped_dict, generate_sep_mapped_dict, \
                     generate_sep_mapped_dict_advanced, advanced_value_extractor

class pipeline_pro(object):
    """
        attributes {list}: 待处理的数值属性名
        all_product {str}: 属性语料文件
        similard_attr {str}: 相似属性文件
        main_unit {dict}: 各属性主单位字典
        align_dict {dict}: 单位对齐字典
    """

    def __init__(self, attributes, all_product, similard_attr, main_unit_dict, align_dict):
        self._unit_ex = None
        self._value_ex = None
        self.unit_analyzer = None
        self.value_analyzer = None

        self._run(attributes, all_product, similard_attr, main_unit_dict, align_dict)

    def _run(self, attributes, all_product, similard_attr, main_unit_dict, align_dict):

        attrd = get_attribute(all_product)
        attrs = generate_mapped_dict(similard_attr)
        self._unit_ex = unit_extractor(attrs, attrd)
        
        unit_dict = self._unit_ex.get_unit_dict(attributes)
        total_unit_counter, sep_counter = self._unit_ex.units()
        self.unit_analyzer = Unit_Analyzer(sep_counter, total_unit_counter, unit_dict, align_dict)

        self._value_ex = advanced_value_extractor(attrs, attrd, self.unit_analyzer, align_dict, main_unit_dict)

        self._value_ex.get_value_dict(attributes)

        value_count_dict = self._value_ex.save()

        self.value_analyzer = Value_Analyzer(value_count_dict, self.unit_analyzer, align_dict, main_unit_dict)

    def save(self, out_dir='', outpath=''):
        """将现有分桶结果保存在特定路径
        """
        return self.value_analyzer.save(out_dir, outpath)

    def parse_sentence_align(self, w: str, attrname: str, change=False):
        """单位数值属性对齐分桶
        """
        return self.value_analyzer.parse_sentence_align(w, attrname, change)

    def parse_unit(self, w):
        return self.unit_analyzer.parse_unit(w)

    def get_unit_extractor(self):
        return self._unit_ex