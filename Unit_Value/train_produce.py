"""制作训练模型和读取训练模型
"""

from typing import List, Tuple
import jieba.posseg as pseg
import pandas as pd
from .IO import txt2excel

def produce_train_data(s_dict: dict, attribute_names: List[str], \
                    label_tag: str = " ", outfile='./kongtiao/train_data.txt', mode = 'a') -> None:
    """@param {dict} s_dict: the dict of all attributes
       @param {List[str]} attribute_names: the attribute names of data to train
       @param {str} label_tag: the DIY tag to add for the data
       @prarm {str} outfile: the path to output the txt train data
       @param {str} mode: the file mode to produce the output

       参数 {dict} s_dict: 所有属性值的对应字典
       参数 {List[str]} attribute_names: 产生训练数据的属性名
       参数 {str} label_tag: 自动标的值
       参数 {str} outfile: 产出的txt训练文件
       参数 {str} mode: 对训练文件的读写模式
    """
    with open(outfile, mode, encoding='utf-8') as f:
        for attr_name in attribute_names:
            # all values of the key
            all_values = s_dict[attr_name]

            for value in all_values:
                words = pseg.cut(value)
                for word in words:
                    f.write(word.word + '\t' + word.flag + '\t' + label_tag + '\n')
                f.write('\n')

def read_train_data_txt(file_name: str) -> Tuple['train_data', 'label']:
    """读取txt训练文件, 最终输出格式是train_data: [[(a1, b1), (a2, b2)...], 
                                            [(a3, b3), (a4, b4)...], 
                                            ...]
                                label: [l1, l2, l3 ...]
    """
    train_data, label = [], []
    with open(file_name, 'r') as f:
        current_train, current_label = [], []
        for line in f:
            if line != '\n':
                eles = line.strip('\n').split('\t')
                current_train.append((eles[0], eles[1]))
                current_label.append(eles[2])
            else:
                train_data.append(current_train)
                label.append(current_label)
                current_train, current_label = [], []
    return train_data, label

def read_train_data_excel(file_name: str) -> Tuple['train_data', 'label']:
    """读取excel形式的训练文件
        最终输出格式: train_data: [[(a1, b1), (a2, b2)...], 
                                            [(a3, b3), (a4, b4)...], 
                                            ...]
                    label: [l1, l2, l3 ...]
    """

    train_data, label = [], []
    df = pd.read_excel(file_name, header=None).fillna('')
    current_train, current_label = [], []
    # display(df)
    for i, row in df.iterrows():
        if row[0] or row[1]:
            current_train.append((row[0], row[1]))
            current_label.append(row[2])
        else:
            train_data.append(current_train)
            label.append(current_label)
            current_train, current_label = [], []
    return train_data, label

if __name__ == '__main__':
    # read_train_data_txt('./kongtiao/train_data.txt')
    # print(read_train_data_excel('./kongtiao/train_data.xlsx')[0])
    # txt2excel('./kongtiao/train_data.txt', './kongtiao/train_data.xlsx')
    df = pd.read_excel('./attribute_dict_all_bingxiang.xlsx', header=None)
    df = df.set_index(0).fillna('')
    # display(df)
    attrd = {}
    attrd['耗电量(KWh/24h)'] = [x for x in df.loc['耗电量(KWh/24h)'].tolist() if x]
    # attrd['适用面积'] = [x for x in df.loc['适用面积'].tolist() if x]
    produce_train_data(attrd, ['耗电量(KWh/24h)'], label_tag='n', outfile='./bingxiang_train.txt')
    txt2excel('./bingxiang_train.txt', './bingxiang_train.xlsx')