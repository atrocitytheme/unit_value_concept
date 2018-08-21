"""模型方法
"""

import sklearn
import pandas as pd
import numpy as np
import sklearn_crfsuite
import jieba.posseg as pseg
import re
from scipy import stats
from sklearn.externals import joblib
from .train_produce import *

checked = {'分贝分贝', '//', 'W/4400W', 'W瓦', 'w瓦', 'GW/H150+N3', 'W(300W', 'W/W'}

Chinese_finder = re.compile(r'[\u4e00-\u9fff]')
dot_finder = re.compile(r'[\.]')
digit_finder = re.compile(r'[0-9]')
back_slash_digit_finder = re.compile(r'/\d*(?<=[^\d])')

"""Implementation of Conditional Random Field
"""

def trim_unit(r):
    r = digit_finder.sub('', r)
    r = dot_finder.sub('', r)
    return r

def generate_dict(df):
    """将all_attribute_dict转化成属性字典
    """
    o = {}
    # display(df)
    for key in df.index:
        # print(key)
        o[key] = [x for x in df.loc[key].tolist() if x]
        # print(df.loc[key])
    print('s_dict ready!')
    return o

def word2features(sent, i):
    """字转化为特征
    """
    try:
        word = sent[i][0]
        postag = sent[i][1]
    except IndexError:
        print('error occur:...')
        print(sent)
        raise Exception('format error')

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        # 'length': len(word),
        'word[:3]': word[:3],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    """句子转化为特征
    """
    return [word2features(sent, i) for i in range(len(sent))]


# class Anomaly_detect_unit(object):

#     def _word2features(self, w: str):
#         """convert w into features
#         """
#         return np.array([])

class CrfTrain(object):
    """ @attr: self._CRF
        @method:
            predict: 预测多个句子
            score: 评定分数
            predict_single: 预测单个句子 
            dump: 储存CRF模型
        @private:
            _sent2features
    """
    def __init__(self, X, Y):
        """X: train_data input format:
        [[(a1, b1), (a2, b2),...], 
        [(a3, b3), (a4, b4),...], ...]
            Y: label_data input format:
            [l1, l2, l3, l4 ...]
        """
        self._CRF = sklearn_crfsuite.CRF(
                        algorithm='lbfgs', \
                        c1=1,
                        c2=1e-3,
                        max_iterations=150,
                        all_possible_transitions=True
                            )
        X = [sent2features(x) for x in X]
        self._CRF.fit(X, Y)

    def predict(self, X):
        """
            X: 格式同 __init__相同
            returns: [[l1, l2, l3...],
                      [l4, l5, l6], 
                                  ...]
        """
        X = self._sent2features(X)
        return self._CRF.predict(X)

    def score(self, X, Y):
        """对训练完成的模型进行评分
        """
        X = self._sent2features(X)

        return self._CRF.score(X, Y)

    def predict_single(self, X):
        """预测句子结果
           X format: [(a1, b1), (a2, b2) ...]
           returns [l1, l2, l3 ...]
        """
        X = self._sent2features([X])
        result = self._CRF.predict(X)
        return result[0]

    def dump(self, file_path):
        """储存模型
        """
        joblib.dump(self._CRF, file_path)

    def _sent2features(self, X):
        return [sent2features(x) for x in X]

class CrfModel(CrfTrain):
    """直接套用模型，非训练版本
        inherited: CrfTrain
    """
    def __init__(self, model_path):
        self._CRF = joblib.load(model_path)

class UnitGetter_Train(CrfTrain):
    """单位抽取
        @attr: 
            Parent[CrfTrain]
        @method: 
            Parent[CrfTrain]
            gather_unit: 抽取句子
        @private:
            _fit_sentence: 将句子转化成可适配CrfTrain的 X 格式
    """
    def __init__(self, X, Y):
        CrfTrain.__init__(self, X, Y)

    def _fit_sentence(self, w: str) -> list:
        """w: the sentence
            returns the form: [(word1, pos_tag), (word2, pos_tag)...]
        """
        target, r = pseg.cut(w), []

        for word in target:
            r.append((word.word, word.flag))
        return r

    def gather_unit(self, w: str, target_tag='y', limit=4, start_pos=0) -> str:
        """returns the unit of the word consisting of target_tag
        start_pos represents the position of start point
        limit is the length of the unit
        reject_tag is the opposite tag
        """
        r, is_unit = '', False

        sent = self._fit_sentence(w[start_pos:])
        # print(sent)
        tag_results = self.predict_single(sent)

        # print(sent)

        words = [x[0] for x in sent] if sent else []

        # print(words)

        counter = 0

        # 拼接

        print(list(zip(words, tag_results)))

        for word, tag in zip(words, tag_results):
            #过大就舍弃该单位后续
            if counter >= limit and \
                not bool(digit_finder.search(r)):

                return r

            elif counter >= limit + 3:

                # r = trim_unit(r)
                if r in checked:
                    print('found!!!!!')
                    print(w)
                print(tag_results)
                print(r)

                return r

            # print(word)

            if tag == target_tag and is_unit == False:
                is_unit = True
                r += word
                counter += 1
            elif tag == target_tag and is_unit == True:
                r += word
                counter += 1
            elif tag != target_tag and is_unit == True:
                # r = trim_unit(r)
                if r in checked:
                    print('found!!!!!')
                    print(w)
                print(tag_results)
                print(r)

                return r

        # r = trim_unit(r)

        if r in checked:
            print('found!!!!!')
            print(w)
        print(tag_results)
        print(r)
        
        return r

class UnitGetter_Model(UnitGetter_Train):
    """直接加载模型，不进行训练
    """
    def __init__(self, model_path):
        self._CRF = joblib.load(model_path)

if __name__ == '__main__':
    """对模型进行训练
    """
    # X, Y = read_train_dasta_excel('./assets/train_data.xlsx')
    # getter = UnitGetter_Train(X, Y)
    # print('doing')
    # getter.gather_unit('660kwg/24h')
    # getter.dump('./kongtiao/models/test4.pkl') # save the model

    model = UnitGetter_Model('./kongtiao/models/test4.pkl')
    print(model.gather_unit('0.49Kwh/24h'))
