import os
import re
import thulac
import jieba
import time
import jieba.posseg as psg
import codecs
import random
import numpy as np
import pandas as pd
from random import choice
from itertools import combinations
from itertools import chain
from collections import Counter,OrderedDict
# from Similarity import edit_similar,jaccard_distance, weighted_jaccard_distance, seg_sentence_jaccard_distance
# from scipy.spatial.distance import cosine, jaccard
# from Datawash import strQ2B
# from Basetools import Dostring,Timestring
from .IO import load_data_from_excel_with_sheet, build_excel, dataLst2txt, readtxt2Lst

def similarl2d(similar_l):
    similar_d = {}
    for l in similar_l:
        if l[0] in similar_d:
            print(l[0])
            similar_d[l[0]].extend(l[1:])
        else:
            similar_d[l[0]] = l[1:]
    return similar_d

def similard2l(similar_d):
    attrd_l = []
    for k, v in similar_d.items():
        tmp = [k]
        tmp.extend(list(set(v)))
        attrd_l.append(tmp)
    return attrd_l

def similard2excel(similar_d, out_similar_f):
    similar_l = []
    for d in similar_d:
        tmp = [d]
        tmp.extend(similar_d[d])
        similar_l.append(tmp)
    build_excel(out_similar_f, similar_l)

def similarl2excel(similar_l, out_similar_f):
    build_excel(out_similar_f, similar_l)

def excel2similard(out_similar_f):
    data_l = load_data_from_excel_with_sheet(out_similar_f, sheet_index=[0])[0]
    similar_d = similarl2d(data_l)
    return similar_d