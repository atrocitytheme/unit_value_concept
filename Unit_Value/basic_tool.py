"""basic functions
"""
from .IO import build_excel

def similard2excel(similar_d, out_similar_f):
    similar_l = []
    for d in similar_d:
        tmp = [d]
        tmp.extend(similar_d[d])
        similar_l.append(tmp)
    build_excel(out_similar_f, similar_l)