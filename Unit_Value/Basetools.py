import os
import re
import codecs
import shutil
import gensim
import pandas as pd

class Dostring(object):

    @staticmethod
    def strInlist_feature(string, feat_l):
        for s in feat_l:
            if string == str(s).split(':')[0]:
                return s
        return ''

    @staticmethod
    def strInlist_value(string, feat_l):
        for s in feat_l:
            if string == str(s).split(':')[1]:
                return s
        return ''

    @staticmethod
    def strInlist(string, str_l):
        return [s for s in str_l if string == str(s)]

    @staticmethod
    def strIFinlist(string, str_l):
        for s in str_l:
            if string == s:
                return True
        return False

    @staticmethod
    def strIFinlist_part(string, str_l):
        for s in str_l:
            if string in s:
                return True
        return False

    @staticmethod
    def strInlist_part(string, str_l):
        return [s for s in str_l if string in s]

    @staticmethod
    def strInlistIndex_part(string, str_l):
        return [i for i,s in enumerate(str_l) if string in s]

    @staticmethod
    def strInlistIndex(string, str_l):
        return [i for i, s in enumerate(str_l) if string == s]

    @staticmethod
    def get_feature_value(string):
        return string.split(':')[1]

    @staticmethod
    def extend_lst_uplowalpha(v_l):
        v_o_l = v_l + [v.lower() for v in v_l] + [v.upper() for v in v_l] + [v.capitalize() for v in v_l]
        v_o_l = [v.strip() for v in v_o_l if v.strip()]
        return v_o_l

    @staticmethod
    def extend_special_surround(v_l, special_surround=('(',')')):
        return ['{0[0]}{1}{0[1]}'.format(special_surround, v.strip()) for v in list(set(v_l)) if v.strip()]

    @staticmethod
    def feature_align(w, similar_d):
        for k,l in similar_d.items():
            if w in l:
                return k
        return w

    @staticmethod
    def word_str_dedup(word, ignore_patt=r'[\(\)]+'):
        if ignore_patt:
            word = re.sub(ignore_patt, '', word)
        if len(word) <= 2:
            return word
        else:
            if word[:len(word) // 2].lower() == word[len(word) // 2:].lower():
                return word[:len(word) // 2]
            else:
                return word


class Timestring(Dostring):

    @staticmethod
    def normalize_year(year_s):
        if len(year_s) == 2:
            return '20' + year_s
        elif len(year_s) == 3:
            return '2' + year_s
        else:
            return year_s

class Dofile(object):

    @staticmethod
    def getSplit_tmpfile(bisReport, outDir, outPrefix='split', linesize=100000):
        outTmpDir = '{0}/{1}_tmp'.format(outDir, outPrefix)
        if not os.path.isdir(outTmpDir):
            os.mkdir(outTmpDir)

        fileName = os.path.split(os.path.abspath(bisReport).strip('/'))[1].split('.')[0]

        def readFile(linesize, fa):
            tmpData = ""
            while True:
                count = 0
                partData = []
                if tmpData:
                    partData.append(tmpData)
                    count += 1
                tmpData = ""
                for line in fa:
                    if not line:
                        break
                    if count < linesize:
                        partData.append(line)
                        count += 1
                    else:
                        tmpData = line
                        break
                if partData:
                    if count >= linesize:
                        yield partData
                    else:
                        yield partData
                        break
                else:
                    break

        with codecs.open(bisReport, 'r', encoding='utf-8') as fa:
            partNum = 0
            for partData in readFile(linesize, fa):
                partFileName = '{0}/{1}_tmp_{2}'.format(outTmpDir, fileName, partNum)
                with codecs.open(partFileName, 'w', encoding='utf-8') as fp:
                    for line in partData:
                        fp.write(line)
                partNum += 1
        return outTmpDir

def get_word_num(glove_vector_file):
    count = 0
    with codecs.open(glove_vector_file, 'r', encoding='utf-8') as fr:
        while 1:
            buffer = fr.read(1024 * 8192)
            if not buffer:
                break
            count += buffer.count('\n')
    return count

def get_self_dict(new_words_f_l, out_dict, filt_word_freq=5):
    word_l = []
    for f in new_words_f_l:
        word_df = pd.read_csv(f, sep='\t', header=None)
        new_word = word_df[word_df.iloc[:, 1] > filt_word_freq]
        word_l.extend(new_word.iloc[:, 0].tolist())
    pd.Series(word_l).to_csv(out_dict, header=None, index=False)

def load_vectors(glove_vector_file, out_gensim_file, dim=50):
    word_count = get_word_num(glove_vector_file)
    fline = "{} {}\n".format(word_count, dim)
    with codecs.open(glove_vector_file, 'r', encoding='utf-8') as fr:
        with codecs.open(out_gensim_file, 'w', encoding='utf-8') as fw:
            fw.write(fline)
            shutil.copyfileobj(fr, fw)
    model = gensim.models.KeyedVectors.load_word2vec_format(out_gensim_file, binary=False)
    return model

''' just filt some illegal words. '''
def get_naive_prodict(word_dict_f, word_dict_fout, filt_word_freq=5):
    with codecs.open(word_dict_f, 'r', encoding='utf-8') as fr:
        words_l = fr.read().splitlines()
    words_d = {l.split('\t')[0]:eval(l.split('\t')[1]) for l in words_l if l}
    new_words_d = {w:words_d[w] for w in words_d if len(w) > 1 and not str(w).isdigit() and words_d[w] > filt_word_freq}
    pd.Series(new_words_d).to_csv(word_dict_fout, encoding='utf-8', sep='\t', header=None)

''' get keywords relate words to produce prowords dict '''
def get_keywords_relate_prodict(keywords_f, gensim_model, keyre_word_dict_fout, topn=50):
    key_w = pd.read_csv(keywords_f, header=None, encoding='utf-8', squeeze=True)
    pro_d = set()
    for w in key_w.values:
        pro_d.update([item[0] for item in gensim_model.most_similar(w, topn=topn)])
    pd.Series(pro_d).to_csv(keyre_word_dict_fout, header=None, encoding='utf-8')
