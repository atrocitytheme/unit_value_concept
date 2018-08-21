import re, json
import pandas as pd
from .Feature_align import similard2excel
from collections import defaultdict
from collections import Counter
condition1 = ['+', '-', '/', '\\']

WIFI_detect = r'(\(WIFI.*\){1}.{0,4}$)|(WIFI$)|(\(WIFI$)'
map_dict = defaultdict(list) # 映射字典，可在if__name__ == '__main__'中重新设计

def find_bracket(s: str) -> str:
    """@param {str} s: 待处理字符串
    排除括号情况, 现在可针对")", 针对"("可将"(",")"调换，将processed.extend换为stack.extend
    """
    r, stack, processed = '', [], []

    con = list(enumerate(list(s)))
    for i, l in con:
        if l == '(':
            stack.append((i, l))
        if l == ')':
            if stack:
                stack.pop()
            else:
                processed.append((i, l))
    processed.extend(stack)
    start_to_add = False if processed else True
    for i, l in con:
        is_stacked = True
        for j, k in processed:
            if j == i:
                is_stacked = False
                break
        if is_stacked and start_to_add:
            r += l
            continue
        if i >= processed[0][0]:
            start_to_add = True
    return r

def null_return(s:str) -> str:
    """
    @param {str} s: 待处理字符串
    空字符返回
    """
    return ''

def no_null(f):
    """
    装饰器
    提前处理字符串，如果不满足条件则直接返回空字符
    """

    def new_func(s: str) -> str:

        if not s:
            return null_return(s)

        if not isinstance(s, str):
            return null_return(s)

        if len(s) <= 1:
            return null_return(s)

        s = str(s).strip()

        return f(s)

    return new_func

@no_null
def check_condition_1(s: str) -> str:
    """
    @param {str} s: 待处理字符串
    排除尾部符号
    """
    return s[:-1] if s[-1] in condition1 else s

@no_null
def check_condition_2(s: str) -> str:
    """
    @param {str} s: 待处理字符串
    排除
    """
    return find_bracket(s)

@no_null
def check_condition_3(s: str) -> str:
    """@param {str} s: 待处理字符串
    """
    return s.split(' ')[0] if ' ' in s else s

@no_null
def check_condition_4(s: str) -> str:
    """
    @param {str} s: 待处理字符串
    特例排除，所有特例均可在此排除
    """
    # if '/' in s:
    #     return s.replace('/', '')
    # return s
    return s[1:] if s[0] in condition1 else s

@no_null
def check_condition_5(s: str) -> str:
    """check the WIFI condition
    """
    return check_condition_1(re.sub(WIFI_detect, '', s, flags=re.IGNORECASE))

def final_check(s: str, all_sit: list) -> None:

    """@param {str} s: 待处理字符串
       @param {list} all_sit: 将产出的情况集合list
    """

    if len(s) < 2:
        return ''

    s = check_condition_1(s) # 去尾部
    s = check_condition_2(s) # 去括号
    s = check_condition_3(s) # 去空格
    s = check_condition_4(s) # 去特殊

    h = Counter(s)

    # 进行筛选

    if len(s.strip()) < 2:
        return

    if s.isdigit() and len(s) <= 2:
        return

    if "+" in s:
        m = s.split('+')[0]

        if m.isdigit() and len(m) <= 2:
            return

        s = check_condition_5(s)
        m = check_condition_5(m)

        map_dict[s].append(m)
        all_sit.append(m)

    if "-" in s and h['-'] >= 2:
        c = s.split('-')
        if len(c[-1]) > 1:
            m = s.split('-')[-1]
            if m.isdigit() and len(m) <= 2:
                pass
            else:
                # comment的部分为加入例外情况
                s = check_condition_5(s)
                m = check_condition_5(m)
                all_sit.append(m)
                map_dict[s].append(m)
                pass
        if len(c[0]) == 0:
            m = '-'.join(c[1:])
            if m.isdigit() and len(m) <= 2:
                pass
            else:
                """comment的部分为加入例外情况
                """
                check_condition_5(s)
                m = check_condition_5(m)
                map_dict[s].append(m)
                all_sit.append(m)
                pass
            return
    if s[0] == '+':
        return s[1:] # 补充最后情况

    s = check_condition_5(s)

    all_sit.append(s)

if __name__ == '__main__':
    all_sit = [] # 所有情况
    # final_result = set(all_sit)
    with open('./data/predict_xh_all_verify.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            # print(line)
            final_check(line, all_sit)

    final_result = set(all_sit)
    print(final_result)

    with open('./data/comp_new_all.txt', 'w') as f:
        for ele in final_result:
            if ele.strip():
                f.write(ele + '\n')
    print(map_dict)
    back_up_dict = json.dumps(map_dict)
    with open('backup_dict.json', 'w') as f:
        f.write(back_up_dict)
    similard2excel(map_dict, './data/xh_all_map_dict.xlsx')
