import os
import sys
import math
import subprocess
import pandas as pd
from bisect import bisect_left

from utils import *

def _numtest(floatn):
    return "{:.3f}".format(floatn)


def set_at_depth(lst, depth):
    ans = set()
    for v in lst[:depth]:
        if isinstance(v, set):
            ans.update(v)
        else:
            ans.add(v)
    return ans


def raw_overlap(list1, list2, depth):
    """Overlap as defined in the article.
    """
    set1, set2 = set_at_depth(list1, depth), set_at_depth(list2, depth)
    return len(set1.intersection(set2)), len(set1), len(set2)


def overlap(list1, list2, depth):

    return agreement(list1, list2, depth) * min(depth, len(list1), len(list2))



def agreement(list1, list2, depth):

    len_intersection, len_set1, len_set2 = raw_overlap(list1, list2, depth)
    return 2 * len_intersection / (len_set1 + len_set2)


def cumulative_agreement(list1, list2, depth):
    return (agreement(list1, list2, d) for d in range(1, depth + 1))


def average_overlap(list1, list2, depth=None):

    depth = min(len(list1), len(list2)) if depth is None else depth
    return sum(cumulative_agreement(list1, list2, depth)) / depth


def rbo_at_k(list1, list2, p, depth=None):

    depth = min(len(list1), len(list2)) if depth is None else depth
    d_a = enumerate(cumulative_agreement(list1, list2, depth))
    return (1 - p) * sum(p ** d * a for (d, a) in d_a)


def rbo_min(list1, list2, p, depth=None):

    depth = min(len(list1), len(list2)) if depth is None else depth
    x_k = overlap(list1, list2, depth)
    log_term = x_k * math.log(1 - p)
    sum_term = sum(
        p ** d / d * (overlap(list1, list2, d) - x_k) for d in range(1, depth + 1)
    )
    return (1 - p) / p * (sum_term - log_term)


def rbo_res(list1, list2, p):

    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    # since overlap(...) can be fractional in the general case of ties and f
    f = int(math.ceil(l + s - x_l))
    # upper bound of range() is non-inclusive, therefore + 1 is needed
    term1 = s * sum(p ** d / d for d in range(s + 1, f + 1))
    term2 = l * sum(p ** d / d for d in range(l + 1, f + 1))
    term3 = x_l * (math.log(1 / (1 - p)) - sum(p ** d / d for d in range(1, f + 1)))
    return p ** s + p ** l - p ** f - (1 - p) / p * (term1 + term2 + term3)


def rbo_ext(list1, list2, p):

    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    x_s = overlap(list1, list2, s)

    sum1 = sum(p ** d * agreement(list1, list2, d) for d in range(1, l + 1))
    sum2 = sum(p ** d * x_s * (d - s) / s / d for d in range(s + 1, l + 1))
    term1 = (1 - p) / p * (sum1 + sum2)
    term2 = p ** l * ((x_l - x_s) / l + x_s / s)
    return term1 + term2


def rbo(list1, list2, p):

    if not 0 <= p <= 1:
        raise ValueError("The ``p`` parameter must be between 0 and 1.")
    args = (list1, list2, p)
    return {'min': rbo_min(*args), 'res': rbo_res(*args), 'ext': rbo_ext(*args)}


def sort_dict(dct):
    scores = []
    items = []
    # items should be unique, scores don't have to
    for item, score in dct.items():
        score = -score
        i = bisect_left(scores, score)
        if i == len(scores):
            scores.append(score)
            items.append(item)
        elif scores[i] == score:
            existing_item = items[i]
            if isinstance(existing_item, set):
                existing_item.add(item)
            else:
                items[i] = {existing_item, item}
        else:
            scores.insert(i, score)
            items.insert(i, item)
    return items


def rbo_dict(dict1, dict2, p):

    list1, list2 = sort_dict(dict1), sort_dict(dict2)
    return rbo(list1, list2, p)


def create_retrieval_stats(
        interval_freq,
        interval_lookup_method,
        interval_start_month,
        amount_of_snapshot_limit):

    query_retrn_files_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ranked_docs/'
    trec_eval_path = "/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/trec_eval/trec_eval-9.0.7/trec_eval"
    qrels_file_path = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc"

    save_dirname = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/retrival_stats/"
    print('Interval Feaq: ' + interval_freq)
    print('Lookup method: ' + interval_lookup_method)
    addition = ""
    if interval_start_month != 1:
        addition = "_" + str(interval_start_month) + "SM_"

    if amount_of_snapshot_limit is not None and amount_of_snapshot_limit > 1:
        addition += "_SnapLimit_" + str(amount_of_snapshot_limit)

    interval_list = build_interval_list('2008', interval_freq, add_clueweb=True, start_month=interval_start_month)

    last_not_clueweb_interval = interval_list[len(interval_list) - 2]
    for interval in interval_list:
        eval_file_path = os.path.join(os.path.dirname(query_retrn_files_path[:-1]), interval + '_' + interval_freq + '_' + interval_lookup_method + addition +'_Results_evaluation.txt')
        if not os.path.exists(eval_file_path):
            bashCommand = trec_eval_path + ' -q ' + qrels_file_path + ' ' + \
                          eval_file_path.replace('_evaluation', '') + ' > ' + eval_file_path
            output = subprocess.check_output(['bash', '-c', bashCommand])

    summary_df = pd.DataFrame(columns = ['Query_Num', 'Interval', 'Map','P@5','P@10', 'Overlap@5_ClueWeb',
                                         'Overlap@5_Prev', 'Overlap@5_Last', 'RBO_0.95_Res_ClueWeb', 'RBO_0.95_Min_ClueWeb',
                                         'RBO_0.95_Ext_ClueWeb', 'RBO_0.975_Res_ClueWeb', 'RBO_0.975_Min_ClueWeb',
                                         'RBO_0.975_Ext_ClueWeb','RBO_0.99_Res_ClueWeb', 'RBO_0.99_Min_ClueWeb',
                                         'RBO_0.99_Ext_ClueWeb','RBO_0.95_Res_Prev', 'RBO_0.95_Min_Prev',
                                         'RBO_0.95_Ext_Prev', 'RBO_0.975_Res_Prev', 'RBO_0.975_Min_Prev',
                                         'RBO_0.975_Ext_Prev','RBO_0.99_Res_Prev', 'RBO_0.99_Min_Prev',
                                         'RBO_0.99_Ext_Prev','RBO_0.95_Res_Last', 'RBO_0.95_Min_Last',
                                         'RBO_0.95_Ext_Last', 'RBO_0.975_Res_Last', 'RBO_0.975_Min_Last',
                                         'RBO_0.975_Ext_Last','RBO_0.99_Res_Last', 'RBO_0.99_Min_Last',
                                         'RBO_0.99_Ext_Last'])
    next_index = 0
    for query_num in range(1,201):
        no_res_list = []
        print('Query: ' + str(query_num))
        sys.stdout.flush()
        ranked_list_dict = {}
        for interval in interval_list:
            with open(os.path.join(query_retrn_files_path, str(query_num) + '_' + interval_freq + '_' + interval + '_' + interval_lookup_method + addition + '_Results.txt'), 'r') as f:
                trec_str = f.read()
            ranked_list_dict[interval] = convert_trec_to_ranked_list(trec_str)
            if len(ranked_list_dict[interval]) == 0:
                no_res_list.append(interval)
            # print (ranked_list_dict[interval] )


        prev_interval = None
        for interval in interval_list:
            eval_file_path = os.path.join(os.path.dirname(query_retrn_files_path[:-1]),
                                          interval + '_' + interval_freq + '_' +interval_lookup_method + addition +'_Results_evaluation.txt')
            print('interval: ' + str(interval))
            insert_row = [query_num, interval]
            if interval in no_res_list:
                print("No Results to analys")
                continue
            if query_num not in [100,95]:
                with open(eval_file_path, 'r') as f:
                    evel_str = f.read()

                for line in evel_str.split('\n'):
                    splitted_line = line.split('\t')
                    splitted_line = list(filter(None, splitted_line))
                    # print (splitted_line )
                    # if splitted_line[1] == 'all':
                    #     map = None
                    #     p_5 = None
                    #     p_10 = None
                    #     break
                    if int(splitted_line[1]) == query_num:
                        if splitted_line[0].replace(' ' ,'') == 'map':
                            map = float(splitted_line[2])
                        elif splitted_line[0].replace(' ' ,'') == 'P_5':
                            p_5 = float(splitted_line[2])
                        elif splitted_line[0].replace(' ' ,'') == 'P_10':
                            p_10 = float(splitted_line[2])
                            break
            else:
                map = None
                p_5 = None
                p_10 = None

            insert_row.extend([map, p_5, p_10])

            if interval != 'ClueWeb09':
                overlap_5_clueweb = overlap(ranked_list_dict[interval], ranked_list_dict['ClueWeb09'], 5)
                rbo_095_clueweb_dict = rbo(ranked_list_dict[interval], ranked_list_dict['ClueWeb09'], p=0.95)
                rbo_0975_clueweb_dict = rbo(ranked_list_dict[interval], ranked_list_dict['ClueWeb09'], p=0.975)
                rbo_099_clueweb_dict = rbo(ranked_list_dict[interval], ranked_list_dict['ClueWeb09'], p=0.99)
            else:
                overlap_5_clueweb = None
                rbo_095_clueweb_dict = None
                rbo_0975_clueweb_dict = None
                rbo_099_clueweb_dict = None

            if prev_interval is not None:
                overlap_5_prev = overlap(ranked_list_dict[interval], ranked_list_dict[prev_interval], 5)
                rbo_095_prev_dict = rbo(ranked_list_dict[interval], ranked_list_dict[prev_interval], p=0.95)
                rbo_0975_prev_dict = rbo(ranked_list_dict[interval], ranked_list_dict[prev_interval], p=0.975)
                rbo_099_prev_dict = rbo(ranked_list_dict[interval], ranked_list_dict[prev_interval], p=0.99)
            else:
                overlap_5_prev = None
                rbo_095_prev_dict = None
                rbo_0975_prev_dict = None
                rbo_099_prev_dict = None

            if interval != last_not_clueweb_interval:
                overlap_5_last = overlap(ranked_list_dict[interval], ranked_list_dict[last_not_clueweb_interval], 5)
                rbo_095_last_dict = rbo(ranked_list_dict[interval], ranked_list_dict[last_not_clueweb_interval], p=0.95)
                rbo_0975_last_dict = rbo(ranked_list_dict[interval], ranked_list_dict[last_not_clueweb_interval], p=0.975)
                rbo_099_last_dict = rbo(ranked_list_dict[interval], ranked_list_dict[last_not_clueweb_interval], p=0.99)
            else:
                overlap_5_last = None
                rbo_095_last_dict = None
                rbo_0975_last_dict = None
                rbo_099_last_dict = None

            insert_row.extend([overlap_5_clueweb,overlap_5_prev, overlap_5_last])
            rbo_dict_list = [rbo_095_clueweb_dict, rbo_0975_clueweb_dict, rbo_099_clueweb_dict,
                             rbo_095_prev_dict,  rbo_0975_prev_dict, rbo_099_prev_dict,
                             rbo_095_last_dict, rbo_0975_last_dict, rbo_099_last_dict]

            for dict in rbo_dict_list:
                if dict is None:
                    insert_row.extend([None, None, None])
                else:
                    insert_row.extend([dict['res'], dict['min'], dict['ext']])

            summary_df.loc[next_index] = insert_row
            next_index += 1
            prev_interval = interval

    summary_df.to_csv(os.path.join(save_dirname ,interval_freq + '_' + interval_lookup_method + addition +'_Per_query_stats.tsv'), sep = '\t', index = False)















