import os
import math
import subprocess
import pandas as pd
from bisect import bisect_left


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
    """Overlap which accounts for possible ties.
    This isn't mentioned in the paper but should be used in the ``rbo*()``
    functions below, otherwise overlap at a given depth might be > depth which
    inflates the result.
    There are no guidelines in the paper as to what's a good way to calculate
    this, but a good guess is agreement scaled by the minimum between the
    requested depth and the lengths of the considered lists (overlap shouldn't
    be larger than the number of ranks in the shorter list, otherwise results
    are conspicuously wrong when the lists are of unequal lengths -- rbo_ext is
    not between rbo_min and rbo_min + rbo_res.
    >>> overlap("abcd", "abcd", 3)
    3.0
    >>> overlap("abcd", "abcd", 5)
    4.0
    >>> overlap(["a", {"b", "c"}, "d"], ["a", {"b", "c"}, "d"], 2)
    2.0
    >>> overlap(["a", {"b", "c"}, "d"], ["a", {"b", "c"}, "d"], 3)
    3.0
    """
    return agreement(list1, list2, depth) * min(depth, len(list1), len(list2))
    # NOTE: comment the preceding and uncomment the following line if you want
    # to stick to the algorithm as defined by the paper
    # return raw_overlap(list1, list2, depth)[0]


def agreement(list1, list2, depth):
    """Proportion of shared values between two sorted lists at given depth.
    >>> _numtest(agreement("abcde", "abdcf", 1))
    '1.000'
    >>> _numtest(agreement("abcde", "abdcf", 3))
    '0.667'
    >>> _numtest(agreement("abcde", "abdcf", 4))
    '1.000'
    >>> _numtest(agreement("abcde", "abdcf", 5))
    '0.800'
    >>> _numtest(agreement([{1, 2}, 3], [1, {2, 3}], 1))
    '0.667'
    >>> _numtest(agreement([{1, 2}, 3], [1, {2, 3}], 2))
    '1.000'
    """
    len_intersection, len_set1, len_set2 = raw_overlap(list1, list2, depth)
    return 2 * len_intersection / (len_set1 + len_set2)


def cumulative_agreement(list1, list2, depth):
    return (agreement(list1, list2, d) for d in range(1, depth + 1))


def average_overlap(list1, list2, depth=None):
    """Calculate average overlap between ``list1`` and ``list2``.
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 1))
    '0.000'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 2))
    '0.000'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 3))
    '0.222'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 4))
    '0.292'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 5))
    '0.313'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 6))
    '0.317'
    >>> _numtest(average_overlap("abcdefg", "zcavwxy", 7))
    '0.312'
    """
    depth = min(len(list1), len(list2)) if depth is None else depth
    return sum(cumulative_agreement(list1, list2, depth)) / depth


def rbo_at_k(list1, list2, p, depth=None):
    # ``p**d`` here instead of ``p**(d - 1)`` because enumerate starts at
    # 0
    depth = min(len(list1), len(list2)) if depth is None else depth
    d_a = enumerate(cumulative_agreement(list1, list2, depth))
    return (1 - p) * sum(p ** d * a for (d, a) in d_a)


def rbo_min(list1, list2, p, depth=None):
    """Tight lower bound on RBO.
    See equation (11) in paper.
    >>> _numtest(rbo_min("abcdefg", "abcdefg", .9))
    '0.767'
    >>> _numtest(rbo_min("abcdefgh", "abcdefg", .9))
    '0.767'
    """
    depth = min(len(list1), len(list2)) if depth is None else depth
    x_k = overlap(list1, list2, depth)
    log_term = x_k * math.log(1 - p)
    sum_term = sum(
        p ** d / d * (overlap(list1, list2, d) - x_k) for d in range(1, depth + 1)
    )
    return (1 - p) / p * (sum_term - log_term)


def rbo_res(list1, list2, p):
    """Upper bound on residual overlap beyond evaluated depth.
    See equation (30) in paper.
    NOTE: The doctests weren't verified against manual computations but seem
    plausible. In particular, for identical lists, ``rbo_min()`` and
    ``rbo_res()`` should add up to 1, which is the case.
    >>> _numtest(rbo_res("abcdefg", "abcdefg", .9))
    '0.233'
    >>> _numtest(rbo_res("abcdefg", "abcdefghijklmnopqrstuvwxyz", .9))
    '0.239'
    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    # since overlap(...) can be fractional in the general case of ties and f
    # must be an integer → math.ceil()
    f = math.ceil(l + s - x_l)
    # upper bound of range() is non-inclusive, therefore + 1 is needed
    term1 = s * sum(p ** d / d for d in range(s + 1, f + 1))
    term2 = l * sum(p ** d / d for d in range(l + 1, f + 1))
    term3 = x_l * (math.log(1 / (1 - p)) - sum(p ** d / d for d in range(1, f + 1)))
    return p ** s + p ** l - p ** f - (1 - p) / p * (term1 + term2 + term3)


def rbo_ext(list1, list2, p):
    """RBO point estimate based on extrapolating observed overlap.
    See equation (32) in paper.
    NOTE: The doctests weren't verified against manual computations but seem
    plausible.
    >>> _numtest(rbo_ext("abcdefg", "abcdefg", .9))
    '1.000'
    >>> _numtest(rbo_ext("abcdefg", "bacdefg", .9))
    '0.900'
    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    x_s = overlap(list1, list2, s)
    # the paper says overlap(..., d) / d, but it should be replaced by
    # agreement(..., d) defined as per equation (28) so that ties are handled
    # properly (otherwise values > 1 will be returned)
    # sum1 = sum(p**d * overlap(list1, list2, d)[0] / d for d in range(1, l + 1))
    sum1 = sum(p ** d * agreement(list1, list2, d) for d in range(1, l + 1))
    sum2 = sum(p ** d * x_s * (d - s) / s / d for d in range(s + 1, l + 1))
    term1 = (1 - p) / p * (sum1 + sum2)
    term2 = p ** l * ((x_l - x_s) / l + x_s / s)
    return term1 + term2


def rbo(list1, list2, p):
    """Complete RBO analysis (lower bound, residual, point estimate).
    ``list`` arguments should be already correctly sorted iterables and each
    item should either be an atomic value or a set of values tied for that
    rank. ``p`` is the probability of looking for overlap at rank k + 1 after
    having examined rank k.
    """
    if not 0 <= p <= 1:
        raise ValueError("The ``p`` parameter must be between 0 and 1.")
    args = (list1, list2, p)
    return dict(min=rbo_min(*args), res=rbo_res(*args), ext=rbo_ext(*args))


def sort_dict(dct):
    scores = []
    items = []
    # items should be unique, scores don't have to
    for item, score in dct.items():
        # sort in descending order, i.e. according to ``-score``
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
    """Wrapper around ``rbo()`` for dict input.
    Each dict maps items to be sorted to the score according to which they
    should be sorted.
    """
    list1, list2 = sort_dict(dict1), sort_dict(dict2)
    return rbo(list1, list2, p)


def build_interval_list(
        work_year,
        frequency):

    interval_list = []
    for i in range(1, 13):
        if frequency == '2W':
            interval_list.extend(
                [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01',
                 work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-16'])
        else:
            raise Exception('build_interval_dict: Unknoen frequency...')

    return interval_list

def convert_trec_to_ranked_list(trec_str):
    docno_ordered_list = []
    splitted_trec = trec_str.split('\n')
    for line in splitted_trec:
        docno = line.split(' ')[2]
        docno_ordered_list.append(docno)
    return docno_ordered_list


if __name__=="__main__":
    query_retrn_files_path = '/lv_local/home/zivvasilisky/ziv/results/ranked_docs/'
    trec_eval_path = "/lv_local/home/zivvasilisky/ziv/env/indri/trec_eval/trec_eval-9.0.7/trec_eval"
    qrels_file_path = "/lv_local/home/zivvasilisky/ziv/results/qrels/qrels.adhoc"
    interval_lookup_method = 'Forward'

    interval_list = build_interval_list('2008', '2W')
    interval_list.append('ClueWeb09')

    last_not_clueweb_interval = interval_list[len(interval_list) - 2]
    for interval in interval_list:
        eval_file_path = os.path.join(os.path.dirname(query_retrn_files_path[:-1]), interval + '_' + interval_lookup_method + '_Results_evaluation.txt')
        if not os.path.exists(eval_file_path):
            bashCommand = '.' + trec_eval_path + ' -q ' + qrels_file_path + ' ' + \
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
        print('Query: ' + str(query_num))
        ranked_list_dict = {}
        for interval in interval_list:
            with open(os.path.join(query_retrn_files_path, str(query_num) + '_' + interval + '_' + interval_lookup_method + '_Results.txt'), 'r') as f:
                trec_str = f.read()
            ranked_list_dict[interval] = convert_trec_to_ranked_list(trec_str)
        eval_file_path = os.path.join(os.path.dirname(query_retrn_files_path[:-1]),
                                      interval + '_' + interval_lookup_method + '_Results_evaluation.txt')


        prev_interval = None
        for interval in interval_list:
            print('interval: ' + str(interval))
            insert_row = [query_num, interval]
            if query_num not in [100,95]:
                with open(eval_file_path, 'r') as f:
                    evel_str = f.read()

                for line in evel_str.split('\n'):
                    splitted_line = line(' ')
                    splitted_line = list(filter(None, splitted_line))
                    if int(splitted_line[1]) == query_num:
                        if splitted_line[0] == 'map':
                            map = float(splitted_line[2])
                        elif splitted_line[0] == 'P_5':
                            p_5 = float(splitted_line[2])
                        elif splitted_line[0] == 'P_10':
                            p_10 = float(splitted_line[2])
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

    summary_df.to_csv(os.path.join(os.path.dirname(query_retrn_files_path[:-1]),interval_lookup_method + '_Per_query_stats.tsv', sep = '\t', index = False)















