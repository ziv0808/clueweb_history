import os
import sys
import ast
import math
import random
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
from scipy import spatial
from itertools import chain, combinations


WORK_YEAR = '2008'
TREC_EVAL_PATH = "/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/trec_eval/trec_eval-9.0.7/trec_eval"
if WORK_YEAR == '2011':
    QRELS_FILE_PATH = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels_cw12.adhoc"
else:
    QRELS_FILE_PATH = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc"

INNER_FOLD = 'cw12_1000_per_q'
QUERY_ENETITY_LIST = [1, 2, 5, 6, 7, 12, 15, 16, 21, 22, 23, 29, 31, 36, 37, 39, 41, 44, 46, 48, 52,
                      54, 56, 62, 65, 67, 68, 71, 72, 80, 81, 84, 85, 86, 97, 100, 101, 102, 103, 104,
                      105, 107, 108, 109, 110, 111, 114, 115, 116, 122, 123, 124, 126, 127, 129, 136,
                      138, 139, 140, 141, 142, 143, 148, 149, 150, 151, 152, 157, 159, 163, 168, 171, 183,
                      190, 191, 196, 197, 198, 200, 201, 202, 203, 207, 212, 215, 216, 219, 220, 221, 223,
                      228, 230, 231, 232, 234, 237, 238, 239, 243, 245, 246, 250, 254, 256, 257, 259,
                      260, 264, 269, 270, 271, 273, 275, 276, 281, 282, 283, 285, 288, 290, 291]
QUERY_ENETITY_LIST_MANUAL = [1,2,5,15,16,21,22,23,27,35,39,41,44,46,48,54,56,60,62,
                           65,71,73,81,85,97,101,102,103,104,105,107,108,109,110,
                           112,114,115,116,122,127,129,130,140,142,148,149,150,156,159,
                           163,167,171,180,183,186,187,190,191,192,197,198,200]

QUERY_PERSON_LOC_ORG_LIST = [1, 5, 6, 7, 12, 15, 16, 21, 22, 23, 31, 36, 37, 39, 41, 44,
                             48, 52, 54, 56, 62, 65, 67, 68, 71, 72, 80, 81, 84, 85, 86,
                             97, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 114,
                             115, 116, 122, 123, 126, 127, 129, 136, 138, 139, 140, 141, 142, 143,
                             148, 149, 150, 152, 157, 159, 163, 168, 171, 183, 190, 191, 196, 197,
                             198, 200, 201, 202, 203, 207, 212, 215, 216, 219, 220, 221, 223, 230,
                             232, 234, 237, 238, 239, 243, 246, 250, 254, 256, 257, 259, 264, 269,
                             270, 273, 281, 282, 283, 285, 288, 290, 291]

QUERY_PERSON_LOC_ORG_LIST_MANUAL = [1, 2, 5, 15, 16,21,22,23, 27, 35, 39, 41, 44, 46, 48, 54, 56, 60, 62,
                                    71, 73, 81, 85, 97, 101, 102, 103, 104, 105, 107, 108, 109, 110,
                                    114, 115, 116, 122, 127, 129, 130, 140, 142, 148, 149, 150, 156,
                                    159, 163 ,167, 171, 180, 183, 187, 190,191,192, 197,198,200]

CHECK_QUERIRES_LIST = [(QUERY_ENETITY_LIST, 'QUERY_ENETITY_LIST'),(QUERY_ENETITY_LIST_MANUAL, 'QUERY_ENETITY_LIST_MANUAL'),
                        (QUERY_PERSON_LOC_ORG_LIST, 'QUERY_PERSON_LOC_ORG_LIST'),(QUERY_PERSON_LOC_ORG_LIST_MANUAL, 'QUERY_PERSON_LOC_ORG_LIST_MANUAL') ]

def build_interval_list_asrc(
        asrc_round,
        add_last = True):

    interval_list = []
    for i in list(reversed(range(1, asrc_round))):
        interval_list.append(str((-1)*i))

    if (add_last == True):
        interval_list.append('ClueWeb09')

    return interval_list

def build_interval_list(
        work_year,
        frequency,
        add_clueweb = False,
        start_month = 1,
        end_month   = 12):
    # create interval list for work year according to required frequency
    interval_list = []
    if frequency.startswith('SIM'):
        for i in list(reversed(range(0, -49, -1)))[:-1]:
            interval_list.append(str(i))
        if int(start_month) < 0:
            interval_list = interval_list[(start_month):]
        else:
            interval_list = interval_list[(start_month - 1) * 4:]
    else:
        for i in range(start_month, end_month + 1):
            if frequency == '2W':
                interval_list.extend(
                    [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01',
                     work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-16'])
            elif frequency == '1W':
                interval_list.extend(
                    [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01',
                     work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-08',
                     work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-16',
                     work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-23'])
            elif frequency == '1M':
                interval_list.extend(
                    [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01'])
            elif frequency == '2M':
                if i % 2 == 1:
                    interval_list.extend(
                        [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01'])
            else:
                raise Exception('build_interval_dict: Unknoen frequency...')

    if (add_clueweb == True):
        interval_list.append('ClueWeb09')

    return interval_list


def get_word_diriclet_smoothed_probability(
        tf_in_doc,
        doc_len,
        collection_count_for_word,
        collection_len,
        mue):
    # return the dirichlet smoothed doc probability
    return (tf_in_doc + mue * (float(collection_count_for_word) / collection_len)) / (float(doc_len + mue))


def calc_cosine(
        dict_1,
        dict_2):
    # calcs cosine similarity from tf-idf dicts
    list_1 = []
    list_2 = []
    for stem in dict_1:
        if stem in dict_2:
            list_1.append(dict_1[stem])
            list_2.append(dict_2[stem])
        else:
            list_1.append(dict_1[stem])
            list_2.append(0.0)
    for stem in dict_2:
        if stem not in dict_1:
            list_1.append(0.0)
            list_2.append(dict_2[stem])
    return (1.0 - spatial.distance.cosine(list_1, list_2))


def convert_df_to_trec(
        df):
    # converts pandas df to terc str
    trec_str = ""
    for index, row in df.iterrows():
        trec_str += str(row['Query_ID']) + " " + row['Iteration'] + " " + \
                    row['Docno'] + " " + str(row['Rank']) + " " + str(row['Score']) + \
                    " " + row['Method'] + '\n'
    return trec_str

def convert_trec_to_ranked_list(trec_str):
    # converts terc str to docno ordered list
    docno_ordered_list = []
    splitted_trec = trec_str.split('\n')
    for line in splitted_trec:
        if line != '':
            docno = line.split(' ')[2]
            docno_ordered_list.append(docno)
    return docno_ordered_list

def get_doc_snapshot_by_lookup_method(
        doc_dict,
        interval_list,
        curr_interval_idx,
        interval_lookup_method,
        filter_params = {}):
    # gets the right interval snapshot according to lookup method
    doc_interval_dict = verify_snapshot_for_doc(doc_dict, interval_list[curr_interval_idx],filter_params)
    if doc_interval_dict is None:
        if interval_list[curr_interval_idx] == "ClueWeb09":
            raise Exception("ClueWeb09 needs lookup..")

        if interval_lookup_method == "Forward":
            addition = 1
            while doc_interval_dict is None:
                doc_interval_dict = verify_snapshot_for_doc(doc_dict,interval_list[curr_interval_idx + addition],filter_params)
                addition += 1

        elif interval_lookup_method == "Backward":
            addition = 1
            while (doc_interval_dict is None) and ((curr_interval_idx - addition) >= 0):
                doc_interval_dict = verify_snapshot_for_doc(doc_dict,interval_list[curr_interval_idx - addition],filter_params)
                addition += 1
            if doc_interval_dict is None:
                addition = 1
                while doc_interval_dict is None:
                    doc_interval_dict = verify_snapshot_for_doc(doc_dict,interval_list[curr_interval_idx + addition],filter_params)
                    addition += 1

        elif interval_lookup_method == "OnlyBackward":
            addition = 1
            while (doc_interval_dict is None) and ((curr_interval_idx - addition) >= 0):
                doc_interval_dict = verify_snapshot_for_doc(doc_dict,interval_list[curr_interval_idx - addition],filter_params)
                addition += 1

        elif interval_lookup_method == "NoLookup":
            pass

        else:
            raise Exception("Unknown lookup method..")

    return doc_interval_dict


def create_cc_dict(
        stemmed_query_collection_counts='/mnt/bi-strg3/v/zivvasilisky/ziv/data/StemsCollectionCounts.tsv'):
    # create easy to use index for collection counts - only for query stems
    query_stems_cc_df = pd.read_csv(stemmed_query_collection_counts, sep='\t', index_col=False)
    cc_dict = {}
    for index, row in query_stems_cc_df.iterrows():
        cc_dict[row['Stem']] = float(row['CollectionCount'])

    return cc_dict


def convert_query_to_tf_dict(
        query):
    # converts query string to tf dict
    query_dict = {}
    splitted_query = query.split(' ')
    for stem in splitted_query:
        if stem in query_dict:
            query_dict[stem] += 1
        else:
            query_dict[stem] = 1
    return query_dict

def create_stemmed_queries_df(
        stemmed_query_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/Stemmed_Query_Words',
        sw_rmv = False,
        limited_q_list = None):
    df = pd.read_csv(stemmed_query_file, sep = '\t', index_col = False)
    df['QueryInt'] = df['QueryNum'].apply(lambda x: int(x))
    if WORK_YEAR == '2011':
        df = df[df['QueryInt'] > 200]
    else:
        df = df[df['QueryInt'] <= 200]
    if limited_q_list is not None:
        df['Filter'] = df['QueryInt'].apply(lambda x: True if x not in limited_q_list else False)
        df = df[df['Filter'] == False]
        del df['Filter']
    del df['QueryInt']
    if sw_rmv == True:
        sw_list = get_stopword_list()
        df['QueryStems'] = df['QueryStems'].apply(lambda x: rmv_sw_from_string(x, sw_list))
    return df

def rmv_sw_from_string(
        strng,
        sw_list):
    new_strng = ""
    for word in strng.split(' '):
        if word not in sw_list:
            new_strng += word + " "
    return new_strng[:-1]

def create_query_to_doc_mapping_df(
        inner_fold = None):
    if inner_fold is None:
        query_to_doc_mapping_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + INNER_FOLD + '/all_urls_no_spam_filtered.tsv'
    else:
        query_to_doc_mapping_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + inner_fold + '/all_urls_no_spam_filtered.tsv'
    return pd.read_csv(query_to_doc_mapping_file, sep = '\t', index_col = False)

def convert_str_to_bool(strng):
    if strng == 'False':
        return False
    elif strng == 'True':
        return True
    else:
        raise Exception("convert_str_to_bool: Not supported string")

def convert_trec_results_file_to_pandas_df(
        results_file_path):
    # converts results file in trec format to pandas df
    res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])
    next_index = 0

    with open(results_file_path, "r") as f:
        f_lines = f.readlines()

    for line in f_lines:
        line = line.strip()
        broken_line = line.split(" ")
        if len(broken_line) > 1:
            res_df.loc[next_index] = broken_line
            next_index += 1

    return  res_df

def get_ranking_effectiveness_for_res_file(
        file_path,
        filename,
        qrel_filepath = QRELS_FILE_PATH):
    bashCommand = TREC_EVAL_PATH + ' ' + qrel_filepath + ' ' + \
                  os.path.join(file_path, filename)

    output = subprocess.check_output(['bash', '-c', bashCommand])
    output_lines = output.split('\n')
    for line in output_lines[:-1]:
        splitted_line = line.split('\t')
        splitted_line = list(filter(None, splitted_line))
        if splitted_line[1] == 'all':
            if splitted_line[0].replace(' ', '') == 'map':
                map = float(splitted_line[2])
            elif splitted_line[0].replace(' ', '') == 'P_5':
                p_5 = float(splitted_line[2])
            elif splitted_line[0].replace(' ', '') == 'P_10':
                p_10 = float(splitted_line[2])

    return {'Map': map, 'P_5': p_5, 'P_10': p_10}


def get_ranking_effectiveness_for_res_file_per_query(
        file_path,
        filename,
        qrel_filepath=QRELS_FILE_PATH,
        calc_ndcg_mrr = False,
        remove_low_quality = False):
    bashCommand = TREC_EVAL_PATH + ' -q ' + qrel_filepath + ' ' + \
                  os.path.join(file_path, filename)

    output = subprocess.check_output(['bash', '-c', bashCommand])
    output_lines = output.split('\n')
    res_dict = {}
    res_dict['all'] = {}
    for line in output_lines[:-1]:
        splitted_line = line.split('\t')
        splitted_line = list(filter(None, splitted_line))
        if (splitted_line[1] != 'all') and (int(splitted_line[1]) not in res_dict):
            res_dict[int(splitted_line[1])] = {}
        else:
            if splitted_line[1] == 'all':
                curr_q = 'all'
            else:
                curr_q = int(splitted_line[1])
            if splitted_line[0].replace(' ', '') == 'map':
                map = float(splitted_line[2])
                res_dict[curr_q]['Map'] = map
            elif splitted_line[0].replace(' ', '') == 'P_5':
                p_5 = float(splitted_line[2])
                res_dict[curr_q]['P_5'] = p_5
            elif splitted_line[0].replace(' ', '') == 'P_10':
                p_10 = float(splitted_line[2])
                res_dict[curr_q]['P_10'] = p_10
    if calc_ndcg_mrr == True:
        ndcg_mrr_dict = calc_ndcg_mrr_for_file(
            filename=filename,
            filepath=file_path,
            qrel_file=qrel_filepath,
            remove_low_quality=remove_low_quality)
        for key in ndcg_mrr_dict:
            for inner_key in ndcg_mrr_dict[key]:
                res_dict[key][inner_key] = ndcg_mrr_dict[key][inner_key]
    return res_dict

def get_ranking_effectiveness_for_res_file_for_all_query_groups(
        file_path,
        filename):
    bashCommand = TREC_EVAL_PATH + ' -q ' + QRELS_FILE_PATH + ' ' + \
                  os.path.join(file_path, filename)

    output = subprocess.check_output(['bash', '-c', bashCommand])
    output_lines = output.split('\n')
    res_dict = {}
    res_dict['all'] = {}
    for line in output_lines[:-1]:
        splitted_line = line.split('\t')
        splitted_line = list(filter(None, splitted_line))
        if (splitted_line[1] != 'all') and (int(splitted_line[1]) not in res_dict):
            res_dict[int(splitted_line[1])] = {}
        else:
            if splitted_line[1] == 'all':
                curr_q = 'all'
            else:
                curr_q = int(splitted_line[1])
            if splitted_line[0].replace(' ', '') == 'map':
                map = float(splitted_line[2])
                res_dict[curr_q]['Map'] = map
            elif splitted_line[0].replace(' ', '') == 'P_5':
                p_5 = float(splitted_line[2])
                res_dict[curr_q]['P_5'] = p_5
            elif splitted_line[0].replace(' ', '') == 'P_10':
                p_10 = float(splitted_line[2])
                res_dict[curr_q]['P_10'] = p_10

    return_res_dict = {}
    for q_list in CHECK_QUERIRES_LIST:
        q_list_name = q_list[1]
        query_list = q_list[0]
        return_res_dict[q_list_name] = {'Map': 0.0, 'P_5': 0.0, 'P_10': 0.0}
        denom = 0.0
        for query_num in query_list:
            if query_num in res_dict:
                denom += 1
                return_res_dict[q_list_name]['Map'] += res_dict[query_num]['Map']
                return_res_dict[q_list_name]['P_5'] += res_dict[query_num]['P_5']
                return_res_dict[q_list_name]['P_10'] += res_dict[query_num]['P_10']
        if denom > 0:
            return_res_dict[q_list_name]['Map'] = return_res_dict[q_list_name]['Map'] / denom
            return_res_dict[q_list_name]['P_5'] = return_res_dict[q_list_name]['P_5'] / denom
            return_res_dict[q_list_name]['P_10'] = return_res_dict[q_list_name]['P_10'] / denom

    return_res_dict['all'] = res_dict['all']
    return return_res_dict

def create_per_interval_cc_dict(
        cc_dict_file='/mnt/bi-strg3/v/zivvasilisky/ziv/data/'+INNER_FOLD+'/cc_per_interval_dict.json',
        interval_freq = None,
        lookup_method = None):

    with open(cc_dict_file, 'r') as f:
        cc_dict = ast.literal_eval(f.read())

    if interval_freq is not None:
        cc_dict = cc_dict[interval_freq]
        if lookup_method is not None:
            cc_dict = cc_dict[lookup_method]
    return  cc_dict


def create_per_interval_df_dict(
        cc_dict_file='/mnt/bi-strg3/v/zivvasilisky/ziv/data/'+INNER_FOLD+'/df_per_interval_dict.json',
        interval_freq=None,
        lookup_method=None):

    with open(cc_dict_file, 'r') as f:
        df_dict = ast.literal_eval(f.read())

    if interval_freq is not None:
        df_dict = df_dict[interval_freq]
        if lookup_method is not None:
            df_dict = df_dict[lookup_method]
    return df_dict

def get_number_of_snapshots_for_doc(
        doc_dict):
    num_of_snapshots = 0
    for snapshot in doc_dict:
        if doc_dict[snapshot] is not None:
            num_of_snapshots += 1
    return num_of_snapshots

def verify_snapshot_for_doc(
        doc_dict,
        interval,
        filter_params):

    if (interval == 'ClueWeb09') or (doc_dict[interval] is None):
        return doc_dict[interval]

    if (('Sim_Upper' in filter_params) and (filter_params['Sim_Upper'] is not None)) or (('Sim_Lower' in filter_params) and (filter_params['Sim_Lower'] is not None)):
        similarity = calc_cosine(doc_dict[interval]['TfIdf'], doc_dict['ClueWeb09']['TfIdf'])
        if ('Sim_Upper' in filter_params) and (filter_params['Sim_Upper'] is not None):
            if similarity > filter_params['Sim_Upper']:
                return None
        if ('Sim_Lower' in filter_params) and (filter_params['Sim_Lower'] is not None):
            if similarity < filter_params['Sim_Lower']:
                return None

    txt_diff = doc_dict['ClueWeb09']['NumWords'] - doc_dict[interval]['NumWords']
    if ('TxtDiff_Upper' in filter_params) and (filter_params['TxtDiff_Upper'] is not None):
        if txt_diff > filter_params['TxtDiff_Upper']:
            return None
    if ('TxtDiff_Lower' in filter_params) and (filter_params['TxtDiff_Lower'] is not None):
        if txt_diff < filter_params['TxtDiff_Lower']:
            return None

    if (('%TxtDiff_Upper' in filter_params) and (filter_params['%TxtDiff_Upper'] is not None)) or (('%TxtDiff_Lower' in filter_params) and (filter_params['%TxtDiff_Lower'] is not None)):
        txt_diff = txt_diff/float(doc_dict['ClueWeb09']['NumWords'])
        if ('%TxtDiff_Upper' in filter_params) and (filter_params['%TxtDiff_Upper'] is not None):
            if txt_diff > filter_params['%TxtDiff_Upper']:
                return None
        if ('%TxtDiff_Lower' in filter_params) and (filter_params['%TxtDiff_Lower'] is not None):
            if txt_diff < filter_params['%TxtDiff_Lower']:
                return None

    return doc_dict[interval]

def create_filter_params_txt_addition(
        filter_params):

    str_addition = ""
    for val in ["Sim_Lower", "%TxtDiff_Lower", "TxtDiff_Lower", "%TxtDiff_Upper","Sim_Upper", "TxtDiff_Upper"]:
        if val in filter_params:
            str_addition += val + '_' + str(filter_params[val]) + '_'

    return str_addition

def get_stopword_list():
    stop_word_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/Stemmed_Stop_Words'
    with open(stop_word_file, 'r') as f:
        stopword_str = f.read()
    stopword_list = stopword_str.split('\n')
    stopword_list.remove('')
    return stopword_list

def calc_shannon_entopy(tf_list):
    total = float(sum(tf_list))
    return sum(float(freq) / total * math.log(total / freq, 2) for freq in tf_list)


def calc_tfidf_dict(
        stem_list,
        tf_list,
        df_list):

    res_dict = {}
    for i in range(1, len(stem_list)):
        if df_list[i] > 0:
            res_dict[stem_list[i]] = round(tf_list[i] * math.log10(500000000.0 / float(df_list[i])), 6)
        else:
            print("Prob Stem: " + stem_list[i])
            res_dict[stem_list[i]] = 0.0

    return res_dict

def create_full_doc_dict_from_fulltext(
        curr_fulltext_list,
        concatenated_stem_list,
        concatenated_df_list,
        concatenated_cc_list,
        stopword_list):

    res_dict = {}
    res_dict['StemList'] = ['[[OOV]']
    res_dict['IndexList'] = []
    res_dict['NumStopWords'] = 0
    res_dict['NumWords'] = 0
    res_dict['TfList'] = [0]
    res_dict['DfList'] = [0]
    res_dict['CCList'] = [0]
    res_dict['Fulltext'] = ""
    res_dict['TfDict'] = {}

    for stem in curr_fulltext_list:
        if stem not in res_dict['TfDict']:
            res_dict['StemList'].append(stem)
            res_dict['TfDict'][stem] = 1
        else:
            res_dict['TfDict'][stem] += 1

        res_dict['IndexList'].append(res_dict['StemList'].index(stem))
        if stem in stopword_list:
            res_dict['NumStopWords'] += 1

        res_dict['NumWords'] += 1
        res_dict['Fulltext'] += stem + " "

    res_dict['Fulltext'] = res_dict['Fulltext'][:-1]

    for stem in res_dict['StemList'][1:]:
        res_dict['TfList'].append(res_dict['TfDict'][stem])
        res_dict['CCList'].append(concatenated_cc_list[concatenated_stem_list.index(stem)])
        res_dict['DfList'].append(concatenated_df_list[concatenated_stem_list.index(stem)])

    res_dict['Entropy'] = calc_shannon_entopy(res_dict['TfList'][1:])

    res_dict['TfIdf'] = calc_tfidf_dict(
        res_dict['StemList'],
        res_dict['TfList'],
        res_dict['DfList'])

    return res_dict


def create_decaying_wieghts_list(
        interval_list,
        decaying_factor,
        skip_idx_list,
        reverse):
    # create decaying weights for the interval list without skiped indexes
    p = len(interval_list) - len(skip_idx_list)
    wieght_list = [0.0] * len(interval_list)
    k = 0.0
    if reverse == True:
        k = float(p - 1)
    for i in range(len(wieght_list)):
        if i not in skip_idx_list:
            wieght_list[i] = (1.0 - k * (decaying_factor / float(p)))
            if reverse == True:
                k -= 1
            else:
                k += 1

    return pd.np.array(wieght_list)

def create_uniform_wieghts_list(
        interval_list,
        skip_idx_list):
    # create uniform weights for the interval list without skiped indexes
    denominator = float(len(interval_list) - len(skip_idx_list))
    wieght_list = [0.0] * len(interval_list)
    for i in range(len(wieght_list)):
        if i not in skip_idx_list:
            wieght_list[i] = 1.0 / denominator
    return pd.np.array(wieght_list)

def calc_releational_measure(
        measure_obs,
        reletional_measure_obs):

    if reletional_measure_obs == 0.0:
        if measure_obs != 0.0:
            return 1.0
        else:
            return 0.0
    else:
        return float(measure_obs - reletional_measure_obs) / float(reletional_measure_obs)

def check_statistical_significance(
        res_dict_1,
        res_dict_2,
        ndcg_mrr = False,
        sinificance_type = 'TTest'):
    try:
        q_list = list(res_dict_1.keys())
        if 'all' in q_list:
            q_list.remove('all')
        res_dict = {}
        additional_measures = []
        if ndcg_mrr == True:
            additional_measures = ['NDCG@1', 'NDCG@3','NDCG@5', 'MRR','nMRR']
        for measure in ['Map', 'P_5', 'P_10'] + additional_measures:
            l1 = []
            l2 = []
            for q in q_list:
                if np.isnan(res_dict_1[q][measure]) and np.isnan(res_dict_2[q][measure]):
                    continue
                l1.append(res_dict_1[q][measure])
                l2.append(res_dict_2[q][measure])
            if sinificance_type == 'TTest':
                t_stat, p_val = stats.ttest_rel(l1, l2)
            elif sinificance_type == 'Perm':
                t_stat, p_val = pemutation_test(l1, l2)
            res_dict[measure] = {}
            if round(p_val,2) <= 0.05:
                res_dict[measure]['Significant'] = True
            else:
                res_dict[measure]['Significant'] = False

            res_dict[measure]['Pval'] = round(p_val,2)
            res_dict[measure]['%Better'] = round(pd.np.sum(pd.np.array(l1) > pd.np.array(l2))/float(len(l1)),2)
            res_dict[measure]['%BetterOrEqual'] = round(pd.np.sum(pd.np.array(l1) >= pd.np.array(l2)) / float(len(l1)), 2)
    except Exception as e:
        raise Exception('check_statistical_significance: ' + str(e))

    return res_dict


def bm25_score_doc_for_query(
        query_stem_dict,
        df_dict,
        doc_dict,
        k1=1.0,
        b=0.5):

    bm25_score = 0.0
    work_stem_list = list(query_stem_dict.keys())

    for stem in work_stem_list:
        doc_stem_tf = 0
        if 'TfDict' in doc_dict:
            if stem in doc_dict['TfDict']:
                doc_stem_tf = float(doc_dict['TfDict'][stem])

        if stem not in df_dict:
            if doc_stem_tf == 0:
                continue
            else:
                raise Exception('Unexpected Situation on ' + str(stem))
        if df_dict[stem] == 0:
            df_dict[stem] = 1

        idf = math.log(df_dict['ALL_DOCS_COUNT'] / float(df_dict[stem]), 10)
        stem_d_proba = (doc_stem_tf * (k1 + 1)) / (
        doc_stem_tf + k1 * ((1 - b) + b * (float(doc_dict['NumWords']) / df_dict['AVG_DOC_LEN'])))

        bm25_score += idf * stem_d_proba

    return bm25_score


def lm_score_doc_for_query(
        query_stem_dict,
        cc_dict,
        doc_dict,
        mue=1000.0):

    kl_score = 0.0
    work_stem_list = list(query_stem_dict.keys())

    for stem in work_stem_list:
        doc_stem_tf = 0
        if stem in doc_dict['TfDict']:
            doc_stem_tf = float(doc_dict['TfDict'][stem])


        if stem not in cc_dict:
            if doc_stem_tf == 0:
                continue
            else:
                raise Exception('Unexpected Situation on ' + str(stem))

        query_tf = 0
        if stem in query_stem_dict:
            query_tf = query_stem_dict[stem]

        stem_q_prob = float(query_tf) / sum(list(query_stem_dict.values()))

        stem_d_proba = get_word_diriclet_smoothed_probability(
            tf_in_doc=doc_stem_tf,
            doc_len=doc_dict['NumWords'],
            collection_count_for_word=cc_dict[stem],
            collection_len=cc_dict['ALL_TERMS_COUNT'],
            mue=mue)

        kl_score += (-1) * stem_q_prob * (math.log((stem_q_prob / stem_d_proba), 2))

    return kl_score


def get_relevant_docs_df_utils(
        qurls_path='/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc'):


    relevant_docs_df = pd.DataFrame(columns=['Query', 'Docno', 'Relevance'])
    next_index = 0
    with open(qurls_path, 'r') as f:
        file_lines = f.readlines()

    for line in file_lines:
        line = line.strip()
        splitted_line = line.split(' ')
        relevant_docs_df.loc[next_index] = [splitted_line[0], splitted_line[2], splitted_line[3]]
        next_index += 1

    return relevant_docs_df

def remove_low_quality_docs_from_df(
        df):
    low_qulity_docs_list = ['EPOCH-05-069-PLB6ZI', 'EPOCH-02-004-02', 'EPOCH-03-195-3RXCQH', 'EPOCH-04-011-GUMSZY', 'EPOCH-02-098-UONKT7',
                            'EPOCH-08-004-48', 'EPOCH-05-177-YA7EE4', 'EPOCH-02-029-15', 'EPOCH-05-011-U4XHBD', 'EPOCH-01-144-DX0IY8',
                            'EPOCH-06-029-41', 'EPOCH-03-017-22', 'EPOCH-05-180-MEP5Y4', 'EPOCH-05-144-06', 'EPOCH-03-124-34PBG2',
                            'EPOCH-02-048-RXK8H3', 'EPOCH-02-045-71QP3V', 'EPOCH-06-017-01', 'EPOCH-03-098-UONKT7', 'EPOCH-08-029-16',
                            'EPOCH-02-017-22', 'EPOCH-05-013-SBHV4U', 'EPOCH-05-167-H4PVOF', 'EPOCH-05-010-9DWMMZ', 'EPOCH-02-011-GUMSZY',
                            'EPOCH-02-182-01', 'EPOCH-01-011-GUMSZY', 'EPOCH-08-017-01', 'EPOCH-08-002-43', 'EPOCH-02-011-09', 'EPOCH-07-069-43',
                            'EPOCH-01-124-34PBG2', 'EPOCH-02-004-BL2KLC', 'EPOCH-07-029-41', 'EPOCH-06-069-43', 'EPOCH-03-011-GUMSZY',
                            'EPOCH-05-144-15', 'EPOCH-07-002-13', 'EPOCH-05-124-S4SDDG', 'EPOCH-04-124-34PBG2', 'EPOCH-08-144-06',
                            'EPOCH-03-144-15', 'EPOCH-05-013-T7KWL2', 'EPOCH-01-180-37', 'EPOCH-08-048-27', 'EPOCH-06-193-01',
                            'EPOCH-01-048-RXK8H3', 'EPOCH-03-069-41', 'EPOCH-03-124-S4SDDG', 'EPOCH-05-033-SA9WFV', 'EPOCH-02-059-14T9OZ',
                            'EPOCH-01-034-6LWZ77', 'EPOCH-08-045-38', 'EPOCH-02-124-39', 'EPOCH-08-032-16', 'EPOCH-01-098-15', 'EPOCH-05-182-W129AB',
                            'EPOCH-07-193-01', 'EPOCH-05-032-7XUGJ0', 'EPOCH-02-195-3RXCQH', 'EPOCH-05-048-RXK8H3', 'EPOCH-07-048-27', 'EPOCH-02-078-WVSJJH',
                            'EPOCH-05-051-6QL968', 'EPOCH-05-033-KP5G43', 'EPOCH-01-034-20', 'EPOCH-06-018-03', 'EPOCH-02-144-T14SSS', 'EPOCH-02-124-S4SDDG',
                            'EPOCH-04-018-609XQD', 'EPOCH-07-144-06', 'EPOCH-01-144-T14SSS', 'EPOCH-03-004-48', 'EPOCH-01-098-UONKT7', 'EPOCH-04-144-06',
                            'EPOCH-08-069-45', 'EPOCH-08-069-43', 'EPOCH-08-144-24', 'EPOCH-05-161-FCT04D', 'EPOCH-03-048-RXK8H3', 'EPOCH-05-195-3RXCQH',
                            'EPOCH-05-098-UONKT7', 'EPOCH-06-029-16', 'EPOCH-06-029-15', 'EPOCH-07-017-01', 'EPOCH-06-195-43', 'EPOCH-01-059-14T9OZ',
                            'EPOCH-04-124-S4SDDG', 'EPOCH-01-011-U4XHBD', 'EPOCH-07-059-45', 'EPOCH-02-144-24', 'EPOCH-07-059-41', 'EPOCH-04-144-T14SSS',
                            'EPOCH-04-032-7XUGJ0', 'EPOCH-04-098-UONKT7', 'EPOCH-07-002-43', 'EPOCH-06-166-03', 'EPOCH-04-048-RXK8H3', 'EPOCH-04-195-35',
                            'EPOCH-02-144-15', 'EPOCH-03-029-15', 'EPOCH-04-069-41', 'EPOCH-01-033-KP5G43', 'EPOCH-03-098-SPSHA3', 'EPOCH-04-009-19',
                            'EPOCH-08-002-13', 'EPOCH-03-144-52', 'EPOCH-03-144-T14SSS', 'EPOCH-05-018-9PFCP4', 'EPOCH-06-059-41', 'EPOCH-02-144-09',
                            'EPOCH-06-144-06', 'EPOCH-01-017-VKBYGU', 'EPOCH-04-045-71QP3V', 'EPOCH-01-029-15', 'EPOCH-01-124-S4SDDG', 'EPOCH-03-180-19',
                            'EPOCH-08-029-15', 'EPOCH-08-009-33', 'EPOCH-05-144-T14SSS', 'EPOCH-08-078-04', 'EPOCH-07-029-16', 'EPOCH-01-004-0YGZO0',
                            'EPOCH-05-018-609XQD', 'EPOCH-06-009-02', 'EPOCH-03-011-U4XHBD', 'EPOCH-06-048-27', 'EPOCH-01-195-3RXCQH', 'EPOCH-03-018-609XQD',
                            'EPOCH-02-078-14', 'EPOCH-04-195-3RXCQH', 'EPOCH-08-010-40', 'EPOCH-04-004-48', 'EPOCH-05-011-GUMSZY', 'EPOCH-03-144-24',
                            'EPOCH-04-029-15', 'EPOCH-08-164-31', 'EPOCH-06-002-43', 'EPOCH-04-059-14T9OZ', 'EPOCH-03-017-32', 'EPOCH-05-048-V0HAX1',
                            'EPOCH-04-144-24', 'EPOCH-07-029-25', 'EPOCH-03-078-WVSJJH']
    idx_drop_list = []
    for idx, row in df.iterrows():
        if row['Docno'] in low_qulity_docs_list:
            idx_drop_list.append(idx)
    df.drop(idx_drop_list, inplace=True)
    return df

def calc_ndcg_mrr_for_file(
        filepath,
        filename,
        qrel_file,
        remove_low_quality= False):
    qrel_df = get_relevant_docs_df_utils(qrel_file)
    qrel_df['Relevance'] = qrel_df['Relevance'].apply(lambda x: int(x))
    res_df = convert_trec_results_file_to_pandas_df(os.path.join(filepath, filename))
    if remove_low_quality == True:
        print("Got Here")
        sys.stdout.flush()
        res_df = remove_low_quality_docs_from_df(res_df)
        print("Got Out")
        sys.stdout.flush()
    res_df.rename(columns = {'Query_ID' : 'Query'}, inplace = True)
    res_df = pd.merge(
        res_df,
        qrel_df,
        on = ['Query', 'Docno'],
        how = 'left')
    res_df.fillna(0, inplace = True)
    res_dict = {'all' : {'NDCG@1'  :0.0 , 'NDCG@3'  :0.0 , 'NDCG@5'  :0.0 , 'MRR' : 0.0, 'nMRR' : 0.0}}
    all_q = list(res_df['Query'].drop_duplicates())
    mrr_denom = 0
    for q in all_q:
        q_df = res_df[res_df['Query'] == q]
        q = int(q)
        res_dict[q] = {}
        true_val_list = q_df['Relevance'].values
        ndcg1 = calc_ndcg(true_val_list, 1)
        ndcg3 = calc_ndcg(true_val_list, 3)
        ndcg5 = calc_ndcg(true_val_list, 5)
        mrr, nmrr = calc_mrr_nmrr(true_val_list)
        res_dict[q]['NDCG@1'] = ndcg1
        res_dict[q]['NDCG@3'] = ndcg3
        res_dict[q]['NDCG@5'] = ndcg5
        res_dict[q]['MRR'] = mrr
        res_dict[q]['nMRR'] = nmrr
        for measure in ['NDCG@1','NDCG@3','NDCG@5','nMRR']:
            res_dict['all'][measure] += res_dict[q][measure]
        if not np.isnan(mrr):
            res_dict['all']['MRR'] += mrr
            mrr_denom += 1

    for measure in ['NDCG@1', 'NDCG@3','NDCG@5', 'nMRR']:
        res_dict['all'][measure] = res_dict['all'][measure] / float(len(all_q))
    if mrr_denom > 0:
        res_dict['all']['MRR'] = res_dict['all']['MRR'] / float(mrr_denom)
    else:
        res_dict['all']['MRR'] = np.nan
    return res_dict


def calc_ndcg_at_x_for_file(
        file_path,
        filename,
        qrel_filepath,
        x = 5):
    qrel_df = get_relevant_docs_df_utils(qrel_filepath)
    qrel_df['Relevance'] = qrel_df['Relevance'].apply(lambda x: int(x))
    res_df = convert_trec_results_file_to_pandas_df(os.path.join(file_path, filename))
    res_df.rename(columns={'Query_ID': 'Query'}, inplace=True)
    res_df = pd.merge(
        res_df,
        qrel_df,
        on=['Query', 'Docno'],
        how='left')
    res_df.fillna(0, inplace=True)
    all_q = list(res_df['Query'].drop_duplicates())

    res_dict = {'NDCG@X': 0.0}
    for q in all_q:
        q_df = res_df[res_df['Query'] == q]
        true_val_list = q_df['Relevance'].values
        ndcgx = calc_ndcg(true_val_list, x)
        res_dict['NDCG@X'] += ndcgx
    res_dict['NDCG@X'] = res_dict['NDCG@X'] / float(len(all_q))
    return res_dict

def calc_ndcg(
        y_true,
        k):
    discount = 1 / np.log2(np.arange(len(y_true[:k])) + 2)
    dcg = np.dot(np.array(y_true[:k]), discount)
    best_dcg = np.dot(np.array(sorted(y_true,reverse =True)[:k]), discount)
    if best_dcg == 0.0:
        return 0.0
    return dcg / best_dcg

def calc_mrr_nmrr(
        y_true):

    min_val = y_true.min()
    if min_val <= 0:
        ret_idx = np.where(y_true <= 0)[0].min()
        return 1.0/(ret_idx + 1), 1.0/(ret_idx + 1)
    else:
        ret_idx = np.where(y_true <= min_val)[0].min()
        return np.nan, 1.0/(ret_idx + 1)


def get_asrc_q_list_and_fold_list(
        inner_fold,
        train_leave_one_out):
    if ('asrc' in inner_fold) or ('comp2020' in inner_fold):
        fold_list = [(2, 9), (10, 17), (18, 32), (33, 36), (45, 51), (59, 78),
                     (98, 144), (161, 166), (167, 180), (182, 195)]

        q_list = [2, 4, 9, 10, 11, 17, 18, 29, 32, 33, 34, 36, 45, 48, 51, 59, 69, 78, 98, 124, 144, 161, 164, 166, 167, 177,
                  180, 182, 188, 193, 195]

    elif 'bot' in inner_fold:
        fold_list = [(10, 13), (18, 32), (33, 34), (48, 51), (69, 167), (177, 177),
                     (180, 180), (182, 182), (193, 193), (195, 195)]

        q_list = [10, 13, 18, 32, 33, 34, 48, 51, 69, 167, 177, 180, 182, 193, 195]

    elif 'herd_control' in inner_fold:
        fold_list = [(4, 10), (11, 17), (18, 32), (33, 45), (48, 59), (69, 98),
                     (124, 161), (164, 167), (177, 182), (188, 195)]

        q_list = [4, 9, 10, 11, 13, 17, 18, 29, 32, 33, 34, 45, 48, 51, 59, 69, 78, 98, 124, 144, 161, 164, 166, 167, 177,
                  180, 182, 188, 193, 195]

    elif 'united' in inner_fold:
        fold_list = [(2, 9), (10, 17), (18, 32), (33, 36), (45, 51), (59, 78),
                     (98, 144), (161, 166), (167, 180), (182, 195)]

        q_list = [2, 4, 9, 10, 11, 13, 17, 18, 29, 32, 33, 34, 36, 45, 48, 51, 59, 69, 78, 98, 124, 144, 161, 164, 166, 167, 177,
                  180, 182, 188, 193, 195]
    else:
        raise Exception('get_asrc_q_list_and_fold_list: Unknown inner fold')

    if train_leave_one_out == True:
        fold_list = []
        for q in q_list:
            fold_list.append((q, q))

    return q_list, fold_list

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))



def pemutation_test(test_group, control_group, total_number=1000):
    """

    :param test_group:  vector of test group results (originally sorted by qid)
    :param control_group: vector of control group results (originally sorted by qid)
    :param total_number: if number of permutations is too high, this parameter should be different than None.
    the parameter states the number of permutations sampled
    :return:
    """
    # np.random.seed(9002)
    diff = abs(np.mean(test_group) - np.mean(control_group))
    indicator_sum=0
    counter = 0
    if total_number is None:
        indices = range(len(test_group))
        indices_set = set(indices)
        for permutation in powerset(indices):
            control_shifted_indices = indices_set - set(permutation)
            permuted_test = [test_group[i] for i in permutation]
            permuted_test.extend([control_group[j] for j in control_shifted_indices])
            permuted_control = [test_group[i] for i in control_shifted_indices]
            permuted_control.extend([control_group[j] for j in permutation])
            permutation_diff = abs(np.mean(permuted_test)-np.mean(permuted_control))
            if permutation_diff>=diff:
                indicator_sum+=1
            counter+=1
    else:
        indices = range(len(test_group))
        for i in range(total_number):
            permuted_test = []
            permuted_control = []
            sample_indices = list(np.random.choice([0, 1], p=[0.5, 0.5], size=(1, len(indices)))[0])
            for index, choice in enumerate(sample_indices):
                if choice == 1:
                    permuted_test.append(test_group[index])
                    permuted_control.append(control_group[index])
                else:
                    permuted_test.append(control_group[index])
                    permuted_control.append(test_group[index])
            permutation_diff = abs(np.mean(permuted_test) - np.mean(permuted_control))
            if permutation_diff >= diff:
                indicator_sum += 1

        counter=total_number


    return diff,indicator_sum/float(counter)

def create_feature_list_shortcut_string(feat_list):
    ret_str = ''
    feat_short_dict = {
        'CoverQueryNum' : 'CQN',
        'CoverQueryRatio' : 'CQR',
        'Ent' : 'E',
        'FracStops' : 'FS',
        'IDF' : 'IDF',
        'Len' : 'L',
        'LMIR.DIR' : 'LMD',
        'LMIR.JM' : 'LMJ',
        'StopCover' : 'SC',
        'TFSum' : 'TFSm',
        'TFMin' : 'TFMn',
        'TFMax': 'TFMx',
        'TFMean': 'TFAv',
        'TFStd': 'TFSd',
        'TFIDFSum': 'TFDFSm',
        'TFIDFMin': 'TFDFMn',
        'TFIDFMax': 'TFDFMx',
        'TFIDFMean': 'TFDFAv',
        'TFIDFStd': 'TFDFSd',
        'TFNormSum': 'TFNSm',
        'TFNormMin': 'TFNMn',
        'TFNormMax': 'TFNMx',
        'TFNormMean': 'TFNAv',
        'TFNormStd': 'TFNSd',
        'SimClueWeb': 'SCw',
        'BM25Score': 'BM25',
        'BERTScore' : 'BRT',
        '-Query-SW' : 'nQSw',
        'JMPrevWinner': 'MJmPw',
        'JMPrev2Winners': 'MJmP2w',
        'JMPrev3Winners': 'MJmP3w',
        'JMPrevBestImprove': 'MJmBi',
        'JMPrev2BestImprove': 'MJmB2i',
        'JMPrev3BestImprove': 'MJmB3i',
        'DIRPrevWinner': 'MDrPw',
        'DIRPrev2Winners': 'MDrP2w',
        'DIRPrev3Winners': 'MDrP3w',
        'DIRPrevBestImprove': 'MDrBi',
        'DIRPrev2BestImprove': 'MDrB2i',
        'DIRPrev3BestImprove': 'MDrB3i',
        'JMOnlyReservoir' : 'MJmOr',
        'DIROnlyReservoir': 'MDrOr',
        'JMPrevWinnerK1': 'MJmPwK1',
        'DIRPrevWinnerK1': 'MDrPwK1',
        'JMPrevWinnerK3': 'MJmPwK3',
        'DIRPrevWinnerK3': 'MDrPwK3',
        'JMPrevWinnerRand': 'MJmPwRd',
        'DIRPrevWinnerRand': 'MDrPwRd',
        'DIRPrevBestImproveK1': 'MDrBiK1',
        'JMPrevBestImproveK1': 'MJmBiK1',
        'DIRPrevBestImproveK3': 'MDrBiK3',
        'JMPrevBestImproveK3': 'MJmBiK3',
        'DIRPrevBestImproveRand': 'MDrBiRd',
        'JMPrevBestImproveRand': 'MJmBiRd',
    }
    for feat in feat_list:
        ret_str += '_' + feat_short_dict[feat]
    return ret_str


def create_cc_dict_with_cw09(
        dataset_name):

    cc_df = pd.read_csv('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + dataset_name + '/All_Collection_Counts.tsv', sep='\t',
                        index_col=False)
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + dataset_name + '/cc_per_interval_dict.json', 'r') as f:
        cc_dict = ast.literal_eval(f.read())

    for index, row in cc_df.iterrows():
        cc_dict[row['Stem']] += row['CollectionCount']

    return cc_dict