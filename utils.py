import os
import ast
import math
import subprocess
import pandas as pd
from scipy import spatial


TREC_EVAL_PATH = "/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/trec_eval/trec_eval-9.0.7/trec_eval"
QRELS_FILE_PATH = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc"


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
            print ("here")
            interval_list[(start_month - 1):]
        else:
            print("there")
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
        interval_lookup_method):
    # gets the right interval snapshot according to lookup method
    doc_interval_dict = doc_dict[interval_list[curr_interval_idx]]
    if doc_interval_dict is None:
        if interval_list[curr_interval_idx] == "ClueWeb09":
            raise Exception("ClueWeb09 needs lookup..")

        if interval_lookup_method == "Forward":
            addition = 1
            while doc_interval_dict is None:
                doc_interval_dict = doc_dict[interval_list[curr_interval_idx + addition]]
                addition += 1

        elif interval_lookup_method == "Backward":
            addition = 1
            while (doc_interval_dict is None) and ((curr_interval_idx - addition) >= 0):
                doc_interval_dict = doc_dict[interval_list[curr_interval_idx - addition]]
                addition += 1
            if doc_interval_dict is None:
                addition = 1
                while doc_interval_dict is None:
                    doc_interval_dict = doc_dict[interval_list[curr_interval_idx + addition]]
                    addition += 1

        elif interval_lookup_method == "OnlyBackward":
            addition = 1
            while (doc_interval_dict is None) and ((curr_interval_idx - addition) >= 0):
                doc_interval_dict = doc_dict[interval_list[curr_interval_idx - addition]]
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
        stemmed_query_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/Stemmed_Query_Words'):
    return pd.read_csv(stemmed_query_file, sep = '\t', index_col = False)

def create_query_to_doc_mapping_df(
        query_to_doc_mapping_file='/mnt/bi-strg3/v/zivvasilisky/ziv/data/all_urls_no_spam_filtered_50_per_query.tsv'):
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
        filename):
    bashCommand = TREC_EVAL_PATH + ' ' + QRELS_FILE_PATH + ' ' + \
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

def create_per_interval_cc_dict(
        cc_dict_file='/mnt/bi-strg3/v/zivvasilisky/ziv/data/cc_per_interval_dict.json',
        interval_freq = None,
        lookup_method = None):

    with open(cc_dict_file, 'r') as f:
        cc_dict = ast.literal_eval(f.read())

    if interval_freq is not None:
        cc_dict = cc_dict[interval_freq]
        if lookup_method is not None:
            cc_dict = cc_dict[lookup_method]
    return  cc_dict

def get_number_of_snapshots_for_doc(
        doc_dict):
    num_of_snapshots = 0
    for snapshot in doc_dict:
        if doc_dict[snapshot] is not None:
            num_of_snapshots += 1
    return num_of_snapshots


