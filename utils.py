import os
import ast
import math
import pandas as pd
from scipy import spatial


def build_interval_list(
        work_year,
        frequency,
        add_clueweb = False,
        start_month = 1,
        end_month   = 12):
    # create interval list for work year according to required frequency
    interval_list = []
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

    if add_clueweb == True:
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
        stemmed_query_collection_counts='/lv_local/home/zivvasilisky/ziv/data/StemsCollectionCounts.tsv'):
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
        stemmed_query_file = '/lv_local/home/zivvasilisky/ziv/data/Stemmed_Query_Words'):
    return pd.read_csv(stemmed_query_file, sep = '\t', index_col = False)

def create_query_to_doc_mapping_df(
        query_to_doc_mapping_file='/lv_local/home/zivvasilisky/ziv/data/all_urls_no_spam_filtered.tsv'):
    return pd.read_csv(query_to_doc_mapping_file, sep = '\t', index_col = False)

def convert_str_to_bool(strng):
    if strng == 'False':
        return False
    elif strng == 'True':
        return True
    else:
        raise Exception("convert_str_to_bool: Not supported string")