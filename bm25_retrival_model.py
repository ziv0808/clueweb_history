import os
import ast
import sys
import math
import subprocess
import pandas as pd

ALL_WORDS = False
from utils import *
from retrival_stats_creator import create_retrieval_stats

def score_doc_for_query(
        query_stem_dict,
        df_dict,
        doc_dict,
        k1,
        b):

    bm25_score = 0.0
    work_stem_list = list(query_stem_dict.keys())

    for stem in work_stem_list:
        doc_stem_tf = 0
        if 'TfDict' in doc_dict:
            if stem in doc_dict['TfDict']:
                doc_stem_tf = float(doc_dict['TfDict'][stem])

        if stem not in df_dict:
            # raise Exception('Unexpected Situation')
            df_dict[stem] = 1


        idf = math.log(df_dict['ALL_DOCS_COUNT'] / float(df_dict[stem]) , 10)
        stem_d_proba = (doc_stem_tf * (k1 + 1)) / (doc_stem_tf + k1*((1-b) + b*(float(doc_dict['NumWords'])/df_dict['AVG_DOC_LEN']) ))

        bm25_score += idf * stem_d_proba

    return bm25_score

def get_scored_df_for_query(
        query_num,
        query,
        query_doc_df,
        interval_idx,
        interval_lookup_method,
        processed_docs_dict,
        interval_list,
        df_dict,
        k1,
        b,
        amount_of_snapshot_limit):

    res_df= pd.DataFrame(columns = ['Query_ID','Iteration', 'Docno', 'Rank', 'Score', 'Method'])
    next_index = 0
    query_dict = convert_query_to_tf_dict(query)
    for index, row in query_doc_df.iterrows():
        docno = row['Docno']
        # retrive docno dict
        doc_dict = processed_docs_dict[docno]
        if amount_of_snapshot_limit is not None:
            if amount_of_snapshot_limit > get_number_of_snapshots_for_doc(doc_dict):
                continue
        # find the interval to look for
        doc_interval_dict = get_doc_snapshot_by_lookup_method(
            doc_dict=doc_dict,
            interval_list=interval_list,
            interval_lookup_method=interval_lookup_method,
            curr_interval_idx=interval_idx)

        if doc_interval_dict is None:
            continue

        # print(doc_interval_dict)
        doc_score = score_doc_for_query(
            query_stem_dict=query_dict,
            df_dict=df_dict,
            doc_dict=doc_interval_dict,
            k1=k1,
            b=b)
        res_df.loc[next_index] = ["0"*(3 - len(str(query_num)))+ str(query_num), 'Q0', docno, 0, doc_score, 'indri']
        next_index += 1
    if res_df.empty == False:
        res_df.sort_values('Score', ascending=False, inplace=True)
        res_df['Rank'] = list(range(1, next_index + 1))
    return res_df


if __name__=='__main__':
    frequency = sys.argv[1]
    interval_lookup_method = sys.argv[2]
    interval_start_month = int(sys.argv[3])
    amount_of_snapshot_limit = ast.literal_eval(sys.argv[4])
    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/2008/' +frequency + '/'
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ranked_docs/'
    addition = ""
    print('Interval Feaq: ' + frequency)
    print('Lookup method: ' + interval_lookup_method)
    print ("interval_start_month: " +str(interval_start_month))
    if interval_start_month != 1:
        addition = "_" + str(interval_start_month) + "SM_"

    if amount_of_snapshot_limit is not None and amount_of_snapshot_limit > 1:
        addition += "_SnapLimit_" + str(amount_of_snapshot_limit)

    k1 = 1
    b = 0.5
    interval_list = build_interval_list('2008', frequency, add_clueweb = True, start_month=interval_start_month)
    # retrieve necessary dataframes
    query_to_doc_mapping_df = create_query_to_doc_mapping_df()
    stemmed_queries_df = create_stemmed_queries_df()
    # create easy to use index for cc
    df_dict = create_per_interval_df_dict(interval_freq=frequency, lookup_method=interval_lookup_method)

    big_df_dict = {}
    full_bench = ""
    for index, row in stemmed_queries_df.iterrows():
        query_num = int(row['QueryNum'])
        print("Query: " + str(query_num))
        sys.stdout.flush()
        query_txt = row['QueryStems']
        relevant_df = query_to_doc_mapping_df[query_to_doc_mapping_df['QueryNum'] == query_num].copy()
        relevant_doc_dict = {}
        for index, row in relevant_df.iterrows():
            docno_ = row['Docno']
            # retrive docno dict
            with open(os.path.join(processed_docs_folder, docno_ + '.json'), 'r') as f:
                doc_dict = ast.literal_eval(f.read())
            relevant_doc_dict[docno_] = doc_dict
        for j in range(len(interval_list)):
            print("Interval: " + str(interval_list[j]))
            sys.stdout.flush()
            res_df = get_scored_df_for_query(
                query_num=query_num,
                query=query_txt,
                query_doc_df=relevant_df,
                interval_idx=j,
                interval_list=interval_list,
                interval_lookup_method=interval_lookup_method,
                processed_docs_dict=relevant_doc_dict,
                df_dict=df_dict[interval_list[j]],
                k1=k1,
                b=b,
                amount_of_snapshot_limit=amount_of_snapshot_limit)

            with open(os.path.join(save_folder, 'BM25_' + str(query_num) + "_" + frequency + '_' + str(interval_list[j] + "_" + interval_lookup_method + addition +"_Results.txt")), 'w') as f:
                f.write(convert_df_to_trec(res_df))
            if interval_list[j] in big_df_dict:
                big_df_dict[interval_list[j]] = big_df_dict[interval_list[j]].append(res_df, ignore_index=True)
            else:
                big_df_dict[interval_list[j]] = res_df

    for interval in interval_list:
        with open(os.path.join(os.path.dirname(save_folder[:-1]), 'BM25_' + interval +  "_" + frequency + '_' + interval_lookup_method + addition +"_Results.txt"), 'w') as f:
            f.write(convert_df_to_trec(big_df_dict[interval]))

    print("Creating Retrival stats...")
    sys.stdout.flush()
    create_retrieval_stats(
        interval_freq=frequency,
        interval_lookup_method=interval_lookup_method,
        interval_start_month=interval_start_month,
        amount_of_snapshot_limit=amount_of_snapshot_limit,
        is_bm25=True)