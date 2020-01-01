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
        cc_dict,
        doc_dict,
        mue):

    kl_score = 0.0
    work_stem_list = list(query_stem_dict.keys())

    for stem in work_stem_list:
        doc_stem_tf = 0
        if 'TfDict' in doc_dict:
            if stem in doc_dict['TfDict']:
                doc_stem_tf = float(doc_dict['TfDict'][stem])
        else:
            for i in range(len(doc_dict['StemList'])):
                if doc_dict['StemList'][i] == stem:
                    doc_stem_tf = doc_dict['TfList'][i]

        if stem not in cc_dict:
            # raise Exception('Unexpected Situation')
            cc_dict[stem] = 1

        # if cc_dict[stem] == 0:
        #     continue

        query_tf = 0
        if stem in query_stem_dict:
            query_tf = query_stem_dict[stem]

        stem_q_prob = float(query_tf)/sum(list(query_stem_dict.values()))

        stem_d_proba = get_word_diriclet_smoothed_probability(
            tf_in_doc = doc_stem_tf,
            doc_len = doc_dict['NumWords'],
            collection_count_for_word=cc_dict[stem],
            collection_len=cc_dict['ALL_TERMS_COUNT'],
            mue=mue)

        kl_score += (-1)*stem_q_prob*(math.log((stem_q_prob/stem_d_proba) , 2))

    return kl_score

def get_scored_df_for_query(
        query_num,
        query,
        query_doc_df,
        interval_idx,
        interval_lookup_method,
        processed_docs_dict,
        interval_list,
        cc_dict,
        mue,
        amount_of_snapshot_limit,
        filter_params):

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
            curr_interval_idx=interval_idx,
            filter_params=filter_params)

        if doc_interval_dict is None:
            continue

        # print(doc_interval_dict)
        doc_score = score_doc_for_query(
            query_stem_dict=query_dict,
            cc_dict=cc_dict,
            doc_dict=doc_interval_dict,
            mue=mue)
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
    filter_params = ast.literal_eval(sys.argv[5])
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

    if filter_params is not None and len(filter_params) > 0:
        addition += create_filter_params_txt_addition(filter_params)

    mue = 1000.0
    interval_list = build_interval_list('2008', frequency, add_clueweb = True, start_month=interval_start_month)
    # retrieve necessary dataframes
    query_to_doc_mapping_df = create_query_to_doc_mapping_df()
    stemmed_queries_df = create_stemmed_queries_df()
    # create easy to use index for cc
    cc_dict = create_per_interval_cc_dict(interval_freq=frequency, lookup_method=interval_lookup_method)

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
                cc_dict=cc_dict[interval_list[j]],
                mue=mue,
                amount_of_snapshot_limit=amount_of_snapshot_limit,
                filter_params=filter_params)

            with open(os.path.join(save_folder, str(query_num) + "_" + frequency + '_' + str(interval_list[j] + "_" + interval_lookup_method + addition +"_Results.txt")), 'w') as f:
                f.write(convert_df_to_trec(res_df))
            if interval_list[j] in big_df_dict:
                big_df_dict[interval_list[j]] = big_df_dict[interval_list[j]].append(res_df, ignore_index=True)
            else:
                big_df_dict[interval_list[j]] = res_df

            if interval_list[j] == 'ClueWeb09':
                grep_term = ""
                for docno in res_df['Docno']:
                    grep_term += docno + "\|"
                bashCommand = 'cat /mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/query/query_res/query_' + "0"*(3 - len(str(query_num)))+ str(query_num) +\
                              '_res.txt | grep "' + grep_term[:-2] + '"'
                output = subprocess.check_output(['bash', '-c', bashCommand])
                full_bench += output + '\n'
                with open(os.path.join(save_folder, str(query_num) + "_" + str(interval_list[j] + "_Indri_Out.txt")),
                          'w') as f:
                    f.write(output)

    with open(os.path.join(os.path.dirname(save_folder[:-1]), "ClueWeb09_Indri_Out.txt"),'w') as f:
        f.write(full_bench)
    for interval in interval_list:
        with open(os.path.join(os.path.dirname(save_folder[:-1]), interval +  "_" + frequency + '_' + interval_lookup_method + addition +"_Results.txt"), 'w') as f:
            f.write(convert_df_to_trec(big_df_dict[interval]))

    print("Creating Retrival stats...")
    sys.stdout.flush()
    create_retrieval_stats(
        interval_freq=frequency,
        interval_lookup_method=interval_lookup_method,
        interval_start_month=interval_start_month,
        amount_of_snapshot_limit=amount_of_snapshot_limit,
        addition=addition)