import os
import ast
import sys
import math
import subprocess
import pandas as pd

from utils import *
BM25 = True

def score_doc_for_query_lm(
        query_stem_dict,
        cc_dict,
        doc_dict,
        interval_list,
        params):
    mue = params['Mue']
    kl_score = 0.0
    work_stem_list = list(query_stem_dict.keys())

    for stem in work_stem_list:
        doc_stem_tf = 0
        doc_len = 0
        active_intervals = 0.0
        for interval in interval_list:
            if doc_dict[interval] is not None:
                doc_len += float(doc_dict[interval]['NumWords'])
                active_intervals += 1
                if stem in doc_dict[interval]['TfDict']:
                    doc_stem_tf += float(doc_dict[interval]['TfDict'][stem])

        query_tf = 0
        if stem in query_stem_dict:
            query_tf = query_stem_dict[stem]

        stem_q_prob = float(query_tf)/sum(list(query_stem_dict.values()))

        stem_d_proba = get_word_diriclet_smoothed_probability(
            tf_in_doc = doc_stem_tf/active_intervals,
            doc_len = doc_len/active_intervals,
            collection_count_for_word=cc_dict[stem],
            collection_len=cc_dict['ALL_TERMS_COUNT'],
            mue=mue)

        kl_score += (-1)*stem_q_prob*(math.log((stem_q_prob/stem_d_proba) , 2))

    return kl_score


def score_doc_for_query_bm25(
        query_stem_dict,
        df_dict,
        doc_dict,
        interval_list,
        params,
        filter_params):
    k1 = params['K1']
    b = params['b']
    bm25_score = 0.0
    work_stem_list = list(query_stem_dict.keys())

    for stem in work_stem_list:
        doc_stem_tf = 0
        doc_len = 0.0
        avg_doc_len = 0.0
        active_intervals = 0.0
        for interval in interval_list:
            if verify_snapshot_for_doc(doc_dict,interval,filter_params) is not None:
                doc_len += float(doc_dict[interval]['NumWords'])
                avg_doc_len += df_dict[interval]['AVG_DOC_LEN']
                active_intervals += 1
                if stem in doc_dict[interval]['TfDict']:
                    doc_stem_tf += float(doc_dict[interval]['TfDict'][stem])


        if stem not in df_dict:
            # raise Exception('Unexpected Situation')
            df_dict[stem] = 1

        doc_stem_tf = doc_stem_tf/float(active_intervals)
        doc_len = doc_len/float(active_intervals)
        avg_doc_len = avg_doc_len/float(active_intervals)
        all_docs_count = df_dict['ClueWeb09']['ALL_DOCS_COUNT']
        stem_df =  df_dict['ClueWeb09'][stem]
        idf = math.log(all_docs_count / float(stem_df), 10)
        stem_d_proba = (doc_stem_tf * (k1 + 1)) / (
        doc_stem_tf + k1 * ((1 - b) + b * (float(doc_len) / avg_doc_len)))

        bm25_score += idf * stem_d_proba

    return bm25_score



def get_scored_df_for_query(
        query_num,
        query,
        query_doc_df,
        processed_docs_path,
        interval_list,
        cc_dict,
        params,
        filter_params,
        amount_of_snapshot_limit):

    res_df= pd.DataFrame(columns = ['Query_ID','Iteration', 'Docno', 'Rank', 'Score', 'Method'])
    next_index = 0
    query_dict = convert_query_to_tf_dict(query)
    for index, row in query_doc_df.iterrows():
        docno = row['Docno']
        # retrive docno dict
        with open(os.path.join(processed_docs_path, docno + '.json'), 'r') as f:
            doc_dict = ast.literal_eval(f.read())

        if amount_of_snapshot_limit is not None:
            if amount_of_snapshot_limit > get_number_of_snapshots_for_doc(doc_dict):
                continue

        if BM25 == True:
            doc_score = score_doc_for_query_bm25(
                query_stem_dict = query_dict,
                df_dict = cc_dict,
                doc_dict = doc_dict,
                interval_list =interval_list,
                params= params,
                filter_params=filter_params)
        else:
            # print(doc_interval_dict)
            doc_score = score_doc_for_query_lm(
                query_stem_dict=query_dict,
                cc_dict=cc_dict,
                doc_dict=doc_dict,
                interval_list=interval_list,
                params=params)
        res_df.loc[next_index] = ["0"*(3 - len(str(query_num)))+ str(query_num), 'Q0', docno, 0, doc_score, 'indri']
        next_index += 1

    if res_df.empty == False:
        res_df.sort_values('Score', ascending=False, inplace=True)
        res_df['Rank'] = list(range(1, next_index + 1))
    return res_df


if __name__=='__main__':
    frequency = sys.argv[1]
    # interval_start_month = int(sys.argv[2])
    amount_of_snapshot_limit = ast.literal_eval(sys.argv[2])
    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/2008/' +frequency + '/'
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/concat_model_res/'
    if BM25 == True:
        cc_dict = create_per_interval_df_dict(interval_freq=frequency, lookup_method='NoLookup')
        params = {'K1' : 1 , 'b' : 0.5}
        affix = "BM25_"
    else:
        # create easy to use index for cc
        cc_dict = create_per_interval_cc_dict(interval_freq=frequency, lookup_method='NoLookup')
        params = {'Mue' : 1000.0}
        affix  = ""

    filter_params_options = {
        "Sim_Upper" : [None, 0.999, 0.99, 0.98, 0.97],
        "Sim_Lower" : [None, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8],
        "TxtDiff_Upper" : [None, 500, 1000, 2000, 3000, 5000],
        "TxtDiff_Lower" : [None, -500, -1000, -2000, -3000, -5000],
        "%TxtDiff_Upper" : [None,0.4,0.5, 0.6, 0.8, 1.0],
        "%TxtDiff_Lower" : [None, -0.4, -0.5, -0.6, -0.8, -1.0]}

    curr_filter_params = {
        "Sim_Upper": None,
        "Sim_Lower": None,
        "TxtDiff_Upper": None,
        "TxtDiff_Lower": None,
        "%TxtDiff_Upper": None,
        "%TxtDiff_Lower": None}

    best_map = 0.0
    for param_kind in list(filter_params_options.keys()):
        print (param_kind )
        sys.stdout.flush()
        best = None
        for param_option in filter_params_options[param_kind]:
            curr_filter_params[param_kind] = param_option

            addition = ""
            print('Interval Freq: ' + frequency)

            interval_list = build_interval_list('2008', frequency, add_clueweb = True, start_month=1)
            # retrieve necessary dataframes
            query_to_doc_mapping_df = create_query_to_doc_mapping_df()
            stemmed_queries_df = create_stemmed_queries_df()

            big_df = pd.DataFrame({})
            for index, row in stemmed_queries_df.iterrows():
                query_num = int(row['QueryNum'])
                print("Query: " + str(query_num))
                sys.stdout.flush()
                query_txt = row['QueryStems']
                relevant_df = query_to_doc_mapping_df[query_to_doc_mapping_df['QueryNum'] == query_num].copy()
                res_df = get_scored_df_for_query(
                    query_num=query_num,
                    query=query_txt,
                    query_doc_df=relevant_df,
                    interval_list=interval_list,
                    processed_docs_path=processed_docs_folder,
                    cc_dict=cc_dict,
                    params=params,
                    filter_params=curr_filter_params,
                    amount_of_snapshot_limit=amount_of_snapshot_limit)

                big_df = big_df.append(res_df, ignore_index=True)


            with open(os.path.join(save_folder,affix + frequency + '_' + addition + "_Results.txt"), 'w') as f:
                f.write(convert_df_to_trec(big_df))

            res_dict = get_ranking_effectiveness_for_res_file(file_path=save_folder,filename=affix +frequency + '_' + addition + "_Results.txt")
            print(curr_filter_params)
            print("SCORE : " + str(res_dict))
            sys.stdout.flush()
            if res_dict['Map'] > best_map:
                best_map = res_dict['Map']
                best = param_option

        curr_filter_params[param_kind] = best