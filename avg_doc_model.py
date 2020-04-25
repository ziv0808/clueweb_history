import os
import ast
import sys
import math
import datetime
import subprocess
import pandas as pd
import numpy as np

from utils import *

def score_doc_for_query_lm(
        query_stem_dict,
        cc_dict,
        doc_dict,
        params):
    mue = params['Mue']
    kl_score = 0.0
    work_stem_list = list(query_stem_dict.keys())
    q_normalizer = sum(list(query_stem_dict.values()))

    for stem in work_stem_list:
        doc_stem_tf = float(doc_dict[stem])
        doc_len = doc_dict['NumWords']
        query_tf = query_stem_dict[stem]

        stem_q_prob = float(query_tf)/q_normalizer
        collection_len = cc_dict['ALL_TERMS_COUNT']
        collection_count_for_word = cc_dict[stem]

        stem_d_proba = get_word_diriclet_smoothed_probability(
            tf_in_doc=doc_stem_tf ,
            doc_len=doc_len,
            collection_count_for_word=collection_count_for_word,
            collection_len=collection_len,
            mue=mue)

        kl_score += (-1)*stem_q_prob*(math.log((stem_q_prob/stem_d_proba) , 2))

    return kl_score


def score_doc_for_query_bm25(
        query_stem_dict,
        df_dict,
        doc_dict,
        params):

    k1 = params['K1']
    b = params['b']
    bm25_score = 0.0
    work_stem_list = list(query_stem_dict.keys())

    for stem in work_stem_list:
        doc_stem_tf = float(doc_dict[stem])
        doc_len = float(doc_dict['NumWords'])

        all_docs_count = df_dict['ALL_DOCS_COUNT']
        if stem in df_dict:
            stem_df = df_dict[stem]
        else:
            raise Exception('Problem!')
        avg_doc_len = df_dict['AVG_DOC_LEN']

        idf = math.log(all_docs_count / float(stem_df), 10)
        stem_d_proba = (doc_stem_tf * (k1 + 1)) / (
        doc_stem_tf + k1 * ((1 - b) + b * (float(doc_len) / avg_doc_len)))

        bm25_score += idf * stem_d_proba

    return bm25_score


def get_scored_df_for_query(
        query_num,
        query,
        query_doc_df,
        cc_dict,
        params,
        model,
        tmp_weight_df,
        all_global_params_dict
        ):

    res_df= pd.DataFrame(columns = ['Query_ID','Iteration', 'Docno', 'Rank', 'Score', 'Method'])
    next_index = 0
    query_dict = convert_query_to_tf_dict(query)
    for index, row in query_doc_df.iterrows():
        docno = row['Docno']
        # retrive docno dict
        doc_dict = {}
        for stem_ in list(query_dict.keys()) + ['NumWords']:
            doc_dict[stem_] = np.sum(all_global_params_dict[stem_].loc[docno] * tmp_weight_df.loc[docno])
            if np.isnan(doc_dict[stem_]):
                raise Exception('Problem 222 !')

        if model == 'BM25':
            doc_score = score_doc_for_query_bm25(
                query_stem_dict = query_dict,
                df_dict = cc_dict,
                doc_dict = doc_dict,
                params= params)
        else:
            # print(doc_interval_dict)
            doc_score = score_doc_for_query_lm(
                query_stem_dict=query_dict,
                cc_dict=cc_dict,
                doc_dict=doc_dict,
                params=params)

        res_df.loc[next_index] = ["0"*(3 - len(str(query_num)))+ str(query_num), 'Q0', docno, 0, doc_score, 'indri']
        next_index += 1

    if res_df.empty == False:
        res_df.sort_values('Score', ascending=False, inplace=True)
        res_df['Rank'] = list(range(1, next_index + 1))
    return res_df

def test_queries(
    stemmed_queries_df,
    query_to_doc_mapping_df,
    cc_dict,
    params,
    retrival_model,
    tmp_weight_df,
    all_global_params_dict):

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
            cc_dict=cc_dict,
            params=params,
            model=retrival_model,
            tmp_weight_df=tmp_weight_df,
            all_global_params_dict=all_global_params_dict)

        big_df = big_df.append(res_df, ignore_index=True)

    return big_df

def create_cached_docs_data(
        interval_list,
        stemmed_queires_df,
        query_to_doc_mapping_df,
        processed_docs_path,
        filter_params,
        amount_of_snapshot_limit):

    all_global_params_dict = {}
    all_query_stems = []
    for index, row in stemmed_queires_df.iterrows():
        query_stems = row['QueryStems'].split(' ')
        all_query_stems.extend(query_stems)

    all_query_stems = list(set(all_query_stems))
    for query_stem in all_query_stems + ['NumWords']:
        all_global_params_dict[query_stem] = pd.DataFrame(columns = interval_list)

    for index, row in stemmed_queires_df.iterrows():
        query_num = int(row['QueryNum'])
        tmp_doc_df = query_to_doc_mapping_df[query_to_doc_mapping_df['QueryNum'] == query_num].copy()
        for index, row in tmp_doc_df.iterrows():
            docno = row['Docno']
            if docno in all_global_params_dict['NumWords'].index:
                continue
            with open(os.path.join(processed_docs_path, docno + '.json'), 'r') as f:
                doc_dict = ast.literal_eval(f.read())

            if amount_of_snapshot_limit is not None:
                if amount_of_snapshot_limit > get_number_of_snapshots_for_doc(doc_dict):
                    continue

            tmp_stem_dict = {}
            for query_stem in all_query_stems:
                tmp_stem_dict[query_stem] = []
            tmp_stem_dict['NumWords'] = []

            for interval in interval_list:
                if verify_snapshot_for_doc(doc_dict, interval, filter_params) is not None:
                    for query_stem in all_query_stems:
                        if query_stem in doc_dict[interval]['TfDict']:
                            tmp_stem_dict[query_stem].append(float(doc_dict[interval]['TfDict'][query_stem]))
                        else:
                            tmp_stem_dict[query_stem].append(0.0)
                    tmp_stem_dict['NumWords'].append(float(doc_dict[interval]['NumWords']))
                else:
                    for query_stem in all_query_stems + ['NumWords']:
                        tmp_stem_dict[query_stem].append(pd.np.nan)

            for query_stem in all_query_stems + ['NumWords']:
                all_global_params_dict[query_stem].loc[docno] = tmp_stem_dict[query_stem]

    return all_global_params_dict

def create_weights_matrix(
        all_weights_df,
        weight_list,
        cw_interval_weight):

    tmp_weights_df = all_weights_df * weight_list
    normalize_factor = np.sum(tmp_weights_df.values, axis=1)
    normalize_factor = normalize_factor.reshape((len(tmp_weights_df), 1))
    tmp_weights_df = tmp_weights_df / normalize_factor

    tmp_weights_df = tmp_weights_df * (1.0 - cw_interval_weight)
    tmp_weights_df['ClueWeb09'] = [cw_interval_weight] * len(tmp_weights_df)
    return tmp_weights_df

def create_cc_df_dict(
        tmp_weights_df,
        all_global_params_dict,
        retrival_model):

    df_cc_dict = {}
    for key_ in all_global_params_dict:
        if key_ == 'NumWords':
            tmp_mtrx = np.sum(all_global_params_dict[key_] * tmp_weights_df, axis = 1)
            if retrival_model == 'LM':
                df_cc_dict['ALL_TERMS_COUNT'] =  np.sum(tmp_mtrx)
            elif retrival_model == 'BM25':
                df_cc_dict['AVG_DOC_LEN'] = np.mean(tmp_mtrx)
                df_cc_dict['ALL_DOCS_COUNT'] = len(tmp_mtrx)
        else:
            tmp_mtrx = np.sum(all_global_params_dict[key_] * tmp_weights_df, axis=1)
            if retrival_model == 'LM':
                df_cc_dict[key_] = np.sum(tmp_mtrx)
            elif retrival_model == 'BM25':
                df_cc_dict[key_] = 0.0
                for val_ in list(tmp_mtrx.values):
                    if val_ > 0.0:
                        df_cc_dict[key_] += 1
    return df_cc_dict

def make_weight_list_options(
        ignore_idxs,
        interval_list,
        optional_decay_weights_k):
    weight_list_options = []
    weight_list = create_uniform_wieghts_list(
        interval_list=interval_list,
        skip_idx_list=ignore_idxs)
    weight_list_options.append(weight_list)
    if len(ignore_idxs) < (len(interval_list) - 1):
        for k in optional_decay_weights_k:
            weight_list = create_decaying_wieghts_list(
                interval_list = interval_list,
                decaying_factor = k,
                skip_idx_list = ignore_idxs,
                reverse = True)
            weight_list_options.append(weight_list)
            weight_list = create_decaying_wieghts_list(
                interval_list=interval_list,
                decaying_factor=k,
                skip_idx_list=ignore_idxs,
                reverse=False)
            weight_list_options.append(weight_list)
    return weight_list_options

def run_test_on_config(
        weight_list,
        tmp_weights_df,
        stemmed_queries_df,
        query_to_doc_mapping_df,
        all_global_params_dict,
        retrival_model,
        params,
        save_folder,
        affix,
        frequency,
        addition,
        cw_interval_weight):
    df_cc_dict = create_cc_df_dict(
        tmp_weights_df=tmp_weights_df,
        all_global_params_dict=all_global_params_dict,
        retrival_model=retrival_model)

    big_df = test_queries(
        stemmed_queries_df=stemmed_queries_df,
        query_to_doc_mapping_df=query_to_doc_mapping_df,
        cc_dict=df_cc_dict,
        params=params,
        retrival_model=retrival_model,
        tmp_weight_df=tmp_weights_df,
        all_global_params_dict=all_global_params_dict)

    cur_time = str(datetime.datetime.now())
    curr_file_name = cur_time.replace(' ', '_') + '_' + affix + frequency + '_' + addition + "_Results.txt"
    with open(os.path.join(save_folder + 'inner_res/', curr_file_name), 'w') as f:
        f.write(convert_df_to_trec(big_df))

    res_dict = get_ranking_effectiveness_for_res_file(
        file_path=save_folder + 'inner_res/',
        filename=curr_file_name)
    weight_list = weight_list / np.sum(weight_list)
    weight_list = weight_list * (1 - cw_interval_weight)
    weight_list[-1] = cw_interval_weight

    return res_dict, weight_list

if __name__=='__main__':
    frequency = sys.argv[1]
    interval_start_month = int(sys.argv[2])
    amount_of_snapshot_limit = ast.literal_eval(sys.argv[3])
    retrival_model = sys.argv[4]
    sw_rmv = ast.literal_eval(sys.argv[5])
    filter_params = ast.literal_eval(sys.argv[6])
    run_cv = ast.literal_eval(sys.argv[7])

    if run_cv == True:
        start_test_q = int(sys.argv[8])
        end_test_q = int(sys.argv[9])
        affix = str(start_test_q) + '_' + str(end_test_q) + "_"
    else:
        affix = ""

    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+INNER_FOLD+'/'+WORK_YEAR+'/' +frequency + '/'
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/avg_model_res/'

    if retrival_model == 'BM25':
        params = {'K1' : 1 , 'b' : 0.5}
        affix += "BM25_"

    elif retrival_model == 'LM' :
        params = {'Mue' : 1000.0}
        affix  += ""
    else:
        raise Exception("Unknown model")

    query_to_doc_mapping_df = create_query_to_doc_mapping_df()
    stemmed_queries_df = create_stemmed_queries_df(sw_rmv=sw_rmv)

    wieght_for_last_interval_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    optional_decay_weights_k = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    if run_cv == True:
        addition = ""
        if interval_start_month != 1:
            addition = "_" + str(interval_start_month) + "SM_"

        if filter_params is not None and len(filter_params) > 0:
            addition += create_filter_params_txt_addition(filter_params)

        if sw_rmv == True:
            addition += "_SW_RMV"

        interval_list = build_interval_list(WORK_YEAR, frequency, add_clueweb=True, start_month=interval_start_month)
        cv_summary_df = pd.DataFrame(columns=interval_list + ['Map', 'P@5', 'P@10'])
        next_idx = 0
        all_global_params_dict = create_cached_docs_data(
            interval_list=interval_list,
            stemmed_queires_df=stemmed_queries_df,
            query_to_doc_mapping_df=query_to_doc_mapping_df,
            processed_docs_path=processed_docs_folder,
            filter_params=filter_params,
            amount_of_snapshot_limit=amount_of_snapshot_limit
            )

        test_set_q = list(range(start_test_q, end_test_q + 1))
        stemmed_queries_df['IsTest'] = stemmed_queries_df['QueryNum'].apply(lambda x: 1 if x in test_set_q else 0)
        test_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 1].copy()
        stemmed_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 0]

        all_weights_df = all_global_params_dict['NumWords'].copy()
        all_weights_df = all_weights_df.applymap(lambda x: 0 if pd.np.isnan(x) else 1)
        for element in all_global_params_dict:
            all_global_params_dict[element] = all_global_params_dict[element].applymap(lambda x: 0.0 if np.isnan(x) else x)

        best_map = 0.0
        best_config = {'WList' : None, 'WDf' : None}
        for i in range(len(interval_list) - 1):
            for cw_interval_weight in wieght_for_last_interval_options:
                ignore_idxs = list(range(i))
                # run uniform weights logic
                ignore_idxs.append(len(interval_list) - 1)
                weight_list_options = make_weight_list_options(
                    ignore_idxs=ignore_idxs,
                    interval_list=interval_list,
                    optional_decay_weights_k=optional_decay_weights_k)
                for weight_list in weight_list_options:
                    tmp_weights_df = create_weights_matrix(
                        all_weights_df=all_weights_df,
                        weight_list=weight_list,
                        cw_interval_weight=cw_interval_weight)

                    res_dict, weight_list = run_test_on_config(
                        weight_list=weight_list,
                        tmp_weights_df=tmp_weights_df,
                        stemmed_queries_df=stemmed_queries_df,
                        query_to_doc_mapping_df=query_to_doc_mapping_df,
                        all_global_params_dict=all_global_params_dict,
                        retrival_model=retrival_model,
                        params=params,
                        save_folder=save_folder,
                        affix=affix,
                        frequency=frequency,
                        addition=addition,
                        cw_interval_weight=cw_interval_weight)

                    print(affix + addition + " " + str(weight_list))
                    print(affix + addition + " " + "SCORE : " + str(res_dict))
                    sys.stdout.flush()

                    insert_row = list(weight_list) + [res_dict['Map'], res_dict['P_5'], res_dict['P_10']]
                    cv_summary_df.loc[next_idx] = insert_row
                    next_idx += 1

                    if res_dict['Map'] > best_map:
                        best_map = res_dict['Map']
                        best_config['WList'] = weight_list
                        best_config['WDf'] = tmp_weights_df

            cv_summary_df.to_csv(os.path.join(save_folder, affix + frequency + '_' + addition + "_Results.tsv"), sep = '\t', index=False)
        # test the benchmark case - only last interval
        tmp_weights_df = tmp_weights_df.applymap(lambda x: 0.0)
        tmp_weights_df['ClueWeb09'] = 1.0
        weight_list = np.array([0.0] * len(weight_list))
        weight_list[-1] = 1.0
        res_dict, weight_list = run_test_on_config(
            weight_list=weight_list,
            tmp_weights_df=tmp_weights_df,
            stemmed_queries_df=stemmed_queries_df,
            query_to_doc_mapping_df=query_to_doc_mapping_df,
            all_global_params_dict=all_global_params_dict,
            retrival_model=retrival_model,
            params=params,
            save_folder=save_folder,
            affix=affix,
            frequency=frequency,
            addition=addition,
            cw_interval_weight=1.0)
        print(affix + addition + " " + str(weight_list))
        print(affix + addition + " " + "SCORE : " + str(res_dict))
        sys.stdout.flush()

        insert_row = list(weight_list) + [res_dict['Map'], res_dict['P_5'], res_dict['P_10']]
        cv_summary_df.loc[next_idx] = insert_row
        next_idx += 1

        if res_dict['Map'] > best_map:
            best_map = res_dict['Map']
            best_config['WList'] = weight_list
            best_config['WDf'] = tmp_weights_df

        # run on test set
        res_dict, weight_list = run_test_on_config(
            weight_list=best_config['WList'],
            tmp_weights_df=best_config['WDf'],
            stemmed_queries_df=test_queries_df,
            query_to_doc_mapping_df=query_to_doc_mapping_df,
            all_global_params_dict=all_global_params_dict,
            retrival_model=retrival_model,
            params=params,
            save_folder=save_folder,
            affix=affix,
            frequency=frequency,
            addition=addition,
            cw_interval_weight=best_config['WList'][-1])
        # last row is the test result
        insert_row = list(best_config['WList']) + [res_dict['Map'], res_dict['P_5'], res_dict['P_10']]
        cv_summary_df.loc[next_idx] = insert_row
        next_idx += 1
        cv_summary_df.to_csv(os.path.join(save_folder, affix + frequency + '_' + addition + "_Results.tsv"), sep='\t',
                             index=False)
    else:
        pass
