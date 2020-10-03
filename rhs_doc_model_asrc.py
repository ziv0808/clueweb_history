import os
import ast
import sys
import math
import datetime
import subprocess
import pandas as pd
import numpy as np

from avg_doc_model import create_cached_docs_data, test_queries, create_weights_matrix, create_cc_df_dict, make_weight_list_options
from utils import *


def run_test_on_config(
        weight_list,
        tmp_weights_df,
        stemmed_queries_df,
        query_to_doc_mapping_df,
        all_global_params_dict,
        retrival_model,
        params,
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

    weight_list = weight_list / np.sum(weight_list)
    weight_list = weight_list * (1 - cw_interval_weight)
    weight_list[-1] = cw_interval_weight

    return big_df, weight_list

def get_score_retrieval_score_for_df(
        affix,
        big_df,
        save_folder,
        qrel_filepath):

    curr_file_name =  affix + "_Results.txt"
    with open(os.path.join(save_folder + 'inner_res/', curr_file_name), 'w') as f:
        f.write(convert_df_to_trec(big_df))

    res_dict = get_ranking_effectiveness_for_res_file_per_query(
        file_path=save_folder + 'inner_res/',
        filename=curr_file_name,
        qrel_filepath=qrel_filepath,
        calc_ndcg_mrr=True)

    return res_dict

def create_global_weight_list(
        interval_list,
        alpha):
    weight_list = []
    for j in range(1, len(interval_list) +1):
        weight_list.append(1.0 / float(j**(alpha)))
    return np.array(weight_list)

def create_burst_weight_matrix(
        interval_list,
        beta):
    weight_matrix = np.zeros((len(interval_list), len(interval_list)))
    for i in range(len(interval_list)):
        for j in range(len(interval_list)):
            if i<=j:
                weight_matrix[i,j] = (1.0) / float((j - i + 1)**beta)
    return weight_matrix

def create_burst_indication_matrix(
        doc_version_len_df,
        interval_list,
        alpha_for_burst):
    res_df = pd.DataFrame(columns = interval_list)
    for docno in doc_version_len_df.index:
        row_vals = doc_version_len_df.loc[docno].values
        insert_row = [0.0]
        for i in range(1, len(row_vals)):
            if float(row_vals[i] - row_vals[i - 1])/float(row_vals[i - 1]) > alpha_for_burst:
                insert_row.append(1.0)
            else:
                insert_row.append(0.0)
        res_df.loc[docno] = insert_row

    return res_df

def create_df_cc_dict(
        all_global_params_dict,
        retrival_model):
    res_dict = {}
    if retrival_model == 'LM':
        for query_stem in all_global_params_dict:
            res_dict[query_stem] = all_global_params_dict[query_stem]['ClueWeb09'].sum()
        res_dict['ALL_TERMS_COUNT'] = res_dict['NumWords']
        del res_dict['NumWords']
    elif retrival_model == 'BM25':
        res_dict['AVG_DOC_LEN'] = all_global_params_dict['NumWords']['ClueWeb09'].mean()
        res_dict['ALL_DOCS_COUNT'] = len(all_global_params_dict['NumWords'])
        for query_stem in all_global_params_dict:
            res_dict[query_stem] = np.sum(all_global_params_dict[query_stem]['ClueWeb09'].apply(lambda x: 1 if x > 0 else 0))
    else:
        raise Exception("Unknown Ret Model !")

    return res_dict


def get_results_for_params(
        stemmed_queries_df,
        all_query_doc_df,
        all_global_params_dict,
        interval_list,
        df_cc_dict,
        retrieval_model,
        ret_model_params,
        alpha_for_burst,
        alpha_decay,
        beta_decay,
        lambda_dict,
        burst_inidcation_mat = None):

    global_w_list = create_global_weight_list(
        interval_list=interval_list,
        alpha=alpha_decay)

    burst_w_matrix = create_burst_weight_matrix(
        interval_list=interval_list,
        beta=beta_decay)

    if burst_inidcation_mat is None:
        burst_inidcation_mat = create_burst_indication_matrix(
            doc_version_len_df=all_global_params_dict['NumWords'],
            interval_list=interval_list,
            alpha_for_burst=alpha_for_burst)

    big_df = pd.DataFrame({})
    for index, row in stemmed_queries_df.iterrows():
        query_num = int(row['QueryNum'])
        print("Query: " + str(query_num))
        sys.stdout.flush()
        query_txt = row['QueryStems']
        relevant_df = all_query_doc_df[all_query_doc_df['QueryNum'] == query_num].copy()
        res_df = get_scored_df_for_query(
            query_num=query_num,
            query=query_txt,
            query_doc_df=relevant_df,
            cc_df_dict=df_cc_dict,
            params=ret_model_params,
            model=retrieval_model,
            all_global_params_dict=all_global_params_dict,
            global_w_list=global_w_list,
            burst_w_matrix=burst_w_matrix,
            burst_inidcation_mat=burst_inidcation_mat,
            lambda_dict=lambda_dict)

        big_df = big_df.append(res_df, ignore_index=True)

    return big_df

def get_scored_df_for_query(
        query_num,
        query,
        query_doc_df,
        cc_df_dict,
        params,
        model,
        all_global_params_dict,
        global_w_list,
        burst_w_matrix,
        burst_inidcation_mat,
        lambda_dict):

    res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])
    next_index = 0
    query_dict = convert_query_to_tf_dict(query)
    for index, row in query_doc_df.iterrows():
        docno = row['Docno']

        if model == 'BM25':
            doc_dict = {'TfDict' : {}}
            doc_dict['NumWords'] = float(all_global_params_dict['NumWords'].loc[docno]['ClueWeb09'])
            for stem_ in list(query_dict.keys()):
                global_stem_tf = np.sum(all_global_params_dict[stem_].loc[docno] * global_w_list)
                burst_stem_tf  = np.sum(np.matmul(burst_inidcation_mat.loc[docno].values, burst_w_matrix)*all_global_params_dict[stem_].loc[docno].values)
                reg_stem_tf = float(all_global_params_dict[stem_].loc[docno]['ClueWeb09'])
                doc_dict['TfDict'][stem_] = lambda_dict['Global']*global_stem_tf + lambda_dict['Burst']*burst_stem_tf + lambda_dict['Reg']*reg_stem_tf

            doc_score = bm25_score_doc_for_query(
                query_stem_dict = query_dict,
                df_dict = cc_df_dict,
                doc_dict = doc_dict,
                k1=params['K1'],
                b=params['b'])
        else:
            kl_score = 0.0
            doc_len = float(all_global_params_dict['NumWords'].loc[docno]['ClueWeb09'])
            global_denom_tf = np.sum(all_global_params_dict['NumWords'].loc[docno] * global_w_list)
            burst_denom_tf = np.sum(np.matmul(burst_inidcation_mat.loc[docno].values, burst_w_matrix) *
                                   all_global_params_dict['NumWords'].loc[docno].values)
            for stem_ in list(query_dict.keys()):
                stem_q_prob = float(query_dict[stem_]) / sum(list(query_dict.values()))

                global_stem_tf = np.sum(all_global_params_dict[stem_].loc[docno] * global_w_list)
                burst_stem_tf = np.sum(np.matmul(burst_inidcation_mat.loc[docno].values, burst_w_matrix) *
                                       all_global_params_dict[stem_].loc[docno].values)
                reg_stem_tf = float(all_global_params_dict[stem_].loc[docno]['ClueWeb09'])
                stem_reg_proba = get_word_diriclet_smoothed_probability(
                    tf_in_doc=reg_stem_tf,
                    doc_len=doc_len,
                    collection_count_for_word=cc_df_dict[stem_],
                    collection_len=cc_df_dict['ALL_TERMS_COUNT'],
                    mue=params['Mue'])
                stem_d_proba = lambda_dict['Global']*(global_stem_tf/float(global_denom_tf)) + lambda_dict['Burst']*(burst_stem_tf/float(burst_denom_tf)) + lambda_dict['Reg']*stem_reg_proba
                kl_score += (-1) * stem_q_prob * (math.log((stem_q_prob / stem_d_proba), 2))

            doc_score = kl_score

        res_df.loc[next_index] = ["0" * (3 - len(str(query_num))) + str(query_num), 'Q0', docno, 0, doc_score, 'indri']
        next_index += 1

    if res_df.empty == False:
        res_df.sort_values('Score', ascending=False, inplace=True)
        res_df['Rank'] = list(range(1, next_index + 1))
    return res_df


def get_score_retrieval_score_for_df(
        affix,
        big_df,
        save_folder,
        qrel_filepath):

    curr_file_name = affix + "_Results.txt"
    with open(os.path.join(save_folder + 'inner_res/', curr_file_name), 'w') as f:
        f.write(convert_df_to_trec(big_df))

    res_dict = get_ranking_effectiveness_for_res_file_per_query(
        file_path=save_folder + 'inner_res/',
        filename=curr_file_name,
        qrel_filepath=qrel_filepath,
        calc_ndcg_mrr=True)

    return res_dict

if __name__=='__main__':
    inner_fold = sys.argv[1]
    retrival_model = sys.argv[2]
    sw_rmv = True
    filter_params = {}
    asrc_round = int(inner_fold.split('_')[-1])

    if 'asrc' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel'
    elif 'bot' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents_fixed.relevance'
    elif 'herd_control' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/control.rel'
    elif 'united' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel'

    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+inner_fold+'/2008/SIM/'
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rhs_model_asrc/'

    query_to_doc_mapping_df = create_query_to_doc_mapping_df(inner_fold=inner_fold)
    limited_q_list = list(query_to_doc_mapping_df['QueryNum'].drop_duplicates())


    alpha_for_burst_options = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    alpha_beta_decay_weights= [-2.0, -1.5, -1.3, -1.2,-1.1, -0.9, -0.5, -0.1, 0.1, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2]
    optional_lambda_wights  = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    all_folds_df = pd.DataFrame({})
    all_fold_params_summary = "Fold" + '\t' + "Params" + '\n'
    q_list, fold_list = get_asrc_q_list_and_fold_list(inner_fold)

    for fold in fold_list:
        start_test_q = int(fold[0])
        end_test_q = int(fold[1])
        affix = inner_fold + '_' + str(start_test_q) + '_' + str(end_test_q) + "_"
        if retrival_model == 'BM25':
            params = {'K1' : 1 , 'b' : 0.5}
            init_params = {'Global' : 0.3, 'Burst' : 0.4,'Reg' : 0.3, 'Alpha' : 1.1, 'Beta' : 1.1, 'BAlpha' : 0.1}
            affix += "BM25_"

        elif retrival_model == 'LM' :
            params = {'Mue' : 1000.0}
            init_params = {'Global': 0.3, 'Burst': 0.2, 'Reg': 0.5, 'Alpha' : 1.1, 'Beta' : 1.1, 'BAlpha' : 0.1}
            affix  += ""
        else:
            raise Exception("Unknown model")

        stemmed_queries_df = create_stemmed_queries_df(sw_rmv=sw_rmv, limited_q_list=limited_q_list)
        interval_list = build_interval_list_asrc(asrc_round)
        cv_summary_df = pd.DataFrame(columns=interval_list + ['Map', 'P@5', 'P@10', ''])
        next_idx = 0
        all_global_params_dict = create_cached_docs_data(
            interval_list=interval_list,
            stemmed_queires_df=stemmed_queries_df,
            query_to_doc_mapping_df=query_to_doc_mapping_df,
            processed_docs_path=processed_docs_folder,
            filter_params=filter_params,
            amount_of_snapshot_limit=None
            )

        df_cc_dict = create_df_cc_dict(
            all_global_params_dict=all_global_params_dict,
            retrival_model=retrival_model)

        test_set_q = list(range(start_test_q, end_test_q + 1))
        stemmed_queries_df['IsTest'] = stemmed_queries_df['QueryNum'].apply(lambda x: 1 if x in test_set_q else 0)
        test_queries_df  = stemmed_queries_df[stemmed_queries_df['IsTest'] == 1].copy()
        train_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 0]

        best_config_dict = init_params.copy()
        best_ndcg = 0.0
        print('Optimizing BAlpha')
        for balpha in alpha_for_burst_options:
            init_params['BAlpha'] = balpha
            big_df = get_results_for_params(
                stemmed_queries_df = train_queries_df,
                all_query_doc_df = query_to_doc_mapping_df,
                all_global_params_dict = all_global_params_dict,
                interval_list = interval_list,
                df_cc_dict = df_cc_dict,
                retrieval_model = retrival_model,
                ret_model_params = params,
                alpha_for_burst = init_params['BAlpha'],
                alpha_decay = init_params['Alpha'],
                beta_decay = init_params['Beta'],
                lambda_dict = init_params,
                burst_inidcation_mat=None)

            res_dict = get_score_retrieval_score_for_df(
                affix=affix,
                big_df=big_df,
                qrel_filepath=qrel_filepath,
                save_folder=save_folder)

            if res_dict['all']['NDCG@3'] > best_ndcg:
                print(affix + " " + "SCORE : " + str(res_dict['all']))
                best_ndcg = res_dict['all']['NDCG@3']
                best_config_dict = init_params.copy()
                print(best_config_dict)

        burst_mat =  create_burst_indication_matrix(
            doc_version_len_df=all_global_params_dict['NumWords'],
            interval_list=interval_list,
            alpha_for_burst=best_config_dict['BAlpha'])

        for parameter_to_opt in ['Alpha','Beta']:
            print('Optimizing ' + parameter_to_opt)
            init_params = best_config_dict.copy()
            for param_option in alpha_beta_decay_weights:
                init_params[parameter_to_opt] = param_option
                big_df = get_results_for_params(
                    stemmed_queries_df=train_queries_df,
                    all_query_doc_df=query_to_doc_mapping_df,
                    all_global_params_dict=all_global_params_dict,
                    interval_list=interval_list,
                    df_cc_dict=df_cc_dict,
                    retrieval_model=retrival_model,
                    ret_model_params=params,
                    alpha_for_burst=init_params['BAlpha'],
                    alpha_decay=init_params['Alpha'],
                    beta_decay=init_params['Beta'],
                    lambda_dict=init_params,
                    burst_inidcation_mat=burst_mat)

                res_dict = get_score_retrieval_score_for_df(
                    affix=affix,
                    big_df=big_df,
                    qrel_filepath=qrel_filepath,
                    save_folder=save_folder)

                if res_dict['all']['NDCG@3'] > best_ndcg:
                    print(affix + " " + "SCORE : " + str(res_dict['all']))
                    best_ndcg = res_dict['all']['NDCG@3']
                    best_config_dict = init_params.copy()
                    print(best_config_dict)

        init_params = best_config_dict.copy()
        print('Optimizing Lambda')
        for wieght_option_g in optional_lambda_wights:
            for wieght_option_b in optional_lambda_wights:
                if (wieght_option_g + wieght_option_b) <= 1:
                    wieght_option_r = 1 - (wieght_option_g + wieght_option_b)
                    init_params['Global'] = wieght_option_g
                    init_params['Burst']  = wieght_option_b
                    init_params['Reg']    = wieght_option_r
                    big_df = get_results_for_params(
                        stemmed_queries_df=train_queries_df,
                        all_query_doc_df=query_to_doc_mapping_df,
                        all_global_params_dict=all_global_params_dict,
                        interval_list=interval_list,
                        df_cc_dict=df_cc_dict,
                        retrieval_model=retrival_model,
                        ret_model_params=params,
                        alpha_for_burst=init_params['BAlpha'],
                        alpha_decay=init_params['Alpha'],
                        beta_decay=init_params['Beta'],
                        lambda_dict=init_params,
                        burst_inidcation_mat=burst_mat)

                    res_dict = get_score_retrieval_score_for_df(
                        affix=affix,
                        big_df=big_df,
                        qrel_filepath=qrel_filepath,
                        save_folder=save_folder)

                    if res_dict['all']['NDCG@3'] > best_ndcg:
                        print(affix + " " + "SCORE : " + str(res_dict['all']))
                        best_ndcg = res_dict['all']['NDCG@3']
                        best_config_dict = init_params.copy()
                        print(best_config_dict)
        # test set run
        test_fold_df = get_results_for_params(
            stemmed_queries_df=test_queries_df,
            all_query_doc_df=query_to_doc_mapping_df,
            all_global_params_dict=all_global_params_dict,
            interval_list=interval_list,
            df_cc_dict=df_cc_dict,
            retrieval_model=retrival_model,
            ret_model_params=params,
            alpha_for_burst=best_config_dict['BAlpha'],
            alpha_decay=best_config_dict['Alpha'],
            beta_decay=best_config_dict['Beta'],
            lambda_dict=best_config_dict,
            burst_inidcation_mat=burst_mat)


        all_folds_df = all_folds_df.append(test_fold_df , ignore_index=True)
        all_fold_params_summary += str(start_test_q) + '_' + str(end_test_q) + '\t' + str(best_config_dict) + '\n'

    curr_file_name = inner_fold + '_' + retrival_model + "_Results.txt"
    with open(os.path.join(save_folder + 'final_res/', curr_file_name), 'w') as f:
        f.write(convert_df_to_trec(all_folds_df))
    with open(os.path.join(save_folder + 'final_res/', curr_file_name.replace('_Results', '_Params')), 'w') as f:
        f.write(all_fold_params_summary)




