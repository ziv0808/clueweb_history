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

if __name__=='__main__':
    inner_fold = sys.argv[1]
    retrival_model = sys.argv[2]
    sw_rmv = True
    filter_params = {}
    asrc_round = int(inner_fold.split('_')[1])

    qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel'
    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+inner_fold+'/2008/SIM/'
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/avg_model_res_asrc/'

    query_to_doc_mapping_df = create_query_to_doc_mapping_df(inner_fold=inner_fold)
    limited_q_list = list(query_to_doc_mapping_df['QueryNum'].drop_duplicates())


    wieght_for_last_interval_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    optional_decay_weights_k = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

    all_folds_df_dict = {}
    all_fold_params_summary = {}
    for weight_list_type in ['Uniform', 'Decaying', 'RDecaying']:
        all_folds_df_dict[weight_list_type] = pd.DataFrame({})
        all_fold_params_summary[weight_list_type] = "Fold" + '\t' + "Weights" + '\n'
    q_list, fold_list = get_asrc_q_list_and_fold_list()
    for fold in fold_list:
        start_test_q = int(fold[0])
        end_test_q = int(fold[1])
        affix = inner_fold + '_' + str(start_test_q) + '_' + str(end_test_q) + "_"
        if retrival_model == 'BM25':
            params = {'K1' : 1 , 'b' : 0.5}
            affix += "BM25_"

        elif retrival_model == 'LM' :
            params = {'Mue' : 1000.0}
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

        test_set_q = list(range(start_test_q, end_test_q + 1))
        stemmed_queries_df['IsTest'] = stemmed_queries_df['QueryNum'].apply(lambda x: 1 if x in test_set_q else 0)
        test_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 1].copy()
        stemmed_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 0]

        all_weights_df = all_global_params_dict['NumWords'].copy()
        all_weights_df = all_weights_df.applymap(lambda x: 0 if pd.np.isnan(x) else 1)
        for element in all_global_params_dict:
            all_global_params_dict[element] = all_global_params_dict[element].applymap(lambda x: 0.0 if np.isnan(x) else x)

        best_config_dict = {
            'Uniform': {'BestNDCG': 0.0, 'WDf': None, 'BestWieghts': None},
            'Decaying': {'BestNDCG': 0.0, 'WDf': None, 'BestWieghts': None},
            'RDecaying': {'BestNDCG': 0.0, 'WDf': None, 'BestWieghts': None},
        }
        for i in range(len(interval_list) - 1):
            for cw_interval_weight in wieght_for_last_interval_options:
                ignore_idxs = list(range(i))
                # run uniform weights logic
                ignore_idxs.append(len(interval_list) - 1)
                weight_list_options = make_weight_list_options(
                    ignore_idxs=ignore_idxs,
                    interval_list=interval_list,
                    optional_decay_weights_k=optional_decay_weights_k)
                for weight_list_config in weight_list_options:
                    weight_list = weight_list_config[0]
                    weight_list_type = weight_list_config[1]
                    tmp_weights_df = create_weights_matrix(
                        all_weights_df=all_weights_df,
                        weight_list=weight_list,
                        cw_interval_weight=cw_interval_weight)

                    big_df, weight_list = run_test_on_config(
                        weight_list=weight_list,
                        tmp_weights_df=tmp_weights_df,
                        stemmed_queries_df=stemmed_queries_df,
                        query_to_doc_mapping_df=query_to_doc_mapping_df,
                        all_global_params_dict=all_global_params_dict,
                        retrival_model=retrival_model,
                        params=params,
                        cw_interval_weight=cw_interval_weight)

                    res_dict = get_score_retrieval_score_for_df(
                        affix=affix,
                        big_df=big_df,
                        qrel_filepath=qrel_filepath,
                        save_folder=save_folder)

                    print(affix  + " " + str(weight_list))
                    print(affix  + " " + "SCORE : " + str(res_dict['all']))
                    sys.stdout.flush()

                    if res_dict['all']['NDCG@3'] > best_config_dict[weight_list_type]['BestNDCG']:
                        best_config_dict[weight_list_type]['BestNDCG'] = res_dict['all']['NDCG@3']
                        best_config_dict[weight_list_type]['BestWieghts'] = weight_list
                        best_config_dict[weight_list_type]['WDf'] = tmp_weights_df

        # test the benchmark case - only last interval
        tmp_weights_df = tmp_weights_df.applymap(lambda x: 0.0)
        tmp_weights_df['ClueWeb09'] = 1.0
        weight_list = np.array([0.0] * len(weight_list))
        weight_list[-1] = 1.0
        big_df, weight_list = run_test_on_config(
            weight_list=weight_list,
            tmp_weights_df=tmp_weights_df,
            stemmed_queries_df=stemmed_queries_df,
            query_to_doc_mapping_df=query_to_doc_mapping_df,
            all_global_params_dict=all_global_params_dict,
            retrival_model=retrival_model,
            params=params,
            cw_interval_weight=1.0)

        res_dict = get_score_retrieval_score_for_df(
            affix=affix,
            big_df=big_df,
            qrel_filepath=qrel_filepath,
            save_folder=save_folder)

        print(affix + " " + str(weight_list))
        print(affix + " " + "SCORE : " + str(res_dict['all']))
        sys.stdout.flush()
        for weight_list_type in ['Uniform','Decaying','RDecaying']:
            if res_dict['all']['NDCG@3'] >  best_config_dict[weight_list_type]['BestNDCG']:
                best_config_dict[weight_list_type]['BestNDCG'] = res_dict['all']['NDCG@3']
                best_config_dict[weight_list_type]['BestWieghts'] = weight_list
                best_config_dict[weight_list_type]['WDf'] = tmp_weights_df

        for weight_list_type in ['Uniform', 'Decaying', 'RDecaying']:
            # run on test set
            big_df, weight_list = run_test_on_config(
                weight_list=best_config_dict[weight_list_type]['WList'],
                tmp_weights_df=best_config_dict[weight_list_type]['WDf'],
                stemmed_queries_df=test_queries_df,
                query_to_doc_mapping_df=query_to_doc_mapping_df,
                all_global_params_dict=all_global_params_dict,
                retrival_model=retrival_model,
                params=params,
                cw_interval_weight=best_config_dict[weight_list_type]['WList'][-1])

            all_folds_df_dict[weight_list_type] = all_folds_df_dict[weight_list_type].append(big_df , ignore_index=True)
            all_fold_params_summary[weight_list_type] += start_test_q + '_' + end_test_q + '\t' + str(best_config_dict[weight_list_type]['BestWieghts']) + '\n'

    for weight_list_type in ['Uniform', 'Decaying', 'RDecaying']:
        curr_file_name = inner_fold + '_' + retrival_model + '_' + weight_list_type + "_Results.txt"
        with open(os.path.join(save_folder + 'final_res/', curr_file_name), 'w') as f:
            f.write(convert_df_to_trec(all_folds_df_dict[weight_list_type]))
        with open(os.path.join(save_folder + 'final_res/', curr_file_name.replace('_Results', '_Params')), 'w') as f:
            f.write(all_fold_params_summary[weight_list_type])




