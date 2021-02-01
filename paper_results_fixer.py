from utils import *


def orginize_res_dict_by_q_order(
        tmp_res_dict,
        q_ordered_list):

    res_dict = {
        'NDCG@1' : [],
        'NDCG@3' : [],
        'NDCG@5' : [],
        'MRR'    : []}
    for q in q_ordered_list:
        for key in res_dict:
            res_dict[key].append(tmp_res_dict[q][key])
    return res_dict



def create_reults_dataframe_for_models(
        model_files_dict,
        basline_model_dict,
        dataset,
        qrel_filepath,
        is_svm_rank,
        init_round,
        round_limit):

    big_res_dict = {}
    for round_ in range(init_round, round_limit + 1):
        first = True
        for model in model_files_dict:
            file_path = model_files_dict[model]['Folder'].replace('<DatasetNameUpper>', dataset.upper()).replace('<RoundNum>', str(round_))
            if is_svm_rank == True:
                file_path = file_path.replace('lambdamart_res', 'rank_svm_res')
            filename = model_files_dict[model]['FileTemplate'].replace('<DatasetNameUpper>', dataset.upper()).replace('<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            print(filename)
            sys.stdout.flush()
            tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                file_path=file_path,
                filename=filename,
                qrel_filepath=qrel_filepath,
                calc_ndcg_mrr=True,
                remove_low_quality=False)
            if first == True:
                q_ordered_list = list(tmp_res_dict.keys())
                q_ordered_list.remove('all')
                first = False

            if model not in big_res_dict:
                big_res_dict[model] = {'NDCG@1' : [],
                                       'NDCG@3' : [],
                                       'NDCG@5' : [],
                                       'MRR'    : []}
            orginized_res_dict = orginize_res_dict_by_q_order(tmp_res_dict=tmp_res_dict,q_ordered_list=q_ordered_list)
            for key in orginized_res_dict :
                big_res_dict[model][key].extend(orginized_res_dict[key])

            if 'AlsoLQRmv' in model_files_dict[model]:
                tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                    file_path=file_path,
                    filename=filename,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=True,
                    remove_low_quality=True)
                model_rmv_lq = model + ' RMV LQ'
                if model_rmv_lq not in big_res_dict:
                    big_res_dict[model_rmv_lq] = {'NDCG@1': [],
                                           'NDCG@3': [],
                                           'NDCG@5': [],
                                           'MRR': []}
                    orginized_res_dict = orginize_res_dict_by_q_order(tmp_res_dict=tmp_res_dict, q_ordered_list=q_ordered_list)
                    for key in orginized_res_dict:
                        big_res_dict[model_rmv_lq][key].extend(orginized_res_dict[key])

        for model in basline_model_dict:
            file_path = model_files_dict[model]['Folder'].replace('<DatasetNameUpper>', dataset.upper()).replace('<RoundNum>',
                                                                                                          str(round_))
            if is_svm_rank == True:
                file_path = file_path.replace('lambdamart_res', 'rank_svm_res')
            filename = model_files_dict[model]['FileTemplate'].replace('<DatasetNameUpper>', dataset.upper()).replace(
                '<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            print(filename)
            sys.stdout.flush()
            tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                file_path=file_path,
                filename=filename,
                qrel_filepath=qrel_filepath,
                calc_ndcg_mrr=True,
                remove_low_quality=False)

            if model not in big_res_dict:
                big_res_dict[model] = {'NDCG@1': [],
                                       'NDCG@3': [],
                                       'NDCG@5': [],
                                       'MRR'   : []}
            orginized_res_dict = orginize_res_dict_by_q_order(tmp_res_dict=tmp_res_dict, q_ordered_list=q_ordered_list)
            for key in orginized_res_dict:
                big_res_dict[model][key].extend(orginized_res_dict[key])

            if 'AlsoLQRmv' in model_files_dict[model]:
                tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                    file_path=file_path,
                    filename=filename,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=True,
                    remove_low_quality=True)
                model_rmv_lq = model + ' RMV LQ'
                if model_rmv_lq not in big_res_dict:
                    big_res_dict[model_rmv_lq] = {'NDCG@1': [],
                                                  'NDCG@3': [],
                                                  'NDCG@5': [],
                                                  'MRR'   : []}
                orginized_res_dict = orginize_res_dict_by_q_order(tmp_res_dict=tmp_res_dict,
                                                                  q_ordered_list=q_ordered_list)
                for key in orginized_res_dict:
                    big_res_dict[model_rmv_lq][key].extend(orginized_res_dict[key])

    model_pval_dict = {}
    all_models = list(big_res_dict.keys())
    measures = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'MRR']
    model_pval_col_list = []
    for model in all_models:
        for measure in measures:
            model_pval_col_list.append(model + '_' + measure + '_Pval')
    res_df = pd.DataFrame(columns = ['Model'] + measures + model_pval_col_list)
    next_idx = 0
    for test_model in big_res_dict:
        print(model)
        sys.stdout.flush()
        insert_row = [test_model]
        for measure in measures:
            insert_row.append(np.mean(big_res_dict[test_model][measure]))
        for model in all_models:
            for measure in measures:
                models_set = set([model, test_model])
                if model == test_model:
                    insert_row.append(np.nan)
                elif (models_set in model_pval_dict) and (measure in model_pval_dict[models_set]):
                    insert_row.append(model_pval_dict[models_set][measure])
                else:
                    if models_set not in model_pval_dict:
                        model_pval_dict[models_set] = {}
                    t_stat, p_val = pemutation_test(big_res_dict[model][measure], big_res_dict[test_model][measure],total_number=10000)
                    model_pval_dict[models_set][measure] = p_val
                    insert_row.append(p_val)
        res_df.loc[next_idx] = insert_row
        next_idx += 1

    return res_df

if __name__=='__main__':

    dataset = sys.argv[1]
    is_lts = ast.literal_eval(sys.argv[2])
    is_svm_rank = ast.literal_eval(sys.argv[3])

    model_files_dict = {
        'S'  : {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_Static_All.txt',
                'AlsoLQRmv' : True},
        'S+MSMM+MG': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_Static_M_STD_Min_Max_MG_AllByMonths.txt'},
        'S+MG': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_Static_MG_AllByMonths.txt'},
    }
    basline_model_dict = {
        'F3 BM25 UW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Rank_Uniform_LoO_Results.txt'},
        'F3 BERT UW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Rank_Uniform_LoO_Results.txt'},
        'F3 LM UW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_LM_Rank_Uniform_LoO_Results.txt'},
        'F3 BM25 IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Rank_Decaying_LoO_Results.txt'},
        'F3 BERT IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Rank_Decaying_LoO_Results.txt'},
        'F3 LM IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_LM_Rank_Decaying_LoO_Results.txt'},
        'F3 BM25 DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Rank_RDecaying_LoO_Results.txt'},
        'F3 BERT DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Rank_RDecaying_LoO_Results.txt'},
        'F3 LM DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_LM_Rank_RDecaying_LoO_Results.txt'},
        'ED KL': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/benchmark_sudomay/final_res/',
                  'FileTemplate': '<DatasetName>_0<RoundNum>_KL_LoO_Results.txt'},
        'ED LM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/benchmark_sudomay/final_res/',
                  'FileTemplate': '<DatasetName>_0<RoundNum>_LM_LoO_Results.txt'},
        'RHS BM25': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rhs_model_asrc/final_res/',
                     'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_LoO_Results.txt'},
        'RHS LM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rhs_model_asrc/final_res/',
                   'FileTemplate': '<DatasetName>_0<RoundNum>_LM_LoO_Results.txt'},
        'BERT': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bert/',
                 'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Results.txt'},
        'MM Prev3Winners DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3Winners_MixtureDIR_LoO_Results.txt'},
        'MM Prev3BestImprove DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                    'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3BestImprove_MixtureDIR_LoO_Results.txt'},
        'MM OnlyReservoir DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                 'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureDIR_OnlyReservoir_LoO_Results.txt'},
        'Orig. Ranker': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/data/<DatasetName>/<DatasetName>_0<RoundNum>/RankedLists/',
                            'FileTemplate': 'LambdaMART_<DatasetName>_0<RoundNum>'},
        'LM DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/basic_lm/',
                   'FileTemplate': '<DatasetName>_0<RoundNum>_LMIR.DIR.txt',
                   'AlsoLQRmv': True},
        'BM25': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bm25_model_res/final_res/',
                 'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_LoO_Results.txt',
                 'AlsoLQRmv': True},
        'Old BM25': {
            'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
            'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_BM25_All.txt',
            'AlsoLQRmv': True},
    }

    init_round = 2
    if is_lts == True:
        basline_model_dict = {
            'LTS MA': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/final_res/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_MA_LTS__LoO_Results.txt'},
           'LTS ARMA': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/final_res/',
                        'FileTemplate': '<DatasetName>_0<RoundNum>_ARMA_LTS__LoO_Results.txt'},
            'MM Prev3Winners DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                    'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3Winners_MixtureDIR_LoO_Results.txt'},
            'MM Prev3BestImprove DIR': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3BestImprove_MixtureDIR_LoO_Results.txt'},
            'MM OnlyReservoir DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                     'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureDIR_OnlyReservoir_LoO_Results.txt'},
        }
        init_round = 4

    if dataset == 'asrc':
        round_limit = 8
    elif dataset == 'united':
        round_limit = 5

    if 'asrc' in dataset:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel'
    elif 'united' in dataset:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel'
    elif 'comp2020' in dataset:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/curr_comp.rel'



    res_df = create_reults_dataframe_for_models(
        model_files_dict=model_files_dict,
        basline_model_dict=basline_model_dict,
        dataset=dataset,
        qrel_filepath=qrel_filepath,
        is_svm_rank=is_svm_rank,
        init_round=init_round,
        round_limit=round_limit)

    save_filename = dataset.upper() + '_LambdaMART_All_Results.tsv'
    if is_lts == True:
        save_filename = save_filename.replace('_All_', '_LTS_')
    if is_svm_rank == True:
        save_filename = save_filename.replace('_LambdaMART_', '_SVMRank_')
    res_df.to_csv(save_filename, sep = '\t', index = False)

