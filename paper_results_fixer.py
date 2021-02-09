from utils import *



def orginize_res_dict_by_q_order(
        tmp_res_dict,
        q_ordered_list):

    res_dict = {
        'NDCG@1' : [],
        'NDCG@3' : [],
        'NDCG@5' : []}
    for q in q_ordered_list:
        for key in res_dict:
            res_dict[key].append(tmp_res_dict[q][key])
    return res_dict

def get_ranking_effectiveness_for_res_file_per_query_after_mean_trials(
        file_path,
        filename,
        qrel_filepath,
        calc_ndcg_mrr,
        remove_low_quality):

    first = True
    for trial_num in range(1, 6):
        trial_res_dict = get_ranking_effectiveness_for_res_file_per_query(
            file_path=file_path,
            filename=filename.replace('.txt', '_' + str(trial_num)+ '.txt'),
            qrel_filepath=qrel_filepath,
            calc_ndcg_mrr=calc_ndcg_mrr,
            remove_low_quality=remove_low_quality)
        if first == True:
            current_res_dict = trial_res_dict
            first = False
        else:
            for key in current_res_dict:
                for measure in current_res_dict[key]:
                    current_res_dict[key][measure] = (float(current_res_dict[key][measure]) * (trial_num - 1) + trial_res_dict[key][measure]) / float(trial_num)

    return current_res_dict

def create_per_round_reults_dataframe_for_models(
        model_files_dict,
        basline_model_dict,
        dataset,
        qrel_filepath,
        init_round,
        round_limit):
    big_res_dict = {}

    for round_ in range(init_round, round_limit + 1):
        first = True
        for model in model_files_dict:
            if dataset == 'united' and model in ['MM Prev3Winners DIR' , 'MM Prev3BestImprove DIR', 'MM OnlyReservoir DIR']:
                continue
            file_path = model_files_dict[model]['Folder'].replace('<DatasetNameUpper>', dataset.upper()).replace(
                '<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            if is_svm_rank == True:
                file_path = file_path.replace('lambdamart_res', 'rank_svm_res')
            filename = model_files_dict[model]['FileTemplate'].replace('<DatasetNameUpper>', dataset.upper()).replace(
                '<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            print(model, filename)
            sys.stdout.flush()
            if 'NeedAdjust' in model_files_dict[model] and model_files_dict[model]['NeedAdjust'] == True:
                tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query_after_mean_trials(
                    file_path=file_path,
                    filename=filename,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=True,
                    remove_low_quality=False)
            else:
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
            orginized_res_dict = orginize_res_dict_by_q_order(tmp_res_dict=tmp_res_dict, q_ordered_list=q_ordered_list)
            if model not in big_res_dict:
                big_res_dict[model] = {}
            big_res_dict[model][round_] = orginized_res_dict
        for model in basline_model_dict:
            if dataset == 'united' and model in ['MM Prev3Winners DIR', 'MM Prev3BestImprove DIR', 'MM OnlyReservoir DIR']:
                continue
            file_path = basline_model_dict[model]['Folder'].replace('<DatasetNameUpper>', dataset.upper()).replace(
                '<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            if is_svm_rank == True:
                file_path = file_path.replace('lambdamart_res', 'rank_svm_res')
            filename = basline_model_dict[model]['FileTemplate'].replace('<DatasetNameUpper>', dataset.upper()).replace(
                '<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            print(model, filename)
            sys.stdout.flush()
            if 'NeedAdjust' in basline_model_dict[model] and basline_model_dict[model]['NeedAdjust'] == True:
                tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query_after_mean_trials(
                    file_path=file_path,
                    filename=filename,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=True,
                    remove_low_quality=False)
            else:
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
            orginized_res_dict = orginize_res_dict_by_q_order(tmp_res_dict=tmp_res_dict, q_ordered_list=q_ordered_list)
            if model not in big_res_dict:
                big_res_dict[model] = {}
            big_res_dict[model][round_] = orginized_res_dict

    all_models = list(big_res_dict.keys())
    measures = ['NDCG@1', 'NDCG@3', 'NDCG@5']
    model_pval_col_list = []
    for model in all_models:
        for measure in measures:
            model_pval_col_list.append(model + '_' + measure + '_Pval')
    res_df = pd.DataFrame(columns=['Model', 'Round'] + measures + model_pval_col_list)
    next_idx =0
    for test_model in big_res_dict:
        print(test_model)
        sys.stdout.flush()
        for round_ in big_res_dict[test_model]:
            insert_row = [test_model, round_]
            for measure in measures:
                insert_row.append(np.mean(big_res_dict[test_model][round_][measure]))
            for model in all_models:
                for measure in measures:
                    models_set = tuple(sorted([model, test_model]))
                    if len(big_res_dict[model][round_][measure]) != len(big_res_dict[test_model][round_][measure]):
                        print(models_set, measure)
                        print("Prob!")
                        sys.stdout.flush()
                    if model == test_model:
                        insert_row.append(np.nan)
                    else:
                        t_stat, p_val = pemutation_test(big_res_dict[model][round_][measure], big_res_dict[test_model][round_][measure],
                                                        total_number=10000)
                        insert_row.append(p_val)
            res_df.loc[next_idx] = insert_row
            next_idx += 1
    return res_df

def create_reults_dataframe_for_models(
        model_files_dict,
        basline_model_dict,
        dataset,
        qrel_filepath,
        is_svm_rank,
        init_round,
        round_limit,
        ablation):

    big_res_dict = {}
    for round_ in range(init_round, round_limit + 1):
        first = True
        for model in model_files_dict:
            file_path = model_files_dict[model]['Folder'].replace('<DatasetNameUpper>', dataset.upper()).replace('<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            if is_svm_rank == True:
                file_path = file_path.replace('lambdamart_res', 'rank_svm_res')
            filename = model_files_dict[model]['FileTemplate'].replace('<DatasetNameUpper>', dataset.upper()).replace('<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            print(model, filename)
            sys.stdout.flush()
            if 'NeedAdjust' in model_files_dict[model] and model_files_dict[model]['NeedAdjust'] == True:
                tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query_after_mean_trials(
                    file_path=file_path,
                    filename=filename,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=True,
                    remove_low_quality=False)
            else:
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
                                       'NDCG@5' : []}
            orginized_res_dict = orginize_res_dict_by_q_order(tmp_res_dict=tmp_res_dict,q_ordered_list=q_ordered_list)
            for key in orginized_res_dict :
                big_res_dict[model][key].extend(orginized_res_dict[key])

            if 'AlsoLQRmv' in model_files_dict[model]:
                if 'NeedAdjust' in model_files_dict[model] and model_files_dict[model]['NeedAdjust'] == True:
                    tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query_after_mean_trials(
                        file_path=file_path,
                        filename=filename,
                        qrel_filepath=qrel_filepath,
                        calc_ndcg_mrr=True,
                        remove_low_quality=True)
                else:
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
                                                   'NDCG@5': []}
                orginized_res_dict = orginize_res_dict_by_q_order(tmp_res_dict=tmp_res_dict, q_ordered_list=q_ordered_list)
                for key in orginized_res_dict:
                    big_res_dict[model_rmv_lq][key].extend(orginized_res_dict[key])

        for model in basline_model_dict:
            if dataset == 'united' and model.startswith('MM '):
                continue
            file_path = basline_model_dict[model]['Folder'].replace('<DatasetNameUpper>', dataset.upper()).replace('<RoundNum>',str(round_)).replace('<DatasetName>', dataset)
            if is_svm_rank == True:
                file_path = file_path.replace('lambdamart_res', 'rank_svm_res')
            filename = basline_model_dict[model]['FileTemplate'].replace('<DatasetNameUpper>', dataset.upper()).replace('<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            print(model, filename)
            sys.stdout.flush()
            if 'NeedAdjust' in basline_model_dict[model] and basline_model_dict[model]['NeedAdjust'] == True:
                tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query_after_mean_trials(
                    file_path=file_path,
                    filename=filename,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=True,
                    remove_low_quality=False)
            else:
                tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                    file_path=file_path,
                    filename=filename,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=True,
                    remove_low_quality=False)

            if model not in big_res_dict:
                big_res_dict[model] = {'NDCG@1': [],
                                       'NDCG@3': [],
                                       'NDCG@5': []}
            orginized_res_dict = orginize_res_dict_by_q_order(tmp_res_dict=tmp_res_dict, q_ordered_list=q_ordered_list)
            for key in orginized_res_dict:
                big_res_dict[model][key].extend(orginized_res_dict[key])

            if 'AlsoLQRmv' in basline_model_dict[model]:
                if 'NeedAdjust' in basline_model_dict[model] and basline_model_dict[model]['NeedAdjust'] == True:
                    tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query_after_mean_trials(
                        file_path=file_path,
                        filename=filename,
                        qrel_filepath=qrel_filepath,
                        calc_ndcg_mrr=True,
                        remove_low_quality=True)
                else:
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
                                                  'NDCG@5': []}
                orginized_res_dict = orginize_res_dict_by_q_order(tmp_res_dict=tmp_res_dict,
                                                                  q_ordered_list=q_ordered_list)
                for key in orginized_res_dict:
                    big_res_dict[model_rmv_lq][key].extend(orginized_res_dict[key])

    model_pval_dict = {}
    all_models = list(big_res_dict.keys())
    measures = ['NDCG@1', 'NDCG@3', 'NDCG@5']
    model_pval_col_list = []
    for model in all_models:
        for measure in measures:
            model_pval_col_list.append(model + '_' + measure + '_Pval')
    res_df = pd.DataFrame(columns = ['Model'] + measures + model_pval_col_list)
    next_idx = 0
    for test_model in big_res_dict:
        print(test_model)
        sys.stdout.flush()
        insert_row = [test_model]
        for measure in measures:
            insert_row.append(np.mean(big_res_dict[test_model][measure]))
        for model in all_models:
            for measure in measures:
                models_set = tuple(sorted([model, test_model]))
                if len(big_res_dict[model][measure]) != len(big_res_dict[test_model][measure]):
                    print(models_set,measure )
                    print("Prob!")
                    sys.stdout.flush()
                if model == test_model:
                    insert_row.append(np.nan)
                elif ablation == True and 'S+MSMM' not in models_set:
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
    graphs_per_round = ast.literal_eval(sys.argv[2])
    is_svm_rank = ast.literal_eval(sys.argv[3])
    static_feat_compare = ast.literal_eval(sys.argv[4])
    ablation = ast.literal_eval(sys.argv[5])
    svm_compare = ast.literal_eval(sys.argv[6])

    model_files_dict = {
        'S'  : {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_All.txt',
                'AlsoLQRmv' : True,
                'NeedAdjust': True},
        'S+MSMM+MG': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_M_STD_Min_Max_MG_AllByMonths.txt',
                'AlsoLQRmv': True,
                'NeedAdjust': True},
        'S+MSMM': {
            'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
            'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_M_STD_Min_Max_AllByMonths.txt',
                   'AlsoLQRmv': True,
                    'NeedAdjust': True},
        'S+MG': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_MG_AllByMonths.txt',
                   'AlsoLQRmv': True,
                 'NeedAdjust': True},
    }

    basline_model_dict = {
        # 'F3 BM25 UW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
        #                'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Rank_Uniform_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'F3 BERT UW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
        #                'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Rank_Uniform_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'F3 LM UW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
        #                'FileTemplate': '<DatasetName>_0<RoundNum>_LM_Rank_Uniform_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'F3 BM25 IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
        #                'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Rank_Decaying_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'F3 BERT IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
        #                'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Rank_Decaying_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'F3 LM IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
        #                'FileTemplate': '<DatasetName>_0<RoundNum>_LM_Rank_Decaying_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'F3 BM25 DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
        #                'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Rank_RDecaying_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'F3 BERT DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
        #                'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Rank_RDecaying_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'F3 LM DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
        #                'FileTemplate': '<DatasetName>_0<RoundNum>_LM_Rank_RDecaying_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'ED KL': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/benchmark_sudomay/final_res/',
        #           'FileTemplate': '<DatasetName>_0<RoundNum>_KL_LoO_Results.txt'},
        # 'ED LM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/benchmark_sudomay/final_res/',
        #           'FileTemplate': '<DatasetName>_0<RoundNum>_LM_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'RHS BM25': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rhs_model_asrc/final_res/',
        #              'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'RHS LM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rhs_model_asrc/final_res/',
        #            'FileTemplate': '<DatasetName>_0<RoundNum>_LM_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'BERT': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bert/',
        #          'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Results.txt',
        #            'AlsoLQRmv': True},
        # 'MM Prev3Winners DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
        #                         'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3Winners_MixtureDIR_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'MM Prev3BestImprove DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
        #                             'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3BestImprove_MixtureDIR_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'MM OnlyReservoir DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
        #                          'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureDIR_OnlyReservoir_LoO_Results.txt',
        #            'AlsoLQRmv': True},
        # 'Orig. Ranker': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/data/<DatasetName>/<DatasetName>_0<RoundNum>/RankedLists/',
        #                     'FileTemplate': 'LambdaMART_<DatasetName>_0<RoundNum>'},
        'LM DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/basic_lm/',
                   'FileTemplate': '<DatasetName>_0<RoundNum>_LMIR.DIR.txt',
                   'AlsoLQRmv': True},
        'BM25': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_BM25.txt',
                 'AlsoLQRmv': True},
        'S+DIROnlyReservoir': {
            'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrOr/',
            'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrOr_MinMax_Static_All.txt',
            'AlsoLQRmv': True,
            'NeedAdjust': True},
    }

    init_round = 2
    # if is_lts == True:
    #     basline_model_dict = {
    #         'LTS MA': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/final_res/',
    #                   'FileTemplate': '<DatasetName>_0<RoundNum>_MA_LTS__LoO_Results.txt',
    #                'AlsoLQRmv': True},
    #         'LTS LR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/final_res/',
    #                    'FileTemplate': '<DatasetName>_0<RoundNum>_LR_LTS__LoO_Results.txt',
    #                    'AlsoLQRmv': True},
    #        'LTS ARMA': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/final_res/',
    #                     'FileTemplate': '<DatasetName>_0<RoundNum>_ARMA_LTS__LoO_Results.txt',
    #                'AlsoLQRmv': True},
    #         'MM Prev3Winners DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
    #                                 'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3Winners_MixtureDIR_LoO_Results.txt',
    #                'AlsoLQRmv': True},
    #         'MM Prev3BestImprove DIR': {
    #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
    #             'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3BestImprove_MixtureDIR_LoO_Results.txt',
    #                'AlsoLQRmv': True},
    #         'MM OnlyReservoir DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
    #                                  'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureDIR_OnlyReservoir_LoO_Results.txt',
    #                'AlsoLQRmv': True},
    #         'BM25': {
    #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
    #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_BM25.txt',
    #             'AlsoLQRmv': True},
    #         'F3 BM25 DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
    #                        'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Rank_RDecaying_LoO_Results.txt',
    #                        'AlsoLQRmv': True},
    #         'F3 LM DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
    #                      'FileTemplate': '<DatasetName>_0<RoundNum>_LM_Rank_RDecaying_LoO_Results.txt',
    #                      'AlsoLQRmv': True},
    #     }
    #     init_round = 4

    if static_feat_compare == True:
        model_files_dict = {
            'S': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_All.txt',
                'AlsoLQRmv': True,
                'NeedAdjust': True},
            # 'S+MSMM+MG': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_M_STD_Min_Max_MG_AllByMonths.txt',
            #     'AlsoLQRmv': True,
            #     'NeedAdjust': True},
            # 'S+MSMM+MG': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_Static_M_STD_Min_Max_MG_AllByMonths.txt'},
        }

        basline_model_dict = {
            'S+DIRPrev3Winners': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrP3w/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrP3w_MinMax_Static_All.txt',
                'AlsoLQRmv': True,
                'NeedAdjust': True},
            'S+DIRPrev3BestImprove': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrB3i/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrB3i_MinMax_Static_All.txt',
                'AlsoLQRmv': True,
                'NeedAdjust': True},
            'S+DIROnlyReservoir': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrOr/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrOr_MinMax_Static_All.txt',
                'AlsoLQRmv': True,
                'NeedAdjust': True},
            'S+DIRPrev3Winners+DIRPrev3BestImprove': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrP3w_MDrB3i/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrP3w_MDrB3i_MinMax_Static_All.txt',
                'AlsoLQRmv': True,
                'NeedAdjust': True},
            'S-LM+DIRPrev3BestImprove': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrB3i/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrB3i_MinMax_Static_All.txt',
                'AlsoLQRmv': True,
                'NeedAdjust': True},
            'S-LM+DIRPrev3Winners': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrP3w/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrP3w_MinMax_Static_All.txt',
                'AlsoLQRmv': True,
                'NeedAdjust': True},
            'S-LM+DIROnlyReservoir': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrOr/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrOr_MinMax_Static_All.txt',
                'AlsoLQRmv': True,
                'NeedAdjust': True},
            # 'S+ED KL': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_EDKl/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_EDKl_MinMax_Static_All.txt'},
            # 'S+ED LM': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_EDLm/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_EDLm_MinMax_Static_All.txt',
            #     'AlsoLQRmv': True},
            # 'S+F3 LM DW': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_FuLM/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_FuLM_MinMax_Static_All.txt',
            #     'AlsoLQRmv': True},
            # 'S+F3 BM25 DW': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_FuBm25/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_FuBm25_MinMax_Static_All.txt',
            #     'AlsoLQRmv': True},
            # 'S+RHS BM25': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_RhsBm25/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_RhsBm25_MinMax_Static_All.txt',
            #     'AlsoLQRmv': True},
            # 'S+RHS LM': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_RhsLm/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_RhsLm_MinMax_Static_All.txt',
            #     'AlsoLQRmv': True},
            # 'S+MSMM+MG+DIRPrev3Winners': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrP3w/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrP3w_MinMax_Static_M_STD_Min_Max_MG_AllByMonths.txt',
            #     'AlsoLQRmv': True,
            #     'NeedAdjust': True},
            # 'S+MSMM+MG+DIRPrev3BestImprove': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrB3i/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrB3i_MinMax_Static_M_STD_Min_Max_MG_AllByMonths.txt',
            #     'AlsoLQRmv': True,
            #     'NeedAdjust': True},
            # 'S+MSMM+DIRPrev3Winners': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrP3w/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrP3w_MinMax_Static_M_STD_Min_Max_AllByMonths.txt',
            #     'AlsoLQRmv': True,
            #     'NeedAdjust': True},
            # 'S+MSMM+DIRPrev3BestImprove': {
            #     'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrB3i/',
            #     'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MDrB3i_MinMax_Static_M_STD_Min_Max_AllByMonths.txt',
            #     'AlsoLQRmv': True,
            #     'NeedAdjust': True}

        }

        # if is_lts == True:
        #     basline_model_dict = {
        #         'S+DIRPrev3Winners': {
        #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MDrP3w/',
        #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MDrP3w_MinMax_Static_All.txt',
        #             'AlsoLQRmv': True},
        #         'S+DIRPrev3BestImprove': {
        #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MDrB3i/',
        #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MDrB3i_MinMax_Static_All.txt',
        #             'AlsoLQRmv': True},
        #         'S+DIROnlyReservoir': {
        #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MDrOr/',
        #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MDrOr_MinMax_Static_All.txt',
        #             'AlsoLQRmv': True},
        #         'S+LTS MA': {
        #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_TsMa/',
        #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_TsMa_MinMax_Static_All.txt',
        #             'AlsoLQRmv': True},
        #         'S+LTS LR': {
        #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_TsLr/',
        #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_TsLr_MinMax_Static_All.txt',
        #             'AlsoLQRmv': True},
        #         'S+LTS ARMA': {
        #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_TsArma/',
        #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_TsArma_MinMax_Static_All.txt',
        #             'AlsoLQRmv': True},
        #         }
        #     init_round = 4
    if ablation == True:
        model_files_dict = {
            'S+MSMM': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_M_STD_Min_Max_AllByMonths.txt'}
            }
        abla_feat_list = ['BERTScore', 'BERTScore_MaxXXSnaps','BERTScore_MinXXSnaps','BERTScore_MXXSnaps','BERTScore_STDXXSnaps',
                          'BM25Score','BM25Score_MaxXXSnaps','BM25Score_MinXXSnaps','BM25Score_MXXSnaps','BM25Score_STDXXSnaps',
                          'Ent', 'Ent_MaxXXSnaps', 'Ent_MinXXSnaps','Ent_MXXSnaps', 'Ent_STDXXSnaps',
                          'FracStops', 'FracStops_MaxXXSnaps', 'FracStops_MinXXSnaps','FracStops_MXXSnaps', 'FracStops_STDXXSnaps',
                          'Len', 'Len_MaxXXSnaps', 'Len_MinXXSnaps','Len_MXXSnaps', 'Len_STDXXSnaps',
                          'LMIR.DIR', 'LMIR.DIR_MaxXXSnaps', 'LMIR.DIR_MinXXSnaps','LMIR.DIR_MXXSnaps', 'LMIR.DIR_STDXXSnaps',
                          'SimClueWeb_MaxXXSnaps', 'SimClueWeb_MinXXSnaps','SimClueWeb_MXXSnaps', 'SimClueWeb_STDXXSnaps',
                          'StopCover', 'StopCover_MaxXXSnaps', 'StopCover_MinXXSnaps', 'StopCover_MXXSnaps', 'StopCover_STDXXSnaps',
                          'TFNormSum', 'TFNormSum_MaxXXSnaps', 'TFNormSum_MinXXSnaps', 'TFNormSum_MXXSnaps', 'TFNormSum_STDXXSnaps',
                          'TFSum', 'TFSum_MaxXXSnaps', 'TFSum_MinXXSnaps', 'TFSum_MXXSnaps', 'TFSum_STDXXSnaps',
           ]

        # basline_model_dict_examp = {
        #     'Sample': {
        #         'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_Ablation_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
        #         'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_Abla_<AblaFeat>_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_M_STD_Min_Max_AllByMonths.txt'},
        #     }

        basline_model_dict = {}
        for abla_feat in abla_feat_list:
            basline_model_dict[abla_feat.replace('XXSnaps', '').replace('SimClueWeb', 'Sim')] = {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_Ablation_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_Abla_' + abla_feat+ '_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_M_STD_Min_Max_AllByMonths.txt',
                'NeedAdjust': True}

        print(basline_model_dict)
    # if svm_compare == True:
    #     model_files_dict = {
    #         'S': {
    #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
    #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_Static_All.txt',
    #             },
    #         'S+MSMM+MG': {
    #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
    #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_Static_M_STD_Min_Max_MG_AllByMonths.txt',
    #                 }
    #         }
    #     basline_model_dict = {
    #         'SVM S': {
    #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
    #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_Static_All.txt',
    #         },
    #         'SVM S+MSMM+MG': {
    #             'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
    #             'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_Static_M_STD_Min_Max_MG_AllByMonths.txt',
    #         }
    #     }

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

    if graphs_per_round == True:
        model_files_dict = {
            'S': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_All.txt',
                'NeedAdjust': True},
            'S+MSMM': {
                'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT/',
                'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_BRT_MinMax_Static_M_STD_Min_Max_AllByMonths.txt',
                'NeedAdjust': True},}
        basline_model_dict ={
            'LM DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/basic_lm/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_LMIR.DIR.txt'},

            'MM Prev3Winners DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                    'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3Winners_MixtureDIR_LoO_Results.txt'},
            'MM Prev3BestImprove DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                        'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3BestImprove_MixtureDIR_LoO_Results.txt'},
            'MM OnlyReservoir DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                     'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureDIR_OnlyReservoir_LoO_Results.txt'},
                }
        res_df = create_per_round_reults_dataframe_for_models(
            model_files_dict=model_files_dict,
            basline_model_dict=basline_model_dict,
            dataset=dataset,
            qrel_filepath=qrel_filepath,
            init_round=init_round,
            round_limit=round_limit)

    else:
        res_df = create_reults_dataframe_for_models(
            model_files_dict=model_files_dict,
            basline_model_dict=basline_model_dict,
            dataset=dataset,
            qrel_filepath=qrel_filepath,
            is_svm_rank=is_svm_rank,
            init_round=init_round,
            round_limit=round_limit,
            ablation=ablation)
    save_filename = dataset.upper() + '_LambdaMART_All_Results.tsv'
    if graphs_per_round == True:
        save_filename = save_filename.replace('_All_', '_PerRound_')
    if is_svm_rank == True:
        save_filename = save_filename.replace('_LambdaMART_', '_SVMRank_')
    if static_feat_compare == True:
        save_filename = save_filename.replace('Results', 'S_Features_Results')
    if ablation == True:
        save_filename = save_filename.replace('_Results','_Ablation_Results')
    if svm_compare == True:
        save_filename = save_filename.replace('_Results', '_SVM_COMPARE_Results')
    res_df.to_csv(save_filename, sep = '\t', index = False)

