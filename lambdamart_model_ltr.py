import sys
from utils import *

from rank_svm_model import run_bash_command, prepare_svmr_model_data, turn_df_to_feature_str_for_model, split_to_train_test, get_trec_prepared_df_form_res_df, create_sinificance_df, create_fold_list_for_cv


def run_lambdamart_model(test_file, model_file, predictions_folder):
    predictions_file = os.path.join(predictions_folder, 'Prdictions.txt' )
    command = "java -jar /mnt/bi-strg3/v/zivvasilisky/ziv/env/ranklib/RankLib-2.14.jar -load " + model_file + " -rank " + test_file + " -score " + predictions_file
    print("##Running command: "+command+"##")
    out = run_bash_command(command)
    print("Output of ranking command: "+str(out))
    sys.stdout.flush()
    return predictions_file


def learn_lambdamart_model(train_file, models_folder, tree_num, leaf_num):
    model_file = os.path.join(models_folder , "model.txt")
    command = "java -jar /mnt/bi-strg3/v/zivvasilisky/ziv/env/ranklib/RankLib-2.14.jar -train " + train_file + " -ranker 6 -metric2t NDCG@5 -save " + model_file
    command += " -tree " + str(tree_num) + " -leaf " +str(leaf_num)
    out = run_bash_command(command)
    print(out)
    sys.stdout.flush()
    return model_file


def get_predictions_list(
        predictions_filename):

    with open(predictions_filename, 'r') as f:
        predications = f.read()

    predications_list = []
    for row in predications.split('\n'):
        if row != "":
            predications_list.append(row.split('\t')[2])

    return predications_list

def train_and_test_model_on_config(
        base_feature_filename,
        snapshot_limit,
        feature_list,
        start_test_q,
        end_test_q,
        feature_groupname,
        normalize_method,
        qrel_filepath,
        snap_chosing_method=None,
        snap_calc_limit=None):

    base_res_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/'
    model_inner_folder = base_feature_filename.replace('All_features_with_meta.tsv', '') + 'SNL' + str(snapshot_limit)
    feature_folder = feature_groupname
    # if normalize_relevance == True:
    feature_folder += '_' + normalize_method
    fold_folder = str(start_test_q) + '_' + str(end_test_q) + "_" + str(snap_chosing_method)

    for hirarcy_folder in [model_inner_folder, feature_folder, fold_folder]:
        base_res_folder = os.path.join(base_res_folder, hirarcy_folder)
        if not os.path.exists(base_res_folder):
            os.mkdir(base_res_folder)

    best_snap_num = snap_calc_limit

    feat_df = prepare_svmr_model_data(
        base_feature_filename=base_feature_filename,
        snapshot_limit=int(snapshot_limit),
        feature_list=feature_list,
        normalize_method=normalize_method,
        limited_snaps_num=best_snap_num,
        lambdamart=True)

    print("Model Data Prepared...")
    sys.stdout.flush()
    train_df, test_df, valid_df, seed = split_to_train_test(
        start_test_q=start_test_q,
        end_test_q=end_test_q,
        feat_df=feat_df,
        base_feature_filename=base_feature_filename)


    valid_df_cp = valid_df.copy()
    with open(os.path.join(base_res_folder, 'train.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(train_df, feature_list=feature_list))

    with open(os.path.join(base_res_folder, 'valid.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(valid_df, feature_list=feature_list))

    num_tree_optional_list = [250, 500]
    num_leaf_optional_list = [3, 2, 5]
    best_map = 0.0

    for tree_num in num_tree_optional_list:
        for leaf_num in num_leaf_optional_list:
            print("Running validation tree num: " + str(tree_num)) + " leaf num: " + str(leaf_num)
            model_filename = learn_lambdamart_model(
                train_file=os.path.join(base_res_folder, 'train.dat'),
                models_folder=base_res_folder,
                tree_num=tree_num,
                leaf_num=leaf_num)

            predictions_filename = run_lambdamart_model(
                test_file=os.path.join(base_res_folder, 'valid.dat'),
                model_file=model_filename,
                predictions_folder=base_res_folder)

            predications = get_predictions_list(predictions_filename)

            valid_df['ModelScore'] = predications
            valid_df['ModelScore'] = valid_df['ModelScore'].apply(lambda x: float(x))
            curr_res_df = get_trec_prepared_df_form_res_df(
                scored_docs_df=valid_df,
                score_colname='ModelScore')
            curr_file_name = 'Curr_valid_res.txt'
            with open(os.path.join(base_res_folder, curr_file_name), 'w') as f:
                f.write(convert_df_to_trec(curr_res_df))

            res_dict = get_ranking_effectiveness_for_res_file(
                file_path=base_res_folder,
                filename=curr_file_name,
                qrel_filepath=qrel_filepath)

            if float(res_dict['Map']) > best_map:
                best_map = float(res_dict['Map'])
                best_tree_num = tree_num
                beat_leaf_num = leaf_num

    train_df = train_df.append(valid_df_cp, ignore_index=True)
    train_df.sort_values('QueryNum', inplace=True)

    with open(os.path.join(base_res_folder, 'train.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(train_df, feature_list=feature_list))

    best_params_str = 'SnapLim: ' + str(best_snap_num) + '\n' + "TreeNum: " +str(best_tree_num) +'\n' +"LeafNum: " +str(beat_leaf_num)
    with open(os.path.join(base_res_folder, 'hyper_params.txt'), 'w') as f:
        f.write(best_params_str)

    with open(os.path.join(base_res_folder, 'test.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(test_df, feature_list=feature_list))

    print("Strating Train : " + model_inner_folder + ' ' + feature_folder + ' ' + fold_folder)
    sys.stdout.flush()
    model_filename = learn_lambdamart_model(
        train_file=os.path.join(base_res_folder, 'train.dat'),
        models_folder=base_res_folder,
        tree_num=best_tree_num,
        leaf_num=beat_leaf_num)

    print("Strating Test : " + model_inner_folder + ' ' + feature_folder + ' ' + fold_folder)
    sys.stdout.flush()

    predictions_filename = run_lambdamart_model(
        test_file=os.path.join(base_res_folder, 'test.dat'),
        model_file=model_filename,
        predictions_folder=base_res_folder)

    predications = get_predictions_list(predictions_filename)

    test_df['ModelScore'] = predications
    test_df['ModelScore'] = test_df['ModelScore'].apply(lambda x: float(x))

    params_list = [best_tree_num, beat_leaf_num]
    hyper_params = ['Tree', 'Leaf']
    if best_snap_num is not None:
        hyper_params.append('SnapLimit')
        params_list.append(best_snap_num)
    params_df = pd.DataFrame(columns=['Fold'] + hyper_params)
    params_df.loc[0] = [str(start_test_q) + '_' + str(end_test_q)] + params_list

    return test_df, params_df

def run_cv_for_config(
        base_feature_filename,
        snapshot_limit,
        feature_groupname,
        retrieval_model,
        normalize_method,
        qrel_filepath,
        snap_chosing_method,
        train_leave_one_out,
        snap_calc_limit):

    k_fold, fold_list = create_fold_list_for_cv(
        base_feature_filename=base_feature_filename,
        train_leave_one_out=train_leave_one_out)

    feature_list = []
    broken_feature_groupname = feature_groupname.split('_')
    len_handled = 0
    base_feature_list = ['Boolean.AND', 'Boolean.OR', 'CoverQueryNum', 'CoverQueryRatio', 'Ent', 'FracStops',
                         'IDF', 'Len', 'LMIR.ABS', 'LMIR.DIR', 'LMIR.JM', 'StopCover', 'TFSum', 'TFMin', 'TFMax',
                         'TFMean', 'TFStd',
                         'TFIDFSum', 'TFIDFMin', 'TFIDFMax', 'TFIDFMean', 'TFIDFStd', 'TFNormSum', 'TFNormMin',
                         'TFNormMax',
                         'TFNormMean', 'TFNormStd', 'VSM', 'SimClueWeb','StopwordsRatio','Stopwords','-Query-SW', 'BM25Score']

    if 'Static' in broken_feature_groupname:
        feature_list.extend(base_feature_list)
        len_handled += 1
    if 'M' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_M')
        len_handled += 1
    if 'STD' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_STD')
        len_handled += 1
    if 'RMG' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_RMG')
        len_handled += 1
    if 'MG' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_MG')
        len_handled += 1
    if 'LG' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_LG')
        len_handled += 1

    if 'RMGXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_RMGXXSnaps')
        len_handled += 1
    if 'MGXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_MGXXSnaps')
        len_handled += 1
    if 'MXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_MXXSnaps')
        len_handled += 1
    if 'STDXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_STDXXSnaps')
        len_handled += 1
    if 'MinXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_MinXXSnaps')
        len_handled += 1
    if 'MaxXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_MaxXXSnaps')
        len_handled += 1

    feature_groups_num = len(broken_feature_groupname)
    if snap_calc_limit is not None:
        feature_groups_num = feature_groups_num - 1
    if len_handled != feature_groups_num:
        raise Exception('Undefined feature group!')

    test_score_df = pd.DataFrame({})
    if 'XXSnap' in feature_groupname:
        feature_groupname += 'By' + snap_chosing_method

    for i in range(k_fold):
        init_q = fold_list[i][0]
        end_q = fold_list[i][1]
        fold_test_df, fold_params_df = train_and_test_model_on_config(
            base_feature_filename=base_feature_filename,
            snapshot_limit=snapshot_limit,
            feature_list=feature_list,
            start_test_q=init_q,
            end_test_q=end_q,
            feature_groupname=feature_groupname + '_' + retrieval_model,
            normalize_method=normalize_method,
            qrel_filepath=qrel_filepath,
            snap_chosing_method=snap_chosing_method,
            snap_calc_limit=snap_calc_limit)
        if i == 0:
            params_df = fold_params_df
        else:
            params_df = params_df.append(fold_params_df, ignore_index=True)
        test_score_df = test_score_df.append(fold_test_df, ignore_index=True)
    return test_score_df, params_df


def run_grid_search_over_params_for_config(
        base_feature_filename,
        snapshot_limit,
        retrieval_model,
        normalize_method,
        snap_chosing_method,
        tarin_leave_one_out,
        feat_group_list,
        calc_ndcg_mrr):

    # optional_c_list = [0.2, 0.1, 0.01, 0.001]
    ## num 1
    # optional_feat_groups_list = ['All','Static','MG','LG','M','RMG','Static_LG','Static_MG'
    #                                 ,'Static_M', 'Static_RMG']
    ## num 2
    # optional_feat_groups_list = ['Static','MGXXSnap', 'MXXSnap','RMGXXSnap','Static_MGXXSnap'
    #                                     ,'Static_MXXSnap', 'Static_RMGXXSnap','MGXXSnap_MXXSnap_RMGXXSnap']
    ## num 3
    if feat_group_list is None:
        optional_feat_groups_list = ['Static','Static_MXXSnap_STDXXSnap_MinXXSnap_MaxXXSnap_MGXXSnap_RMGXXSnap',
                                     'Static_MXXSnap_STDXXSnap_MinXXSnap_MaxXXSnap',
                                     'Static_MXXSnap_STDXXSnap_MinXXSnap_MaxXXSnap_MGXXSnap',
                                     'Static_MGXXSnap', 'Static_RMGXXSnap'
                                     ]
    else:
        optional_feat_groups_list = feat_group_list
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/'
    save_summary_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/'
    if '2008' in base_feature_filename:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc"
    elif 'ASRC' in base_feature_filename:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel"
    elif 'BOT' in base_feature_filename:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents_fixed.relevance"
    elif 'HERD_CONTROL' in base_feature_filename:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/control.rel"
    elif 'UNITED' in base_feature_filename:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel'
    else:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels_cw12.adhoc"

    if snap_chosing_method == 'Months':
        snap_limit_options = [
            # '3M', '6M', '9M', '1Y', '1.5Y',
            'All']
    elif snap_chosing_method == 'SnapNum':
        snap_limit_options = [
            # 3, 5, 7, 10, 15,
            'All']
    else:
        raise Exception("Unknown snap_chosing_method!")

    model_base_filename = base_feature_filename.replace('All_features_with_meta.tsv', '') + 'SNL' + str(
        snapshot_limit) + "_" + retrieval_model + "_By" + snap_chosing_method

    if not os.path.exists(os.path.join(save_folder, model_base_filename)):
        os.mkdir(os.path.join(save_folder, model_base_filename))
    save_folder = os.path.join(save_folder, model_base_filename)


    model_base_filename += '_' + normalize_method
    if tarin_leave_one_out == True:
        model_base_filename += '_LoO'
    additional_measures = []
    if calc_ndcg_mrr == True:
        additional_measures =  ['NDCG@1', 'NDCG@3', 'MRR', 'nMRR']
    model_summary_df = pd.DataFrame(columns=['FeatureGroup', 'Map', 'P@5', 'P@10'] +additional_measures)
    next_idx = 0
    per_q_res_dict = {}
    feat_group_list_str = ""
    # for optional_c in optional_c_list:
    for curr_feat_group in optional_feat_groups_list:
        feat_group_list_str +=  "__" + curr_feat_group.replace('XXSnap','')
        if 'XXSnap' in curr_feat_group:
            snap_limit_list = snap_limit_options
        else:
            snap_limit_list = [None]
        for snap_limit in snap_limit_list:
            if snap_limit is None:
                feat_group = curr_feat_group
            else:
                feat_group = curr_feat_group + "_" + str(snap_limit)
            test_res_df, tmp_params_df = run_cv_for_config(
                base_feature_filename=base_feature_filename,
                snapshot_limit=snapshot_limit,
                feature_groupname=feat_group,
                retrieval_model=retrieval_model,
                normalize_method=normalize_method,
                qrel_filepath=qrel_filepath,
                snap_chosing_method=snap_chosing_method,
                train_leave_one_out=tarin_leave_one_out,
                snap_calc_limit=snap_limit)

            tmp_params_df['FeatGroup'] = feat_group
            if 'XXSnap' in feat_group:
                feat_group = feat_group.replace('XXSnap','') + 'By' + snap_chosing_method

            if next_idx == 0:
                curr_res_df = get_trec_prepared_df_form_res_df(
                    scored_docs_df=test_res_df,
                    score_colname=retrieval_model + 'Score')
                insert_row = [retrieval_model]
                curr_file_name = model_base_filename + '_' + retrieval_model + '.txt'
                with open(os.path.join(save_folder, curr_file_name), 'w') as f:
                    f.write(convert_df_to_trec(curr_res_df))

                res_dict = get_ranking_effectiveness_for_res_file_per_query(
                    file_path=save_folder,
                    filename=curr_file_name,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=calc_ndcg_mrr)
                for measure in ['Map', 'P_5', 'P_10']+additional_measures:
                    insert_row.append(res_dict['all'][measure])
                per_q_res_dict[retrieval_model] = res_dict
                model_summary_df.loc[next_idx] = insert_row
                next_idx += 1
                params_df = tmp_params_df
            else:
                params_df = params_df.append(tmp_params_df, ignore_index=True)

            curr_res_df = get_trec_prepared_df_form_res_df(
                scored_docs_df=test_res_df,
                score_colname='ModelScore')
            insert_row = [feat_group.replace('_', '+')]
            curr_file_name = model_base_filename + '_' + feat_group + '.txt'
            with open(os.path.join(save_folder, curr_file_name), 'w') as f:
                f.write(convert_df_to_trec(curr_res_df))

            res_dict = get_ranking_effectiveness_for_res_file_per_query(
                file_path=save_folder,
                filename=curr_file_name,
                qrel_filepath=qrel_filepath,
                calc_ndcg_mrr=calc_ndcg_mrr)
            for measure in ['Map', 'P_5', 'P_10']+additional_measures:
                insert_row.append(res_dict['all'][measure])
            per_q_res_dict[feat_group.replace('_', '+')] = res_dict
            model_summary_df.loc[next_idx] = insert_row
            next_idx += 1

    significance_df = create_sinificance_df(per_q_res_dict, calc_ndcg_mrr)
    model_summary_df = pd.merge(
        model_summary_df,
        significance_df,
        on=['FeatureGroup'],
        how='inner')

    model_summary_df.to_csv(os.path.join(save_summary_folder, model_base_filename + feat_group_list_str + '.tsv'), sep='\t', index=False)
    params_df.to_csv(os.path.join(save_summary_folder, model_base_filename + feat_group_list_str + '_Params.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    operation = sys.argv[1]

    if operation == 'GridSearchParams':
        base_feature_filename = sys.argv[2]
        snapshot_limit = int(sys.argv[3])
        retrieval_model = sys.argv[4]
        normalize_method = sys.argv[5]
        snap_chosing_method = sys.argv[6]
        tarin_leave_one_out = ast.literal_eval(sys.argv[7])
        feat_group_list = ast.literal_eval(sys.argv[8])
        calc_ndcg_mrr = ast.literal_eval(sys.argv[9])

        run_grid_search_over_params_for_config(
            base_feature_filename=base_feature_filename,
            snapshot_limit=snapshot_limit,
            retrieval_model=retrieval_model,
            normalize_method=normalize_method,
            snap_chosing_method=snap_chosing_method,
            tarin_leave_one_out=tarin_leave_one_out,
            feat_group_list=feat_group_list,
            calc_ndcg_mrr=calc_ndcg_mrr)