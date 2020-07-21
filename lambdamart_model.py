import sys
from utils import *

from rank_svm_model import run_bash_command, prepare_svmr_model_data, turn_df_to_feature_str_for_model, split_to_train_test, get_trec_prepared_df_form_res_df, create_sinificance_df


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
    command = "java -jar /mnt/bi-strg3/v/zivvasilisky/ziv/env/ranklib/RankLib-2.14.jar -train " + train_file + " -ranker 6 -metric2t NDCG@20 -save " + model_file
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


def learn_best_num_of_snaps(
        base_feature_filename,
        snapshot_limit,
        feature_list,
        start_test_q,
        end_test_q,
        base_res_folder,
        qrel_filepath,
        normalize_relevance,
        snap_chosing_method):


    if snap_chosing_method == 'SnapNum':
        optional_snap_limit = [2,3,4,5,6,7, 8,9, 10, 15, 'All']
    elif snap_chosing_method == 'Months':
        optional_snap_limit = ['2M','3M','5M','6M','7M','8M','9M','10M','1Y','1.5Y','All']
    else:
        raise Exception('learn_best_num_of_snaps: Unknown snap_chosing_method')


    tree_num = 250
    leaf_num = 5
    best_snap_lim = None
    best_map = 0.0
    seed = None
    for snap_lim in optional_snap_limit:
        print("Running validation snap limit: " + str(snap_lim))
        sys.stdout.flush()
        feat_df = prepare_svmr_model_data(
            base_feature_filename=base_feature_filename,
            snapshot_limit=int(snapshot_limit),
            feature_list=feature_list,
            normalize_relvance=normalize_relevance,
            limited_snaps_num=snap_lim,
            lambdamart=True)

        train_df, test_df, valid_df, seed = split_to_train_test(
            start_test_q=start_test_q,
            end_test_q=end_test_q,
            feat_df=feat_df,
            seed=seed)

        with open(os.path.join(base_res_folder, 'train.dat'), 'w') as f:
            f.write(turn_df_to_feature_str_for_model(train_df, feature_list=feature_list))

        with open(os.path.join(base_res_folder, 'valid.dat'), 'w') as f:
            f.write(turn_df_to_feature_str_for_model(valid_df, feature_list=feature_list))

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
            best_snap_lim = snap_lim

    return best_snap_lim


def train_and_test_model_on_config(
        base_feature_filename,
        snapshot_limit,
        feature_list,
        start_test_q,
        end_test_q,
        feature_groupname,
        normalize_relevance,
        qrel_filepath,
        snap_chosing_method=None):

    base_res_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/'
    model_inner_folder = base_feature_filename.replace('All_features_with_meta.tsv', '') + 'SNL' + str(snapshot_limit)
    feature_folder = feature_groupname
    if normalize_relevance == True:
        feature_folder += '_NR'
    fold_folder = str(start_test_q) + '_' + str(end_test_q) + "_" + str(snap_chosing_method)

    for hirarcy_folder in [model_inner_folder, feature_folder, fold_folder]:
        base_res_folder = os.path.join(base_res_folder, hirarcy_folder)
        if not os.path.exists(base_res_folder):
            os.mkdir(base_res_folder)

    best_snap_num = None
    if 'XXSnap' in feature_groupname:
        best_snap_num = learn_best_num_of_snaps(
            base_feature_filename=base_feature_filename,
            snapshot_limit=snapshot_limit,
            feature_list=feature_list,
            start_test_q=start_test_q,
            end_test_q=end_test_q,
            base_res_folder=base_res_folder,
            qrel_filepath=qrel_filepath,
            normalize_relevance=normalize_relevance,
            snap_chosing_method=snap_chosing_method)

    feat_df = prepare_svmr_model_data(
        base_feature_filename=base_feature_filename,
        snapshot_limit=int(snapshot_limit),
        feature_list=feature_list,
        normalize_relvance=normalize_relevance,
        limited_snaps_num=best_snap_num,
        lambdamart=True)

    print("Model Data Prepared...")
    sys.stdout.flush()
    train_df, test_df, valid_df, seed = split_to_train_test(
        start_test_q=start_test_q,
        end_test_q=end_test_q,
        feat_df=feat_df)


    valid_df_cp = valid_df.copy()
    with open(os.path.join(base_res_folder, 'train.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(train_df, feature_list=feature_list))

    with open(os.path.join(base_res_folder, 'valid.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(valid_df, feature_list=feature_list))

    num_tree_optional_list = [250, 500]
    num_leaf_optional_list = [3, 2, 5, 10]
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
        normalize_relevance,
        qrel_filepath,
        snap_chosing_method,
        train_leave_one_out):

    k_fold = 10
    if '2008' in base_feature_filename:
        init_q = 1
        end_q = 20
        query_bulk = 20
        num_q = 198
    else:
        init_q = 201
        end_q = 210
        query_bulk = 10
        num_q = 100

    if train_leave_one_out == True:
        k_fold = num_q
        query_bulk = 1
        end_q = init_q

    feature_list = []
    broken_feature_groupname = feature_groupname.split('_')
    len_handled = 0
    if feature_groupname == 'All':
        feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                        'QueryWords', 'Stopwords', 'TextLen', '-Query-SW',
                        'QueryTermsRatio_LG', 'StopwordsRatio_LG', 'Entropy_LG', 'SimClueWeb_LG',
                        'QueryWords_LG', 'Stopwords_LG', 'TextLen_LG', '-Query-SW_LG',
                        'QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                        'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG',
                        'QueryTermsRatio_M', 'StopwordsRatio_M', 'Entropy_M', 'SimClueWeb_M',
                        'QueryWords_M', 'Stopwords_M', 'TextLen_M', '-Query-SW_M',
                        'QueryTermsRatio_STD', 'StopwordsRatio_STD', 'Entropy_STD', 'SimClueWeb_STD',
                        'QueryWords_STD', 'Stopwords_STD', 'TextLen_STD', '-Query-SW_STD',
                        'QueryTermsRatio_MRG', 'StopwordsRatio_MRG', 'Entropy_MRG', 'SimClueWeb_MRG',
                        'QueryWords_MRG', 'Stopwords_MRG', 'TextLen_MRG', '-Query-SW_MRG',
                        # 'QueryTermsRatio_M/STD', 'StopwordsRatio_M/STD', 'Entropy_M/STD', 'SimClueWeb_M/STD',
                        # 'QueryWords_M/STD', 'Stopwords_M/STD', 'TextLen_M/STD', '-Query-SW_M/STD'
                        ]
        len_handled += 1

    if 'Static' in broken_feature_groupname:
        feature_list.extend(['QueryTermsRatio', 'StopwordsRatio', 'Entropy',
                             'QueryWords', 'Stopwords', 'TextLen', '-Query-SW'])
        len_handled += 1
    if 'M' in broken_feature_groupname:
        feature_list.extend(['QueryTermsRatio_M', 'StopwordsRatio_M', 'Entropy_M', 'SimClueWeb_M',
                             'QueryWords_M', 'Stopwords_M', 'TextLen_M', '-Query-SW_M'])
        len_handled += 1
    if 'STD' in broken_feature_groupname:
        feature_list.extend(['QueryTermsRatio_STD', 'StopwordsRatio_STD', 'Entropy_STD', 'SimClueWeb_STD',
                             'QueryWords_STD', 'Stopwords_STD', 'TextLen_STD', '-Query-SW_STD'])
        len_handled += 1
    if 'RMG' in broken_feature_groupname:
        feature_list.extend(['QueryTermsRatio_MRG', 'StopwordsRatio_MRG', 'Entropy_MRG', 'SimClueWeb_MRG',
                             'QueryWords_MRG', 'Stopwords_MRG', 'TextLen_MRG', '-Query-SW_MRG'])
        len_handled += 1
    if 'MG' in broken_feature_groupname:
        feature_list.extend(['QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                             'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG'])
        len_handled += 1
    if 'LG' in broken_feature_groupname:
        feature_list.extend(['QueryTermsRatio_LG', 'StopwordsRatio_LG', 'Entropy_LG', 'SimClueWeb_LG',
                             'QueryWords_LG', 'Stopwords_LG', 'TextLen_LG', '-Query-SW_LG'])
        len_handled += 1

    if 'RMGXXSnap' in broken_feature_groupname:
        feature_list.extend(
            ['QueryTermsRatio_RMGXXSnaps', 'StopwordsRatio_RMGXXSnaps', 'Entropy_RMGXXSnaps', 'SimClueWeb_RMGXXSnaps',
             'QueryWords_RMGXXSnaps', 'Stopwords_RMGXXSnaps', 'TextLen_RMGXXSnaps', '-Query-SW_RMGXXSnaps'])
        len_handled += 1
    if 'MGXXSnap' in broken_feature_groupname:
        feature_list.extend(
            ['QueryTermsRatio_MGXXSnaps', 'StopwordsRatio_MGXXSnaps', 'Entropy_MGXXSnaps', 'SimClueWeb_MGXXSnaps',
             'QueryWords_MGXXSnaps', 'Stopwords_MGXXSnaps', 'TextLen_MGXXSnaps', '-Query-SW_MGXXSnaps'])
        len_handled += 1
    if 'MXXSnap' in broken_feature_groupname:
        feature_list.extend(
            ['QueryTermsRatio_MXXSnaps', 'StopwordsRatio_MXXSnaps', 'Entropy_MXXSnaps', 'SimClueWeb_MXXSnaps',
             'QueryWords_MXXSnaps', 'Stopwords_MXXSnaps', 'TextLen_MXXSnaps', '-Query-SW_MXXSnaps'])
        len_handled += 1
    if 'STDXXSnap' in broken_feature_groupname:
        feature_list.extend(
            ['QueryTermsRatio_STDXXSnaps', 'StopwordsRatio_STDXXSnaps', 'Entropy_STDXXSnaps', 'SimClueWeb_STDXXSnaps',
             'QueryWords_STDXXSnaps', 'Stopwords_STDXXSnaps', 'TextLen_STDXXSnaps', '-Query-SW_STDXXSnaps'])
        len_handled += 1
    if 'MinXXSnap' in broken_feature_groupname:
        feature_list.extend(
            ['QueryTermsRatio_MinXXSnaps', 'StopwordsRatio_MinXXSnaps', 'Entropy_MinXXSnaps', 'SimClueWeb_MinXXSnaps',
             'QueryWords_MinXXSnaps', 'Stopwords_MinXXSnaps', 'TextLen_MinXXSnaps', '-Query-SW_MinXXSnaps'])
        len_handled += 1
    if 'MaxXXSnap' in broken_feature_groupname:
        feature_list.extend(
            ['QueryTermsRatio_MaxXXSnaps', 'StopwordsRatio_MaxXXSnaps', 'Entropy_MaxXXSnaps', 'SimClueWeb_MaxXXSnaps',
             'QueryWords_MaxXXSnaps', 'Stopwords_MaxXXSnaps', 'TextLen_MaxXXSnaps', '-Query-SW_MaxXXSnaps'])
        len_handled += 1

    if len_handled != len(broken_feature_groupname):
        raise Exception('Undefined feature group!')

    if retrieval_model == 'LM':
        feature_list.append('LMScore')
    elif retrieval_model == 'BM25':
        feature_list.append('BM25Score')

    test_score_df = pd.DataFrame({})
    if 'XXSnap' in feature_groupname:
        feature_groupname += 'By' + snap_chosing_method

    for i in range(k_fold):
        fold_test_df, fold_params_df = train_and_test_model_on_config(
            base_feature_filename=base_feature_filename,
            snapshot_limit=snapshot_limit,
            feature_list=feature_list,
            start_test_q=init_q,
            end_test_q=end_q,
            feature_groupname=feature_groupname + '_' + retrieval_model,
            normalize_relevance=normalize_relevance,
            qrel_filepath=qrel_filepath,
            snap_chosing_method=snap_chosing_method)
        init_q += query_bulk
        end_q += query_bulk
        if (init_q in [95, 100]) and (init_q == end_q):
            init_q += 1
            end_q += 1
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
        normalize_relevance,
        snap_chosing_method,
        tarin_leave_one_out,
        feat_group_list):

    # optional_c_list = [0.2, 0.1, 0.01, 0.001]
    ## num 1
    # optional_feat_groups_list = ['All','Static','MG','LG','M','RMG','Static_LG','Static_MG'
    #                                 ,'Static_M', 'Static_RMG']
    ## num 2
    # optional_feat_groups_list = ['Static','MGXXSnap', 'MXXSnap','RMGXXSnap','Static_MGXXSnap'
    #                                     ,'Static_MXXSnap', 'Static_RMGXXSnap','MGXXSnap_MXXSnap_RMGXXSnap']
    ## num 3
    if feat_group_list is None:
        optional_feat_groups_list = ['Static','Static_MXXSnap_STDXXSnap_MinXXSnap_MaxXXSnap_MGXXSnap_RMGXXSnap']
    else:
        optional_feat_groups_list = feat_group_list
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/'
    save_summary_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/'
    if '2008' in base_feature_filename:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc"
    else:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels_cw12.adhoc"

    model_base_filename = base_feature_filename.replace('All_features_with_meta.tsv', '') + 'SNL' + str(
        snapshot_limit) + "_" + retrieval_model + "_By" + snap_chosing_method
    if normalize_relevance == True:
        model_base_filename += '_NR'
    if tarin_leave_one_out == True:
        model_base_filename += '_LoO'
    model_summary_df = pd.DataFrame(columns=['FeatureGroup', 'Map', 'P@5', 'P@10'])
    next_idx = 0
    per_q_res_dict = {}
    feat_group_list_str = ""
    # for optional_c in optional_c_list:
    for feat_group in optional_feat_groups_list:
        feat_group_list_str +=  "__" + feat_group
        test_res_df, tmp_params_df = run_cv_for_config(
            base_feature_filename=base_feature_filename,
            snapshot_limit=snapshot_limit,
            feature_groupname=feat_group,
            retrieval_model=retrieval_model,
            normalize_relevance=normalize_relevance,
            qrel_filepath=qrel_filepath,
            snap_chosing_method=snap_chosing_method,
            train_leave_one_out=tarin_leave_one_out)

        tmp_params_df['FeatGroup'] = feat_group
        if 'XXSnap' in feat_group:
            feat_group += 'By' + snap_chosing_method

        if next_idx == 0:
            curr_res_df = get_trec_prepared_df_form_res_df(
                scored_docs_df=test_res_df,
                score_colname=retrieval_model + 'Score')
            insert_row = ['Basic Retrieval']
            curr_file_name = model_base_filename + '_Benchmark.txt'
            with open(os.path.join(save_folder, curr_file_name), 'w') as f:
                f.write(convert_df_to_trec(curr_res_df))

            res_dict = get_ranking_effectiveness_for_res_file_per_query(
                file_path=save_folder,
                filename=curr_file_name,
                qrel_filepath=qrel_filepath)
            for measure in ['Map', 'P_5', 'P_10']:
                insert_row.append(res_dict['all'][measure])
            per_q_res_dict['Basic Retrieval'] = res_dict
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
            qrel_filepath=qrel_filepath)
        for measure in ['Map', 'P_5', 'P_10']:
            insert_row.append(res_dict['all'][measure])
        per_q_res_dict[feat_group.replace('_', '+')] = res_dict
        model_summary_df.loc[next_idx] = insert_row
        next_idx += 1

    significance_df = create_sinificance_df(per_q_res_dict)
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
        normalize_relevance = ast.literal_eval(sys.argv[5])
        snap_chosing_method = sys.argv[6]
        tarin_leave_one_out = ast.literal_eval(sys.argv[7])
        feat_group_list = ast.literal_eval(sys.argv[8])

        run_grid_search_over_params_for_config(
            base_feature_filename=base_feature_filename,
            snapshot_limit=snapshot_limit,
            retrieval_model=retrieval_model,
            normalize_relevance=normalize_relevance,
            snap_chosing_method=snap_chosing_method,
            tarin_leave_one_out=tarin_leave_one_out,
            feat_group_list=feat_group_list)