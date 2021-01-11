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
        feature_list,
        feature_groupname,
        normalize_method,
        qrel_filepath,
        snap_calc_limit=None):

    base_res_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/trained_models/'

    model_inner_folder = base_feature_filename.replace('All_features_', '').replace('with_meta.tsv', '')
    feature_folder = feature_groupname.replace('XXSnap','XS')
    feature_folder += '_' + normalize_method

    for hirarcy_folder in [model_inner_folder, feature_folder]:
        base_res_folder = os.path.join(base_res_folder, hirarcy_folder)
        if not os.path.exists(base_res_folder):
            os.mkdir(base_res_folder)


    best_snap_num = snap_calc_limit

    feat_df = prepare_svmr_model_data(
        base_feature_filename=base_feature_filename,
        snapshot_limit=int(1),
        feature_list=feature_list,
        normalize_method=normalize_method,
        limited_snaps_num=best_snap_num,
        lambdamart=True)

    print("Model Data Prepared...")
    sys.stdout.flush()
    train_df, test_df, valid_df, seed = split_to_train_test(
        start_test_q=2,
        end_test_q=2,
        feat_df=feat_df,
        base_feature_filename=base_feature_filename)

    train_df = train_df.append(test_df, ignore_index=True)
    train_df.sort_values('QueryNum', inplace=True)

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

            res_dict = calc_ndcg_at_x_for_file(
                file_path=base_res_folder,
                filename=curr_file_name,
                qrel_filepath=qrel_filepath)

            if float(res_dict['NDCG@X']) > best_map:
                best_map = float(res_dict['NDCG@X'])
                best_tree_num = tree_num
                beat_leaf_num = leaf_num

    new_feature_list = feature_list[:]

    train_df = train_df.append(valid_df_cp, ignore_index=True)
    train_df.sort_values('QueryNum', inplace=True)

    with open(os.path.join(base_res_folder, 'train.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(train_df, feature_list=new_feature_list))

    best_params_str = 'SnapLim: ' + str(best_snap_num) + '\n' + "TreeNum: " +str(best_tree_num) +'\n' +"LeafNum: " +str(beat_leaf_num)
    with open(os.path.join(base_res_folder, 'hyper_params.txt'), 'w') as f:
        f.write(best_params_str)

    print("Strating Train : " + model_inner_folder + ' ' + feature_folder)
    sys.stdout.flush()

    model_filename = learn_lambdamart_model(
        train_file=os.path.join(base_res_folder, 'train.dat'),
        models_folder=base_res_folder,
        tree_num=best_tree_num,
        leaf_num=beat_leaf_num)


def run_cv_for_config(
        base_feature_filename,
        feature_groupname,
        normalize_method,
        qrel_filepath,
        snap_calc_limit,
        limited_features_list):

    feature_list = []
    broken_feature_groupname = feature_groupname.split('_')
    len_handled = 0


    base_feature_list = limited_features_list

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

    print(feature_list)
    if len_handled != feature_groups_num:
        raise Exception('Undefined feature group!')

    train_and_test_model_on_config(
            base_feature_filename=base_feature_filename,
            feature_list=feature_list,
            feature_groupname=feature_groupname ,
            normalize_method=normalize_method,
            qrel_filepath=qrel_filepath,
            snap_calc_limit=snap_calc_limit)



def run_grid_search_over_params_for_config(
        base_feature_filename,
        normalize_method,
        snap_choosing_config,
        limited_features_list):

    optional_feat_groups_list = ['Static',
                                 'Static_MXXSnap_STDXXSnap_MinXXSnap_MaxXXSnap_MGXXSnap' ]


    if 'ASRC' in base_feature_filename:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel"
    elif 'UNITED' in base_feature_filename:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel'
    elif 'COMP2020' in base_feature_filename:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/curr_comp.rel'

    for curr_feat_group in optional_feat_groups_list:
        snap_limit =snap_choosing_config
        run_cv_for_config(
            base_feature_filename=base_feature_filename,
            feature_groupname=curr_feat_group,
            normalize_method=normalize_method,
            qrel_filepath=qrel_filepath,
            snap_calc_limit=snap_limit,
            limited_features_list=limited_features_list)



if __name__ == '__main__':
    operation = sys.argv[1]

    if operation == 'TrainModels':
        base_feature_filename = sys.argv[2]
        normalize_method = 'MinMax'
        snap_choosing_config = sys.argv[3]
        limited_features_list = ast.literal_eval(sys.argv[4])

        run_grid_search_over_params_for_config(
            base_feature_filename=base_feature_filename,
            normalize_method=normalize_method,
            snap_choosing_config=snap_choosing_config,
            limited_features_list=limited_features_list)
