import os
import sys
import random
import subprocess
import pandas as pd

from utils import *
from rank_svm_model import  train_and_test_model_on_config, get_trec_prepared_df_form_res_df, create_fold_list_for_cv, create_sinificance_df


def run_cv_for_config(
        base_feature_filename,
        snapshot_limit,
        feature_groupname,
        retrieval_model,
        normalize_method,
        qrel_filepath,
        snap_chosing_method,
        train_leave_one_out,
        snap_calc_limit,
        backward_elimination,
        snap_num_as_hyper_param,
        is_new_server):

    k_fold, fold_list = create_fold_list_for_cv(
        base_feature_filename=base_feature_filename,
        train_leave_one_out=train_leave_one_out)

    feature_list = []
    broken_feature_groupname = feature_groupname.split('_')
    len_handled = 0
    # base_feature_list = ['Boolean.AND', 'Boolean.OR', 'CoverQueryNum', 'CoverQueryRatio', 'Ent', 'FracStops',
    #                      'IDF', 'Len', 'LMIR.ABS', 'LMIR.DIR', 'LMIR.JM', 'StopCover','TFSum','TFMin','TFMax','TFMean','TFStd',
    #                      'TFIDFSum','TFIDFMin','TFIDFMax','TFIDFMean','TFIDFStd','TFNormSum','TFNormMin','TFNormMax',
    #                      'TFNormMean','TFNormStd', 'VSM', 'SimClueWeb','BM25Score']
    base_feature_list = ['CoverQueryNum', 'CoverQueryRatio', 'Ent', 'FracStops',
                         'IDF', 'Len', 'LMIR.DIR', 'LMIR.JM', 'StopCover', 'TFSum', 'TFMin', 'TFMax',
                         'TFMean', 'TFStd','TFIDFSum', 'TFIDFMin', 'TFIDFMax', 'TFIDFMean', 'TFIDFStd', 'TFNormSum', 'TFNormMin',
                         'TFNormMax','TFNormMean', 'TFNormStd', 'SimClueWeb', 'BM25Score']

    if 'Static' in broken_feature_groupname:
        feature_list.extend(base_feature_list)
        len_handled += 1
    if 'M' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat +'_M')
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
            feature_list= feature_list,
            start_test_q=init_q,
            end_test_q=end_q,
            feature_groupname=feature_groupname +'_'+retrieval_model,
            normalize_method=normalize_method,
            qrel_filepath=qrel_filepath,
            snap_chosing_method=snap_chosing_method,
            snap_calc_limit=snap_calc_limit,
            backward_elimination=backward_elimination,
            snap_num_as_hyper_param=snap_num_as_hyper_param,
            is_new_server=is_new_server)
        if i == 0:
            params_df = fold_params_df
        else:
            params_df = params_df.append(fold_params_df, ignore_index = True)
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
        calc_ndcg_mrr,
        backward_elimination,
        snap_num_as_hyper_param,
        snap_choosing_config,
        is_new_server):

    if feat_group_list is None:
        optional_feat_groups_list = ['Static','Static_MXXSnap_STDXXSnap_MinXXSnap_MaxXXSnap_MGXXSnap_RMGXXSnap',
                                    'Static_MXXSnap_STDXXSnap_MinXXSnap_MaxXXSnap',
                                     'Static_MXXSnap_STDXXSnap_MinXXSnap_MaxXXSnap_MGXXSnap',
                                    'Static_MGXXSnap','Static_RMGXXSnap']
    else:
        optional_feat_groups_list = feat_group_list
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/ret_res/'
    save_summary_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/'
    if is_new_server == True:
        save_folder = '~/ziv/results/rank_svm_res/ret_res/'
        save_summary_folder = '~/ziv/results/rank_svm_res/'

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
        snap_limit_options = [snap_choosing_config]
    else:
        raise Exception("Unknown snap_chosing_method!")
    model_base_filename = base_feature_filename.replace('All_features_with_meta.tsv', '') + 'SNL' + str(snapshot_limit) + "_" + retrieval_model + "_By" + snap_chosing_method + "_" + snap_choosing_config

    retrieval_model_addition = ""
    if tarin_leave_one_out == True:
        model_base_filename += '_LoO'
        retrieval_model_addition += '_LoO'
    if backward_elimination == True:
        model_base_filename += '_BElim'
        retrieval_model_addition += '_BElim'
    if snap_num_as_hyper_param == True:
        model_base_filename += '_SnapLim'
        retrieval_model_addition += '_SnapLim'

    if not os.path.exists(os.path.join(save_folder, model_base_filename)):
        os.mkdir(os.path.join(save_folder, model_base_filename))
    save_folder = os.path.join(save_folder, model_base_filename)


    model_base_filename += '_' + normalize_method

    additional_measures = []
    if calc_ndcg_mrr == True:
        additional_measures = ['NDCG@1', 'NDCG@3', 'MRR', 'nMRR']
    model_summary_df = pd.DataFrame(columns = ['FeatureGroup', 'Map', 'P@5', 'P@10']+additional_measures)
    next_idx = 0
    per_q_res_dict = {}
    feat_group_list_str = ""
    for curr_feat_group in optional_feat_groups_list:
        feat_group_list_str += "__" + curr_feat_group.replace('XXSnap','')
        if 'XXSnap' in curr_feat_group:
            snap_limit_list = snap_limit_options
        else:
            snap_limit_list = [snap_choosing_config]
        for snap_limit in snap_limit_list:
            if snap_limit is None:
                feat_group = curr_feat_group
            else:
                feat_group = curr_feat_group + "_" + str(snap_limit)

            test_res_df, tmp_params_df = run_cv_for_config(
                base_feature_filename=base_feature_filename,
                snapshot_limit=snapshot_limit,
                feature_groupname=feat_group,
                retrieval_model=retrieval_model + retrieval_model_addition,
                normalize_method=normalize_method,
                qrel_filepath=qrel_filepath,
                snap_chosing_method=snap_chosing_method,
                train_leave_one_out=tarin_leave_one_out,
                snap_calc_limit=snap_limit,
                backward_elimination=backward_elimination,
                snap_num_as_hyper_param=snap_num_as_hyper_param,
                is_new_server=is_new_server)

            tmp_params_df['FeatGroup'] = feat_group
            if 'XXSnap' in feat_group:
                feat_group = feat_group.replace('XXSnap','') + 'By' + snap_chosing_method

            if next_idx == 0:
                curr_res_df = get_trec_prepared_df_form_res_df(
                    scored_docs_df=test_res_df,
                    score_colname=retrieval_model+'Score')
                insert_row = [retrieval_model]
                curr_file_name =  model_base_filename + '_' + retrieval_model + '.txt'
                with open(os.path.join(save_folder ,curr_file_name), 'w') as f:
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
                next_idx+=1
                params_df = tmp_params_df
            else:
                params_df= params_df.append(tmp_params_df, ignore_index = True)

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

    significance_df = create_sinificance_df(
        per_q_res_dict,
        calc_ndcg_mrr =calc_ndcg_mrr)
    model_summary_df = pd.merge(
        model_summary_df,
        significance_df,
        on = ['FeatureGroup'],
        how = 'inner')
    model_summary_df.to_csv(os.path.join(save_summary_folder, model_base_filename +feat_group_list_str+'.tsv'), sep = '\t', index = False)
    params_df.to_csv(os.path.join(save_summary_folder, model_base_filename +feat_group_list_str+'_Params.tsv'), sep = '\t', index = False)


def create_all_x_snap_aggregations(
        base_feature_filename,
        possible_num_snaps_options_list =None,
        history_only_in_msmm_calc = True):

    base_file_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/'
    base_feature_list = ['Boolean.AND', 'Boolean.OR', 'CoverQueryNum', 'CoverQueryRatio', 'Ent', 'FracStops',
                         'IDF', 'Len', 'LMIR.ABS', 'LMIR.DIR', 'LMIR.JM', 'StopCover','TFSum','TFMin','TFMax','TFMean','TFStd',
                         'TFIDFSum','TFIDFMin','TFIDFMax','TFIDFMean','TFIDFStd','TFNormSum','TFNormMin','TFNormMax',
                         'TFNormMean','TFNormStd', 'VSM','SimClueWeb','StopwordsRatio','Stopwords','-Query-SW', 'BM25Score']

    work_df = pd.read_csv(os.path.join(base_file_folder, base_feature_filename), sep = '\t', index_col = False)
    if '2008' in base_feature_filename:
        start_interval = '2008-12-01'
    else:
        start_interval = '2012-01-01'

    for feature in base_feature_list:
        del work_df[feature + '_Shift']

    possible_num_snaps = [1,2,3,4,5,6,7,8,9,10,12,15,20,'All']
    possible_monthes = ['2M','3M','4M','5M','6M','7M','8M','9M','10M','1Y','1.5Y']
    if possible_num_snaps_options_list is None:
        possible_num_snaps_options_list = possible_num_snaps + possible_monthes

    for num_snaps in possible_num_snaps_options_list:
        if type(num_snaps) == str:
            if num_snaps == 'All':
                tmp_df = work_df.copy()
            else:
                broken_interval = start_interval.split('-')
                if 'M' in num_snaps:
                    rel_month = str(int(broken_interval[1]) - int(num_snaps.replace('M', '')) + 1)
                    curr_interval = broken_interval[0] + '-' + '0'*(2 - len(rel_month)) + rel_month + '-01'
                elif num_snaps == '1Y':
                    if broken_interval[0] == '2008':
                        curr_interval = '2008-01-01'
                    else:
                        curr_interval = '2011-02-01'
                else:
                    if broken_interval[0] == '2008':
                        curr_interval = '2007-07-01'
                    else:
                        curr_interval = '2010-08-01'
                tmp_df = work_df[work_df['Interval'] >= curr_interval].copy()
        else:
            tmp_df = work_df[work_df['SnapNum'] >= (-1)*num_snaps].copy()
        if history_only_in_msmm_calc == True:
            tmp_df = tmp_df[tmp_df['Interval'] != 'ClueWeb09']
        suffix = str(num_snaps) + 'Snaps'
        suffix_col = 'XXSnaps'
        print(suffix)
        sys.stdout.flush()
        save_df = tmp_df[['QueryNum', 'Docno'] + base_feature_list].groupby(['QueryNum', 'Docno']).mean()
        rename_dict_1 = {}
        rename_dict_2 = {}
        rename_dict_3 = {}
        rename_dict_4 = {}
        for feature in base_feature_list:
            rename_dict_1[feature] = feature + '_M' + suffix_col
            rename_dict_2[feature] = feature + '_STD' + suffix_col
            rename_dict_3[feature] = feature + '_Min' + suffix_col
            rename_dict_4[feature] = feature + '_Max' + suffix_col

        std_df = tmp_df[['QueryNum', 'Docno'] + base_feature_list].groupby(['QueryNum', 'Docno']).std()
        save_df = pd.merge(
            save_df.rename(columns = rename_dict_1),
            std_df.rename(columns = rename_dict_2),
            right_index=True,
            left_index=True)

        min_df = tmp_df[['QueryNum', 'Docno'] + base_feature_list].groupby(['QueryNum', 'Docno']).min()
        save_df = pd.merge(
            save_df,
            min_df.rename(columns=rename_dict_3),
            right_index=True,
            left_index=True)

        max_df = tmp_df[['QueryNum', 'Docno'] + base_feature_list].groupby(['QueryNum', 'Docno']).max()
        save_df = pd.merge(
            save_df,
            max_df.rename(columns=rename_dict_4),
            right_index=True,
            left_index=True)

        tmp_df = tmp_df[tmp_df['Interval'] != 'ClueWeb09']
        grad_list = []
        rgrad_list = []
        rename_dict_1 = {}
        rename_dict_2 = {}
        for feature in base_feature_list:
            grad_list.append(feature + '_Grad')
            rename_dict_1[feature + '_Grad'] = feature + '_MG' + suffix_col
            rgrad_list.append(feature + '_RGrad')
            rename_dict_2[feature + '_RGrad'] = feature + '_RMG' + suffix_col

        grad_df = tmp_df[['QueryNum', 'Docno'] + grad_list].groupby(['QueryNum', 'Docno']).mean()
        rgrad_df = tmp_df[['QueryNum', 'Docno'] + rgrad_list].groupby(['QueryNum', 'Docno']).mean()

        save_df = pd.merge(
            save_df.reset_index(),
            grad_df.rename(columns=rename_dict_1).reset_index(),
            on=['QueryNum', 'Docno'],
            how='left')

        save_df = pd.merge(
            save_df,
            rgrad_df.rename(columns=rename_dict_2).reset_index(),
            on=['QueryNum', 'Docno'],
            how='left')

        save_df.to_csv(os.path.join(os.path.join(base_file_folder, 'feat_ref'), base_feature_filename.replace('_all_snaps.tsv', suffix + '.tsv')), sep = '\t', index = False)


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
        backward_elimination = ast.literal_eval(sys.argv[10])
        snap_num_as_hyper_param = ast.literal_eval(sys.argv[11])
        snap_choosing_config = sys.argv[12]
        is_new_server = ast.literal_eval(sys.argv[13])

        run_grid_search_over_params_for_config(
            base_feature_filename=base_feature_filename,
            snapshot_limit=snapshot_limit,
            retrieval_model=retrieval_model,
            normalize_method=normalize_method,
            snap_chosing_method=snap_chosing_method,
            tarin_leave_one_out=tarin_leave_one_out,
            feat_group_list=feat_group_list,
            calc_ndcg_mrr =calc_ndcg_mrr,
            backward_elimination=backward_elimination,
            snap_num_as_hyper_param=snap_num_as_hyper_param,
            snap_choosing_config=snap_choosing_config,
            is_new_server=is_new_server)

    elif operation == 'LimitedSnapFeatures':
        base_feature_filename = sys.argv[2]
        possible_num_snaps_options_list = ast.literal_eval(sys.argv[3])
        history_only_in_msmm_calc = ast.literal_eval(sys.argv[4])
        create_all_x_snap_aggregations(base_feature_filename, possible_num_snaps_options_list, history_only_in_msmm_calc)