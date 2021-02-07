from utils import *
from rank_svm_model import prepare_svmr_model_data, get_trec_prepared_df_form_res_df

def get_feature_list():
    base_feature_list = ['CoverQueryNum', 'CoverQueryRatio', 'Ent', 'FracStops',
                         'IDF', 'Len', 'LMIR.DIR', 'LMIR.JM', 'StopCover', 'TFSum', 'TFMin', 'TFMax',
                         'TFMean', 'TFStd', 'TFIDFSum', 'TFIDFMin', 'TFIDFMax', 'TFIDFMean', 'TFIDFStd', 'TFNormSum',
                         'TFNormMin',
                         'TFNormMax', 'TFNormMean', 'TFNormStd', 'SimClueWeb', 'BM25Score']


    feature_list = []
    feature_groupname = 'Static_MXXSnap_STDXXSnap_MinXXSnap_MaxXXSnap_MGXXSnap'
    broken_feature_groupname = feature_groupname.split('_')
    if 'Static' in broken_feature_groupname:
        feature_list.extend(base_feature_list)

    if 'M' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_M')
    if 'STD' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_STD')
    if 'RMG' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_RMG')
    if 'MG' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_MG')
    if 'LG' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_LG')

    if 'RMGXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_RMGXXSnaps')
    if 'MGXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_MGXXSnaps')
    if 'MXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_MXXSnaps')
    if 'STDXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_STDXXSnaps')
    if 'MinXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_MinXXSnaps')
    if 'MaxXXSnap' in broken_feature_groupname:
        for base_feat in base_feature_list:
            feature_list.append(base_feat + '_MaxXXSnaps')
    return feature_list

def rank_by_feature(
        save_folder,
        work_df,
        curr_feat):

    fin_res_dict = {}
    asc_df = get_trec_prepared_df_form_res_df(work_df, curr_feat)
    curr_file_name = 'TmpRes.txt'
    with open(os.path.join(save_folder, curr_file_name), 'w') as f:
        f.write(convert_df_to_trec(asc_df))
    res_dict = get_ranking_effectiveness_for_res_file_per_query(
        file_path=save_folder,
        filename=curr_file_name,
        qrel_filepath=qrel_filepath,
        calc_ndcg_mrr=True)
    fin_res_dict['Asc'] = res_dict['all']['NDCG@5']

    work_df[curr_feat] = work_df[curr_feat].apply(lambda x: x*(-1))
    dec_df = get_trec_prepared_df_form_res_df(work_df, curr_feat)
    with open(os.path.join(save_folder, curr_file_name), 'w') as f:
        f.write(convert_df_to_trec(dec_df))
    res_dict = get_ranking_effectiveness_for_res_file_per_query(
        file_path=save_folder,
        filename=curr_file_name,
        qrel_filepath=qrel_filepath,
        calc_ndcg_mrr=True)
    fin_res_dict['Desc'] = res_dict['all']['NDCG@5']

    return fin_res_dict


if __name__=='__main__':
    inner_fold = sys.argv[1]
    round_limit = int(sys.argv[2])

    sw_rmv = True
    filter_params = {}
    # asrc_round = int(inner_fold.split('_')[-1])

    if 'asrc' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel'
    elif 'bot' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents_fixed.relevance'
    elif 'herd_control' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/control.rel'
    elif 'united' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel'
    elif 'comp2020' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/curr_comp.rel'

    feature_list = get_feature_list()
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/feature_test/'

    summary_df = pd.DataFrame(columns = ['Feature', 'Round', 'Dec', 'Asc'])
    next_idx = 0
    for round_ in range(2, round_limit + 1):
        base_filename = inner_fold.upper() + '_LTR_All_features_Round0' + str(round_) + '_with_meta.tsv'
        feature_df = prepare_svmr_model_data(
            base_feature_filename=base_filename,
            snapshot_limit=1,
            feature_list=feature_list,
            normalize_relvance=False,
            limited_snaps_num='All')
        for feat in feature_list:
            scores_dict = rank_by_feature(save_folder=save_folder,work_df=feature_df[['QueryNum', 'Docno'] + [feat]],curr_feat=feat)

            insert_row = [feat, round_, scores_dict['Dec'], scores_dict['Asc']]
            summary_df.loc[next_idx] = insert_row
            next_idx += 1

    summary_df = summary_df[['Feature', 'Dec', 'Asc']].groupby(['Feature']).mean()
    summary_df = summary_df.reset_index()
    summary_df.to_csv('Feature_Test_df.tsv', sep = '\t', index = False)


