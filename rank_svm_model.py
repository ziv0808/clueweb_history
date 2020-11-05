import os
import sys
import random
import subprocess
import pandas as pd

from utils import *
from jobs import get_relevant_docs_df

def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)
    out, err = p.communicate()
    return out

def run_svm_rank_model(test_file, model_file, predictions_folder):
    predictions_file = os.path.join(predictions_folder, 'Prdictions.txt' )
    command = "/mnt/bi-strg3/v/zivvasilisky/ziv/env/svm_rank/svm_rank_classify " + test_file + " " + model_file + " " + predictions_file
    print("##Running command: "+command+"##")
    out = run_bash_command(command)
    print("Output of ranking command: "+str(out))
    sys.stdout.flush()
    return predictions_file


def learn_svm_rank_model(train_file, models_folder, C):
    model_file = os.path.join(models_folder , "model_" + str(C) + ".txt")
    command = "/mnt/bi-strg3/v/zivvasilisky/ziv/env/svm_rank/svm_rank_learn -c " + str(C) + " " + train_file + " " + model_file
    out = run_bash_command(command)
    print(out)
    sys.stdout.flush()
    return model_file


def create_base_feature_file_for_configuration(
        year_list,
        last_interval,
        interval_freq,
        inner_fold,
        retrival_scores_inner_fold):

    data_path = "/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/"
    retrival_scores_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/'
    save_folder ='/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/'
    if inner_fold != "":
        data_path += inner_fold + "/"

    if retrival_scores_inner_fold != "":
        retrival_scores_folder += retrival_scores_inner_fold +"/"

    year_df_dict = {}
    doc_query_df = pd.DataFrame({})
    for year in year_list:
        year_df_dict[year] = pd.read_csv(
            os.path.join(data_path, 'Summay_snapshot_stats_' + interval_freq + '_' + year + '.tsv'), sep='\t',
            index_col=False)
        del year_df_dict[year]['CluWebStemDiff']
        del year_df_dict[year]['StemDiffCluWeb']
        doc_query_df = doc_query_df.append(year_df_dict[year][['Docno', 'QueryNum']].drop_duplicates(), ignore_index=True)
        year_df_dict[year]['QueryTermsRatio'] = year_df_dict[year].apply(
            lambda row: row['QueryWords'] / float(row['TextLen'] - row['QueryWords']),
            axis=1)
        year_df_dict[year]['StopwordsRatio'] = year_df_dict[year].apply(
            lambda row: row['#Stopword'] / float(row['TextLen'] - row['#Stopword']),
            axis=1)
        year_df_dict[year]['-Query-SW'] = year_df_dict[year].apply(
            lambda row: row['TextLen'] - (row['#Stopword'] + row['QueryWords']),
            axis=1)
        year_df_dict[year].rename(columns={'#Stopword': 'Stopwords', 'SimToClueWeb': 'SimClueWeb'}, inplace=True)

    print("data retrieved!")
    sys.stdout.flush()

    if '2008' in year_list:
        lm_scores_ref_df = convert_trec_results_file_to_pandas_df(
            os.path.join(retrival_scores_folder, 'ClueWeb09_1M_Backward_Results.txt'))
        bm25_scores_ref_df = convert_trec_results_file_to_pandas_df(
            os.path.join(retrival_scores_folder, 'BM25_ClueWeb09_1M_Backward_Results.txt'))
        rel_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc')
    else:
        lm_scores_ref_df = convert_trec_results_file_to_pandas_df(
            os.path.join(retrival_scores_folder, 'ClueWeb09_1M_Backward_SW_RMV_Results.txt'))
        bm25_scores_ref_df = convert_trec_results_file_to_pandas_df(
            os.path.join(retrival_scores_folder, 'BM25_ClueWeb09_1M_Backward_SW_RMV_Results.txt'))
        rel_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels_cw12.adhoc')

    lm_scores_ref_df['Query_ID'] = lm_scores_ref_df['Query_ID'].apply(lambda x: int(x))
    bm25_scores_ref_df['Query_ID'] = bm25_scores_ref_df['Query_ID'].apply(lambda x: int(x))
    rel_df['Query'] = rel_df['Query'].apply(lambda x: int(x))

    meta_data_df = pd.merge(
        lm_scores_ref_df[['Query_ID', 'Docno', 'Score']].rename(columns = {'Query_ID' : 'QueryNum', 'Score': 'LMScore'}),
        bm25_scores_ref_df[['Query_ID', 'Docno', 'Score']].rename(columns={'Query_ID': 'QueryNum', 'Score': 'BM25Score'}),
        on = ['QueryNum', 'Docno'],
        how = 'inner')

    meta_data_df = pd.merge(
        meta_data_df,
        rel_df.rename(columns = {'Query' : 'QueryNum'}),
        on=['QueryNum', 'Docno'],
        how='left')

    if len(meta_data_df) != len(lm_scores_ref_df):
        raise Exception('retrieval data not Not allienged')

    meta_data_df.fillna(0, inplace = True)
    del lm_scores_ref_df
    del bm25_scores_ref_df
    meta_data_df[['BM25Score', 'LMScore']] = meta_data_df[['BM25Score', 'LMScore']].applymap(lambda x: float(x))

    print ("Meta data retrieved!")
    sys.stdout.flush()

    fin_df = pd.DataFrame(
        columns=['NumSnapshots','QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                 'QueryWords','Stopwords','TextLen','-Query-SW',
                 'QueryTermsRatio_M', 'StopwordsRatio_M', 'Entropy_M', 'SimClueWeb_M',
                 'QueryWords_M', 'Stopwords_M', 'TextLen_M', '-Query-SW_M',
                 'QueryTermsRatio_M_5Snap', 'StopwordsRatio_M_5Snap', 'Entropy_M_5Snap', 'SimClueWeb_M_5Snap',
                 'QueryWords_M_5Snap', 'Stopwords_M_5Snap', 'TextLen_M_5Snap', '-Query-SW_M_5Snap',
                 'QueryTermsRatio_STD', 'StopwordsRatio_STD', 'Entropy_STD', 'SimClueWeb_STD',
                 'QueryWords_STD', 'Stopwords_STD', 'TextLen_STD', '-Query-SW_STD',
                 'QueryTermsRatio_LG', 'StopwordsRatio_LG', 'Entropy_LG', 'SimClueWeb_LG',
                 'QueryWords_LG', 'Stopwords_LG', 'TextLen_LG', '-Query-SW_LG',
                 'QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                 'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG',
                 'QueryTermsRatio_STDG', 'StopwordsRatio_STDG', 'Entropy_STDG', 'SimClueWeb_STDG',
                 'QueryWords_STDG', 'Stopwords_STDG', 'TextLen_STDG', '-Query-SW_STDG',
                 'QueryTermsRatio_MRG', 'StopwordsRatio_MRG', 'Entropy_MRG', 'SimClueWeb_MRG',
                 'QueryWords_MRG', 'Stopwords_MRG', 'TextLen_MRG', '-Query-SW_MRG',
                 'QueryTermsRatio_MRG_5Snap', 'StopwordsRatio_MRG_5Snap', 'Entropy_MRG_5Snap', 'SimClueWeb_MRG_5Snap',
                 'QueryWords_MRG_5Snap', 'Stopwords_MRG_5Snap', 'TextLen_MRG_5Snap', '-Query-SW_MRG_5Snap',

                 # 'LMScore','BM25Score', 'Relevance',
                 'QueryNum', 'Docno'])

    all_snaps_df = pd.DataFrame({})
    base_feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                 'QueryWords','Stopwords','TextLen','-Query-SW']
    next_index = 0
    doc_query_df = doc_query_df.drop_duplicates()
    for row in doc_query_df.itertuples():
        docno = row.Docno
        query = row.QueryNum
        print(docno)
        sys.stdout.flush()
        first = True
        tmp_doc_df = pd.DataFrame({})
        for year in list(reversed(year_list)):
            tmp_doc_df_y = year_df_dict[year][year_df_dict[year]['Docno'] == docno].copy()
            tmp_doc_df_y = tmp_doc_df_y[tmp_doc_df_y['QueryNum'] == query]
            if first == True:
                first = False
                if last_interval is not None and last_interval != 'None':
                    tmp_doc_df_y['Filter'] = tmp_doc_df_y['Interval'].apply(
                        lambda x: 1 if x > last_interval and x != 'ClueWeb09' else 0)
                    tmp_doc_df_y = tmp_doc_df_y[tmp_doc_df_y['Filter'] == 0]
                    del tmp_doc_df_y['Filter']
            else:
                tmp_doc_df_y = tmp_doc_df_y[tmp_doc_df_y['Interval'] != 'ClueWeb09']

            tmp_doc_df = tmp_doc_df_y.append(tmp_doc_df, ignore_index=True)

        bench_df = tmp_doc_df[tmp_doc_df['Interval'] == 'ClueWeb09']
        insert_row = [len(tmp_doc_df)]
        for feature in base_feature_list:
            insert_row.append(list(bench_df[feature])[0])

        for feature in base_feature_list:
            insert_row.append(tmp_doc_df[feature].mean())

        for feature in base_feature_list:
            insert_row.append(tmp_doc_df[-5:][feature].mean())

        for feature in base_feature_list:
            insert_row.append(tmp_doc_df[feature].std())

        if len(tmp_doc_df) == 1:
            insert_row.extend([pd.np.nan]*(len(base_feature_list)*5))
        else:
            for feature in base_feature_list:
                tmp_doc_df[feature + '_Shift'] = tmp_doc_df[feature].shift(-1)
                tmp_doc_df[feature + '_Grad']  = tmp_doc_df.apply(lambda row_: calc_releational_measure(row_[feature + '_Shift'], row_[feature]), axis = 1)
                tmp_doc_df[feature + '_RGrad'] = tmp_doc_df.apply(lambda row_: calc_releational_measure(row_[feature], list(bench_df[feature])[0]), axis=1)

            tmp_doc_df = tmp_doc_df[tmp_doc_df['Interval'] != 'ClueWeb09']
            for feature in base_feature_list:
                insert_row.append(list(tmp_doc_df[feature + '_Grad'])[-1])

            for feature in base_feature_list:
                insert_row.append(tmp_doc_df[feature + '_Grad'].mean())

            for feature in base_feature_list:
                curr_std = tmp_doc_df[feature + '_Grad'].std()
                if pd.np.isnan(curr_std):
                    curr_std = 0.0
                insert_row.append(curr_std)

            for feature in base_feature_list:
                insert_row.append(tmp_doc_df[feature + '_RGrad'].mean())

            for feature in base_feature_list:
                insert_row.append(tmp_doc_df[-5:][feature + '_RGrad'].mean())

        insert_row.extend([query, docno])
        fin_df.loc[next_index] = insert_row
        next_index += 1

        tmp_doc_df['NumSnapshots'] = len(tmp_doc_df)
        tmp_doc_df['SnapNum'] = list(range((len(tmp_doc_df) - 1) * (-1), 1))
        all_snaps_df = all_snaps_df.append(tmp_doc_df, ignore_index=True)

    filename = interval_freq + "_"
    for year_ in year_list:
        filename += year_ + '_'

    filename += last_interval + '_All_features'

    # fin_df.to_csv(os.path.join(save_folder, filename + '_raw.tsv'), sep = '\t', index = False)
    # meta_data_df.to_csv(os.path.join(save_folder, filename + '_Meatdata.tsv'), sep = '\t', index = False)

    fin_df['QueryNum'] = fin_df['QueryNum'].apply(lambda x: int(x))
    fin_df = pd.merge(
        fin_df,
        meta_data_df,
        on=['QueryNum', 'Docno'],
        how = 'inner')

    fin_df.to_csv(os.path.join(save_folder, filename + '_with_meta.tsv'), sep='\t', index=False)
    all_snaps_df.to_csv(os.path.join(save_folder, filename + '_all_snaps.tsv'), sep='\t', index=False)

def prepare_svmr_model_data(
        base_feature_filename,
        snapshot_limit,
        feature_list,
        normalize_relvance = False,
        limited_snaps_num  = None,
        normalize_method = 'MinMax',
        lambdamart = False):
    base_feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                         'QueryWords', 'Stopwords', 'TextLen', '-Query-SW']
    data_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/'
    work_df = pd.read_csv(os.path.join(data_folder, base_feature_filename), sep = '\t', index_col = False)
    # enforce snapshot limit
    work_df = work_df[work_df['NumSnapshots'] >= snapshot_limit]
    if '2008' in base_feature_filename:
        work_df['Filter'] = work_df['QueryNum'].apply(lambda x: True if int(x) in [95,100] else False)
        work_df = work_df[work_df['Filter'] == False]
        del work_df['Filter']
    # add m/s features
    need_m_div_std = False
    for feature in feature_list:
        if 'M/STD' in feature:
            need_m_div_std = True
    if need_m_div_std == True:
        for feature in base_feature_list:
            work_df[feature + '_M/STD'] = work_df.apply(lambda row: row[feature +'_M']
                                                        if (row[feature +'_STD'] == 0.0 or pd.np.isnan(row[feature +'_STD']))
                                                        else float(row[feature +'_M']) / row[feature +'_STD'], axis = 1)
    if limited_snaps_num is not None:
        limited_snaps_df = pd.read_csv(os.path.join(os.path.join(data_folder, 'feat_ref'),
                                                    base_feature_filename.replace('_with_meta', str(limited_snaps_num) +'Snaps')), sep = '\t', index_col = False)
        work_df = pd.merge(
            work_df,
            limited_snaps_df,
            on = ['QueryNum', 'Docno'],
            how = 'left')
    # cat relevant features
    work_df = work_df[['QueryNum', 'Docno', 'Relevance'] + feature_list]
    # adapt relevance column
    if normalize_relvance == True:
        work_df['Relevance'] = work_df['Relevance'].apply(lambda x: 0 if int(x) <= 0 else 1)

    if lambdamart == True:
        work_df['Relevance'] = work_df['Relevance'].apply(lambda x: 0 if int(x) <= 0 else x)
    # minmax normalize per query
    all_queries = list(work_df['QueryNum'].drop_duplicates())
    fin_df = pd.DataFrame({})
    for q in sorted(all_queries):
        tmp_q_df = work_df[work_df['QueryNum'] == q].copy()
        if len(tmp_q_df[tmp_q_df['Relevance'] > 0]) == 0:
            continue
        for feature in feature_list:
            if normalize_method == 'MinMax':
                min_feat = tmp_q_df[feature].min()
                max_feat = tmp_q_df[feature].max()
                tmp_q_df[feature] = tmp_q_df[feature].apply(lambda x: (x - min_feat)/ float(max_feat - min_feat) if (max_feat - min_feat) != 0 else 0.0)
            elif normalize_method == 'ZScore':
                mean_feat = tmp_q_df[feature].mean()
                std_feat = tmp_q_df[feature].std()
                tmp_q_df[feature] = tmp_q_df[feature].apply(lambda x: (x - mean_feat) / float(std_feat) if (std_feat) != 0 else 0.0)
            else:
                raise Exception('Unknown normalization...')
        fin_df = fin_df.append(tmp_q_df, ignore_index=True)

    fin_df.fillna(0.0, inplace=True)
    return fin_df


def turn_df_to_feature_str_for_model(
        df,
        feature_list):

    feature_str = ""
    for index, row in df.iterrows():
        feature_str += str(int(row['Relevance'])) + " qid:" + str(int(row['QueryNum']))
        feat_num = 1
        for feature in feature_list:
            feature_str += " " + str(feat_num) + ":" + str(row[feature])
            feat_num += 1

        feature_str += '\n'

    return feature_str

def split_to_train_test(
        start_test_q,
        end_test_q,
        feat_df,
        base_feature_filename,
        seed = None):

    test_set_q = list(range(start_test_q, end_test_q + 1))
    feat_df['IsTest'] = feat_df['QueryNum'].apply(lambda x: 1 if x in test_set_q else 0)
    test_df = feat_df[feat_df['IsTest'] == 1].copy()
    train_df = feat_df[feat_df['IsTest'] == 0]

    k_fold, potential_folds = create_fold_list_for_cv(
        base_feature_filename=base_feature_filename,
        train_leave_one_out=False)

    for potential_fold in potential_folds[:]:
        if start_test_q in range(potential_fold[0], potential_fold[1]+1):
            potential_folds.remove(potential_fold)
    if seed is None:
        seed = random.randint(0,len(potential_folds) - 1)
    valid_fold = potential_folds[seed]
    valid_set_q = list(range(valid_fold[0], valid_fold[1] + 1))

    train_df['IsValid'] = train_df['QueryNum'].apply(lambda x: 1 if x in valid_set_q else 0)
    valid_df = train_df[train_df['IsValid'] == 1]
    train_df = train_df[train_df['IsValid'] == 0]

    del valid_df['IsValid']
    del train_df['IsValid']

    return train_df, test_df, valid_df, seed

def get_trec_prepared_df_form_res_df(
        scored_docs_df,
        score_colname):

    all_q = sorted(list(scored_docs_df['QueryNum'].drop_duplicates()))
    big_df = pd.DataFrame({})
    for query_num in all_q:
        res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])
        next_index = 0
        query_df = scored_docs_df[scored_docs_df['QueryNum'] == query_num].copy()
        for index, row in query_df.iterrows():
            res_df.loc[next_index] = ["0" * (3 - len(str(query_num))) + str(query_num), 'Q0', row['Docno'], 0, row[score_colname],
                                      'indri']
            next_index += 1

        if res_df.empty == False:
            res_df.sort_values('Score', ascending=False, inplace=True)
            res_df['Rank'] = list(range(1, next_index + 1))
            big_df = big_df.append(res_df, ignore_index=True)

    return big_df

def get_svm_weights(
    model_file,
    feature_list = []):

    with open(model_file, 'r') as f:
        file_str = f.read()
    broken_str = file_str.split('\n')
    broken_str = broken_str[-2].split(' ')
    wieght_list = []
    if len(feature_list) > 0:
        uncovered_features = feature_list[:]
        for elem in broken_str[1:-1]:
            wieght_list.append(elem.split(':')[1])
            uncovered_features.remove(feature_list[int(elem.split(':')[0])-1])

        for feature in uncovered_features:
            feature_list.remove(feature)

        return wieght_list, feature_list
    else:
        for elem in broken_str[1:-1]:
            wieght_list.append(elem.split(':')[1])
        return wieght_list



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
        optional_snap_limit = [2,3,4,5,6,7,8,9,10,15,'All']
    elif snap_chosing_method == 'Months':
        optional_snap_limit = ['2M','3M','5M','6M','7M','8M','9M','10M','1Y','1.5Y','All']
    else:
        raise Exception('learn_best_num_of_snaps: Unknown snap_chosing_method')

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
            limited_snaps_num=snap_lim)

        train_df, test_df, valid_df, seed = split_to_train_test(
            start_test_q=start_test_q,
            end_test_q=end_test_q,
            feat_df=feat_df,
            seed = seed)

        with open(os.path.join(base_res_folder, 'train.dat'), 'w') as f:
            f.write(turn_df_to_feature_str_for_model(train_df, feature_list=feature_list))

        with open(os.path.join(base_res_folder, 'valid.dat'), 'w') as f:
            f.write(turn_df_to_feature_str_for_model(valid_df, feature_list=feature_list))

        model_filename = learn_svm_rank_model(
            train_file=os.path.join(base_res_folder, 'train.dat'),
            models_folder=base_res_folder,
            C=0.2)

        predictions_filename = run_svm_rank_model(
            test_file=os.path.join(base_res_folder, 'valid.dat'),
            model_file=model_filename,
            predictions_folder=base_res_folder)

        with open(predictions_filename, 'r') as f:
            predications = f.read()

        predications = predications.split('\n')
        if '' in predications:
            predications = predications[:-1]

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
        normalize_method,
        qrel_filepath,
        snap_chosing_method = None,
        snap_calc_limit = None):

    base_res_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/'
    model_inner_folder = base_feature_filename.replace('All_features_with_meta.tsv', '') + 'SNL' + str(snapshot_limit)
    feature_folder = feature_groupname
    feature_folder += '_' + normalize_method
    fold_folder = str(start_test_q) + '_' + str(end_test_q) + "_" + str(snap_chosing_method)

    for hirarcy_folder in [model_inner_folder, feature_folder, fold_folder]:
        base_res_folder = os.path.join(base_res_folder, hirarcy_folder)
        if not os.path.exists(base_res_folder):
            os.mkdir(base_res_folder)

    best_snap_num = snap_calc_limit
    # if 'XXSnap' in feature_groupname:
    #     best_snap_num = learn_best_num_of_snaps(
    #         base_feature_filename=base_feature_filename,
    #         snapshot_limit=snapshot_limit,
    #         feature_list=feature_list,
    #         start_test_q=start_test_q,
    #         end_test_q=end_test_q,
    #         base_res_folder=base_res_folder,
    #         qrel_filepath=qrel_filepath,
    #         normalize_relevance=normalize_relevance,
    #         snap_chosing_method=snap_chosing_method)

    feat_df = prepare_svmr_model_data(
        base_feature_filename=base_feature_filename,
        snapshot_limit=int(snapshot_limit),
        feature_list=feature_list,
        normalize_method=normalize_method,
        limited_snaps_num=best_snap_num)

    print("Model Data Prepared...")
    sys.stdout.flush()
    train_df, test_df, valid_df, seed = split_to_train_test(
        start_test_q=start_test_q,
        end_test_q=end_test_q,
        feat_df=feat_df,
        base_feature_filename=base_feature_filename)

    with open(os.path.join(base_res_folder, 'train.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(train_df, feature_list=feature_list))

    with open(os.path.join(base_res_folder, 'valid.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(valid_df, feature_list=feature_list))

    valid_cp_df = valid_df.copy()

    best_c = None
    best_map = 0.0
    for potential_c in [0.01, 0.1, 0.2, 0.5, 1]:
        print("Running valid on C : " +str(potential_c))
        model_filename = learn_svm_rank_model(
            train_file=os.path.join(base_res_folder, 'train.dat'),
            models_folder=base_res_folder,
            C=potential_c)

        predictions_filename = run_svm_rank_model(
            test_file=os.path.join(base_res_folder, 'valid.dat'),
            model_file=model_filename,
            predictions_folder=base_res_folder)

        with open(predictions_filename, 'r') as f:
            predications = f.read()

        predications = predications.split('\n')
        if '' in predications:
            predications = predications[:-1]

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
            best_c = potential_c

    best_params_str = "C: " +str(best_c) +' \nSnapLim: ' + str(best_snap_num)
    with open(os.path.join(base_res_folder, 'hyper_params.txt'), 'w') as f:
        f.write(best_params_str)

    train_df = train_df.append(valid_cp_df, ignore_index = True)
    train_df.sort_values('QueryNum', inplace = True)
    with open(os.path.join(base_res_folder, 'train.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(train_df, feature_list=feature_list))

    with open(os.path.join(base_res_folder, 'test.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(test_df, feature_list=feature_list))

    print("Strating Train : " + model_inner_folder + ' ' + feature_folder + ' ' + fold_folder)
    sys.stdout.flush()
    model_filename = learn_svm_rank_model(
        train_file=os.path.join(base_res_folder, 'train.dat'),
        models_folder=base_res_folder,
        C=best_c)

    print("Strating Test : " + model_inner_folder + ' ' + feature_folder + ' ' + fold_folder)
    sys.stdout.flush()

    predictions_filename = run_svm_rank_model(
        test_file=os.path.join(base_res_folder, 'test.dat'),
        model_file=model_filename,
        predictions_folder=base_res_folder)

    with open(predictions_filename, 'r') as f:
        predications = f.read()

    predications = predications.split('\n')
    if '' in predications:
        predications = predications[:-1]

    test_df['ModelScore'] = predications
    test_df['ModelScore'] = test_df['ModelScore'].apply(lambda x: float(x))

    feature_list_cp = feature_list[:]
    wieghts_list, feature_list_cp = get_svm_weights(model_filename,feature_list_cp)
    hyper_params = ['C']
    wieghts_list.append(best_c)
    if best_snap_num is not None:
        hyper_params.append('SnapLimit')
        wieghts_list.append(best_snap_num)
    params_df = pd.DataFrame(columns=['Fold'] + feature_list_cp + hyper_params)
    params_df.loc[0] = [str(start_test_q) + '_' + str(end_test_q)] + wieghts_list

    return test_df, params_df


def create_fold_list_for_cv(
        base_feature_filename,
        train_leave_one_out):

    k_fold = 10
    if '2008' in base_feature_filename:
        fold_list = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 120),
                     (121, 140), (141, 160), (161, 180), (181, 200)]
        q_list = list(range(1,201))
        q_list.remove(95)
        q_list.remove(100)
        query_bulk = 20
    elif 'ASRC' in base_feature_filename:
        fold_list = [(2, 9), (10, 17), (18, 32), (33, 36), (45, 51), (59, 78),
                     (98, 144), (161, 166), (167, 180), (182, 195)]
        q_list = [2, 4, 9, 10, 11, 17, 18, 29, 32, 33, 34, 36, 45, 48, 51, 59, 69, 78, 98, 124, 144, 161, 164, 166, 167, 177, 180, 182, 188, 193, 195]
        query_bulk = 3
    elif 'BOT' in base_feature_filename:
        fold_list = [(10, 13), (18, 32), (33, 34), (48, 51), (69, 167), (177, 177),
                     (180, 180), (182, 182), (193, 193), (195, 195)]

        q_list = [10, 13, 18, 32, 33, 34, 48, 51, 69, 167, 177, 180, 182, 193, 195]
    elif 'HERD_CONTROL' in base_feature_filename:
        fold_list = [(4, 10), (11, 17), (18, 32), (33, 45), (48, 59), (69, 98),
                     (124, 161), (164, 167), (177, 182), (188, 195)]

        q_list = [4, 9, 10, 11, 13, 17, 18, 29, 32, 33, 34, 45, 48, 51, 59, 69, 78, 98, 124, 144, 161, 164, 166, 167, 177,
                  180, 182, 188, 193, 195]
    elif 'UNITED' in base_feature_filename:
        fold_list = [(2, 9), (10, 17), (18, 32), (33, 36), (45, 51), (59, 78),
                     (98, 144), (161, 166), (167, 180), (182, 195)]

        q_list = [2, 4, 9, 10, 11, 13, 17, 18, 29, 32, 33, 34, 36, 45, 48, 51, 59, 69, 78, 98, 124, 144, 161, 164, 166, 167, 177,
                  180, 182, 188, 193, 195]
    else:
        fold_list = [(201, 210), (211, 220), (221, 230), (231, 240), (241, 250),
                   (251, 260), (261, 270), (271, 280), (281, 290), (291, 300)]
        q_list = list(range(201, 301))
        query_bulk = 10

    if train_leave_one_out == True:
        k_fold = len(q_list)
        query_bulk = 1
        fold_list = []
        for q in q_list:
            fold_list.append((q,q))

    return  k_fold, fold_list

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
    if feature_groupname == 'All':
        feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                        'QueryWords','Stopwords','-Query-SW',
                        'QueryTermsRatio_LG', 'StopwordsRatio_LG', 'Entropy_LG', 'SimClueWeb_LG',
                        'QueryWords_LG', 'Stopwords_LG', '-Query-SW_LG',
                        'QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                        'QueryWords_MG', 'Stopwords_MG', '-Query-SW_MG',
                        'QueryTermsRatio_M', 'StopwordsRatio_M', 'Entropy_M', 'SimClueWeb_M',
                        'QueryWords_M', 'Stopwords_M', '-Query-SW_M',
                        'QueryTermsRatio_STD', 'StopwordsRatio_STD', 'Entropy_STD', 'SimClueWeb_STD',
                        'QueryWords_STD', 'Stopwords_STD', '-Query-SW_STD',
                        'QueryTermsRatio_MRG', 'StopwordsRatio_MRG', 'Entropy_MRG', 'SimClueWeb_MRG',
                        'QueryWords_MRG', 'Stopwords_MRG', '-Query-SW_MRG',
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
        feature_list.extend(['QueryTermsRatio_RMGXXSnaps', 'StopwordsRatio_RMGXXSnaps', 'Entropy_RMGXXSnaps', 'SimClueWeb_RMGXXSnaps',
                             'QueryWords_RMGXXSnaps', 'Stopwords_RMGXXSnaps', 'TextLen_RMGXXSnaps', '-Query-SW_RMGXXSnaps'])
        len_handled += 1
    if 'MGXXSnap' in broken_feature_groupname:
        feature_list.extend(['QueryTermsRatio_MGXXSnaps', 'StopwordsRatio_MGXXSnaps', 'Entropy_MGXXSnaps', 'SimClueWeb_MGXXSnaps',
                             'QueryWords_MGXXSnaps', 'Stopwords_MGXXSnaps', 'TextLen_MGXXSnaps', '-Query-SW_MGXXSnaps'])
        len_handled += 1
    if 'MXXSnap' in broken_feature_groupname:
        feature_list.extend(['QueryTermsRatio_MXXSnaps', 'StopwordsRatio_MXXSnaps', 'Entropy_MXXSnaps', 'SimClueWeb_MXXSnaps',
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

    feature_groups_num = len(broken_feature_groupname)
    if snap_calc_limit is not None:
        feature_groups_num = feature_groups_num - 1
    if len_handled != feature_groups_num:
        raise Exception('Undefined feature group!')

    if retrieval_model == 'LM':
        feature_list.append('LMScore')
    elif retrieval_model == 'BM25':
        feature_list.append('BM25Score')

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
            snap_calc_limit=snap_calc_limit)
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
                                    'Static_MGXXSnap','Static_RMGXXSnap']
    else:
        optional_feat_groups_list = feat_group_list
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/ret_res/'
    save_summary_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/'

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
    model_base_filename = base_feature_filename.replace('All_features_with_meta.tsv', '') + 'SNL' + str(snapshot_limit) + "_" + retrieval_model + "_By" + snap_chosing_method

    if not os.path.exists(os.path.join(save_folder, model_base_filename)):
        os.mkdir(os.path.join(save_folder, model_base_filename))
    save_folder = os.path.join(save_folder, model_base_filename)


    model_base_filename += '_' + normalize_method
    if tarin_leave_one_out == True:
        model_base_filename += '_LoO'
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


def create_sinificance_df(
        per_q_res_dict,
        calc_ndcg_mrr = False,
        sinificance_type ='TTest'):

    additional_measures = []
    if calc_ndcg_mrr == True:
        additional_measures = ['NDCG@1', 'NDCG@3', 'MRR', 'nMRR']

    measure_list = ['Map', 'P_5', 'P_10'] + additional_measures
    measure_col_list = []
    for measure in measure_list:
        measure_col_list.extend([measure + '_sign', measure + '_vsS'])

    significance_df = pd.DataFrame(
        columns=['FeatureGroup'] + measure_col_list)
    next_idx = 0

    for key in per_q_res_dict:
        sinificance_list_dict = {}
        for col in measure_col_list:
            sinificance_list_dict[col] = ""
        for key_2 in per_q_res_dict:
            if key != key_2:
                sinificance_dict = check_statistical_significance(res_dict_1=per_q_res_dict[key], res_dict_2=per_q_res_dict[key_2], ndcg_mrr=calc_ndcg_mrr,sinificance_type=sinificance_type)
                for measure in measure_list:
                    if sinificance_dict[measure]['Significant'] == True:
                        if per_q_res_dict[key]['all'][measure] > per_q_res_dict[key_2]['all'][measure]:
                            sinificance_list_dict[measure + '_sign'] += key_2+ ','
                    if (key_2 == 'Static') or (key_2 in ['LM','BM25']):
                        sinificance_list_dict[measure + "_vsS"] += key_2 + " " +str(sinificance_dict[measure]['Pval']) +\
                                                          "(" + str(sinificance_dict[measure]['%Better'])+ ")(" + str(sinificance_dict[measure]['%BetterOrEqual'])+ "),"
        insert_row = [key]
        for measure in measure_col_list:
            insert_row.append(sinificance_list_dict[measure])
        significance_df.loc[next_idx] = insert_row
        next_idx += 1
    return significance_df

def fix_statistical_sinificance(
        base_feature_filename,
        snapshot_limit,
        retrieval_model):

    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/ret_res/'
    save_summary_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/'
    if '2008' in base_feature_filename:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc"
    else:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels_cw12.adhoc"

    model_base_filename = base_feature_filename.replace('All_features_with_meta.tsv', '') + 'SNL' + str(snapshot_limit) + "_" + retrieval_model
    optional_feat_groups_list = ['All', 'Static', 'MG', 'LG', 'M_STD', 'Static_LG', 'Static_MG',
                                 'Static_M_STD', 'MG_M_STD', 'Static_MG_M_STD']

    per_q_res_dict = {}
    curr_file_name = model_base_filename + '_Benchmark.txt'
    res_dict = get_ranking_effectiveness_for_res_file_per_query(
        file_path=save_folder,
        filename=curr_file_name,
        qrel_filepath=qrel_filepath)

    per_q_res_dict['Basic Retrieval'] = res_dict
    for feat_group in optional_feat_groups_list:
        curr_file_name = model_base_filename + '_' + feat_group + '.txt'

        res_dict = get_ranking_effectiveness_for_res_file_per_query(
            file_path=save_folder,
            filename=curr_file_name,
            qrel_filepath=qrel_filepath)

        per_q_res_dict[feat_group.replace('_', '+')] = res_dict

    model_summary_df = pd.read_csv(os.path.join(save_summary_folder, model_base_filename +'.tsv'), sep = '\t', index_col = False)
    del model_summary_df['Map_sign']
    del model_summary_df['P@5_sign']
    del model_summary_df['P@10_sign']
    significance_df = pd.DataFrame(columns=['FeatureGroup', 'Map_sign', 'P@5_sign', 'P@10_sign'])
    next_idx = 0
    for key in per_q_res_dict:
        sinificance_list_dict = {'Map': "", "P_5": "", "P_10": ""}
        for key_2 in per_q_res_dict:
            sinificance_dict = check_statistical_significance(per_q_res_dict[key], per_q_res_dict[key_2])
            for measure in ['Map', 'P_5', 'P_10']:
                if sinificance_dict[measure] == True:
                    sinificance_list_dict[measure] += key_2 + ','
        insert_row = [key]
        for measure in ['Map', 'P_5', 'P_10']:
            insert_row.append(sinificance_list_dict[measure])
        significance_df.loc[next_idx] = insert_row
        next_idx += 1
    model_summary_df = pd.merge(
        model_summary_df,
        significance_df,
        on=['FeatureGroup'],
        how='inner')
    model_summary_df.to_csv(os.path.join(save_summary_folder, model_base_filename + '_Fixed_Sign.tsv'), sep='\t', index=False)

def create_all_x_snap_aggregations(
        base_feature_filename,
        possible_num_snaps_options_list =None):

    base_file_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/'
    base_feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                         'QueryWords', 'Stopwords', 'TextLen', '-Query-SW','LMScore','BM25Score']

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

    if operation == 'BaseFeatureFile':
        year_list = ast.literal_eval(sys.argv[2])
        last_interval = sys.argv[3]
        interval_freq = sys.argv[4]
        inner_fold = sys.argv[5]
        retrival_scores_inner_fold= sys.argv[6]
        create_base_feature_file_for_configuration(
            year_list=year_list,
            last_interval=last_interval,
            interval_freq=interval_freq,
            inner_fold=inner_fold,
            retrival_scores_inner_fold=retrival_scores_inner_fold)

    elif operation == 'GridSearchParams':
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
            calc_ndcg_mrr =calc_ndcg_mrr)

    elif operation == 'FixSinificance':
        base_feature_filename = sys.argv[2]
        snapshot_limit = int(sys.argv[3])
        retrieval_model = sys.argv[4]

        fix_statistical_sinificance(
            base_feature_filename=base_feature_filename,
            snapshot_limit=snapshot_limit,
            retrieval_model=retrieval_model)

    elif operation == 'LimitedSnapFeatures':
        base_feature_filename = sys.argv[2]
        possible_num_snaps_options_list = ast.literal_eval(sys.argv[3])
        create_all_x_snap_aggregations(base_feature_filename, possible_num_snaps_options_list)