import os
import sys
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
                 'QueryTermsRatio_STD', 'StopwordsRatio_STD', 'Entropy_STD', 'SimClueWeb_STD',
                 'QueryWords_STD', 'Stopwords_STD', 'TextLen_STD', '-Query-SW_STD',
                 'QueryTermsRatio_LG', 'StopwordsRatio_LG', 'Entropy_LG', 'SimClueWeb_LG',
                 'QueryWords_LG', 'Stopwords_LG', 'TextLen_LG', '-Query-SW_LG',
                 'QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                 'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG',
                 'QueryTermsRatio_STDG', 'StopwordsRatio_STDG', 'Entropy_STDG', 'SimClueWeb_STDG',
                 'QueryWords_STDG', 'Stopwords_STDG', 'TextLen_STDG', '-Query-SW_STDG',
                 # 'LMScore','BM25Score', 'Relevance',
                 'QueryNum', 'Docno'])
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
            insert_row.append(tmp_doc_df[feature].std())

        if len(tmp_doc_df) == 1:
            insert_row.extend([pd.np.nan]*(len(base_feature_list)*3))
        else:
            for feature in base_feature_list:
                tmp_doc_df[feature + '_Shift'] = tmp_doc_df[feature].shift(-1)
                tmp_doc_df[feature + '_Grad'] = tmp_doc_df.apply(lambda row_: (row_[feature + '_Shift'] - row_[feature]) / row_[feature + '_Shift'] , axis = 1)

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

        insert_row.extend([query, docno])
        fin_df.loc[next_index] = insert_row
        next_index += 1

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

def prepare_svmr_model_data(
        base_feature_filename,
        snapshot_limit,
        feature_list,
        normalize_relvance = False):

    data_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/'
    work_df = pd.read_csv(os.path.join(data_folder, base_feature_filename), sep = '\t', index_col = False)
    # enforce snapshot limit
    work_df = work_df[work_df['NumSnapshots'] >= snapshot_limit]
    # cat relevant features
    work_df = work_df[['QueryNum', 'Docno', 'Relevance'] + feature_list]
    # adapt relevance column
    if normalize_relvance == True:
        work_df['Relevance'] = work_df['Relevance'].apply(lambda x: 0 if int(x) <= 0 else 1)
    # minmax normalize per query
    all_queries = list(work_df['QueryNum'].drop_duplicates())
    fin_df = pd.DataFrame({})
    for q in sorted(all_queries):
        tmp_q_df = work_df[work_df['QueryNum'] == q].copy()
        for feature in feature_list:
            min_feat = tmp_q_df[feature].min()
            max_feat = tmp_q_df[feature].max()
            tmp_q_df[feature] = tmp_q_df[feature].apply(lambda x: (x - min_feat)/ float(max_feat - min_feat) if (max_feat - min_feat) != 0 else 0.0)

        fin_df = fin_df.append(tmp_q_df, ignore_index=True)

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
        feat_df):

    test_set_q = list(range(start_test_q, end_test_q + 1))
    feat_df['IsTest'] = feat_df['QueryNum'].apply(lambda x: 1 if x in test_set_q else 0)
    test_df = feat_df[feat_df['IsTest'] == 1].copy()
    train_df = feat_df[feat_df['IsTest'] == 0]

    return train_df, test_df

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

def train_and_test_model_on_config(
        base_feature_filename,
        snapshot_limit,
        feature_list,
        start_test_q,
        end_test_q,
        C,
        feature_groupname = 'All',
        normalize_relevance=False):

    feat_df = prepare_svmr_model_data(
        base_feature_filename=base_feature_filename,
        snapshot_limit=int(snapshot_limit),
        feature_list=feature_list,
        normalize_relvance=normalize_relevance)

    print("Model Data Prepared...")
    sys.stdout.flush()
    train_df, test_df = split_to_train_test(
        start_test_q=start_test_q,
        end_test_q=end_test_q,
        feat_df=feat_df)

    base_res_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/'
    model_inner_folder = base_feature_filename.replace('All_features_with_meta.tsv','') + 'SNL' + str(snapshot_limit) + '_C' + str(C)
    feature_folder = feature_groupname
    if normalize_relevance == True:
        feature_folder += '_NR'
    fold_folder = str(start_test_q) + '_' + str(end_test_q)

    for hirarcy_folder in [model_inner_folder, feature_folder, fold_folder]:
        base_res_folder = os.path.join(base_res_folder, hirarcy_folder)
        if not os.path.exists(base_res_folder):
            os.mkdir(base_res_folder)

    with open(os.path.join(base_res_folder, 'train.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(train_df, feature_list=feature_list))

    with open(os.path.join(base_res_folder, 'test.dat'), 'w') as f:
        f.write(turn_df_to_feature_str_for_model(test_df, feature_list=feature_list))

    print("Strating Train : " + model_inner_folder + ' ' + feature_folder + ' ' + fold_folder)
    sys.stdout.flush()
    model_filename = learn_svm_rank_model(
        train_file=os.path.join(base_res_folder, 'train.dat'),
        models_folder=base_res_folder,
        C=C)

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
    return test_df


def run_cv_for_config(
        base_feature_filename,
        snapshot_limit,
        feature_groupname,
        C,
        retrieval_model,
        normalize_relevance):

    k_fold = 10
    if '2008' in base_feature_filename:
        init_q = 1
        end_q = 20
        query_bulk = 20
    else:
        init_q = 201
        end_q = 210
        query_bulk = 10

    if feature_groupname == 'All':
        feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                        'QueryWords','Stopwords','TextLen','-Query-SW',
                        'QueryTermsRatio_LG', 'StopwordsRatio_LG', 'Entropy_LG', 'SimClueWeb_LG',
                        'QueryWords_LG', 'Stopwords_LG', 'TextLen_LG', '-Query-SW_LG',
                        'QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                        'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG',
                        # 'QueryTermsRatio_STDG', 'StopwordsRatio_STDG', 'Entropy_STDG', 'SimClueWeb_STDG',
                        # 'QueryWords_STDG', 'Stopwords_STDG', 'TextLen_STDG', '-Query-SW_STDG'
                        ]

    elif feature_groupname == 'Static':
        feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                        'QueryWords', 'Stopwords', 'TextLen', '-Query-SW']

    elif feature_groupname == 'Static_LG':
        feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                        'QueryWords','Stopwords','TextLen','-Query-SW',
                        'QueryTermsRatio_LG', 'StopwordsRatio_LG', 'Entropy_LG', 'SimClueWeb_LG',
                        'QueryWords_LG', 'Stopwords_LG', 'TextLen_LG', '-Query-SW_LG']

    elif feature_groupname == 'Static_MG':
        feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                        'QueryWords', 'Stopwords', 'TextLen', '-Query-SW',
                        'QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                        'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG']

    elif feature_groupname == 'Static_STDG':
        feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                        'QueryWords', 'Stopwords', 'TextLen', '-Query-SW',
                        'QueryTermsRatio_STDG', 'StopwordsRatio_STDG', 'Entropy_STDG', 'SimClueWeb_STDG',
                        'QueryWords_STDG', 'Stopwords_STDG', 'TextLen_STDG', '-Query-SW_STDG']

    elif feature_groupname == 'Static_MG_STDG':
        feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                        'QueryWords', 'Stopwords', 'TextLen', '-Query-SW',
                        'QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                        'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG',
                        'QueryTermsRatio_STDG', 'StopwordsRatio_STDG', 'Entropy_STDG', 'SimClueWeb_STDG',
                        'QueryWords_STDG', 'Stopwords_STDG', 'TextLen_STDG', '-Query-SW_STDG']

    elif feature_groupname == 'MG_STDG':
        feature_list = ['QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                        'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG',
                        'QueryTermsRatio_STDG', 'StopwordsRatio_STDG', 'Entropy_STDG', 'SimClueWeb_STDG',
                        'QueryWords_STDG', 'Stopwords_STDG', 'TextLen_STDG', '-Query-SW_STDG']

    elif feature_groupname == 'MG':
        feature_list = ['QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                        'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG']

    elif feature_groupname == 'LG':
        feature_list = ['QueryTermsRatio_LG', 'StopwordsRatio_LG', 'Entropy_LG', 'SimClueWeb_LG',
                        'QueryWords_LG', 'Stopwords_LG', 'TextLen_LG', '-Query-SW_LG']

    elif feature_groupname == 'STDG':
        feature_list = ['QueryTermsRatio_STDG', 'StopwordsRatio_STDG', 'Entropy_STDG', 'SimClueWeb_STDG',
                        'QueryWords_STDG', 'Stopwords_STDG', 'TextLen_STDG', '-Query-SW_STDG']

    if retrieval_model == 'LM':
        feature_list.append('LMScore')
    elif retrieval_model == 'BM25':
        feature_list.append('BM25Score')

    test_score_df = pd.DataFrame({})

    for i in range(k_fold):
        fold_test_df = train_and_test_model_on_config(
            base_feature_filename=base_feature_filename,
            snapshot_limit=snapshot_limit,
            feature_list= feature_list,
            start_test_q=init_q,
            end_test_q=end_q,
            C=C,
            feature_groupname=feature_groupname +'_'+retrieval_model,
            normalize_relevance=normalize_relevance)
        init_q += query_bulk
        end_q += query_bulk
        test_score_df = test_score_df.append(fold_test_df, ignore_index=True)
    return test_score_df

def run_grid_search_over_params_for_config(
        base_feature_filename,
        snapshot_limit,
        retrieval_model,
        normalize_relevance):

    optional_c_list = [0.2, 0.1, 0.01, 0.001]
    optional_feat_groups_list = ['All','Static','Static_LG','Static_MG','MG','LG']

    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/ret_res/'
    save_summary_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/'
    if '2008' in base_feature_filename:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc"
    else:
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels_cw12.adhoc"

    model_base_filename = base_feature_filename.replace('All_features_with_meta.tsv', '') + 'SNL' + str(snapshot_limit) + "_" + retrieval_model
    if normalize_relevance == True:
        model_base_filename += '_NR'
    model_summary_df = pd.DataFrame(columns = ['FeatureGroup', 'C', 'Map', 'P@5', 'P@10'])
    next_idx = 0
    for optional_c in optional_c_list:
        for feat_group in optional_feat_groups_list:
            test_res_df = run_cv_for_config(
                base_feature_filename=base_feature_filename,
                snapshot_limit=snapshot_limit,
                feature_groupname=feat_group,
                C=optional_c,
                retrieval_model=retrieval_model,
                normalize_relevance=normalize_relevance)

            if next_idx == 0:
                curr_res_df = get_trec_prepared_df_form_res_df(
                    scored_docs_df=test_res_df,
                    score_colname=retrieval_model+'Score')
                insert_row = ['Benchmark', pd.np.nan]
                curr_file_name =  model_base_filename + '_Benchmark.txt'
                with open(os.path.join(save_folder ,curr_file_name), 'w') as f:
                    f.write(convert_df_to_trec(curr_res_df))

                res_dict = get_ranking_effectiveness_for_res_file(
                    file_path=save_folder,
                    filename=curr_file_name,
                    qrel_filepath=qrel_filepath)
                for measure in ['Map', 'P_5', 'P_10']:
                    insert_row.append(res_dict[measure])

                model_summary_df.loc[next_idx] = insert_row
                next_idx+=1

            curr_res_df = get_trec_prepared_df_form_res_df(
                scored_docs_df=test_res_df,
                score_colname='ModelScore')
            insert_row = [feat_group, optional_c]
            curr_file_name = model_base_filename + '_' + feat_group + '_' + str(optional_c) + '.txt'
            with open(os.path.join(save_folder, curr_file_name), 'w') as f:
                f.write(convert_df_to_trec(curr_res_df))

            res_dict = get_ranking_effectiveness_for_res_file(
                file_path=save_folder,
                filename=curr_file_name,
                qrel_filepath=qrel_filepath)
            for measure in ['Map', 'P_5', 'P_10']:
                insert_row.append(res_dict[measure])

            model_summary_df.loc[next_idx] = insert_row
            next_idx += 1

    model_summary_df.to_csv(os.path.join(save_summary_folder, model_base_filename +'.tsv'), sep = '\t', index = False)



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

    if operation == 'GridSearchParams':
        base_feature_filename = sys.argv[2]
        snapshot_limit = int(sys.argv[3])
        retrieval_model = sys.argv[4]
        normalize_relevance = ast.literal_eval(sys.argv[5])

        run_grid_search_over_params_for_config(
            base_feature_filename=base_feature_filename,
            snapshot_limit=snapshot_limit,
            retrieval_model=retrieval_model,
            normalize_relevance=normalize_relevance)