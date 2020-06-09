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
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)
    predictions_file = predictions_folder + os.path.basename(model_file)
    command = "./svm_rank_classify " + test_file + " " + model_file + " " + predictions_file
    print("##Running command: "+command+"##")
    out = run_bash_command(command)
    print("Output of ranking command: "+str(out),flush=True)
    return predictions_file


def learn_svm_rank_model(train_file, fold, C):
    models_folder = "svm_rank_models/" + str(fold) + "/"
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    model_file = models_folder + "model_" + str(C) + ".txt"
    command = "./svm_rank_learn -c " + str(C) + " " + train_file + " " + model_file
    out = run_bash_command(command)
    print(out)
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
        year_df_dict[year].rename(columns = {'#Stopword' : 'Stopwords', 'SimToClueWeb' : 'SimClueWeb'}, inplace = True)
        year_df_dict[year]['-Query-SW'] = year_df_dict[year].apply(
            lambda row: row['TextLen'] - (row['#Stopword'] + row['QueryWords']),
            axis=1)

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

    lm_scores_ref_df['Query_ID'] = lm_scores_ref_df['Query_ID'].applymap(lambda x: int(x))
    bm25_scores_ref_df['Query_ID'] = bm25_scores_ref_df['Query_ID'].applymap(lambda x: int(x))
    rel_df['Query'] = rel_df['Query'].applymap(lambda x: int(x))

    meta_data_df = pd.merge(
        lm_scores_ref_df[['Query_ID', 'Docno', 'Score']].rename(columns = {'Query_ID' : 'QueryNum', 'Score': ''}),
        bm25_scores_ref_df[['Query_ID', 'Docno', 'Score']].rename(columns={'Query_ID': 'QueryNum', 'Score': 'BM25Score'}),
        on = ['QueryNum', 'Docno'],
        how = 'inner')

    meta_data_df = pd.merge(
        meta_data_df,
        rel_df.rename(columns = {'Query' : 'QueryNum'}),
        on=['QueryNum', 'Docno'],
        how='left')

    meta_data_df.fillna(0, inplace = True)
    del lm_scores_ref_df
    del bm25_scores_ref_df
    meta_data_df[['BM25Score', 'LMScore']] = meta_data_df[['BM25Score', 'LMScore']].applymap(lambda x: float(x))

    if len(meta_data_df) != len(lm_scores_ref_df):
        raise Exception('retrieval data not Not allienged' )

    print ("Meta data retrieved!")
    sys.stdout.flush()

    fin_df = pd.DataFrame(
        columns=['NumSnapshots','QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                 'QueryWords','Stopwords','TextLen','-Query-SW',
                 'QueryTermsRatio_LG', 'StopwordsRatio_LG', 'Entropy_LG', 'SimClueWeb_LG',
                 'QueryWords_LG', 'Stopwords_LG', 'TextLen_LG', '-Query-SW_LG',
                 'QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                 'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG',
                 # 'LMScore','BM25Score', 'Relevance',
                 'QueryNum', 'Docno'])
    base_feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                 'QueryWords','Stopwords','TextLen','-Query-SW']
    next_index = 0
    doc_query_df = doc_query_df.drop_duplicates()
    for row in doc_query_df.itertuples():
        docno = row.Docno
        query = int(row.QueryNum)
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
            tmp_doc_df[feature + '_Shift'] = tmp_doc_df[feature].shift(-1)
            tmp_doc_df[feature + '_Grad'] = tmp_doc_df.apply(lambda row_: row_[feature + '_Shift'] - row_[feature] , axis = 1)

        tmp_doc_df = tmp_doc_df[tmp_doc_df['Interval'] != 'ClueWeb09']
        for feature in base_feature_list:
            insert_row.append(list(tmp_doc_df[feature + '_Grad'])[-1])

        for feature in base_feature_list:
            insert_row.append(tmp_doc_df[feature + '_Grad'].mean())

        insert_row.extend([docno, query])
        fin_df.loc[next_index] = insert_row
        next_index += 1

    filename = interval_freq + "_"
    for year_ in year_list:
        filename += year_ + '_'

    filename += last_interval + '_All_features'

    fin_df.to_csv(os.path.join(save_folder, filename + '_raw.tsv'), sep = '\t', index = False)
    meta_data_df.to_csv(os.path.join(save_folder, filename + '_Meatdata.tsv'), sep = '\t', index = False)

    fin_df = pd.merge(
        fin_df,
        meta_data_df,
        on=['QueryNum', 'Docno'],
        how = 'inner')

    fin_df.to_csv(os.path.join(save_folder, filename + '_with_meta.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    operation = sys.argv[1]

    if operation == 'MultiYearFile':
        year_list = ast.literal_eval(sys.argv[2])
        last_interval = sys.argv[3]
        interval_freq = sys.argv[4]
        inner_fold = sys.argv[5]
        retrival_scores_inner_fold= sys.argv[6]
        create_base_feature_file_for_configuration(year_list=year_list, last_interval=last_interval, interval_freq=interval_freq,
                                        inner_fold=inner_fold, retrival_scores_inner_fold=retrival_scores_inner_fold)
