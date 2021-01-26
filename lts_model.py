from utils import *

from avg_doc_model import create_cached_docs_data

def get_cw09_cc_dict(
        dataset_name):
    cc_df = pd.read_csv('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + dataset_name + '/All_Collection_Counts.tsv', sep='\t', index_col=False)
    cc_dict = {}
    for index, row in cc_df.iterrows():
        cc_dict[row['Stem']] = int(row['CollectionCount'])

    return cc_dict


def score_doc_for_query_bm25(
        query_stem_dict,
        df_dict,
        doc_dict,
        params,
        stem_time_series_wieghts_dict,
        ts_model_type):

    k1 = params['K1']
    b = params['b']
    k3 = params['K3']
    q = params['q']
    bm25_score = 0.0
    work_stem_list = list(query_stem_dict.keys())

    for stem in work_stem_list:
        if stem in doc_dict:
            doc_stem_tf = float(doc_dict[stem])
        else:
            doc_stem_tf = 0.0
        doc_len = float(doc_dict['NumWords'])

        stem_df = stem_time_series_wieghts_dict[stem][ts_model_type]

        avg_doc_len = df_dict['AVG_DOC_LEN']

        idf = stem_df
        stem_d_proba = (doc_stem_tf * (k1 + 1)) / (
            doc_stem_tf + k1 * ((1 - b) + b * (float(doc_len) / avg_doc_len)))
        k3_q_formula = ((k3+ 1)*q*doc_stem_tf) / float(k3 + q*doc_stem_tf)
        bm25_score += idf * stem_d_proba * k3_q_formula

    return bm25_score


def get_scored_df_for_query(
        query_num,
        query,
        query_doc_df,
        df_dict,
        all_docs_tf_dict,
        stem_time_series_wieghts_dict,
        params,
        ts_model_type):

    res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])
    next_index = 0
    query_dict = convert_query_to_tf_dict(query)
    for index, row in query_doc_df.iterrows():
        docno = row['Docno']
        # retrive docno dict
        doc_dict = all_docs_tf_dict[docno]
        doc_score = score_doc_for_query_bm25(
            query_stem_dict=query_dict,
            df_dict=df_dict,
            doc_dict=doc_dict,
            params=params,
            stem_time_series_wieghts_dict=stem_time_series_wieghts_dict,
            ts_model_type=ts_model_type)

        res_df.loc[next_index] = ["0" * (3 - len(str(query_num))) + str(query_num), 'Q0', docno, 0, doc_score, 'indri']
        next_index += 1

    if res_df.empty == False:
        res_df.sort_values('Score', ascending=False, inplace=True)
        res_df['Rank'] = list(range(1, next_index + 1))
    return res_df

def test_queries(
    stemmed_queries_df,
    query_to_doc_mapping_df,
    df_dict,
    all_docs_tf_dict,
    stem_time_series_wieghts_dict,
    params,
    ts_model_type):

    big_df = pd.DataFrame({})
    for index, row in stemmed_queries_df.iterrows():
        query_num = int(row['QueryNum'])
        print("Query: " + str(query_num))
        sys.stdout.flush()
        query_txt = row['QueryStems']
        relevant_df = query_to_doc_mapping_df[query_to_doc_mapping_df['QueryNum'] == query_num].copy()
        res_df = get_scored_df_for_query(
            query_num=query_num,
            query=query_txt,
            query_doc_df=relevant_df,
            df_dict=df_dict,
            all_docs_tf_dict=all_docs_tf_dict,
            stem_time_series_wieghts_dict=stem_time_series_wieghts_dict,
            params=params,
            ts_model_type=ts_model_type)

        big_df = big_df.append(res_df, ignore_index=True)

    return big_df

def create_all_docs_tf_dict(
        processed_docs_path,
        query_to_doc_mapping_df):

    big_doc_dict = {}
    for index, row in query_to_doc_mapping_df.iterrows():
        docno = row['Docno']
        with open(os.path.join(processed_docs_path, docno + '.json'), 'r') as f:
            doc_dict = ast.literal_eval(f.read())
        big_doc_dict[docno] = doc_dict['ClueWeb09']['TfDict']
        big_doc_dict[docno]['NumWords'] = doc_dict['ClueWeb09']['NumWords']
    return big_doc_dict

def create_doc_avg_len_dict(
        all_global_params_dict):

    return {'AVG_DOC_LEN' : all_global_params_dict['NumWords']['ClueWeb09'].mean()}


def get_score_retrieval_score_for_df(
        affix,
        big_df,
        save_folder,
        qrel_filepath):

    curr_file_name = affix + "_Results.txt"
    with open(os.path.join(save_folder + 'inner_res/', curr_file_name), 'w') as f:
        f.write(convert_df_to_trec(big_df))

    res_dict = get_ranking_effectiveness_for_res_file_per_query(
        file_path=save_folder + 'inner_res/',
        filename=curr_file_name,
        qrel_filepath=qrel_filepath,
        calc_ndcg_mrr=True)

    return res_dict
if __name__=='__main__':
    inner_fold = sys.argv[1]
    train_leave_one_out = ast.literal_eval(sys.argv[2])

    sw_rmv = True
    filter_params = {}
    asrc_round = int(inner_fold.split('_')[-1])

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

    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+inner_fold+'/2008/SIM/'
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/'

    query_to_doc_mapping_df = create_query_to_doc_mapping_df(inner_fold=inner_fold)
    limited_q_list = list(query_to_doc_mapping_df['QueryNum'].drop_duplicates())


    q_list, fold_list = get_asrc_q_list_and_fold_list(inner_fold, train_leave_one_out=train_leave_one_out)

    stemmed_queries_df = create_stemmed_queries_df(sw_rmv=sw_rmv, limited_q_list=limited_q_list)
    interval_list = build_interval_list_asrc(asrc_round)

    all_global_params_dict = create_cached_docs_data(
        interval_list=interval_list,
        stemmed_queires_df=stemmed_queries_df,
        query_to_doc_mapping_df=query_to_doc_mapping_df,
        processed_docs_path=processed_docs_folder,
        filter_params=filter_params,
        amount_of_snapshot_limit=None
    )
    cw_cc_dict = get_cw09_cc_dict(dataset_name=inner_fold.split('_')[0])
    bash_command = "python3 lts_pre_process.py " + inner_fold
    p = subprocess.Popen(bash_command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)
    out, err = p.communicate()
    print(out)
    sys.stdout.flush()
    with open(save_folder + 'preprocessed_dicts/' + inner_fold + '_preprocessed_dict.json', 'r') as f:
        stem_time_series_wieghts_dict = ast.literal_eval(f.read())
    # stem_time_series_wieghts_dict = create_rmse_scores_per_term(
    #                                     all_global_params_dict=all_global_params_dict,
    #                                     cc_dict=cw_cc_dict)
    all_docs_tf_dict = create_all_docs_tf_dict(
                            processed_docs_path=processed_docs_folder,
                            query_to_doc_mapping_df=query_to_doc_mapping_df)
    df_dict = create_doc_avg_len_dict(all_global_params_dict)

    all_folds_df_dict = {}
    all_fold_params_summary = {}
    for ts_model_type in ['MA', 'LR', 'ARMA']:
        all_folds_df_dict[ts_model_type] = pd.DataFrame({})
        all_fold_params_summary[ts_model_type] = "Fold" + '\t'  + "Params" + '\n'


    for fold in fold_list:
        start_test_q = int(fold[0])
        end_test_q = int(fold[1])
        affix = inner_fold + '_' + str(start_test_q) + '_' + str(end_test_q) + "_"

        test_set_q = list(range(start_test_q, end_test_q + 1))
        stemmed_queries_df['IsTest'] = stemmed_queries_df['QueryNum'].apply(lambda x: 1 if x in test_set_q else 0)
        test_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 1].copy()
        stemmed_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 0]

        k1_option_list = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        b_option_list = [0.3,0.45,0.5,0.55,0.6,0.75,0.9]
        k3_option_list = [1,2,3,4,5,6,7,8,9,10]
        q_option_list = [0.3,0.45,0.5,0.55,0.6,0.75,0.9]

        best_ndcg_dict = {}
        best_config_dict = {}
        for ts_model_type in ['MA', 'LR', 'ARMA']:
            best_ndcg_dict[ts_model_type] = 0.0
        for k1 in k1_option_list:
            for b in b_option_list:
                params = {'K1' : k1,
                          'b'  : b,
                          'K3' : 7,
                          'q'  : 0.5}

                for ts_model_type in ['MA', 'LR', 'ARMA']:
                    big_df = test_queries(
                        stemmed_queries_df=stemmed_queries_df,
                        query_to_doc_mapping_df=query_to_doc_mapping_df,
                        df_dict=df_dict,
                        all_docs_tf_dict=all_docs_tf_dict,
                        stem_time_series_wieghts_dict=stem_time_series_wieghts_dict,
                        params=params,
                        ts_model_type=ts_model_type)

                    res_dict = get_score_retrieval_score_for_df(
                        affix=affix,
                        big_df=big_df,
                        qrel_filepath=qrel_filepath,
                        save_folder=save_folder)

                    if res_dict['all']['NDCG@5'] > best_ndcg_dict[ts_model_type]:
                        best_ndcg_dict[ts_model_type] = res_dict['all']['NDCG@5']
                        best_config_dict[ts_model_type] = params

        for k3 in k3_option_list:
            for q in q_option_list:
                for ts_model_type in ['MA', 'LR', 'ARMA']:
                    params = {'K1': best_config_dict[ts_model_type]['K1'],
                              'b': best_config_dict[ts_model_type]['b'],
                              'K3': k3,
                              'q': q}
                    big_df = test_queries(
                        stemmed_queries_df=stemmed_queries_df,
                        query_to_doc_mapping_df=query_to_doc_mapping_df,
                        df_dict=df_dict,
                        all_docs_tf_dict=all_docs_tf_dict,
                        stem_time_series_wieghts_dict=stem_time_series_wieghts_dict,
                        params=params,
                        ts_model_type=ts_model_type)

                    res_dict = get_score_retrieval_score_for_df(
                        affix=affix,
                        big_df=big_df,
                        qrel_filepath=qrel_filepath,
                        save_folder=save_folder)

                    if res_dict['all']['NDCG@5'] > best_ndcg_dict[ts_model_type]:
                        best_ndcg_dict[ts_model_type] = res_dict['all']['NDCG@5']
                        best_config_dict[ts_model_type] = params

        for ts_model_type in ['MA', 'LR', 'ARMA']:
            big_df = test_queries(
                stemmed_queries_df=test_queries_df,
                query_to_doc_mapping_df=query_to_doc_mapping_df,
                df_dict=df_dict,
                all_docs_tf_dict=all_docs_tf_dict,
                stem_time_series_wieghts_dict=stem_time_series_wieghts_dict,
                params=best_config_dict[ts_model_type],
                ts_model_type=ts_model_type)

            all_folds_df_dict[ts_model_type] = all_folds_df_dict[ts_model_type].append(big_df , ignore_index=True)
            all_fold_params_summary[ts_model_type] += str(start_test_q) + '_' + str(end_test_q) + '\t' + str(best_config_dict[ts_model_type]) + '\n'

    filenam_addition = ""
    if train_leave_one_out == True:
        filenam_addition += "_LoO"

    for ts_model_type in ['MA', 'LR', 'ARMA']:
        curr_file_name = inner_fold + '_' + ts_model_type + '_LTS_'+ filenam_addition + "_Results.txt"
        with open(os.path.join(save_folder + 'final_res/', curr_file_name), 'w') as f:
            f.write(convert_df_to_trec(all_folds_df_dict[ts_model_type]))
        with open(os.path.join(save_folder + 'final_res/', curr_file_name.replace('_Results', '_Params')), 'w') as f:
            f.write(all_fold_params_summary[ts_model_type])