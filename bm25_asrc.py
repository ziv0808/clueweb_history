from utils import *


def score_doc_for_query_bm25(
        query_stem_dict,
        df_dict,
        doc_dict,
        params):

    k1 = params['K1']
    b = params['b']
    bm25_score = 0.0
    work_stem_list = list(query_stem_dict.keys())

    for stem in work_stem_list:
        if stem in doc_dict:
            doc_stem_tf = float(doc_dict[stem])
        else:
            doc_stem_tf = 0.0
        doc_len = float(doc_dict['NumWords'])

        stem_df = df_dict[stem]

        avg_doc_len = df_dict['AVG_DOC_LEN']
        all_docs_count = df_dict['ALL_DOCS_COUNT']
        idf = math.log(all_docs_count / float(stem_df), 10)

        stem_d_proba = (doc_stem_tf * (k1 + 1)) / (
            doc_stem_tf + k1 * ((1 - b) + b * (float(doc_len) / avg_doc_len)))
        bm25_score += idf * stem_d_proba

    return bm25_score


def get_scored_df_for_query(
        query_num,
        query,
        query_doc_df,
        df_dict,
        all_docs_tf_dict,
        params):
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
            params=params)

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
        params):

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
            params=params)

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
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bm25_model_res/'

    query_to_doc_mapping_df = create_query_to_doc_mapping_df(inner_fold=inner_fold)
    limited_q_list = list(query_to_doc_mapping_df['QueryNum'].drop_duplicates())


    q_list, fold_list = get_asrc_q_list_and_fold_list(inner_fold, train_leave_one_out=train_leave_one_out)

    stemmed_queries_df = create_stemmed_queries_df(sw_rmv=sw_rmv, limited_q_list=limited_q_list)
    interval_list = build_interval_list_asrc(asrc_round)

    df_dict = create_df_dict_with_cw09(inner_fold.split('_')[0])

    all_folds_df = pd.DataFrame({})
    all_fold_params_summary = "Fold" + '\t' + "Params" + '\n'

    all_docs_tf_dict = create_all_docs_tf_dict(
        processed_docs_path=processed_docs_folder,
        query_to_doc_mapping_df=query_to_doc_mapping_df)

    for fold in fold_list:
        start_test_q = int(fold[0])
        end_test_q = int(fold[1])
        affix = inner_fold + '_' + str(start_test_q) + '_' + str(end_test_q) + "_"

        test_set_q = list(range(start_test_q, end_test_q + 1))
        stemmed_queries_df['IsTest'] = stemmed_queries_df['QueryNum'].apply(lambda x: 1 if x in test_set_q else 0)
        test_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 1].copy()
        train_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 0].copy()

        k1_option_list = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        b_option_list = [0.3, 0.45, 0.5, 0.55, 0.6, 0.75, 0.9]

        best_ndcg = 0.0
        best_config = None
        for k1 in k1_option_list:
            for b in b_option_list:
                params = {'K1': k1,
                          'b': b}

                big_df = test_queries(
                    stemmed_queries_df=train_queries_df,
                    query_to_doc_mapping_df=query_to_doc_mapping_df,
                    df_dict=df_dict,
                    all_docs_tf_dict=all_docs_tf_dict,
                    params=params)

                res_dict = get_score_retrieval_score_for_df(
                    affix=affix,
                    big_df=big_df,
                    qrel_filepath=qrel_filepath,
                    save_folder=save_folder)

                if res_dict['all']['NDCG@5'] > best_ndcg:
                    best_ndcg = res_dict['all']['NDCG@5']
                    best_config = params
        # test queries
        test_fold_df = test_queries(
            stemmed_queries_df=test_queries_df,
            query_to_doc_mapping_df=query_to_doc_mapping_df,
            df_dict=df_dict,
            all_docs_tf_dict=all_docs_tf_dict,
            params=best_config)

        all_folds_df = all_folds_df.append(test_fold_df, ignore_index=True)
        all_fold_params_summary += str(start_test_q) + '_' + str(end_test_q) + '\t' + str(best_config) + '\n'

    filenam_addition = ""
    if train_leave_one_out == True:
        filenam_addition += "_LoO"
    curr_file_name = inner_fold + '_BM25' + filenam_addition + "_Results.txt"
    with open(os.path.join(save_folder + 'final_res/', curr_file_name), 'w') as f:
        f.write(convert_df_to_trec(all_folds_df))
    with open(os.path.join(save_folder + 'final_res/', curr_file_name.replace('_Results', '_Params')), 'w') as f:
        f.write(all_fold_params_summary)