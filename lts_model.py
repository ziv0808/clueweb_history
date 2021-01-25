from utils import *

from avg_doc_model import create_cached_docs_data


if __name__=='__main__':
    inner_fold = sys.argv[1]
    retrival_model = sys.argv[2]
    train_leave_one_out = ast.literal_eval(sys.argv[3])

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
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/avg_model_res_asrc/'

    query_to_doc_mapping_df = create_query_to_doc_mapping_df(inner_fold=inner_fold)
    limited_q_list = list(query_to_doc_mapping_df['QueryNum'].drop_duplicates())

    all_folds_df = pd.DataFrame({})
    all_fold_params_summary = "Fold" + '\t' + "Params" + '\n'
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
    stem_time_series_dict = {}
    for stem in all_global_params_dict:
        print(stem)
        stem_time_series_dict[stem] = np.array(all_global_params_dict[stem].sum(axis = 0))
        print(stem_time_series_dict[stem])
        # normalize
        normalize_factor = np.sqrt(np.sum(np.square(stem_time_series_dict[stem])))
        stem_time_series_dict[stem] = stem_time_series_dict[stem] / normalize_factor
        # diff series
        print(stem_time_series_dict[stem])
        new_stem_ts = []
        for i in range(1, len(stem_time_series_dict[stem])):
            new_stem_ts.append(stem_time_series_dict[stem][i] - stem_time_series_dict[stem][i-1])
        stem_time_series_dict[stem] = np.array(new_stem_ts)
        print(stem_time_series_dict[stem])


    for fold in fold_list:
        start_test_q = int(fold[0])
        end_test_q = int(fold[1])
        affix = inner_fold + '_' + str(start_test_q) + '_' + str(end_test_q) + "_"

        if retrival_model == 'BM25':
            params = {'K1': 1, 'b': 0.5}
            affix += "BM25_"

        elif retrival_model == 'LM':
            params = {'Mue': 1000.0}
            affix += ""
        else:
            raise Exception("Unknown model")


        test_set_q = list(range(start_test_q, end_test_q + 1))
        stemmed_queries_df['IsTest'] = stemmed_queries_df['QueryNum'].apply(lambda x: 1 if x in test_set_q else 0)
        test_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 1].copy()
        stemmed_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 0]

