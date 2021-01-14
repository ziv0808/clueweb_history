from utils import *

EM_HOLT_DIFF = 0.005

def create_united_lm_for_doc_dict_list(
        doc_dict_list):

    res_dict = {'NumWords' : 0}
    for doc_dict in doc_dict_list:
        for term in doc_dict['TfDict']:
            if term in res_dict:
                res_dict[term] += int(doc_dict['TfDict'][term])
            else:
                res_dict[term] = int(doc_dict['TfDict'][term])
        res_dict['NumWords'] += int(doc_dict['NumWords'])

    return res_dict

def em_step_for_doc(
        doc_tf_vec,
        curr_proba_vec,
        adverserail_vec,
        lambda_1,
        collection_vec = None,
        lambda_2 = 0):
    # e step
    if lambda_2 == 0:
        proba_z_0_vec = lambda_1*adverserail_vec / (lambda_1*adverserail_vec + (1- lambda_1)* curr_proba_vec)
    else:
        proba_z_0_vec = (lambda_1 * adverserail_vec + lambda_2 * collection_vec)/ (lambda_1 * adverserail_vec + lambda_2 * collection_vec + (1 - (lambda_1 + lambda_2)) * curr_proba_vec)
    proba_z_0_vec = 1.0 - proba_z_0_vec
    # m step
    res_proba_vec = proba_z_0_vec * doc_tf_vec
    res_proba_vec = res_proba_vec / np.sum(res_proba_vec)
    return res_proba_vec

def run_em_for_doc(
        doc_dict,
        adveserial_dict,
        lambda_1,
        collection_dict=None,
        lambda_2=0):

    adverserail_vec = []
    collection_vec  = None
    if collection_dict is not None:
        collection_vec = []
    stem_list       = doc_dict['StemList'][1:]
    doc_tf_vec      = np.array(doc_dict['TfList'][1:])
    num_stems       = float(len(stem_list))
    # init uniform probabilities
    curr_proba_vec = np.array([1.0/num_stems]*int(num_stems))
    # set adverserail_vec and collection_vec to contain values in correct order by stem_list
    for stem in stem_list:
        if stem in adveserial_dict:
            adverserail_vec.append(float(adveserial_dict[stem]) / adveserial_dict['NumWords'])
        else:
            adverserail_vec.append(0.0)
        if collection_dict is not None:
            collection_vec.append(float(collection_dict[stem]) / collection_dict['ALL_TERMS_COUNT'])
    adverserail_vec = np.array(adverserail_vec)
    collection_vec = np.array(collection_vec)
    em_running = True
    num_steps = 1
    while em_running == True:
        # print("EM Step: " + str(num_steps))
        # sys.stdout.flush()
        new_proba_vec = em_step_for_doc(
                                doc_tf_vec=doc_tf_vec,
                                curr_proba_vec=curr_proba_vec,
                                adverserail_vec=adverserail_vec,
                                lambda_1=lambda_1,
                                collection_vec=collection_vec,
                                lambda_2=lambda_2)
        diff = np.sum(np.abs(new_proba_vec - curr_proba_vec))
        curr_proba_vec = new_proba_vec
        num_steps += 1
        if diff < EM_HOLT_DIFF:
            em_running = False
        if num_steps > 50:
            raise Exception("Non-Converging " + str(diff) + '\nStemList: ' + str(stem_list) + '\nTFVec: ' + str(doc_tf_vec) + '\nCollecVec: ' + str(collection_vec) + '\nAdvereVec: ' + str(adverserail_vec) + '\nProbaVec: ' + str(curr_proba_vec) )
    doc_proba_dict = {}
    for i in range(len(stem_list)):
        doc_proba_dict[stem_list[i]] = curr_proba_vec[i]

    return doc_proba_dict


def get_kl_result_for_doc(
        query_stem_dict,
        doc_proba_dict,
        collection_dict,
        params,
        model_to_run,
        doc_len):

    kl_score = 0.0
    work_stem_list = list(query_stem_dict.keys())
    q_normalizer = sum(list(query_stem_dict.values()))
    for stem in work_stem_list:
        collection_len = collection_dict['ALL_TERMS_COUNT']
        collection_count_for_word = collection_dict[stem]

        if stem in doc_proba_dict:
            stem_d_proba = doc_proba_dict[stem]
        else:
            stem_d_proba = 0.0
        if 'JM' in model_to_run:
            beta = params['Beta']
        elif 'DIR' in model_to_run:
            beta = (1.0 - (params['Mue']/ float(params['Mue'] + doc_len)))
        stem_d_proba = beta*stem_d_proba + (1.0-beta)*(float(collection_count_for_word) / collection_len)
        query_tf = query_stem_dict[stem]
        stem_q_prob = float(query_tf) / q_normalizer
        kl_score += (-1) * stem_q_prob * (math.log((stem_q_prob / stem_d_proba), 2))

    return kl_score

def create_mle_dict_for_doc(
        doc_dict):
    res_dict = {}
    for stem in doc_dict['TfDict']:
        res_dict[stem] = float(doc_dict['TfDict'][stem]) / float(doc_dict['NumWords'])
    return res_dict

def get_scored_df_for_query(
        query_num,
        query,
        query_doc_df,
        query_adveserial_dict,
        collection_dict,
        processed_docs_path,
        params,
        model_to_run):

    res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])
    next_index = 0
    query_dict = convert_query_to_tf_dict(query)
    for index, row in query_doc_df.iterrows():
        docno = row['Docno']
        # retrive docno dict
        with open(os.path.join(processed_docs_path, docno + '.json'), 'r') as f:
            doc_dict = ast.literal_eval(f.read())
        doc_dict = doc_dict['ClueWeb09']
        if 'Mixture' in model_to_run:
            doc_proba_dict = run_em_for_doc(
                doc_dict=doc_dict,
                adveserial_dict=query_adveserial_dict,
                lambda_1=params['Lambda1'],
                collection_dict=collection_dict,
                lambda_2=params['Lambda2'])
        elif model_to_run in ['JM', 'DIR']:
            doc_proba_dict = create_mle_dict_for_doc(doc_dict)

        doc_score = get_kl_result_for_doc(
            query_stem_dict=query_dict,
            doc_proba_dict=doc_proba_dict,
            collection_dict=collection_dict,
            params=params,
            model_to_run=model_to_run,
            doc_len=doc_dict['NumWords'])


        res_df.loc[next_index] = ["0" * (3 - len(str(query_num))) + str(query_num), 'Q0', docno, 0, doc_score, 'indri']
        next_index += 1

    if res_df.empty == False:
        res_df.sort_values('Score', ascending=False, inplace=True)
        res_df['Rank'] = list(range(1, next_index + 1))
    return res_df

def test_queries(
    stemmed_queries_df,
    query_to_doc_mapping_df,
    all_query_adveserial_dict,
    collection_dict,
    params,
    processed_docs_path,
    model_to_run):

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
            collection_dict=collection_dict,
            query_adveserial_dict=all_query_adveserial_dict[str(query_num).zfill(3)],
            params=params,
            processed_docs_path=processed_docs_path,
            model_to_run=model_to_run)

        big_df = big_df.append(res_df, ignore_index=True)

    return big_df

def create_cc_dict(
        dataset_name):

    cc_df = pd.read_csv('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + dataset_name + '/All_Collection_Counts.tsv', sep = '\t', index_col = False)
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + dataset_name + '/cc_per_interval_dict.json', 'r') as f:
        cc_dict = ast.literal_eval(f.read())

    for index, row in cc_df.iterrows():
        cc_dict[row['Stem']] += row['CollectionCount']

    return cc_dict

def create_adveserial_dict_from_docno_list_for_q(
        docno_list,
        dataset_name):

    doc_dict_list = []
    for docno in docno_list:
        docno_round = docno.split('-')[1]
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + dataset_name + '_' + docno_round + '/2008/SIM/' + docno +'.json', 'r') as f:
            doc_json = ast.literal_eval(f.read())
        doc_dict_list.append(doc_json['ClueWeb09'])

    return create_united_lm_for_doc_dict_list(doc_dict_list)


def make_adverserial_dict_by_method(
        dataset_name,
        curr_round,
        adverserial_method):

    prev_rounds_dict = {}
    for round_ in range(1, int(curr_round)):
        round_str = str(round_).zfill(2)
        res_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + dataset_name + '/' + dataset_name + '_' + round_str + '/RankedLists/LambdaMART_' + dataset_name + '_' + round_str
        prev_rounds_dict[round_] = convert_trec_results_file_to_pandas_df(res_file)
        prev_rounds_dict[round_]['Query-User'] = prev_rounds_dict[round_]['Docno'].apply(lambda x: x.split('-')[2] + '-' + x.split('-')[3])
        prev_rounds_dict[round_]['Rank'] = prev_rounds_dict[round_]['Rank'].apply(lambda x: int(x))

    adverserial_dict = {}
    if adverserial_method == 'PrevWinner' or int(curr_round) == 2:
        rel_round = int(curr_round) - 1
        rel_df = prev_rounds_dict[rel_round]
        for q in list(rel_df['Query_ID'].drop_duplicates()):
            q_df = rel_df[rel_df['Query_ID'] == q].copy()
            q_df = q_df[q_df['Rank'] == 1]
            docno_list = list(q_df['Docno'])
            adverserial_dict[str(q).zfill(3)] = create_adveserial_dict_from_docno_list_for_q(
                                                    docno_list=docno_list,
                                                    dataset_name=dataset_name)
    elif adverserial_method == 'Prev2Winners':
        rel_df = pd.DataFrame({})
        for round_ in range(max(1, int(curr_round) - 2), int(curr_round)):
            rel_df = rel_df.append(prev_rounds_dict[round_], ignore_index=True)
        for q in list(rel_df['Query_ID'].drop_duplicates()):
            q_df = rel_df[rel_df['Query_ID'] == q].copy()
            q_df = q_df[q_df['Rank'] == 1]
            docno_list = list(q_df['Docno'])
            adverserial_dict[str(q).zfill(3)] = create_adveserial_dict_from_docno_list_for_q(
                docno_list=docno_list,
                dataset_name=dataset_name)

    elif adverserial_method == 'Prev3Winners':
        rel_df = pd.DataFrame({})
        for round_ in range(max(1, int(curr_round) -3), int(curr_round)):
            rel_df = rel_df.append(prev_rounds_dict[round_], ignore_index=True)
        for q in list(rel_df['Query_ID'].drop_duplicates()):
            q_df = rel_df[rel_df['Query_ID'] == q].copy()
            q_df = q_df[q_df['Rank'] == 1]
            docno_list = list(q_df['Docno'])
            adverserial_dict[str(q).zfill(3)] = create_adveserial_dict_from_docno_list_for_q(
                docno_list=docno_list,
                dataset_name=dataset_name)

    elif adverserial_method == 'PrevBestImprove':
        rel_round = int(curr_round) - 1
        rel_df = prev_rounds_dict[rel_round]
        rel_df = pd.merge(
            rel_df,
            prev_rounds_dict[rel_round -1][['Query-User', 'Rank']].rename(columns={'Rank' : 'PrevRank'}),
            on = ['Query-User'],
            how = 'inner')
        rel_df['RankDiff'] = rel_df['PrevRank'] - rel_df['Rank']
        for q in list(rel_df['Query_ID'].drop_duplicates()):
            q_df = rel_df[rel_df['Query_ID'] == q].copy()
            q_df = q_df[q_df['RankDiff'] == q_df['RankDiff'].max()]
            if len(q_df) > 1:
                q_df = q_df[q_df['Rank'] == q_df['Rank'].max()]
            docno_list = list(q_df['Docno'])
            adverserial_dict[str(q).zfill(3)] = create_adveserial_dict_from_docno_list_for_q(
                docno_list=docno_list,
                dataset_name=dataset_name)

    elif adverserial_method == 'Prev2BestImprove':
        rel_round = int(curr_round) - 1
        rel_df = prev_rounds_dict[rel_round]
        rel_df = pd.merge(
            rel_df,
            prev_rounds_dict[max(rel_round - 2, 1)][['Query-User', 'Rank']].rename(columns={'Rank': 'PrevRank'}),
            on=['Query-User'],
            how='inner')
        rel_df['RankDiff'] = rel_df['PrevRank'] - rel_df['Rank']
        for q in list(rel_df['Query_ID'].drop_duplicates()):
            q_df = rel_df[rel_df['Query_ID'] == q].copy()
            q_df = q_df[q_df['RankDiff'] == q_df['RankDiff'].max()]
            if len(q_df) > 1:
                q_df = q_df[q_df['Rank'] == q_df['Rank'].max()]
            docno_list = list(q_df['Docno'])
            if (rel_round - 2) >= 1:
                docno_list.append('EPOCH-' + str(rel_round - 1).zfill(2) + '-' + list(q_df['Query-User'])[0])
            adverserial_dict[str(q).zfill(3)] = create_adveserial_dict_from_docno_list_for_q(
                docno_list=docno_list,
                dataset_name=dataset_name)

    elif adverserial_method == 'Prev3BestImprove':
        rel_round = int(curr_round) - 1
        rel_df = prev_rounds_dict[rel_round]
        rel_df = pd.merge(
            rel_df,
            prev_rounds_dict[max(rel_round - 3, 1)][['Query-User', 'Rank']].rename(columns={'Rank': 'PrevRank'}),
            on=['Query-User'],
            how='inner')
        rel_df['RankDiff'] = rel_df['PrevRank'] - rel_df['Rank']
        for q in list(rel_df['Query_ID'].drop_duplicates()):
            q_df = rel_df[rel_df['Query_ID'] == q].copy()
            q_df = q_df[q_df['RankDiff'] == q_df['RankDiff'].max()]
            if len(q_df) > 1:
                q_df = q_df[q_df['Rank'] == q_df['Rank'].max()]
            docno_list = list(q_df['Docno'])
            if (rel_round - 2) >= 1:
                docno_list.append('EPOCH-' + str(rel_round - 1).zfill(2) + '-' +list(q_df['Query-User'])[0])
            if (rel_round - 3) >= 1:
                docno_list.append('EPOCH-' + str(rel_round - 2).zfill(2) + '-' + list(q_df['Query-User'])[0])
            adverserial_dict[str(q).zfill(3)] = create_adveserial_dict_from_docno_list_for_q(
                docno_list=docno_list,
                dataset_name=dataset_name)
    else:
        raise Exception("unknown method!")

    return adverserial_dict


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
    model_to_run = sys.argv[1]
    inner_fold = sys.argv[2]
    train_leave_one_out = ast.literal_eval(sys.argv[3])

    sw_rmv = True
    asrc_round = inner_fold.split('_')[-1]
    datset_name = inner_fold.split('_')[0]

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

    query_to_doc_mapping_df = create_query_to_doc_mapping_df(inner_fold=inner_fold)
    limited_q_list = list(query_to_doc_mapping_df['QueryNum'].drop_duplicates())

    all_folds_df = pd.DataFrame({})
    all_fold_params_summary = "Fold" + '\t' + "Params" + '\n'

    q_list, fold_list = get_asrc_q_list_and_fold_list(inner_fold, train_leave_one_out=train_leave_one_out)
    cc_dict = create_cc_dict(dataset_name=datset_name)

    if 'Mixture' in model_to_run:
        adverserial_method = sys.argv[4]
        only_reservoir_lambda = ast.literal_eval(sys.argv[5])

        save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/'
        if 'JM' in model_to_run:
            beta_option_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif 'DIR' in model_to_run:
            beta_option_list = [50, 100, 200, 300, 500, 700, 800, 900, 1000, 1200, 1500]

        lambda1_option_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        lambda2_option_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        addition_to_filename = ""
        if only_reservoir_lambda == True:
            lambda1_option_list = [0.0]
        addition_to_filename = "_OnlyReservoir"
        adverserial_dict = make_adverserial_dict_by_method(
            dataset_name=datset_name,
            curr_round=asrc_round,
            adverserial_method=adverserial_method)

        for fold in fold_list:
            start_test_q = int(fold[0])
            end_test_q = int(fold[1])
            affix = inner_fold + '_' + str(start_test_q) + '_' + str(end_test_q) + "_" + adverserial_method + "_" + model_to_run +addition_to_filename

            stemmed_queries_df = create_stemmed_queries_df(sw_rmv=sw_rmv, limited_q_list=limited_q_list)

            test_set_q = list(range(start_test_q, end_test_q + 1))
            stemmed_queries_df['IsTest'] = stemmed_queries_df['QueryNum'].apply(lambda x: 1 if x in test_set_q else 0)
            test_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 1].copy()
            train_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 0]

            best_config_dict = None
            best_ndcg = 0.0
            init_lam_1 = 0.0
            init_lam_2 = 0.0
            for beta in beta_option_list:
                params = {'Lambda1': init_lam_1,
                          'Lambda2': init_lam_2}
                if 'JM' in model_to_run:
                    params['Beta'] = beta
                elif 'DIR' in model_to_run:
                    params['Mue'] = beta
                print(params)
                big_df = test_queries(
                    stemmed_queries_df=train_queries_df,
                    query_to_doc_mapping_df=query_to_doc_mapping_df,
                    all_query_adveserial_dict=adverserial_dict,
                    collection_dict=cc_dict,
                    params=params,
                    processed_docs_path=processed_docs_folder,
                    model_to_run=model_to_run)

                res_dict = get_score_retrieval_score_for_df(
                    affix=affix,
                    big_df=big_df,
                    qrel_filepath=qrel_filepath,
                    save_folder=save_folder)

                if res_dict['all']['NDCG@5'] > best_ndcg:
                    print(affix + " " + "SCORE : " + str(res_dict['all']))
                    best_ndcg = res_dict['all']['NDCG@5']
                    best_config_dict = params.copy()
                    print(best_config_dict)

            for k in range(1,4):
                if adverserial_method == 'PrevWinner':
                    if k == 1:
                        curr_adverserial_method = 'PrevWinner'
                    else:
                        curr_adverserial_method = 'Prev' + str(k) +'Winners'
                    adverserial_dict = make_adverserial_dict_by_method(
                        dataset_name=datset_name,
                        curr_round=asrc_round,
                        adverserial_method=curr_adverserial_method)
                elif adverserial_method == 'PrevBestImprove':
                    if k == 1:
                        curr_adverserial_method = 'PrevBestImprove'
                    else:
                        curr_adverserial_method = 'Prev' + str(k) + 'BestImprove'
                else:
                    raise Exception("Fucker!!!")

                for lambda1 in lambda1_option_list:
                    for lambda2 in lambda2_option_list:
                        if (lambda1 + lambda2) >= 1:
                            continue
                        params = {'Lambda1': lambda1,
                                  'Lambda2': lambda2,
                                  'AdverMethod' : curr_adverserial_method}
                        if 'JM' in model_to_run:
                            params['Beta'] = best_config_dict['Beta']
                        elif 'DIR' in model_to_run:
                            params['Mue'] = best_config_dict['Mue']
                        print(params)
                        big_df = test_queries(
                                    stemmed_queries_df=train_queries_df,
                                    query_to_doc_mapping_df=query_to_doc_mapping_df,
                                    all_query_adveserial_dict=adverserial_dict,
                                    collection_dict=cc_dict,
                                    params=params,
                                    processed_docs_path=processed_docs_folder,
                                    model_to_run=model_to_run)

                        res_dict = get_score_retrieval_score_for_df(
                            affix=affix,
                            big_df=big_df,
                            qrel_filepath=qrel_filepath,
                            save_folder=save_folder)

                        if res_dict['all']['NDCG@5'] > best_ndcg:
                            print(affix + " " + "SCORE : " + str(res_dict['all']))
                            best_ndcg = res_dict['all']['NDCG@5']
                            best_config_dict = params.copy()
                            print(best_config_dict)

            adverserial_dict = make_adverserial_dict_by_method(
                dataset_name=datset_name,
                curr_round=asrc_round,
                adverserial_method=best_config_dict['AdverMethod'])

            test_fold_df = test_queries(
                                    stemmed_queries_df=test_queries_df,
                                    query_to_doc_mapping_df=query_to_doc_mapping_df,
                                    all_query_adveserial_dict=adverserial_dict,
                                    collection_dict=cc_dict,
                                    params=best_config_dict,
                                    processed_docs_path=processed_docs_folder,
                                    model_to_run=model_to_run)

            all_folds_df = all_folds_df.append(test_fold_df, ignore_index=True)
            all_fold_params_summary += str(start_test_q) + '_' + str(end_test_q) + '\t' + str(best_config_dict) + '\n'

        filenam_addition = "_" + model_to_run + addition_to_filename
        if train_leave_one_out == True:
            filenam_addition += "_LoO"
        curr_file_name = inner_fold + '_' + adverserial_method + filenam_addition + "_Results.txt"
        with open(os.path.join(save_folder + 'final_res/', curr_file_name), 'w') as f:
            f.write(convert_df_to_trec(all_folds_df))
        with open(os.path.join(save_folder + 'final_res/', curr_file_name.replace('_Results', '_Params')), 'w') as f:
            f.write(all_fold_params_summary)

    else:
        save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/basic_lm/'
        if model_to_run == 'JM':
            beta_option_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif model_to_run == 'DIR':
            beta_option_list = [50, 100, 200, 300, 500, 700, 800, 900, 1000, 1200, 1500, 2000]
        for fold in fold_list:
            start_test_q = int(fold[0])
            end_test_q = int(fold[1])
            affix = inner_fold + '_' + str(start_test_q) + '_' + str(end_test_q) + "_" + model_to_run

            stemmed_queries_df = create_stemmed_queries_df(sw_rmv=sw_rmv, limited_q_list=limited_q_list)

            test_set_q = list(range(start_test_q, end_test_q + 1))
            stemmed_queries_df['IsTest'] = stemmed_queries_df['QueryNum'].apply(lambda x: 1 if x in test_set_q else 0)
            test_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 1].copy()
            train_queries_df = stemmed_queries_df[stemmed_queries_df['IsTest'] == 0]

            # mock adverserial dict
            adverserial_dict = make_adverserial_dict_by_method(
                dataset_name=datset_name,
                curr_round=asrc_round,
                adverserial_method='PrevWinner')

            best_config_dict = None
            best_ndcg = 0.0
            for beta in beta_option_list:
                params ={}
                if 'JM' in model_to_run:
                    params['Beta'] = beta
                elif 'DIR' in model_to_run:
                    params['Mue'] = beta

                big_df = test_queries(
                    stemmed_queries_df=train_queries_df,
                    query_to_doc_mapping_df=query_to_doc_mapping_df,
                    all_query_adveserial_dict=adverserial_dict,
                    collection_dict=cc_dict,
                    params=params,
                    processed_docs_path=processed_docs_folder,
                    model_to_run=model_to_run)

                res_dict = get_score_retrieval_score_for_df(
                    affix=affix,
                    big_df=big_df,
                    qrel_filepath=qrel_filepath,
                    save_folder=save_folder)

                if res_dict['all']['NDCG@5'] > best_ndcg:
                    print(affix + " " + "SCORE : " + str(res_dict['all']))
                    best_ndcg = res_dict['all']['NDCG@5']
                    best_config_dict = params.copy()
                    print(best_config_dict)

            test_fold_df = test_queries(
                stemmed_queries_df=test_queries_df,
                query_to_doc_mapping_df=query_to_doc_mapping_df,
                all_query_adveserial_dict=adverserial_dict,
                collection_dict=cc_dict,
                params=best_config_dict,
                processed_docs_path=processed_docs_folder,
                model_to_run=model_to_run)

            all_folds_df = all_folds_df.append(test_fold_df, ignore_index=True)
            all_fold_params_summary += str(start_test_q) + '_' + str(end_test_q) + '\t' + str(best_config_dict) + '\n'


        curr_file_name = inner_fold + '_LMIR.'  + model_to_run + ".txt"
        with open(os.path.join(save_folder , curr_file_name), 'w') as f:
            f.write(convert_df_to_trec(all_folds_df))
        with open(os.path.join(save_folder , curr_file_name.replace('.txt', '_Params.txt')), 'w') as f:
            f.write(all_fold_params_summary)