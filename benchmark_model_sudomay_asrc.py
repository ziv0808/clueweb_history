import os
import ast
import sys
import time
import datetime
import subprocess
import pandas as pd
import murmurhash

from utils import *

class Benchmark:
    def __init__(
            self,
            inner_fold,
            retrieval_model = 'KL'):

        # create interval list
        asrc_round = int(inner_fold.split('_')[-1])
        self.interval_list = build_interval_list_asrc(
            asrc_round=asrc_round,
            add_last=True)

        self.retrieval_model = retrieval_model
        dataset_name = inner_fold.split('_')[0]
        # init usefull dfs and dicts
        self.cc_dict = create_cc_dict_with_cw09(dataset_name)
        self.query_to_doc_mapping_df = create_query_to_doc_mapping_df(inner_fold=inner_fold)
        limited_q_list               = list(self.query_to_doc_mapping_df['QueryNum'].drop_duplicates())
        self.stemmed_queries_df      = create_stemmed_queries_df(sw_rmv=True, limited_q_list=limited_q_list)
        self.query_num_to_tf_dict    = {}
        for index, row in self.stemmed_queries_df.iterrows():
            self.query_num_to_tf_dict[int(row['QueryNum'])] = convert_query_to_tf_dict(row['QueryStems'])
        # preprocess all_docs
        processed_docs_path = os.path.join(os.path.join(os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors', inner_fold), '2008'), 'SIM')
        self.all_docs_dict = self.preprocess_docs(
                    processed_docs_path=processed_docs_path,
                    interval_list = self.interval_list)

        main_dirname = os.path.join(os.path.dirname('/mnt/bi-strg3/v/zivvasilisky/ziv/results/'), 'benchmark_sudomay')
        self.save_dirname = os.path.join(main_dirname, inner_fold + '_' + retrieval_model)
        if not os.path.exists(self.save_dirname):
            os.mkdir(self.save_dirname)

        self.all_words_current_cc = {}
        total_count = 0
        for docno in self.all_docs_dict:
            for stem in self.all_docs_dict[docno]:
                if stem in ['GROUP_S_LENGH','GROUP_M_LENGH','GROUP_L_LENGH']:
                    continue
                if stem not in self.all_words_current_cc:
                    self.all_words_current_cc[stem] = self.all_docs_dict[docno][stem]['TF']
                else:
                    self.all_words_current_cc[stem] += self.all_docs_dict[docno][stem]['TF']
                total_count += self.all_docs_dict[docno][stem]['TF']

        self.all_words_current_cc['TOTAL_COUNT'] = total_count

        self.run_log = ""

    def save_log(self):
        with open(os.path.join(self.save_dirname, self.retrieval_model + '_grid_search_results.txt'), 'w') as f:
            f.write(str(self.run_log))

    def add_to_log(self, strng):
        self.run_log += strng + '\n'

    def preprocess_docs(
            self,
            processed_docs_path,
            interval_list):

        doc_count = 0
        all_docs_dict = {}
        for filename in os.listdir(processed_docs_path):
            if filename.endswith('.json'):
                with open(os.path.join(processed_docs_path, filename), 'r') as f:
                    doc_dict = ast.literal_eval(f.read())
                docno = filename.replace('.json', '')
                all_docs_dict[docno] = self.process_doc_for_S_M_L_groups(
                    interval_list=interval_list,
                    doc_dict=doc_dict)
                doc_count += 1
            if doc_count % 100 == 0:
                print("Docs Processed : " + str(doc_count))

        print("Docs Processed : " + str(doc_count))
        return all_docs_dict

    def minhash(self, text, num_shingels, window=25):  # assume len(text) > 50
        hashes = [murmurhash.hash(text[i:i + window]) for i in range(len(text) - window + 1)]
        return set(sorted(hashes)[0:num_shingels])

    def shingleprint_doc_similarity(
            self,
            doc_1_text,
            doc_2_text,
            num_shingels = 84):

        hashes1 = self.minhash(doc_1_text, num_shingels)
        hashes2 = self.minhash(doc_2_text, num_shingels)

        return len(hashes1 & hashes2) / float(len(hashes1))


    def process_doc_for_S_M_L_groups(
            self,
            interval_list,
            doc_dict):

        working_snapshots_list = []
        shing_score = 0.0
        for i in range(len(interval_list)):
            relevant_snapshot_dict = get_doc_snapshot_by_lookup_method(
                doc_dict=doc_dict,
                interval_list=interval_list,
                interval_lookup_method='NoLookup',
                curr_interval_idx=i)
            if i > 0:
                shing_score += self.shingleprint_doc_similarity(relevant_snapshot_dict['Fulltext'], working_snapshots_list[i-1]['Fulltext'])
            # if relevant_snapshot_dict is not None:
            working_snapshots_list.append(relevant_snapshot_dict)

        shing_score = shing_score / float(len(interval_list) - 1)
        stem_snapshot_count_dict = {}
        stem_total_count_dict = {}
        for snapshot_dict in working_snapshots_list:
            for i in range(1, len(snapshot_dict['StemList'])):
                stem = snapshot_dict['StemList'][i]
                if snapshot_dict['DfList'][i] > 0:
                    if stem in stem_snapshot_count_dict:
                        stem_snapshot_count_dict[stem] += 1
                        stem_total_count_dict[stem]['TF'] += snapshot_dict['TfList'][i]
                    else:
                        stem_snapshot_count_dict[stem] = 1
                        stem_total_count_dict[stem] = {}
                        stem_total_count_dict[stem]['TF'] = snapshot_dict['TfList'][i]

        amount_of_snapshots = len(working_snapshots_list)
        group_S_bar = amount_of_snapshots * 0.5
        group_M_bar = amount_of_snapshots * 0.9
        group_S_len = 0
        group_M_len = 0
        group_L_len = 0
        for stem in stem_snapshot_count_dict:
            if stem_snapshot_count_dict[stem] < group_S_bar:
                stem_total_count_dict[stem]['Group'] = 'S'
                group_S_len += stem_total_count_dict[stem]['TF']

            elif stem_snapshot_count_dict[stem] < group_M_bar:
                stem_total_count_dict[stem]['Group'] = 'M'
                group_M_len += stem_total_count_dict[stem]['TF']

            else:
                stem_total_count_dict[stem]['Group'] = 'L'
                group_L_len += stem_total_count_dict[stem]['TF']

        stem_total_count_dict['GROUP_S_LENGH'] = group_S_len
        stem_total_count_dict['GROUP_M_LENGH'] = group_M_len
        stem_total_count_dict['GROUP_L_LENGH'] = group_L_len
        stem_total_count_dict['ShingDiffScore'] = 1.0 - shing_score

        return stem_total_count_dict


    def score_doc_for_query(
            self,
            query_stem_dict,
            cc_dict,
            doc_dict,
            hyper_param_dict):

        kl_score = 0.0
        if self.retrieval_model == 'LM':
            kl_score = 1.0

        work_stem_list = list(query_stem_dict.keys())
        for stem in work_stem_list:
            if stem not in doc_dict:
                # no meaning to group
                doc_dict[stem] = {}
                doc_dict[stem]['TF'] = 0.0
                doc_dict[stem]['Group'] = 'S'

            stem_d_proba = 0.0
            for curr_group in ['S','M','L']:
                if curr_group == doc_dict[stem]['Group']:
                    stem_d_group_proba = get_word_diriclet_smoothed_probability(
                            tf_in_doc=doc_dict[stem]['TF'],
                            doc_len=doc_dict['GROUP_' + curr_group + '_LENGH'],
                            collection_count_for_word=cc_dict[stem],
                            collection_len=cc_dict['ALL_TERMS_COUNT'],
                            mue=hyper_param_dict[curr_group]['Mue'])
                else:
                    stem_d_group_proba = get_word_diriclet_smoothed_probability(
                            tf_in_doc=0.0,
                            doc_len=doc_dict['GROUP_' + curr_group + '_LENGH'],
                            collection_count_for_word=cc_dict[stem],
                            collection_len=cc_dict['ALL_TERMS_COUNT'],
                            mue=hyper_param_dict[curr_group]['Mue'])
                stem_d_proba += (hyper_param_dict[curr_group]['Lambda']) * stem_d_group_proba

            if self.retrieval_model == 'KL':
                query_tf = query_stem_dict[stem]
                stem_q_prob = float(query_tf) / sum(list(query_stem_dict.values()))
                kl_score += (-1) * stem_q_prob * (math.log((stem_q_prob / stem_d_proba), 2))

            elif self.retrieval_model == 'LM':
                kl_score = kl_score * stem_d_proba
            else:
                raise Exception('Unknown retrival Model')
        if self.retrieval_model == 'LM':
            kl_score = kl_score * (doc_dict['ShingDiffPrior'])
        return kl_score

    def get_scored_df_for_query(
            self,
            query_num,
            query_doc_df,
            cc_dict,
            hyper_param_dict):

        res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])
        next_index = 0
        query_dict = self.query_num_to_tf_dict[query_num]
        for index, row in query_doc_df.iterrows():
            docno = row['Docno']
            doc_dict = self.all_docs_dict[docno]
            doc_score = self.score_doc_for_query(
                query_stem_dict=query_dict,
                cc_dict=cc_dict,
                doc_dict=doc_dict,
                hyper_param_dict=hyper_param_dict)
            res_df.loc[next_index] = ["0"*(3 - len(str(query_num)))+ str(query_num), 'Q0', docno, 0, doc_score, 'indri']
            next_index += 1

        res_df.sort_values('Score', ascending=False, inplace=True)
        res_df['Rank'] = list(range(1, next_index + 1))
        return res_df

    def score_queries(
            self,
            query_list,
            hyper_param_dict):

        big_df = pd.DataFrame({})
        for index, row in self.stemmed_queries_df.iterrows():
            query_num = int(row['QueryNum'])
            if (query_num in query_list):
                print("Query: " + str(query_num))
                sys.stdout.flush()
                relevant_df = self.query_to_doc_mapping_df[self.query_to_doc_mapping_df['QueryNum'] == query_num].copy()
                res_query_df = self.get_scored_df_for_query(
                    query_num=query_num,
                    query_doc_df=relevant_df,
                    cc_dict=self.cc_dict,
                    hyper_param_dict=hyper_param_dict)

                big_df = big_df.append(res_query_df, ignore_index=True)

        return big_df

    def calc_shin_diff_prior(
            self,
            theta):

        prior_sum = 0.0
        for docno in self.all_docs_dict:
            shinscore = self.all_docs_dict[docno]['ShingDiffScore'] + 1.0
            shinscore = shinscore ** theta
            self.all_docs_dict[docno]['ShingDiffPrior'] = shinscore
            prior_sum += shinscore

        for docno in self.all_docs_dict:
            self.all_docs_dict[docno]['ShingDiffPrior'] = self.all_docs_dict[docno]['ShingDiffPrior'] / float(prior_sum)


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

if __name__=="__main__":
    inner_fold = sys.argv[1]
    retrieval_model = sys.argv[2]
    train_leave_one_out = ast.literal_eval(sys.argv[3])

    print('Retrivel Model: ' + retrieval_model)

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


    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/benchmark_sudomay/'

    print('Time: ' + str(datetime.datetime.now()))

    benchmark_obj = Benchmark(
            inner_fold=inner_fold,
            retrieval_model=retrieval_model)
    print("Obj Created!")
    print('Time: ' + str(datetime.datetime.now()))
    sys.stdout.flush()
    query_list, fold_list = get_asrc_q_list_and_fold_list(inner_fold,train_leave_one_out=train_leave_one_out)
    all_folds_df = pd.DataFrame({})
    for fold in fold_list:
        start_test_q = int(fold[0])
        end_test_q = int(fold[1])
        affix = inner_fold + '_' + str(start_test_q) + '_' + str(end_test_q) + "_" + retrieval_model + '_'
        train_q_list = query_list[:]
        test_q_list = []
        for q in range(start_test_q, end_test_q + 1):
            if q in train_q_list:
                train_q_list.remove(q)
                test_q_list.append(q)
        hyper_param_dict = {'S': {'Mue': 1500, 'Lambda': 0.45},
                            'M': {'Mue': 1500, 'Lambda': 0.45},
                            'L': {'Mue': 5, 'Lambda': 0.1}}
        optional_mue_list = [5, 10, 50, 100, 200, 300, 500, 700, 800, 900, 1000, 1200, 1500]
        optional_lambda_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        theta_option_list = [0.0, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.6, 1.7, 2.0, 2.3, 2.5]

        max_ndcg = 0.0
        best_config = None

        if retrieval_model == 'LM':
            for theta in theta_option_list:
                hyper_param_dict['Theta'] = theta
                benchmark_obj.calc_shin_diff_prior(theta)
                print("Prior calculated")
                sys.stdout.flush()
                res_dict = get_score_retrieval_score_for_df(
                    affix=affix,
                    big_df=big_df,
                    qrel_filepath=qrel_filepath,
                    save_folder=save_folder)
                print(res_dict['all'])
                sys.stdout.flush()
                if res_dict['all']['NDCG@5'] > max_ndcg:
                    max_ndcg = res_dict['all']['NDCG@5']
                    best_config = hyper_param_dict

            benchmark_obj.calc_shin_diff_prior(best_config['Theta'])

        l_labmda = 0.1
        for s_mue in optional_mue_list:
            for m_mue in optional_mue_list:
                for l_mue in optional_mue_list:
                    hyper_param_dict = {'S': {'Mue': s_mue, 'Lambda': (1-l_labmda)/2.0},
                                        'M': {'Mue': m_mue, 'Lambda': (1-l_labmda)/2.0},
                                        'L': {'Mue': l_mue, 'Lambda': l_labmda}}

                    print(hyper_param_dict)
                    big_df = benchmark_obj.score_queries(query_list=train_q_list,
                                                         hyper_param_dict=hyper_param_dict)

                    res_dict = get_score_retrieval_score_for_df(
                        affix=affix,
                        big_df=big_df,
                        qrel_filepath=qrel_filepath,
                        save_folder=save_folder)
                    print(res_dict['all'])
                    sys.stdout.flush()
                    if res_dict['all']['NDCG@5'] > max_ndcg:
                        max_ndcg = res_dict['all']['NDCG@5']
                        best_config = hyper_param_dict
        for l_labmda in optional_lambda_list:
            hyper_param_dict =  {'S': {'Mue': best_config['S']['Mue'], 'Lambda': (1-l_labmda)/2.0},
                                 'M': {'Mue': best_config['M']['Mue'], 'Lambda': (1-l_labmda)/2.0},
                                 'L': {'Mue': best_config['L']['Mue'], 'Lambda': l_labmda}}

            big_df = benchmark_obj.score_queries(query_list=train_q_list,
                                                hyper_param_dict=hyper_param_dict)

            res_dict = get_score_retrieval_score_for_df(
                affix=affix,
                big_df=big_df,
                qrel_filepath=qrel_filepath,
                save_folder=save_folder)
            print(res_dict['all'])
            sys.stdout.flush()
            if res_dict['all']['NDCG@5'] > max_ndcg:
                max_ndcg = res_dict['all']['NDCG@5']
                best_config = hyper_param_dict

        print("Best Config: " + str(best_config) + " NDCG@5 : " + str(max_ndcg))
        sys.stdout.flush()
        big_df = benchmark_obj.score_queries(query_list=test_q_list,
                                             hyper_param_dict=best_config)

        all_folds_df = all_folds_df.append(big_df, ignore_index=True)
    filenam_addition = ""
    if train_leave_one_out == True:
        filenam_addition += "_LoO"
    curr_file_name = inner_fold + '_' + retrieval_model + filenam_addition + "_Results.txt"
    with open(os.path.join(save_folder + 'final_res/', curr_file_name), 'w') as f:
        f.write(convert_df_to_trec(all_folds_df))
