import os
import ast
import sys
import time
import datetime
import subprocess
import pandas as pd

from utils import *

PROCESSED_DOCS_MAIN_FOLDER = '/lv_local/home/zivvasilisky/ziv/data/processed_document_vectors'
TREC_EVAL_PATH = "/lv_local/home/zivvasilisky/ziv/env/indri/trec_eval/trec_eval-9.0.7/trec_eval"
QRELS_FILE_PATH = "/lv_local/home/zivvasilisky/ziv/results/qrels/qrels.adhoc"
COLLECTION_MODEL_BY_PAPER = True


class Benchmark:
    def __init__(
            self,
            interval_lookup_method,
            work_year,
            interval_freq,
            save_all_doc_dict = False,
            get_all_doc_dict_from_file = True,
            retrieval_model = 'KL'):
        # create interval list
        self.interval_list = build_interval_list(
            work_year=work_year,
            frequency=interval_freq,
            add_clueweb=True)

        self.retrieval_model = retrieval_model
        self.interval_freq = interval_freq
        self.interval_lookup_method = interval_lookup_method
        # init usefull dfs and dicts
        self.cc_dict                 = create_cc_dict()
        self.query_to_doc_mapping_df = create_query_to_doc_mapping_df()
        self.stemmed_queries_df      = create_stemmed_queries_df()
        self.query_num_to_tf_dict    = {}
        for index, row in self.stemmed_queries_df.iterrows():
            self.query_num_to_tf_dict[int(row['QueryNum'])] = convert_query_to_tf_dict(row['QueryStems'])
        # preprocess all_docs
        processed_docs_path = os.path.join(os.path.join(PROCESSED_DOCS_MAIN_FOLDER, work_year), interval_freq)
        if get_all_doc_dict_from_file == False:
            self.all_docs_dict = self.preprocess_docs(
                    processed_docs_path=processed_docs_path,
                    interval_list = self.interval_list,
                    interval_lookup_method=self.interval_lookup_method)
        main_dirname = os.path.join(os.path.dirname(PROCESSED_DOCS_MAIN_FOLDER), 'benchmark')
        self.save_dirname = os.path.join(main_dirname, work_year + '_' + interval_freq + '_' + interval_lookup_method)
        if not os.path.exists(self.save_dirname):
            os.mkdir(self.save_dirname)

        if get_all_doc_dict_from_file == True:
            with open(os.path.join(self.save_dirname, 'all_docs_dict.json'), 'r') as f:
                self.all_docs_dict = ast.literal_eval(f.read())

        if save_all_doc_dict == True:
            with open(os.path.join(self.save_dirname, 'all_docs_dict.json'), 'w') as f:
                f.write(str(self.all_docs_dict))

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
            interval_list,
            interval_lookup_method):

        doc_count = 0
        all_docs_dict = {}
        for filename in os.listdir(processed_docs_path):
            if filename.endswith('.json'):
                with open(os.path.join(processed_docs_path, filename), 'r') as f:
                    doc_dict = ast.literal_eval(f.read())
                docno = filename.replace('.json', '')
                all_docs_dict[docno] = self.process_doc_for_S_M_L_groups(
                    interval_list=interval_list,
                    interval_lookup_method=interval_lookup_method,
                    doc_dict=doc_dict)
                doc_count += 1
            if doc_count % 100 == 0:
                print("Docs Processed : " + str(doc_count))

        print("Docs Processed : " + str(doc_count))
        return all_docs_dict

    def process_doc_for_S_M_L_groups(
            self,
            interval_list,
            interval_lookup_method,
            doc_dict):

        working_snapshots_list = []
        for i in range(len(interval_list)):
            relevant_snapshot_dict = get_doc_snapshot_by_lookup_method(
                doc_dict=doc_dict,
                interval_list=interval_list,
                interval_lookup_method=interval_lookup_method,
                curr_interval_idx=i)
            if relevant_snapshot_dict is not None:
                working_snapshots_list.append(relevant_snapshot_dict)

        stem_snapshot_count_dict = {}
        stem_total_count_dict = {}
        for snapshot_dict in working_snapshots_list:
            for i in range(len(snapshot_dict['StemList'])):
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
            res_df.loc[next_index] = [query_num, 'Q0', docno, 0, doc_score, 'indri']
            next_index += 1

        res_df.sort_values('Score', ascending=False, inplace=True)
        res_df['Rank'] = list(range(1, next_index + 1))
        return res_df

    def score_queries(
            self,
            query_list,
            output_folder,
            hyper_param_dict):

        big_df = pd.DataFrame({})
        for index, row in self.stemmed_queries_df.iterrows():
            query_num = int(row['QueryNum'])
            if (query_num in query_list) and (query_num not in [100, 95]):
                relevant_df = self.query_to_doc_mapping_df[self.query_to_doc_mapping_df['QueryNum'] == query_num].copy()
                res_query_df = self.get_scored_df_for_query(
                    query_num=query_num,
                    query_doc_df=relevant_df,
                    cc_dict=self.cc_dict,
                    hyper_param_dict=hyper_param_dict)

                big_df = big_df.append(res_query_df, ignore_index=True)

        results_trec_str = convert_df_to_trec(big_df)
        cur_time = str(datetime.datetime.now())
        with open(os.path.join(output_folder, 'curr_run_results_'  + cur_time.replace(' ', '_') + '.txt'), 'w') as f:
            f.write(results_trec_str)

        bashCommand = TREC_EVAL_PATH + ' ' + QRELS_FILE_PATH + ' ' + \
                      os.path.join(output_folder, 'curr_run_results_' + cur_time.replace(' ', '_') + '.txt')

        output = subprocess.check_output(['bash', '-c', bashCommand])
        output_lines = output.split('\n')
        for line in output_lines[:-1]:
            splitted_line = line.split('\t')
            splitted_line = list(filter(None, splitted_line))
            if splitted_line[1] == 'all':
                if splitted_line[0].replace(' ', '') == 'map':
                    map = float(splitted_line[2])
                elif splitted_line[0].replace(' ', '') == 'P_5':
                    p_5 = float(splitted_line[2])
                elif splitted_line[0].replace(' ', '') == 'P_10':
                    p_10 = float(splitted_line[2])

        results_dict = {'TimeStamp' : cur_time, 'Map' : map, 'P_5' : p_5, 'P_10' : p_10}
        # self.run_log += str(hyper_param_dict) + '\n' + str(results_dict) + '\n'

        return results_dict

    def score_queries_cv(
            self,
            query_list,
            output_folder,
            hyper_param_dict,
            k):
        left_out_chunk_size = len(query_list)/int(k)
        k_results_dict = {'Map' : 0.0, 'P_5' : 0.0, 'P_10' : 0.0}
        for i in (range(k)):
            query_list_cp = query_list[:]
            for left_out_query_num in query_list[i*left_out_chunk_size:(i+1)*left_out_chunk_size]:
                query_list_cp.remove(left_out_query_num)
            # print('K:' + str(i) + ' ' + str(query_list_cp))
            curr_result_dict = self.score_queries(
                    query_list=query_list_cp,
                    output_folder=output_folder,
                    hyper_param_dict=hyper_param_dict)

            for measure in k_results_dict:
                k_results_dict[measure] += curr_result_dict[measure]

        for measure in k_results_dict:
            k_results_dict[measure] = k_results_dict[measure]/float(k)

        cur_time = str(datetime.datetime.now())
        self.run_log += str(cur_time) + '\n' + str(hyper_param_dict) + '\n' + str(k_results_dict) + '\n'

        return k_results_dict

if __name__=="__main__":
    work_year = '2008'
    interval_freq = sys.argv[1]
    interval_lookup_method = sys.argv[2]
    save_all_doc_dict = convert_str_to_bool(sys.argv[3])
    get_all_doc_dict_from_file = convert_str_to_bool(sys.argv[4])
    retrieval_model = sys.argv[5]

    k = 5
    print('Interval Feaq: ' + interval_freq)
    print('Lookup method: ' + interval_lookup_method)
    print('Retrivel Model: ' + retrieval_model)

    output_folder = '/lv_local/home/zivvasilisky/ziv/results/benchmark'


    print('Time: ' + str(datetime.datetime.now()))

    benchmark_obj = Benchmark(
            interval_lookup_method=interval_lookup_method,
            work_year=work_year,
            interval_freq=interval_freq,
            save_all_doc_dict=save_all_doc_dict,
            get_all_doc_dict_from_file=get_all_doc_dict_from_file,
            retrieval_model=retrieval_model)
    print("Obj Created!")
    print('Time: ' + str(datetime.datetime.now()))
    query_list = list(range(1,201))
    if len(sys.argv) > 6:
        test_best = convert_str_to_bool(sys.argv[6])
        print ('Testing best config:')
        if test_best == True:
            with open(os.path.join(benchmark_obj.save_dirname , 'grid_search_results'), 'r' ) as f:
                all_grid_res = f.read()
            for line in all_grid_res.split('\n'):
                if 'Best Config:' in line:
                    hyper_param_dict = ast.literal_eval(line.split('Best Config: ')[1].split(' Map :')[0])
            res_dict = benchmark_obj.score_queries(query_list=query_list,
                                              output_folder=output_folder,
                                              hyper_param_dict=hyper_param_dict)
            print(res_dict)
    else:
        hyper_param_dict = {'S': {'Mue': 1500, 'Lambda': 0.45},
                            'M': {'Mue': 1500, 'Lambda': 0.45},
                            'L': {'Mue': 5, 'Lambda': 0.1}}
        optional_mue_list = [5, 10, 100, 500, 800, 1000, 1200, 1500, 1800]
        optional_lambda_list = [0.1]
        max_map = 0.0
        best_config = None
        for s_mue in optional_mue_list:
            for m_mue in optional_mue_list:
                for l_mue in optional_mue_list:
                    for l_labmda in optional_lambda_list:
                        hyper_param_dict = {'S': {'Mue': s_mue, 'Lambda': (1-l_labmda)/2.0},
                                            'M': {'Mue': m_mue, 'Lambda': (1-l_labmda)/2.0},
                                            'L': {'Mue': l_mue, 'Lambda': l_labmda}}

                        print(hyper_param_dict)
                        res_dict = benchmark_obj.score_queries_cv(query_list=query_list,
                                                             output_folder=output_folder,
                                                             hyper_param_dict=hyper_param_dict,
                                                             k=k)
                        print(res_dict)
                        if res_dict['Map'] > max_map:
                            max_map = res_dict['Map']
                            best_config = hyper_param_dict
                        print(str(datetime.datetime.now()))
        print("Best Config: " + str(best_config) + " Map : " + str(max_map))
        benchmark_obj.add_to_log("Best Config: " + str(best_config) + " Map : " + str(max_map))
        benchmark_obj.save_log()

