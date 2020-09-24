
import sys
import itertools
import datetime
import numpy as np

from utils import *


class WeightedListRanker():
    def __init__(
            self,
            inner_fold,
            rank_or_score,
            qrel_filepath,
            base_save_folder,
            retrieval_model = 'BM25'):

        asrc_round = int(inner_fold.split('_')[-1])
        self.interval_list = build_interval_list_asrc(
            asrc_round=asrc_round,
            add_last=True)

        self.affix = inner_fold + "_" + retrieval_model + '_' + rank_or_score
        self.rank_or_score = rank_or_score
        self.qrel_filepath = qrel_filepath

        base_dirname = base_save_folder
        self.save_dirname = os.path.join(base_dirname,self.affix)
        if not os.path.exists(self.save_dirname):
            os.mkdir(self.save_dirname)
        self.run_log = ""
        # self.test_set_q = list(range(test_set_queries[0], test_set_queries[1] + 1))
        # for i in range(len(self.test_set_q)):
        #     self.test_set_q[i] = '0'*(3-len(str(self.test_set_q[i]))) + str(self.test_set_q[i])

        self.pre_process_data(rank_or_score=rank_or_score,retrieval_model=retrieval_model,inner_fold=inner_fold)

    def save_log(self):
        with open(os.path.join(self.save_dirname, "Log_" + self.save_files_suffix ), 'a') as f:
            f.write(str(self.run_log))

        self.run_log = ""

    def add_to_log(self, strng):
        self.run_log += self.log_affix + strng + '\n'

    def pre_process_data(
            self,
            rank_or_score,
            retrieval_model,
            inner_fold):
        # initiate all scores df
        first = True
        for interval in self.interval_list[::-1]:
            print("Interval : " + interval)
            sys.stdout.flush()
            if interval == 'ClueWeb09':
                filename = os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/',inner_fold.replace('_0'+str(len(self.interval_list)),'').upper()+'_All_features_Round0' +str(len(self.interval_list)) + '_with_meta.tsv')
                curr_df = pd.read_csv(filename, sep = '\t', index_col = False)
            else:
                filename = os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/',
                                        inner_fold.replace('_0'+str(len(self.interval_list)),'').upper()+ '_All_features_Round0' + str(len(self.interval_list)) + '_all_snaps.tsv')
                curr_df = pd.read_csv(filename, sep='\t', index_col=False)
                curr_df = curr_df[curr_df['Interval'] == int(interval)]
            print("filename : " + filename)
            sys.stdout.flush()
            curr_df = curr_df[['QueryNum','Docno',  retrieval_model + 'Score']].rename(
                                                        columns = {'QueryNum'                 : 'Query_ID',
                                                                    retrieval_model + 'Score' : 'Score'})
            if rank_or_score == 'Rank':
                rank_df = pd.DataFrame({})
                for query in list(curr_df['Query_ID'].drop_duplicates()):
                    q_df = curr_df[curr_df['Query_ID'] == query].copy()
                    q_df.sort_values('Score', ascending=False, inplace=True)
                    q_df['Rank'] = list(range(1, len(q_df) + 1))
                    rank_df = rank_df.append(q_df, ignore_index=True)
                curr_df = rank_df

            curr_df = curr_df[['Query_ID', 'Docno', rank_or_score]]
            curr_df[rank_or_score] = curr_df[rank_or_score].apply(lambda x: float(x))
            if True == first:
                self.data_df = curr_df.rename(columns = {rank_or_score : interval})
                first = False
            else:
                self.data_df = pd.merge(
                    self.data_df,
                    curr_df.rename(columns = {rank_or_score : interval}),
                    on=['Query_ID', 'Docno'],
                    how='outer')
        # initiate wieght multiplier df
        self.data_df = self.data_df[['Query_ID', 'Docno'] + self.interval_list]
        self.wieght_multiplier_df = self.data_df.copy()
        self.wieght_multiplier_df[self.interval_list] = self.wieght_multiplier_df[self.interval_list].applymap(lambda x: 0 if np.isnan(float(x)) else 1.0)
        self.data_df[self.interval_list] = self.data_df[self.interval_list].applymap(lambda x: 0.0 if np.isnan(float(x)) else float(x))
        self.data_df.to_csv(os.path.join(self.save_dirname, 'Data_df.tsv'), sep = '\t', index = False)


    def get_score_for_weight_vector(
            self,
            work_data_df,
            wieght_multiplier_df,
            weight_list,
            rank_at_k = None):

        wieght_vector = np.zeros((len(weight_list), 1))
        for i in range(len(weight_list)):
            wieght_vector[i, 0] = weight_list[i]

        wieght_vector = wieght_vector / np.sum(wieght_vector)
        all_wieghts = wieght_multiplier_df.values * wieght_vector.transpose()

        if self.rank_or_score == 'Rank':
            work_data_df[self.interval_list] = work_data_df[self.interval_list].applymap(lambda x: x if np.isnan(x) else 1.0/(x + float(rank_at_k)))
        new_score = np.sum(work_data_df[self.interval_list].values * all_wieghts, axis=1)
        new_score_df = work_data_df[['Query_ID', 'Docno']].copy()

        new_score_df['Score'] = new_score
        new_score_df['Iteration'] = 'Q0'
        new_score_df['Method'] = 'indri'

        res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])

        all_queries = list(new_score_df['Query_ID'].drop_duplicates())
        for query in all_queries:
            query_df = new_score_df[new_score_df['Query_ID'] == query].copy()
            if self.rank_or_score == 'Rank':
                query_df.sort_values('Score', ascending=False, inplace=True)
            elif self.rank_or_score == 'Score':
                query_df.sort_values('Score', ascending=False, inplace=True)
            else:
                raise Exception("Uknown measure...")
            query_df['Query_ID'] = query_df['Query_ID'].apply(lambda x: "0" * (3 - len(str(x))) + str(x))
            query_df['Rank'] = list(range(1, len(query_df) + 1))
            res_df = res_df.append(query_df, ignore_index=True)

        return res_df

    def create_decaying_wieghts(
            self,
            decaying_factor,
            skip_idx_list,
            reverse):
        # create decaying weights for the interval list without skiped indexes
        p = len(self.interval_list) - len(skip_idx_list)
        wieght_list = [0.0]*len(self.interval_list)
        k = 0.0
        if reverse == True:
            k = float(p - 1)
        for i in range(len(wieght_list)):
            if i not in skip_idx_list:
                wieght_list[i] = (1.0 - k*(decaying_factor/float(p)))
                if reverse == True:
                    k -= 1
                else:
                    k += 1

        return  wieght_list

    def create_uniform_wieghts(
            self,
            skip_idx_list):
        # create uniform weights for the interval list without skiped indexes
        denominator = float(len(self.interval_list) - len(skip_idx_list))
        wieght_list = [0.0] * len(self.interval_list)
        for i in range(len(wieght_list)):
            if i not in skip_idx_list:
                wieght_list[i] = 1.0 / denominator
        return wieght_list

    def check_wieght_options_for_fold(
            self,
            test_q_list):

        best_config_dict = {
            'Uniform'   : {'BestNDCG' : 0.0, 'BestK' : None, 'BestWieghts' : None},
            'Decaying'  : {'BestNDCG': 0.0, 'BestK': None, 'BestWieghts': None},
            'RDecaying' : {'BestNDCG': 0.0, 'BestK': None, 'BestWieghts': None},
        }
        all_inteval_indexs = list(range(len(self.interval_list)))
        self.data_df['IsTest'] = self.data_df['Query_ID'].apply(lambda x: True if x in test_q_list else False)
        self.wieght_multiplier_df['IsTest'] = self.wieght_multiplier_df['Query_ID'].apply(lambda x: True if x in test_q_list else False)
        train_df = self.data_df[self.data_df['IsTest'] == False].copy()
        test_df = self.data_df[self.data_df['IsTest'] == True].copy()
        train_wieght_mul_df = self.wieght_multiplier_df[self.wieght_multiplier_df['IsTest'] == False][self.interval_list].copy()
        test_wieght_mul_df = self.wieght_multiplier_df[self.wieght_multiplier_df['IsTest'] == True][self.interval_list].copy()

        if self.rank_or_score == 'Rank':
            k_list = [10,20,30,40,50,60,70,80,90,100]
        else:
            k_list = [None]
        for K in k_list:
            print ("K:" + str(K))
            sys.stdout.flush()
            for L in range(len(all_inteval_indexs)):
                curr_ignore_idx_list = all_inteval_indexs[:L]
                # uniform weights
                list_of_weight_lists = []
                weight_list = self.create_uniform_wieghts(curr_ignore_idx_list)
                list_of_weight_lists.append((weight_list, 'Uniform'))
                for decay_factor in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
                    # decying weights
                    weight_list = self.create_decaying_wieghts(
                        decaying_factor=decay_factor,
                        skip_idx_list=curr_ignore_idx_list,
                        reverse=False)
                    list_of_weight_lists.append((weight_list, 'Decaying'))
                    weight_list = self.create_decaying_wieghts(
                        decaying_factor=decay_factor,
                        skip_idx_list=curr_ignore_idx_list,
                        reverse=True)
                    list_of_weight_lists.append((weight_list, 'RDecaying'))

                for weight_list_config in list_of_weight_lists:
                    weight_list = weight_list_config[0]
                    weight_list_type = weight_list_config[1]
                    res_df = self.get_score_for_weight_vector(
                            work_data_df=train_df,
                            wieght_multiplier_df=train_wieght_mul_df,
                            weight_list=weight_list,
                            rank_at_k=K)
                    res_dict = self.get_score_retrieval_score_for_df(
                        affix=self.affix + str(test_q_list[0]) +'_' + str(test_q_list[-1]),
                        big_df=res_df)

                    if res_dict['all']['NDCG@3'] > best_config_dict[weight_list_type]['BestNDCG']:
                        best_config_dict[weight_list_type]['BestNDCG'] = res_dict['all']['NDCG@3']
                        best_config_dict[weight_list_type]['BestWieghts'] = weight_list
                        best_config_dict[weight_list_type]['BestK'] = K
                        print("Curr Best " + weight_list_type + " config gets NDCG@3: " +str(best_config_dict[weight_list_type]['BestNDCG']))
                        sys.stdout.flush()
        res_df_dict = {}
        for weight_list_type in best_config_dict.keys():
            res_df = self.get_score_for_weight_vector(
                work_data_df=test_df,
                wieght_multiplier_df=test_wieght_mul_df,
                weight_list=best_config_dict[weight_list_type]['BestWieghts'],
                rank_at_k=best_config_dict[weight_list_type]['BestK'])
            res_df_dict[weight_list_type] = res_df

        return res_df_dict, best_config_dict

    def get_score_retrieval_score_for_df(
            self,
            affix,
            big_df):

        curr_file_name = affix + "_Results.txt"
        with open(os.path.join(self.save_dirname, curr_file_name), 'w') as f:
            f.write(convert_df_to_trec(big_df))

        res_dict = get_ranking_effectiveness_for_res_file_per_query(
            file_path=self.save_dirname,
            filename=curr_file_name,
            qrel_filepath=self.qrel_filepath,
            calc_ndcg_mrr=True)

        return res_dict

if __name__=="__main__":
    inner_fold = sys.argv[1]
    retrieval_model = sys.argv[2]
    rank_or_score = sys.argv[3]
    print('Retrivel Model: ' + retrieval_model)
    if 'asrc' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel'
    elif 'bot' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents_fixed.relevance'
    elif 'herd_control' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/control.rel'
    elif 'united' in inner_fold:
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel'

    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/'

    weighted_list_object = WeightedListRanker(
        inner_fold=inner_fold,
        rank_or_score=rank_or_score,
        retrieval_model=retrieval_model,
        qrel_filepath=qrel_filepath,
        base_save_folder=save_folder)

    print("Object Created...")
    sys.stdout.flush()

    query_list, fold_list = get_asrc_q_list_and_fold_list(inner_fold)
    all_folds_df_dict = {}
    all_fold_params_summary = {}
    for weight_list_type in ['Uniform','Decaying','RDecaying']:
        all_folds_df_dict[weight_list_type] = pd.DataFrame({})
        all_fold_params_summary[weight_list_type] = "Fold" + '\t' + "K" + '\t' + "Weights" + '\n'
    for fold in fold_list:
        start_test_q = int(fold[0])
        end_test_q = int(fold[1])
        test_q_list = []
        for q in range(start_test_q, end_test_q + 1):
            if q in query_list:
                test_q_list.append(q)
        fold_df_dict, best_params_dict = weighted_list_object.check_wieght_options_for_fold(test_q_list=test_q_list)

        for weight_list_type in ['Uniform', 'Decaying', 'RDecaying']:
            all_folds_df_dict[weight_list_type] = all_folds_df_dict[weight_list_type].append(fold_df_dict[weight_list_type], ignore_index=True)
            all_fold_params_summary[weight_list_type] += str(start_test_q) + '_' + str(end_test_q) + '\t' + str(best_params_dict[weight_list_type]['BestK']) +\
                                                        '\t' + str(best_params_dict[weight_list_type]['BestWieghts']) + '\n'

    for weight_list_type in ['Uniform','Decaying','RDecaying']:
        curr_file_name = inner_fold + '_' + retrieval_model + '_' + rank_or_score + '_' + weight_list_type + "_Results.txt"
        with open(os.path.join(save_folder + 'final_res/', curr_file_name), 'w') as f:
            f.write(convert_df_to_trec(all_folds_df_dict[weight_list_type]))
        with open(os.path.join(save_folder + 'final_res/', curr_file_name.replace('_Results', '_Params')), 'w') as f:
            f.write(all_fold_params_summary[weight_list_type])




