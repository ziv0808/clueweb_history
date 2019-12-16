
import sys
import itertools
import datetime
import numpy as np

from utils import *


DEBUG = False
DISTRIBUTE_MISSING_WIEGHTS = False
RESULT_FILES_PATH = "/lv_local/home/zivvasilisky/ziv/results/"

class WeightedListRanker():
    def __init__(
            self,
            interval_lookup_method,
            work_year,
            interval_freq,
            rank_or_score,
            start_month = 1,
            amount_of_snapshot_limit = 1,
            test_set_queries = []):

        self.interval_list = build_interval_list(
            work_year=work_year,
            frequency=interval_freq,
            add_clueweb=True,
            start_month=start_month)

        addition = ""
        if start_month != 1:
            addition = "_" + str(start_month) + "SM_"

        if amount_of_snapshot_limit is not None and amount_of_snapshot_limit > 1:
            addition += "_SnapLimit_" + str(amount_of_snapshot_limit)

        self.result_files_sufix = "_" + interval_freq + "_" + interval_lookup_method + addition +  "_Results.txt"
        self.save_files_suffix = "_" + rank_or_score + self.result_files_sufix

        self.rank_or_score = rank_or_score

        self.save_dirname = "/lv_local/home/zivvasilisky/ziv/data/WeightedListRanker/"
        self.save_dirname = os.path.join(self.save_dirname, rank_or_score + "_" + interval_freq + "_" + interval_lookup_method + addition)
        if not os.path.exists(self.save_dirname):
            os.mkdir(self.save_dirname)
        self.run_log = ""
        self.test_set_q = list(range(test_set_queries[0], test_set_queries[1] + 1))
        for i in range(len(self.test_set_q)):
            self.test_set_q[i] = '0'*(3-len(str(self.test_set_q[i]))) + str(self.test_set_q[i])

        self.pre_process_data(rank_or_score=rank_or_score)


    def save_log(self):
        with open(os.path.join(self.save_dirname, "Log_" + self.save_files_suffix ), 'a') as f:
            f.write(str(self.run_log))

        self.run_log = ""

    def add_to_log(self, strng):
        self.run_log += strng + '\n'

    def pre_process_data(
            self,
            rank_or_score):
        # initiate all scores df
        first = True
        for interval in self.interval_list[::-1]:
            curr_df = convert_trec_results_file_to_pandas_df(os.path.join(RESULT_FILES_PATH, interval + self.result_files_sufix))
            curr_df = curr_df[['Query_ID', 'Docno', rank_or_score]]
            curr_df[rank_or_score] = curr_df[rank_or_score].apply(lambda x: float(x))
            if rank_or_score == 'Rank':
                # min max normalize values

                min_df = curr_df[['Query_ID',rank_or_score]].groupby(['Query_ID']).min()
                max_df = curr_df[['Query_ID', rank_or_score]].groupby(['Query_ID']).max()
                curr_df[rank_or_score] = curr_df.apply(lambda row:
                    (row[rank_or_score] - min_df.loc[row['Query_ID']][0])/(max_df.loc[row['Query_ID']][0] - min_df.loc[row['Query_ID']][0])
                                            if not np.isnan(row[rank_or_score]) else row[rank_or_score] , axis = 1)
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
        self.data_df['IsTest'] = self.data_df['Query_ID'].apply(lambda x: 1 if x in self.test_set_q else 0)
        self.wieght_multiplier_df = self.data_df[['IsTest'] + self.interval_list].copy()
        self.wieght_multiplier_df = self.wieght_multiplier_df.applymap(lambda x: 0 if np.isnan(float(x)) else 1.0)
        self.data_df[self.interval_list] = self.data_df[self.interval_list].applymap(lambda x: 0.0 if np.isnan(float(x)) else float(x))
        self.data_df.to_csv(os.path.join(self.save_dirname, 'Data_df.tsv'), sep = '\t', index = False)

        self.data_test = self.data_df[self.data_df['IsTest'] == 1].copy()
        self.wieght_multiplier_test = self.wieght_multiplier_df[self.wieght_multiplier_df['IsTest'] == 1].copy()
        self.data_df = self.data_df[self.data_df['IsTest'] == 0]
        self.wieght_multiplier_df = self.wieght_multiplier_df[self.wieght_multiplier_df['IsTest'] == 0]

        print(self.wieght_multiplier_df)
        print(self.wieght_multiplier_test)

        del self.data_test['IsTest']
        del self.wieght_multiplier_test['IsTest']

        del self.data_df['IsTest']
        del self.wieght_multiplier_df['IsTest']


    def get_score_for_weight_vector(
            self,
            weight_list,
            train = True,
            rank_at_k = None):

        wieght_vector = np.zeros((len(weight_list), 1))
        for i in range(len(weight_list)):
            wieght_vector[i, 0] = weight_list[i]

        if train == True:
            if DISTRIBUTE_MISSING_WIEGHTS == True:
                all_wieghts = self.wieght_multiplier_df.values * wieght_vector.transpose()
                normalize_factor = np.sum(all_wieghts, axis = 1)
                normalize_factor = normalize_factor.reshape((len(all_wieghts), 1))
                all_wieghts = all_wieghts/normalize_factor
            else:
                wieght_vector = wieght_vector/np.sum(wieght_vector)
                all_wieghts = self.wieght_multiplier_df.values * wieght_vector.transpose()
        else:
            if DISTRIBUTE_MISSING_WIEGHTS == True:
                all_wieghts = self.wieght_multiplier_test.values * wieght_vector.transpose()
                normalize_factor = np.sum(all_wieghts, axis=1)
                normalize_factor = normalize_factor.reshape((len(all_wieghts), 1))
                all_wieghts = all_wieghts / normalize_factor
            else:
                wieght_vector = wieght_vector / np.sum(wieght_vector)
                all_wieghts = self.wieght_multiplier_test.values * wieght_vector.transpose()

        if DEBUG == True:
            print("Normalize Factors : ")
            print(normalize_factor)
            print("Normalized wieghts : ")
            print(all_wieghts)
        if train == True:
            work_data = self.data_df[self.interval_list].copy()
        else:
            work_data = self.data_test[self.interval_list].copy()
        if self.rank_or_score == 'Rank':
            work_data = work_data.applymap(lambda x: x if np.isnan(x) else 1.0/(x + float(rank_at_k)))
        new_score = np.sum(work_data.values * all_wieghts, axis=1)
        new_score_df = work_data[['Query_ID', 'Docno']].copy()

        new_score_df['Score'] = new_score
        new_score_df['Iteration'] = 'Q0'
        new_score_df['Method'] = 'indri'

        res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])

        all_queries = list(new_score_df['Query_ID'].drop_duplicates())
        for query in all_queries:
            query_df = new_score_df[new_score_df['Query_ID'] == query].copy()
            if self.rank_or_score == 'Rank':
                query_df['Score'] = query_df['Score'].apply(lambda x: x*(-1))
                query_df.sort_values('Score', ascending=False, inplace=True)
            elif self.rank_or_score == 'Score':
                query_df.sort_values('Score', ascending=False, inplace=True)
            else:
                raise Exception("Uknown measure...")

            query_df['Rank'] = list(range(1, len(query_df) + 1))
            res_df = res_df.append(query_df, ignore_index=True)

        results_trec_str = convert_df_to_trec(res_df[['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method']])
        cur_time = str(datetime.datetime.now())
        curr_file_name = cur_time.replace(' ', '_') + self.save_files_suffix
        with open(os.path.join(os.path.join(RESULT_FILES_PATH, 'wieghted_lists_res'),curr_file_name ), 'w') as f:
            f.write(results_trec_str)
        if DEBUG == True:
            print(curr_file_name)
        results = get_ranking_effectiveness_for_res_file(
            file_path=os.path.join(RESULT_FILES_PATH, 'wieghted_lists_res'),
            filename=curr_file_name)

        return results

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
            wieght_list[i] = 1.0 / denominator
        return wieght_list

    def check_wieght_options(
            self):

        best_map = 0.0
        best_config = None
        all_inteval_indexs = list(range(len(self.interval_list)))
        self.add_to_log("Method\tWieghtList\tScore\tK")
        if self.rank_or_score == 'Rank':
            k_list = [10,20,30,40,50,60,70,80,90,100]
        else:
            k_list = [None]
        for K in k_list:
            for L in range(len(all_inteval_indexs)):
                curr_ignore_idx_list = all_inteval_indexs[:L]
                rest_of_idx_list = all_inteval_indexs[L:]
                ignore_sub_set_len = 0
                if len(rest_of_idx_list) > 2:
                    ignore_sub_set_len = 1
                if len(rest_of_idx_list) > 3:
                    ignore_sub_set_len = 2
                for i in range(ignore_sub_set_len + 1):
                    for subset in itertools.combinations(rest_of_idx_list, i):
                        ignore_idx_list = curr_ignore_idx_list + list(subset)
                        if (len(all_inteval_indexs) - 1) in ignore_idx_list:
                            continue
                        # uniform weights
                        weight_list = self.create_uniform_wieghts(ignore_idx_list)
                        res_dict = self.get_score_for_weight_vector(weight_list=weight_list, rank_at_k=K)
                        self.add_to_log("Uniform\t" + str(weight_list) + "\t" + str(res_dict) +"\t" +str(K))
                        if res_dict['Map'] > best_map:
                            best_map = res_dict['Map']
                            best_config = weight_list + [K]
                            print("Curr Best config gets Map: " +str(best_map))
                            sys.stdout.flush()
                        for decay_factor in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
                            # decying weights
                            weight_list = self.create_decaying_wieghts(
                                decaying_factor=decay_factor,
                                skip_idx_list=ignore_idx_list,
                                reverse=False)
                            res_dict = self.get_score_for_weight_vector(weight_list=weight_list, rank_at_k=K)
                            self.add_to_log("Decay_" + str(decay_factor) + "\t" + str(weight_list) + "\t" + str(res_dict)+"\t" +str(K))
                            if res_dict['Map'] > best_map:
                                best_map = res_dict['Map']
                                best_config = weight_list + [K]
                                print("Curr Best config gets Map: " + str(best_map))
                                sys.stdout.flush()
                            # reverse decaying weights
                            weight_list = self.create_decaying_wieghts(
                                decaying_factor=decay_factor,
                                skip_idx_list=ignore_idx_list,
                                reverse=True)
                            res_dict = self.get_score_for_weight_vector(weight_list=weight_list, rank_at_k=K)
                            self.add_to_log("Decay_" + str(decay_factor) + "\t" + str(weight_list) + "\t" + str(res_dict)+"\t" +str(K))
                            if res_dict['Map'] > best_map:
                                best_map = res_dict['Map']
                                best_config = weight_list + [K]
                                print("Curr Best config gets Map: " + str(best_map))
                                sys.stdout.flush()
                self.save_log()
        if self.rank_or_score == 'Score':
            self.add_to_log("Best\t" + str(best_config) + "\t" + str(self.get_score_for_weight_vector(weight_list=best_config)))
            self.add_to_log("Test\t" + str(best_config) + "\t" + str(self.get_score_for_weight_vector(weight_list=best_config, train=False)))
        else:
            self.add_to_log(
                "Best\t" + str(best_config) + "\t" + str(self.get_score_for_weight_vector(weight_list=best_config[:-1], rank_at_k=best_config[-1])))
            self.add_to_log("Test\t" + str(best_config) + "\t" + str(
                self.get_score_for_weight_vector(weight_list=best_config[:-1], train=False, rank_at_k=best_config[-1])))
        self.save_log()


if __name__=="__main__":
    work_year = '2008'
    interval_freq = sys.argv[1]
    interval_lookup_method = sys.argv[2]
    rank_or_score = sys.argv[3]
    start_month = int(sys.argv[4])
    amount_of_snapshot_limit = ast.literal_eval(sys.argv[5])
    start_test_q = int(sys.argv[6])
    end_test_q = int(sys.argv[7])

    weighted_list_object = WeightedListRanker(
        work_year=work_year,
        interval_lookup_method=interval_lookup_method,
        interval_freq=interval_freq,
        rank_or_score=rank_or_score,
        start_month=start_month,
        amount_of_snapshot_limit=amount_of_snapshot_limit,
        test_set_queries=[start_test_q, end_test_q])
    print("Object Created...")
    sys.stdout.flush()
    if DEBUG == True:
        wieght_list = ast.literal_eval(sys.argv[8])
        print (str(wieght_list))
        print(weighted_list_object.get_score_for_weight_vector(weight_list=wieght_list))
    else:
        weighted_list_object.check_wieght_options()






