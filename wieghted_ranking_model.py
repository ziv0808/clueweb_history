
import sys
import itertools
import datetime
import numpy as np

from utils import *


DEBUG = True

RESULT_FILES_PATH = "/lv_local/home/zivvasilisky/ziv/results/"

class WeightedListRanker():
    def __init__(
            self,
            interval_lookup_method,
            work_year,
            interval_freq,
            rank_or_score,
            start_month = 1):

        self.interval_list = build_interval_list(
            work_year=work_year,
            frequency=interval_freq,
            add_clueweb=True,
            start_month=start_month)

        addition = ""
        if start_month != 1:
            addition = "_" + str(start_month) + "SM_"
        self.result_files_sufix = "_" + interval_freq + "_" + interval_lookup_method + addition +  "_Results.txt"
        self.save_files_suffix = "_" + rank_or_score + self.result_files_sufix

        self.rank_or_score = rank_or_score

        self.save_dirname = "/lv_local/home/zivvasilisky/ziv/data/WeightedListRanker/"
        self.save_dirname = os.path.join(self.save_dirname, rank_or_score + "_" + interval_freq + "_" + interval_lookup_method + addition)
        if not os.path.exists(self.save_dirname):
            os.mkdir(self.save_dirname)
        self.run_log = ""

        self.pre_process_data(rank_or_score=rank_or_score)


    def save_log(self):
        with open(os.path.join(self.save_dirname, "Log_" + self.save_files_suffix ), 'w') as f:
            f.write(str(self.run_log))

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
        self.wieght_multiplier_df = self.data_df[self.interval_list].copy()
        self.wieght_multiplier_df = self.wieght_multiplier_df.applymap(lambda x: 0 if np.isnan(float(x)) else 1.0)
        self.data_df[self.interval_list] = self.data_df[self.interval_list].applymap(lambda x: 0.0 if np.isnan(float(x)) else float(x))

        self.data_df.to_csv(os.path.join(self.save_dirname, 'Data_df.tsv'), sep = '\t', index = False)

    def get_score_for_weight_vector(
            self,
            weight_list):

        wieght_vector = np.zeros((len(weight_list), 1))
        for i in range(len(weight_list)):
            wieght_vector[i, 0] = weight_list[i]

        all_wieghts = self.wieght_multiplier_df.values * wieght_vector.transpose()
        normalize_factor = np.sum(all_wieghts, axis = 1)
        normalize_factor = normalize_factor.reshape((len(all_wieghts), 1))
        all_wieghts = all_wieghts/normalize_factor
        if DEBUG == True:
            print("Normalize Factors : ")
            print(normalize_factor)
            print("Normalized wieghts : ")
            print(all_wieghts)

        new_score = np.sum( self.data_df[self.interval_list].values * all_wieghts, axis = 1)
        new_score_df = self.data_df[['Query_ID', 'Docno']].copy()
        new_score_df['Score'] = new_score
        new_score_df['Iteration'] = 'Q0'
        new_score_df['Method'] = 'indri'

        res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])

        all_queries = list(new_score_df['Query_ID'].drop_duplicates())
        for query in all_queries:
            query_df = new_score_df[new_score_df['Query_ID'] == query].copy()
            if self.rank_or_score == 'Rank':
                query_df.sort_values('Score', ascending=True, inplace=True)
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
        self.add_to_log("Method\tWieghtList\tScore")
        for L in range(len(all_inteval_indexs)):
            for subset in itertools.combinations(all_inteval_indexs, L):
                ignore_idx_list = list(subset)
                # uniform weights
                weight_list = self.create_uniform_wieghts(ignore_idx_list)
                res_dict = self.get_score_for_weight_vector(weight_list)
                self.add_to_log("Uniform\t" + str(weight_list) + "\t" + str(res_dict))
                if res_dict['Map'] > best_map:
                    best_map = res_dict['Map']
                    best_config = weight_list
                    print("Curr Best config gets Map: " +str(best_map))
                    sys.stdout.flush()
                for decay_factor in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                    # decying weights
                    weight_list = self.create_decaying_wieghts(
                        decaying_factor=decay_factor,
                        skip_idx_list=ignore_idx_list,
                        reverse=False)
                    res_dict = self.get_score_for_weight_vector(weight_list)
                    self.add_to_log("Decay_" + str(decay_factor) + "\t" + str(weight_list) + "\t" + str(res_dict))
                    if res_dict['Map'] > best_map:
                        best_map = res_dict['Map']
                        best_config = weight_list
                        print("Curr Best config gets Map: " + str(best_map))
                        sys.stdout.flush()
                    # reverse decaying weights
                    weight_list = self.create_decaying_wieghts(
                        decaying_factor=decay_factor,
                        skip_idx_list=ignore_idx_list,
                        reverse=True)
                    res_dict = self.get_score_for_weight_vector(weight_list)
                    self.add_to_log("Decay_" + str(decay_factor) + "\t" + str(weight_list) + "\t" + str(res_dict))
                    if res_dict['Map'] > best_map:
                        best_map = res_dict['Map']
                        best_config = weight_list
                        print("Curr Best config gets Map: " + str(best_map))
                        sys.stdout.flush()
                self.save_log()

        self.add_to_log("Best\t" + str(best_config) + "\t" + str(best_map))
        self.save_log()


if __name__=="__main__":
    work_year = '2008'
    interval_freq = sys.argv[1]
    interval_lookup_method = sys.argv[2]
    rank_or_score = sys.argv[3]
    start_month = int(sys.argv[4])

    weighted_list_object = WeightedListRanker(
        work_year=work_year,
        interval_lookup_method=interval_lookup_method,
        interval_freq=interval_freq,
        rank_or_score=rank_or_score,
        start_month=start_month)
    print("Object Created...")
    sys.stdout.flush()
    if DEBUG == True:
        wieght_list = ast.literal_eval(sys.argv[5])
        print (str(wieght_list))
        print(weighted_list_object.get_score_for_weight_vector(weight_list=wieght_list))
    else:
        weighted_list_object.check_wieght_options()






