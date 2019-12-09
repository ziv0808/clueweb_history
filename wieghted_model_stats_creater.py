import ast
import sys
from utils import *

WEIGHTED_MODELS_FILES_PATH = "/lv_local/home/zivvasilisky/ziv/data/WeightedListRanker/"


def create_data_df(
        interval_list,
        log_file_path):

    work_df = pd.DataFrame(columns=interval_list + ['Map', 'P_5', 'P_10'])
    next_index = 0
    with open(log_file_path, 'r') as f:
        file_lines = f.readlines()

    for line in file_lines:
        line = line.strip()
        if line == "" or line.startswith("Best"):
            continue

        broken_line = line.split('\t')
        if broken_line[0].startswith('Decay_'):
            weight_list = ast.literal_eval(broken_line[1])
            score_dict = ast.literal_eval(broken_line[2])
            insert_row = weight_list + [score_dict['Map'], score_dict['P_5'],score_dict['P_10']]
            work_df.loc[next_index] = insert_row
            next_index += 1

    return work_df



if __name__=="__main__":
    work_year = '2008'
    interval_freq = sys.argv[1]
    start_month = int(sys.argv[2])
    addition = ""
    if start_month != 1:
        addition = "_" + str(start_month) + "SM_"
    interval_list = build_interval_list(
        work_year=work_year,
        frequency=interval_freq,
        add_clueweb=True,
        start_month=start_month)

    rank_or_score_list = ['Rank', 'Score']
    lookup_method_list = ['Backward', 'NoLookup']

    res_df = pd.DataFrame(columns = ['Measure', 'Lookup'] + interval_list)
    next_idx = 0
    for rank_or_score in rank_or_score_list:
        for lookup_method in lookup_method_list:
            folder_name = rank_or_score + "_" + interval_freq + "_" + lookup_method + addition
            log_filename = "Log__" + folder_name + "_Results.txt"
            curr_corr_df = create_data_df(
                interval_list=interval_list,
                log_file_path=os.path.join(WEIGHTED_MODELS_FILES_PATH, os.path.join(folder_name, log_filename)))

            next_row = [rank_or_score, lookup_method]
            for interval in interval_list:
                corr_  = curr_corr_df[[interval], 'Map'].corr()
                next_row.append(corr_.loc[interval]['Map'])

            res_df.loc[next_idx] = next_row
            next_idx += 1

    res_df.to_csv(os.path.join(WEIGHTED_MODELS_FILES_PATH, interval_freq + "_" + addition + ".tsv") , sep = '\t', index=False)


