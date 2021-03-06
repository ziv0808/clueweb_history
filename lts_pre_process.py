import sys
import pandas as pd
import numpy as np

from utils import *
from sklearn import linear_model
from statsmodels.tsa.arima.model import ARIMA
from avg_doc_model import create_cached_docs_data


def get_cw09_cc_dict(
        dataset_name):
    cc_df = pd.read_csv('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + dataset_name + '/All_Collection_Counts.tsv',
                        sep='\t', index_col=False)

    cc_dict = {}
    for index, row in cc_df.iterrows():
        cc_dict[row['Stem']] = int(row['CollectionCount'])

    return cc_dict


def create_rmse_scores_per_term(
        all_global_params_dict,
        cc_dict):
    stem_time_series_wieghts_dict = {}
    for stem in all_global_params_dict:
        if stem == 'NumWords':
            continue
        print(stem)
        stem_time_series = np.array(all_global_params_dict[stem].sum(axis=0))
        stem_time_series = stem_time_series + cc_dict[stem]
        print(stem_time_series)
        # normalize
        normalize_factor = np.sqrt(np.sum(np.square(stem_time_series)))
        stem_time_series = stem_time_series / normalize_factor
        # diff series
        print(stem_time_series)
        new_stem_ts = []
        for i in range(1, len(stem_time_series)):
            new_stem_ts.append(stem_time_series[i] - stem_time_series[i - 1])
        stem_time_series = np.array(new_stem_ts)
        print(stem_time_series)
        for method in ['MA', 'LR', 'ARMA']:
            curr_score = 0.0
            if method == 'MA':
                for i in range(2, len(stem_time_series)):
                    curr_score += ((0.5 * stem_time_series[i - 2] + 0.5 * stem_time_series[i - 1]) - stem_time_series[
                        i]) ** 2
            elif method == 'LR':
                regr = linear_model.LinearRegression()
                x_series = stem_time_series[:-1]
                y_series = stem_time_series[1:]
                regr.fit(x_series.reshape(-1, 1), y_series.reshape(-1, 1))
                y_pred = regr.predict(x_series.reshape(-1, 1)).reshape(1, -1)
                for i in range(len(y_series)):
                    curr_score += (y_pred[0][i] - y_series[i]) ** 2
            elif method == 'ARMA':
                model = ARIMA(stem_time_series, order=(1, 0, 1))
                model_fit = model.fit()
                curr_score += np.sum(np.square(model_fit.resid[1:]))
            curr_score = np.sqrt(curr_score / float(len(stem_time_series) - 2))
            if stem not in stem_time_series_wieghts_dict:
                stem_time_series_wieghts_dict[stem] = {}
            stem_time_series_wieghts_dict[stem][method] = curr_score
    print(stem_time_series_wieghts_dict)
    return stem_time_series_wieghts_dict


if __name__=='__main__':
    inner_fold = sys.argv[1]

    sw_rmv = True
    filter_params = {}
    asrc_round = int(inner_fold.split('_')[-1])

    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+inner_fold+'/2008/SIM/'
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/preprocessed_dicts/'

    query_to_doc_mapping_df = create_query_to_doc_mapping_df(inner_fold=inner_fold)
    limited_q_list = list(query_to_doc_mapping_df['QueryNum'].drop_duplicates())

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
    stem_time_series_wieghts_dict = create_rmse_scores_per_term(
        all_global_params_dict=all_global_params_dict,
        cc_dict=cw_cc_dict)

    with open(save_folder + inner_fold + '_preprocessed_dict.json', 'w') as f:
        f.write(str(stem_time_series_wieghts_dict))

