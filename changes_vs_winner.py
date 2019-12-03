import os
import sys
import ast
import pandas as pd
from scipy import spatial

from utils import *


def plot_stats_vs_winner(
        stats_file_path,
        interval_freq,
        lookup_method):

    work_df = pd.read_csv(os.path.join(stats_file_path,'Summay_vs_winner_' + interval_freq + '_' + lookup_method + '.tsv'),sep = '\t' ,index_col=False)
    all_queries = list(work_df['QueryNum'].drop_duplicates())
    all_intervals = sorted(list(work_df['Interval'].drop_duplicates()))

    summary_df = pd.DataFrame(
        columns=['QueryNum', 'Interval', 'Docno', 'Round', 'Winner', 'Sim_PW', 'QueryTermsRatio', 'StopwordsRatio',
                 'Entropy', 'Trend'])
    next_index = 0
    spearman_corr_df = pd.DataFrame({})
    for query_num in all_queries:
        print(query_num)
        query_df = work_df[work_df['QueryNum'] == query_num].copy()
        interval_winners_df_dict = {}
        for interval_idx in range(len(all_intervals)):
            interval = all_intervals[interval_idx]
            query_interval_df = query_df[query_df['Interval'] == interval]
            query_interval_df['Sim_PD_Rank'] = query_interval_df['Sim_PW'].rank(ascending=False)
            # interval_winners_df_dict[interval] = query_interval_df[query_interval_df['Rank'] == 1]
            if interval != '2008-01-01':
                spearman_corr_df = spearman_corr_df.append(query_interval_df[['Sim_PD_Rank', 'Rank']])
                addition = 0
                while (interval_idx - addition) >= 0:
                    temp_interval = all_intervals[interval_idx - addition]
                    temp_interval_df = query_df[query_df['Interval'] == temp_interval].copy()
                    temp_interval_df = pd.merge(
                        temp_interval_df,
                        query_interval_df[['Rank', 'Docno']].rename(columns={'Rank': 'CurrRank'}),
                        on=['Docno'],
                        how='inner')
                    for index, row in temp_interval_df.iterrows():
                        trend = '='
                        if row['Rank'] > row['CurrRank']:
                            trend = 'W'
                        elif row['Rank'] < row['CurrRank']:
                            trend = 'L'
                        insert_row = [query_num, interval, row['Docno'], (-1) * addition, row['Rank'] == 1, row['Sim_PW'],
                                      row['QueryTermsRatio'], row['StopwordsRatio'], row['Entropy'], trend]

                        summary_df.loc[next_index] = insert_row
                        next_index += 1
                    addition += 1

    print('Spearman corr: ' + str(spearman_corr_df.corr().loc['Sim_PD_Rank']['Rank']))
    summary_df.to_csv('Summay_vs_winner_processed_' + interval_freq + '_' + lookup_method + '_' +
                      str(round(spearman_corr_df.corr().loc['Sim_PD_Rank']['Rank'], 6)) + '.tsv', sep='\t', index=False)

if __name__=="__main__":
    work_year = '2008'
    interval_freq = sys.argv[1]
    interval_lookup_method = sys.argv[2]
    print('Interval Feaq: ' + interval_freq)
    print('Lookup method: ' + interval_lookup_method)

    query_retrn_files_path = '/lv_local/home/zivvasilisky/ziv/results/ranked_docs/'
    processed_docs_folder = '/lv_local/home/zivvasilisky/ziv/data/processed_document_vectors/' + work_year + '/' + interval_freq + '/'
    stats_file_path = '/lv_local/home/zivvasilisky/ziv/clueweb_history/'

    snapshot_stats_df = pd.read_csv(os.path.join(stats_file_path, 'Summay_snapshot_stats_' + interval_freq + '.tsv'), sep = '\t', index_col = False)
    interval_list = build_interval_list(work_year, interval_freq, add_clueweb=True)

    summary_df = pd.DataFrame(columns = ['QueryNum', 'Interval', 'Docno','Rank','SimClueWeb','QueryTermsRatio', 'StopwordsRatio', 'Entropy','Sim_PW'])
    next_index = 0
    for query_num in range(1, 201):
        no_res_list = []
        print('Query: ' + str(query_num))
        sys.stdout.flush()
        ranked_list_dict = {}
        relevant_to_query_df = snapshot_stats_df[snapshot_stats_df['QueryNum'] == query_num].copy()
        for interval in interval_list:
            with open(os.path.join(query_retrn_files_path, str(
                    query_num) + '_' + interval_freq + '_' + interval + '_' + interval_lookup_method + '_Results.txt'),
                      'r') as f:
                trec_str = f.read()
            ranked_list_dict[interval] = convert_trec_to_ranked_list(trec_str)
            if len(ranked_list_dict[interval]) == 0:
                no_res_list.append(interval)

        prev_winner_dict= None
        for interval_idx in range(len(interval_list)):
            interval = interval_list[interval_idx]
            if interval in no_res_list:
                continue

            curr_interval_df = relevant_to_query_df[relevant_to_query_df['Interval'] == interval].copy()
            curr_interval_docnos = list(curr_interval_df['Docno'].drop_duplicates())
            for i in range(len(ranked_list_dict[interval])):
                docno = ranked_list_dict[interval][i]
                rank = i + 1
                insert_row = [query_num, interval, docno, rank]
                if docno in curr_interval_docnos:
                    curr_docno_interval_df = curr_interval_df[curr_interval_df['Docno'] == docno].copy()
                else:
                    if 'Backward' in interval_lookup_method:
                        curr_docno_df = relevant_to_query_df[relevant_to_query_df['Docno'] == docno].copy()
                        covered_docno_intervals = (curr_docno_df['Interval'].drop_duplicates())
                        addition = 1
                        curr_docno_interval_df = pd.DataFrame({})
                        while (interval_idx - addition) >= 0:
                            if interval_list[interval_idx - addition] in covered_docno_intervals:
                                curr_docno_interval_df = curr_docno_df[curr_docno_df['Interval'] == interval_list[interval_idx - addition]]
                                break
                            addition += 1
                        if interval_lookup_method == 'Backward':
                            if curr_docno_interval_df.empty == True:
                                for addition in range(1, len(interval_list) - interval_idx):
                                    if interval_list[interval_idx + addition] in covered_docno_intervals:
                                        curr_docno_interval_df = curr_docno_df[curr_docno_df['Interval'] == interval_list[interval_idx + addition]]
                                        break

                if curr_docno_interval_df.empty == True:
                        continue
                if len(curr_docno_interval_df) > 1:
                    raise Exception('Unexpected Behavior')

                simcluweb = list(curr_docno_interval_df['SimToClueWeb'])[0]
                entropy   = list(curr_docno_interval_df['Entropy'])[0]
                querywords= list(curr_docno_interval_df['QueryWords'])[0]
                stopwords = list(curr_docno_interval_df['#Stopword'])[0]
                textlen   = list(curr_docno_interval_df['TextLen'])[0]
                ref_interval = list(curr_docno_interval_df['Interval'])[0]

                with open(os.path.join(processed_docs_folder, docno + '.json'), 'r') as f:
                    doc_dict = ast.literal_eval(f.read())

                doc_dict = doc_dict[ref_interval]

                queryword_ratio = float(querywords)/ (textlen - querywords)
                stopword_ratio  = float(stopwords) / (textlen - stopwords)

                sim_to_pw = None
                if prev_winner_dict is not None:
                    sim_to_pw = calc_cosine(doc_dict['TfIdf'], prev_winner_dict['TfIdf'])

                if rank == 1:
                    prev_winner_dict = doc_dict

                insert_row.extend([simcluweb, queryword_ratio, stopword_ratio, entropy, sim_to_pw])
                summary_df.loc[next_index] = insert_row
                next_index += 1

    summary_df.to_csv(os.path.join(stats_file_path, 'Summay_vs_winner_' + interval_freq + '_' + interval_lookup_method + '.tsv'), sep = '\t', index=False)
    plot_stats_vs_winner(
        stats_file_path = stats_file_path,
        interval_freq = interval_freq,
        lookup_method = interval_lookup_method)


