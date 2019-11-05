import os
import sys
import ast
import pandas as pd
from scipy import spatial


def build_interval_list(
        work_year,
        frequency):

    interval_list = []
    for i in range(1, 13):
        if frequency == '2W':
            interval_list.extend(
                [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01',
                 work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-16'])
        elif frequency == '1M':
            interval_list.extend(
                [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01'])
        elif frequency == '2M':
            if i % 2 == 1:
                interval_list.extend(
                    [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01'])
        else:
            raise Exception('build_interval_dict: Unknoen frequency...')

    return interval_list


def calc_cosine(
        dict_1,
        dict_2):
    list_1 = []
    list_2 = []
    for stem in dict_1:
        if stem in dict_2:
            list_1.append(dict_1[stem])
            list_2.append(dict_2[stem])
        else:
            list_1.append(dict_1[stem])
            list_2.append(0.0)
    for stem in dict_2:
        if stem not in dict_1:
            list_1.append(0.0)
            list_2.append(dict_2[stem])
    return (1.0 - spatial.distance.cosine(list_1, list_2))

def convert_trec_to_ranked_list(trec_str):
    docno_ordered_list = []
    splitted_trec = trec_str.split('\n')
    for line in splitted_trec:
        if line != '':
            docno = line.split(' ')[2]
            docno_ordered_list.append(docno)
    return docno_ordered_list

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
    interval_list = build_interval_list(work_year, interval_freq)
    interval_list.append('ClueWeb09')

    summary_df = pd.DataFrame(columns = ['QueryNum', 'Interval', 'Docno','Rank','SimClueWeb','QueryTermsRatio', 'StopwordsRatio', 'Entropy','Sim_PW'])
    next_index = 0
    for query_num in range(1, 201):
        no_res_list = []
        print('Query: ' + str(query_num))
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
                    if interval_lookup_method == 'Backward':
                        curr_docno_df = relevant_to_query_df[relevant_to_query_df['Docno'] == docno].copy()
                        covered_docno_intervals = (curr_docno_df['Interval'].drop_duplicates())
                        addition = 1
                        curr_docno_interval_df = pd.DataFrame({})
                        while (interval_idx - addition) >= 0:
                            if interval_list[interval_idx - addition] in covered_docno_intervals:
                                curr_docno_interval_df = curr_docno_df[curr_docno_df['Interval'] == interval_list[interval_idx - addition]]
                                break
                            addition += 1
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

    summary_df.to_csv('Summay_vs_winner_' + interval_freq + '_' + interval_lookup_method + '.tsv')



