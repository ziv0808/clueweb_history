import os
import ast
import sys
import math
import pandas as pd

from utils import *

def create_doc_json(
        data_path,
        docno,
        stop_word_list):

    # get file and start parsing
    with open(os.path.join(data_path, docno), 'r') as f:
        file_str = f.read()

    broken_file = file_str.split('\n\n\n\n')
    interval_data_list = []
    for piece in broken_file:
        if piece == '':
            continue
        broken_piece = piece.split('\n\n\n')
        interval_data_list.extend(broken_piece)

    doc_json = {}
    for interval_data in interval_data_list:
        splitted_data = interval_data.split('\n')
        if len(splitted_data) == 1:
            doc_json[splitted_data[0]] = None
        else:
            doc_json[splitted_data[0]] = {}
            doc_json[splitted_data[0]]['StemList']  = splitted_data[1].split(', ')
            doc_json[splitted_data[0]]['StemList'][len(doc_json[splitted_data[0]]['StemList']) - 1] = doc_json[splitted_data[0]]['StemList'][len(doc_json[splitted_data[0]]['StemList']) - 1].replace(']', '')
            doc_json[splitted_data[0]]['IndexList'] = ast.literal_eval(splitted_data[2])
            doc_json[splitted_data[0]]['DfList']    = ast.literal_eval(splitted_data[3])
            doc_json[splitted_data[0]]['CCList'] = ast.literal_eval(splitted_data[4])
            fulltext, tf_list, num_of_words, num_stopwords = build_text_and_tf(
                doc_json[splitted_data[0]]['StemList'],
                doc_json[splitted_data[0]]['IndexList'],
                stop_word_list)
            doc_json[splitted_data[0]]['Fulltext'] = fulltext
            doc_json[splitted_data[0]]['TfList'] = tf_list
            doc_json[splitted_data[0]]['NumWords'] = num_of_words
            doc_json[splitted_data[0]]['NumStopWords'] = num_stopwords
            doc_json[splitted_data[0]]['Entropy'] = calc_shannon_entopy(tf_list[1:])
            doc_json[splitted_data[0]]['TfIdf'] = calc_tfidf_dict(
                doc_json[splitted_data[0]]['StemList'],
                tf_list,
                doc_json[splitted_data[0]]['DfList'])

    return doc_json

def calc_shannon_entopy(tf_list):
    total = float(sum(tf_list))
    return sum(float(freq) / total * math.log(total / freq, 2) for freq in tf_list)

def calc_tfidf_dict(
        stem_list,
        tf_list,
        df_list):

    res_dict = {}
    for i in range(1, len(stem_list)):
        if df_list[i] > 0:
            res_dict[stem_list[i]] = round(tf_list[i]*math.log10(500000000.0/float(df_list[i])), 6)
        else:
            print("Prob Stem: " + stem_list[i])
            res_dict[stem_list[i]] = 0.0

    return res_dict


def build_text_and_tf(
        stem_list,
        index_list,
        stop_word_list):

    num_of_words = 0
    num_stopwords = 0
    fulltext = ""
    tf_list  = [0]*len(stem_list)
    for index in index_list:
        fulltext += stem_list[index] + " "
        tf_list[index] += 1
        num_of_words += 1
    for i in range(len(tf_list)):
        if stem_list[i] in stop_word_list:
            num_stopwords += tf_list[i]

    return fulltext[:-1], tf_list, num_of_words, num_stopwords



if __name__=='__main__':
    work_year = sys.argv[1]
    interval_freq = sys.argv[2]
    query_to_doc_file_addition = sys.argv[3]
    similarity_to_clueweb_threshold = 0.05
    # if work_year == '2011':
    #     query_to_doc_file_addition = "_cw12_1000_per_q"
    # else:
    #     query_to_doc_file_addition = ""

    doc_vector_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/document_vectors/' + work_year + '/'
    stop_word_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/Stemmed_Stop_Words'
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + work_year + '/' + interval_freq +'/'
    query_stem_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/Stemmed_Query_Words'
    query_to_doc_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/all_urls_no_spam_filtered'+query_to_doc_file_addition+'.tsv'

    df_query_to_doc = pd.read_csv(query_to_doc_file, sep = '\t', index_col= False)
    df_query_stems = pd.read_csv(query_stem_file, sep = '\t', index_col= False)
    # create query to stems index
    query_to_stem_mapping = {}
    for index, row in df_query_stems.iterrows():
        query_num = ('0' * (3 - len(str(row['QueryNum'])))) + str(row['QueryNum'])
        query_to_stem_mapping[query_num] = row['QueryStems'].split(' ')
    # create query to doc index
    query_to_docno_mapping = {}
    print (query_to_stem_mapping)
    for index, row in df_query_to_doc.iterrows():
        query_num = ('0'*(3 - len(str(row['QueryNum'])))) + str(row['QueryNum'])
        if query_num in query_to_docno_mapping:
            query_to_docno_mapping[query_num].append(row['Docno'])
        else:
            query_to_docno_mapping[query_num] = [row['Docno']]
    # get stopword list
    with open(stop_word_file, 'r') as f:
        stopword_str = f.read()
    stopword_list = stopword_str.split('\n')
    stopword_list.remove('')

    summary_df = pd.DataFrame(columns = ['Docno', 'Interval', 'PrevValidInterval', 'QueryNum','TextLen', '#Stopword', 'QueryWords',
                                         'Entropy', 'SimToClueWeb', 'SimToPrev', 'SimToClosePrev',
                                         'StemDiffCluWeb', 'CluWebStemDiff'])
    next_index = 0
    interval_ordered_list = build_interval_list(work_year, interval_freq, add_clueweb=True)

    print(stopword_list, len(stopword_list))
    for file_name in os.listdir(doc_vector_folder):
        if file_name.startswith('clueweb') and not file_name.endswith('.json'):
            print (file_name)
            sys.stdout.flush()
            doc_json = create_doc_json(doc_vector_folder, file_name, stopword_list)
            prev_interval = None
            for interval in interval_ordered_list:
                if doc_json[interval] is None:
                    if interval_freq == '1W':
                        continue
                    else:
                        if interval_freq == '2W':
                            if interval[-2:] == '01':
                                optional_ref_list = [interval[:8] + '08']
                            elif interval[-2:] == '16':
                                optional_ref_list = [interval[:8] + '23']

                        elif interval_freq in ['1M', '2M']:
                            optional_ref_list = [interval[:8] + '08', interval[:8] + '16', interval[:8] + '23']
                            if interval_freq == '2M':
                                curr_month = int(interval[5:7])
                                next_month = curr_month + 1
                                optional_ref_list.extend([interval[:5] + (2 - len(str(next_month))) * '0' + str(next_month) + '-01',
                                                          interval[:5] + (2 - len(str(next_month))) * '0' + str(next_month) + '-08',
                                                          interval[:5] + (2 - len(str(next_month))) * '0' + str(next_month) + '-16',
                                                          interval[:5] + (2 - len(str(next_month))) * '0' + str(next_month) + '-23'])

                        for potential_ref in optional_ref_list:
                            if doc_json[potential_ref] is not None:
                                if similarity_to_clueweb_threshold > calc_cosine(doc_json[potential_ref]['TfIdf'], doc_json['ClueWeb09']['TfIdf']):
                                    continue
                                doc_json[interval] = doc_json[potential_ref]
                                break

                        if doc_json[interval] is None:
                            continue

                txt_len             = doc_json[interval]['NumWords']
                num_stop_words      = doc_json[interval]['NumStopWords']
                entropy             = doc_json[interval]['Entropy']
                sim_to_clueweb      = calc_cosine(doc_json[interval]['TfIdf'], doc_json['ClueWeb09']['TfIdf'])
                if sim_to_clueweb < similarity_to_clueweb_threshold:
                    doc_json[interval] = None
                    continue
                sim_to_prev         = None
                sim_to_close_prev   = None
                if prev_interval is not None:
                    sim_to_prev = calc_cosine(doc_json[interval]['TfIdf'], doc_json[prev_interval]['TfIdf'])
                    if (pd.to_datetime(interval.replace('ClueWeb09', str(int(work_year)+1)+'-01-01')) - pd.to_datetime(prev_interval)).days <= 31:
                        sim_to_close_prev = calc_cosine(doc_json[interval]['TfIdf'], doc_json[prev_interval]['TfIdf'])
                curr_document_stem_set = set(doc_json[interval]['StemList'])
                clueweb_document_stem_set = set(doc_json['ClueWeb09']['StemList'])
                document_from_clueweb_stem_diff = list(curr_document_stem_set - clueweb_document_stem_set)
                clueweb_from_document_stem_diff = list(clueweb_document_stem_set - curr_document_stem_set)
                found_related_query = False
                for query_num in query_to_docno_mapping:
                    if file_name in query_to_docno_mapping[query_num]:
                        found_related_query = True
                        query_word_num = 0
                        for j in range(len(doc_json[interval]['StemList'])):
                            if doc_json[interval]['StemList'][j] in query_to_stem_mapping[query_num]:
                                query_word_num += doc_json[interval]['TfList'][j]

                        summary_df.loc[next_index] = [file_name, interval, prev_interval, query_num,
                                                      txt_len, num_stop_words, query_word_num, entropy,
                                                      sim_to_clueweb, sim_to_prev, sim_to_close_prev, str(document_from_clueweb_stem_diff),
                                                      str(clueweb_from_document_stem_diff)]
                        next_index += 1

                if found_related_query == False:
                    raise Exception("found_related_query is false for " + file_name)
                prev_interval = interval
            for interval_ in list(doc_json.keys()):
                if interval_ not in interval_ordered_list:
                    del doc_json[interval_]
            with open(os.path.join(save_folder, file_name + '.json'), 'w') as f:
                f.write(str(doc_json))

    summary_df.to_csv('/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/Summay_snapshot_stats_' + interval_freq + '_' + work_year + '.tsv', sep ='\t', index= False)
    # doc_json = create_doc_json('','clueweb09-en0000-68-11648')
    # with open('test.txt', 'w') as f:
    #     f.write(doc_json)





