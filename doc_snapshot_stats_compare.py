import os
import ast
import math
from scipy import spatial
import pandas as pd

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
    return (1.0 - spatial.distance.cosine(list_1,list_2))



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


def build_interval_list(
        work_year,
        frequency):

    interval_list = []
    for i in range(1, 13):
        if frequency == '2W':
            interval_list.extend(
                [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01',
                 work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-16'])
        else:
            raise Exception('build_interval_dict: Unknoen frequency...')

    return interval_list


if __name__=='__main__':
    work_year = '2009'
    doc_vector_folder = '/lv_local/home/zivvasilisky/ziv/data/document_vectors/2009/'
    stop_word_file = '/lv_local/home/zivvasilisky/ziv/data/Stemmed_Stop_Words'
    save_folder = '/lv_local/home/zivvasilisky/ziv/data/processed_document_vectors/2009/'
    query_stem_file = '/lv_local/home/zivvasilisky/ziv/data/Stemmed_Query_Words'
    query_to_doc_file = '/lv_local/home/zivvasilisky/ziv/data/all_urls_no_spam_filtered.tsv'

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
    interval_ordered_list = build_interval_list(work_year, '2W')
    interval_ordered_list.append('ClueWeb09')
    print(stopword_list, len(stopword_list))
    for file_name in os.listdir(doc_vector_folder):
        if file_name.startswith('clueweb09') and not file_name.endswith('.json'):
            print (file_name)
            doc_json = create_doc_json(doc_vector_folder, file_name, stopword_list)
            prev_interval = None
            for interval in interval_ordered_list:
                if doc_json[interval] is None:
                    continue
                txt_len             = doc_json[interval]['NumWords']
                num_stop_words      = doc_json[interval]['NumStopWords']
                entropy             = doc_json[interval]['Entropy']
                sim_to_clueweb      = calc_cosine(doc_json[interval]['TfIdf'], doc_json['ClueWeb09']['TfIdf'])
                sim_to_prev         = None
                sim_to_close_prev   = None
                if prev_interval is not None:
                    sim_to_prev = calc_cosine(doc_json[interval]['TfIdf'], doc_json[prev_interval]['TfIdf'])
                    if (pd.to_datetime(interval.replace('ClueWeb09', '2009-01-01')) - pd.to_datetime(prev_interval)).days <= 31:
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
            with open(os.path.join(save_folder, file_name + '.json'), 'w') as f:
                f.write(str(doc_json))

    summary_df.to_csv('Summay_snapshot_stats.tsv', sep ='\t', index= False)
    # doc_json = create_doc_json('','clueweb09-en0000-68-11648')
    # with open('test.txt', 'w') as f:
    #     f.write(doc_json)





