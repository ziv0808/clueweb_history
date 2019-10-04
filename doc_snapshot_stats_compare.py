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
    total = sum(tf_list)
    return sum(freq / total * math.log(total / freq, 2) for freq in tf_list)

def calc_tfidf_dict(
        stem_list,
        tf_list,
        df_list):

    res_dict = {}
    for i in range(1, len(stem_list)):
        if df_list[i] > 0:
            res_dict[stem_list[i]] = round(tf_list[i]*math.log10(500000000.0/float(df_list[i])), 6)
        else:

            res_dict[stem_list[i]] = round(tf_list[i] * math.log10(500000000.0 / float(50000000)), 6)

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



if __name__=='__main__':
    doc_vector_folder = '/lv_local/home/zivvasilisky/ziv/data/document_vectors/'
    stop_word_file = '/lv_local/home/zivvasilisky/ziv/data/Stemmed_Stop_Words'
    save_folder = '/lv_local/home/zivvasilisky/ziv/data/processed_document_vectors'
    with open(stop_word_file, 'r') as f:
        stopword_str = f.read()
    stopword_list = stopword_str.split('\n')
    stopword_list.remove('')
    print(stopword_list, len(stopword_list))
    for file_name in os.listdir(doc_vector_folder):
        if file_name.startswith('clueweb09') and not file_name.endswith('.json'):
            print (file_name)
            doc_json = create_doc_json(doc_vector_folder, file_name, stopword_list)
            with open(os.path.join(save_folder, file_name + '.json'), 'w') as f:
                f.write(str(doc_json))
    # create_doc_json('','clueweb09-enwp03-26-02724')





