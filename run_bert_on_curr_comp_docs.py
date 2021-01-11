import os
import sys
import ast
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import convert_df_to_trec
from bs4 import BeautifulSoup


def get_query_doc_rel_proba(
        tokenizer,
        model,
        query,
        document,
        fulltext_num_of_last_words_remove = 0):

    try:
        if fulltext_num_of_last_words_remove == 0:
            inputs = tokenizer.encode_plus(query, document, return_tensors="pt")
        else:
            print("Trying with " + str(fulltext_num_of_last_words_remove) + ' Words remove')
            inputs = tokenizer.encode_plus(query, document.rsplit(' ', fulltext_num_of_last_words_remove)[0], return_tensors="pt")
        res = torch.softmax(model(**inputs)[0], dim=1).tolist()[0][1]
    except Exception as e:
        res = get_query_doc_rel_proba(tokenizer = tokenizer,
                                      model = model,
                                      query = query,
                                      document = document,
                                      fulltext_num_of_last_words_remove = fulltext_num_of_last_words_remove + 1)

    return res

def read_current_doc_file(doc_filepath):
    stats = {}
    with open(doc_filepath, 'r') as f:
        soup = BeautifulSoup(f.read())
    all_docs = soup.find_all('doc')

    for doc_ in list(all_docs):
        docno = doc_.find('docno').text
        fulltext = doc_.find('text').text
        query = docno.split('-')[0]
        user =docno.split('-')[1]
        if query not in stats:
            stats[query] = {}

        stats[query][user] = {}
        stats[query][user]['FullText'] = fulltext

    return stats

if __name__=="__main__":
    trectext_filename = sys.argv[1]

    save_path = '/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/FeatureIdx/'
    query_num_to_text = {}
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/query/query_num_to_text.txt', 'r') as f:
        file_content = f.read()

    file_content = file_content.split('\n')
    for line in file_content:
        if ':' in line:
            query_num_to_text[line.split(':')[0]] = line.split(':')[1].strip()

    tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
    model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")


    parsed_file_dict = read_current_doc_file(trectext_filename)

    for query in parsed_file_dict:
        rel_query = query.split('_')[0]
        for user in parsed_file_dict[query]:
            parsed_file_dict[query][user]['BERTScore'] =  get_query_doc_rel_proba(
                                                            tokenizer=tokenizer,
                                                            model=model,
                                                            query=query_num_to_text[rel_query],
                                                            document=parsed_file_dict[query][user]['FullText'])
    with open(save_path + trectext_filename.split('/')[-1] + '_BERT.json', 'w') as f:
        f.write(str(parsed_file_dict))


