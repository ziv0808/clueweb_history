import re
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import *


def get_queries_file_to_df(specif_file):
    df = pd.read_csv('/lv_local/home/zivvasilisky/dataset/queries.'+specif_file+'.tsv', sep = '\t', header = None)
    df.columns = ['qid', 'query']
    return df

def load_fine_tuned_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
    model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

    return model, tokenizer


def create_df_dict_from_raw_passage_file():
    df = pd.read_csv('/lv_local/home/zivvasilisky/dataset/collection.tsv', sep='\t', header=None)
    df.columns = ['pid', 'passage']
    df_dict = {'ALL_DOCS_COUNT' : 0.0,
               'AVG_DOC_LEN'    : 0.0}

    ps = nltk.stemmer.PorterStemmer()
    for row in df.itertuples():
        fulltext = re.sub('[^a-zA-Z0-9 ]', ' ', row.passage.strip())
        curr_fulltext_list = fulltext.split(" ")
        p_len = 0
        curr_dict = {}
        for stem in curr_fulltext_list:
            stem = ps.stem(stem)
            if stem == '' or stem == '\n':
                continue
            p_len += 1
            if stem in curr_dict:
                continue
            else:
                curr_dict[stem] = 1

        for stem in curr_dict:
            if stem in df_dict:
                df_dict[stem] += 1
            else:
                df_dict[stem] = 1
        df_dict['ALL_DOCS_COUNT'] += 1
        df_dict['AVG_DOC_LEN'] += p_len
        print(df_dict['ALL_DOCS_COUNT'])
        sys.stdout.flush()

    df_dict['AVG_DOC_LEN'] = df_dict['AVG_DOC_LEN'] / float(df_dict['ALL_DOCS_COUNT'])

    with open('/lv_local/home/zivvasilisky/dataset/df_dict.json', 'w') as f:
        f.write(str(df_dict))



if __name__=="__main__":
    create_df_dict_from_raw_passage_file()


