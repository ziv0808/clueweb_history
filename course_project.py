import re
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import *


def get_queries_file_to_df(specif_file, as_dict=False):
    df = pd.read_csv('/lv_local/home/zivvasilisky/dataset/queries.'+specif_file+'.tsv', sep = '\t', header = None)
    df.columns = ['qid', 'query']
    if as_dict == False:
        return df
    else:
        q_dict = {}
        for index, row in df.iterrows():
            q_dict[row['query']] = row['qid']
        return q_dict

def load_fine_tuned_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
    model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco", output_hidden_states=True)

    return model, tokenizer


def create_df_dict_from_raw_passage_file():
    df = pd.read_csv('/lv_local/home/zivvasilisky/dataset/collection.tsv', sep='\t', header=None)
    df.columns = ['pid', 'passage']
    df_dict = {'ALL_DOCS_COUNT' : 0.0,
               'AVG_DOC_LEN'    : 0.0}

    ps = nltk.stem.porter.PorterStemmer()
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

def create_query_to_row_idx_index_file():
    q_txt_to_qid_dict = get_queries_file_to_df('train',as_dict=True)
    large_index_dict = {}
    curr_idx = 0
    for df in pd.read_csv('/lv_local/home/zivvasilisky/dataset/triples.train.small.tsv', sep = '\t', chunksize = 50000, header = None):
        df.columns = ['query', 'pospar','negpar']
        for query in df['query']:
            q_num = q_txt_to_qid_dict[query]
            if q_num in large_index_dict:
                large_index_dict[q_num].append(curr_idx)
            else:
                large_index_dict[q_num]= [curr_idx]

            curr_idx += 1
        print(curr_idx)
        sys.stdout.flush()
    with open('/lv_local/home/zivvasilisky/dataset/query_to_train_row_idx.json', 'w') as f:
        f.write(str(large_index_dict))

if __name__=="__main__":
    # create_df_dict_from_raw_passage_file()
    create_query_to_row_idx_index_file()


