import re
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import *


def get_queries_file_to_df(specif_file, as_dict=None, frac =None):
    if frac is None:
        df = pd.read_csv('/lv_local/home/zivvasilisky/dataset/queries.'+specif_file+'.tsv', sep = '\t', header = None)
    else:
        df = pd.read_csv('/lv_local/home/zivvasilisky/dataset/processed_queries/queries.frac.tsva'+ frac, sep='\t', header=None)
    df.columns = ['qid', 'query']
    if as_dict is None:
        return df
    else:
        q_dict = {}
        for index, row in df.iterrows():
            if as_dict == 'Reverse':
                q_dict[row['qid']] = row['query']
            elif as_dict == 'Regular':
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
        fulltext = re.sub('[^a-zA-Z0-9 ]', ' ', row.passage.strip().lower())
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
    q_txt_to_qid_dict = get_queries_file_to_df('train',as_dict='Regular')
    large_index_dict = {}
    curr_idx = 0
    for df in pd.read_csv('/lv_local/home/zivvasilisky/dataset/triples.train.small.tsv', sep = '\t', chunksize = 50000, header = None):
        df.columns = ['query', 'pospar','negpar']
        for query in df['query']:
            q_num = q_txt_to_qid_dict[query.encode('latin1').decode('utf8')]
            if q_num in large_index_dict:
                large_index_dict[q_num].append(curr_idx)
            else:
                large_index_dict[q_num]= [curr_idx]

            curr_idx += 1
        print(curr_idx)
        sys.stdout.flush()
    with open('/lv_local/home/zivvasilisky/dataset/query_to_train_row_idx.json', 'w') as f:
        f.write(str(large_index_dict))

def get_query_to_train_row_index(frac_idx_to_filter_by=None):
    with open('/lv_local/home/zivvasilisky/dataset/query_to_train_row_idx.json', 'r') as f:
        big_idx = ast.literal_eval(f.read())
    del_q_list = []
    if frac_idx_to_filter_by is not None:
        for q in big_idx:
            if q not in frac_idx_to_filter_by:
                del_q_list.append(q)

    for q in del_q_list:
        del big_idx[q]
    return big_idx

def create_tf_dict_bm25_ready(curr_txt, stemmer):
    fulltext = re.sub('[^a-zA-Z0-9 ]', ' ', curr_txt.strip().lower())
    curr_fulltext_list = fulltext.split(" ")
    p_len = 0
    curr_dict = {}
    for stem in curr_fulltext_list:
        stem = stemmer.stem(stem)
        if stem == '' or stem == '\n':
            continue
        p_len += 1
        if stem in curr_dict:
            curr_dict[stem] += 1
        else:
            curr_dict[stem] = 1
        curr_dict['NumWords'] = p_len

    return curr_dict


def create_bm25_and_bert_scores_and_cls_for_train_frac(frac):
    qid_to_q_txt_dict = get_queries_file_to_df('train', as_dict='Reverse', frac=frac)
    print("got qid_to_q_txt_dict")
    sys.stdout.flush()
    q_to_train_row_index = get_query_to_train_row_index(qid_to_q_txt_dict)
    print("got q_to_train_row_index")
    sys.stdout.flush()
    ps = nltk.stem.porter.PorterStemmer()
    for qid in qid_to_q_txt_dict:
        qid_to_q_txt_dict[qid] = create_tf_dict_bm25_ready(qid_to_q_txt_dict[qid], ps)
        del qid_to_q_txt_dict[qid]['NumWords']

    print("fixed qid_to_q_txt_dict")
    sys.stdout.flush()
    q_txt_to_qid_dict = get_queries_file_to_df('train', as_dict='Regular')
    print("got q_txt_to_qid_dict")
    sys.stdout.flush()
    q_res_dict = {}

    with open('/lv_local/home/zivvasilisky/dataset/df_dict.json', 'r') as f:
        df_dict = ast.literal_eval(f.read())
    model, tokenizer = load_fine_tuned_bert_model()
    print("got df_dict, model and tokenizer")
    sys.stdout.flush()
    curr_idx = 0
    for df in pd.read_csv('/lv_local/home/zivvasilisky/dataset/triples.train.small.tsv', sep='\t', chunksize=5000,header=None):
        df.columns = ['query', 'pospar', 'negpar']
        for row in df.itertuples():
            q_num = q_txt_to_qid_dict[row.query.encode('latin1').decode('utf8')]
            if q_num not in q_to_train_row_index:
                curr_idx += 1
                continue
            if q_num not in q_res_dict:
                q_res_dict[q_num] = {}

            bm25_score = bm25_score_doc_for_query(
                query_stem_dict=qid_to_q_txt_dict[q_num],
                df_dict=df_dict,
                doc_dict=create_tf_dict_bm25_ready(row.pospar, ps),
                k1=0.6,
                b=0.62)

            inputs = tokenizer.encode_plus(row.query, row.pospar, return_tensors="pt")
            outputs = model(**inputs)
            proba = torch.softmax(outputs[0], dim=1).tolist()[0][1]
            cls = outputs.hidden_states[-1][0][0]
            q_res_dict[q_num][str(curr_idx) + '_Pos'] = {'BM25': bm25_score,
                                                         'BERT': proba,
                                                         # 'CLS' : cls
                                                         }

            bm25_score = bm25_score_doc_for_query(
                query_stem_dict=qid_to_q_txt_dict[q_num],
                df_dict=df_dict,
                doc_dict=create_tf_dict_bm25_ready(row.negpar, ps),
                k1=0.6,
                b=0.62)
            inputs = tokenizer.encode_plus(row.query, row.negpar, return_tensors="pt")
            outputs = model(**inputs)
            proba = torch.softmax(outputs[0], dim=1).tolist()[0][1]
            cls = outputs.hidden_states[-1][0][0]
            q_res_dict[q_num][str(curr_idx) + '_Neg'] = {'BM25': bm25_score,
                                                         'BERT': proba,
                                                         # 'CLS': cls
                                                         }

            if curr_idx == int(q_to_train_row_index[q_num][-1]):
                with open('/lv_local/home/zivvasilisky/dataset/processed_queries/doc_idx/' + str(q_num) +'.json', 'w') as f:
                    f.write(str(q_res_dict[q_num]))
                del q_res_dict[q_num]

            curr_idx += 1

        print(frac + ':' + str(curr_idx))
        sys.stdout.flush()




if __name__=="__main__":
    create_df_dict_from_raw_passage_file()
    # create_query_to_row_idx_index_file()
    # frac = sys.argv[1]
    # create_bm25_and_bert_scores_and_cls_for_train_frac(frac)

