import re
import torch
import nltk
import time
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

def create_query_to_row_idx_index_file_for_test_set():
    q_txt_to_qid_dict = get_queries_file_to_df('dev',as_dict='Regular')
    large_index_dict = {}
    curr_idx = 0
    for df in pd.read_csv('/lv_local/home/zivvasilisky/dataset/top1000.dev', sep = '\t', chunksize = 50000, header = None):
        df.columns = ['qid', 'pid', 'query', 'par']
        for query in df['query']:
            try:
                q_num = q_txt_to_qid_dict[query.encode('latin1').decode('utf8')]
            except Exception as e:
                q_num = q_txt_to_qid_dict[query.encode('utf-8').decode('utf8')]
            if q_num in large_index_dict:
                large_index_dict[q_num].append(curr_idx)
            else:
                large_index_dict[q_num]= [curr_idx]

            curr_idx += 1
        print(curr_idx)
        sys.stdout.flush()
    with open('/lv_local/home/zivvasilisky/dataset/query_to_train_row_idx_for_test_set.json', 'w') as f:
        f.write(str(large_index_dict))

def get_query_to_train_row_index(frac_idx_to_filter_by=None, test_set=False):
    if test_set == True:
        with open('/lv_local/home/zivvasilisky/dataset/query_to_train_row_idx_for_test_set.json', 'r') as f:
            big_idx = ast.literal_eval(f.read())
    else:
        with open('/lv_local/home/zivvasilisky/dataset/query_to_train_row_idx.json', 'r') as f:
            big_idx = ast.literal_eval(f.read())
    del_q_list = []
    if frac_idx_to_filter_by is not None:
        for q in big_idx:
            if q not in frac_idx_to_filter_by:
                del_q_list.append(q)
    if test_set == True:
        for filename in os.listdir('/lv_local/home/zivvasilisky/dataset/processed_queries/test_doc_idx/'):
            curr_q = int(filename.replace('.json', ''))
            if curr_q in big_idx and curr_q not in del_q_list:
                del_q_list.append(curr_q)
    else:
        for filename in os.listdir('/lv_local/home/zivvasilisky/dataset/processed_queries/doc_idx/'):
            curr_q = int(filename.replace('.json',''))
            if curr_q in big_idx and curr_q not in del_q_list:
                del_q_list.append(curr_q)

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


def bm25_score_doc_for_query_course_proj(
        query_stem_dict,
        df_dict,
        doc_dict,
        k1=1.0,
        b=0.5):


    bm25_score = 0.0
    work_stem_list = list(query_stem_dict.keys())

    for stem in work_stem_list:
        doc_stem_tf = 0
        if stem in doc_dict:
            doc_stem_tf = float(doc_dict[stem])

        if stem not in df_dict:
            if doc_stem_tf == 0:
                continue
            else:
                raise Exception('Unexpected Situation on ' + str(stem))
        if df_dict[stem] == 0:
            df_dict[stem] = 1

        idf = math.log(df_dict['ALL_DOCS_COUNT'] / float(df_dict[stem]), 10)
        stem_d_proba = (doc_stem_tf * (k1 + 1)) / (
            doc_stem_tf + k1 * ((1 - b) + b * (float(doc_dict['NumWords']) / df_dict['AVG_DOC_LEN'])))

        bm25_score += idf * stem_d_proba

    return bm25_score

def create_bm25_and_bert_scores_and_cls_for_train_frac(frac):
    qid_to_q_txt_dict = get_queries_file_to_df('train', as_dict='Reverse', frac=frac)
    print(frac + ": got qid_to_q_txt_dict: " + str(len(qid_to_q_txt_dict)))
    sys.stdout.flush()
    q_to_train_row_index = get_query_to_train_row_index(qid_to_q_txt_dict)
    print(frac + ": got q_to_train_row_index: " + str(len(q_to_train_row_index)))
    sys.stdout.flush()
    ps = nltk.stem.porter.PorterStemmer()
    for qid in qid_to_q_txt_dict:
        qid_to_q_txt_dict[qid] = create_tf_dict_bm25_ready(qid_to_q_txt_dict[qid], ps)
        del qid_to_q_txt_dict[qid]['NumWords']

    print(frac + ": fixed qid_to_q_txt_dict")
    sys.stdout.flush()
    q_txt_to_qid_dict = get_queries_file_to_df('train', as_dict='Regular')
    print(frac + ": got q_txt_to_qid_dict")
    sys.stdout.flush()
    q_res_dict = {}

    with open('/lv_local/home/zivvasilisky/dataset/df_dict.json', 'r') as f:
        df_dict = ast.literal_eval(f.read())
    model, tokenizer = load_fine_tuned_bert_model()
    print("got df_dict, model and tokenizer")
    sys.stdout.flush()

    first_run = True
    chunk_size = 5000
    all_queries_to_handle = list(q_to_train_row_index.keys())
    for chunk_num in range(0, len(all_queries_to_handle), chunk_size):
        curr_idx = 0
        curr_chunk = all_queries_to_handle[chunk_num:chunk_num + chunk_size]
        print(frac + ': Chunk ' + str(chunk_num))
        sys.stdout.flush()
        curr_chunk_check_index = {}
        for q in curr_chunk:
            curr_chunk_check_index[q] = None
        for df in pd.read_csv('/lv_local/home/zivvasilisky/dataset/triples.train.small.tsv', sep='\t', chunksize=50000,header=None):
            df.columns = ['query', 'pospar', 'negpar']
            for row in df.itertuples():
                q_num = q_txt_to_qid_dict[row.query.encode('latin1').decode('utf8')]
                if q_num not in curr_chunk_check_index:
                    curr_idx += 1
                    continue
                if q_num not in q_res_dict:
                    q_res_dict[q_num] = {}
                if first_run == True:
                    print(frac + ": Curr Idx: " + str(curr_idx))
                    print(frac + ": " + str(time.time()))
                    sys.stdout.flush()
                bm25_score = bm25_score_doc_for_query_course_proj(
                    query_stem_dict=qid_to_q_txt_dict[q_num],
                    df_dict=df_dict,
                    doc_dict=create_tf_dict_bm25_ready(row.pospar, ps),
                    k1=0.6,
                    b=0.62)
                try:
                    inputs = tokenizer.encode_plus(row.query, row.pospar, return_tensors="pt")
                    outputs = model(**inputs)
                    proba = torch.softmax(outputs[0], dim=1).tolist()[0][1]
                    cls = outputs.hidden_states[-1][0][0].tolist()
                except Exception as e:
                    print("BERT Exception " + str(e))
                    sys.stdout.flush()
                    proba = None
                    cls = None
                q_res_dict[q_num][str(curr_idx) + '_Pos'] = {'BM25': bm25_score,
                                                             'BERT': proba,
                                                             'CLS' : cls
                                                             }

                bm25_score = bm25_score_doc_for_query_course_proj(
                    query_stem_dict=qid_to_q_txt_dict[q_num],
                    df_dict=df_dict,
                    doc_dict=create_tf_dict_bm25_ready(row.negpar, ps),
                    k1=0.6,
                    b=0.62)
                try:
                    inputs = tokenizer.encode_plus(row.query, row.negpar, return_tensors="pt")
                    outputs = model(**inputs)
                    proba = torch.softmax(outputs[0], dim=1).tolist()[0][1]
                    cls = outputs.hidden_states[-1][0][0].tolist()
                except Exception as e:
                    print("BERT Exception " + str(e))
                    sys.stdout.flush()
                    proba = None
                    cls = None
                q_res_dict[q_num][str(curr_idx) + '_Neg'] = {'BM25': bm25_score,
                                                             'BERT': proba,
                                                             'CLS': cls
                                                             }
                if first_run == True:
                    print(frac + ": " + str(time.time()))
                    sys.stdout.flush()
                    first_run = False

                if curr_idx == int(q_to_train_row_index[q_num][-1]):
                    with open('/lv_local/home/zivvasilisky/dataset/processed_queries/doc_idx/' + str(q_num) +'.json', 'w') as f:
                        f.write(str(q_res_dict[q_num]))
                    del q_res_dict[q_num]
                    del curr_chunk_check_index[q_num]

                curr_idx += 1

            print(frac + ':' + str(curr_idx))
            sys.stdout.flush()

def get_all_relavnt_scores_for_ranked_df(ranked_df):
    mrr  = np.nan
    p_5  = np.nan
    p_10 = np.nan
    map  = 0.0
    map_50 = 0.0
    curr_num_rel = 0.0
    num_rows = 0.0
    for index, row in ranked_df.iterrows():
        num_rows += 1
        if row['Relevance'] == 1:
            if np.isnan(mrr):
                mrr = 1.0 / num_rows
            curr_num_rel += 1

        map += curr_num_rel / float(num_rows)
        if num_rows <= 50:
            map_50 += curr_num_rel / float(num_rows)
        if num_rows == 5:
            p_5 = curr_num_rel / float(num_rows)
        if num_rows == 10:
            p_10 = curr_num_rel / float(num_rows)

    if np.isnan(p_5):
        p_5 = curr_num_rel / float(num_rows)

    if np.isnan(p_10):
        p_10 = curr_num_rel / float(num_rows)

    if num_rows < 50:
        map_50 = map_50 / float(num_rows)
    else:
        map_50 = map_50 / 50.0

    map = map / float(num_rows)

    return {'MRR' : mrr, 'P_5' : p_5, 'P_10' : p_10, 'MAP' : map, 'MAP_50' : map_50}


def sort_df_by_score_and_get_eval(df):
    df.sort_values('Score', inplace=True, ascending=False)
    return get_all_relavnt_scores_for_ranked_df(df)


def get_reciprocal_rank_and_bm25_bert_scores(frac):
    qid_to_q_txt_dict = get_queries_file_to_df('train', as_dict='Reverse', frac=frac)
    for filename in os.listdir('/lv_local/home/zivvasilisky/dataset/processed_queries/doc_idx/'):
        curr_q = int(filename.replace('.json', ''))
        if curr_q in qid_to_q_txt_dict:
            print(curr_q)
            sys.stdout.flush()
            with open(os.path.join('/lv_local/home/zivvasilisky/dataset/processed_queries/doc_idx/', filename), 'r') as f:
                curr_dict = ast.literal_eval(f.read())
            curr_df = pd.DataFrame(columns=['Docno', 'BM25', 'BERT', 'Relevance'])
            curr_idx = 0
            for docno in curr_dict:
                insert_row = [docno, curr_dict[docno]['BM25'], curr_dict[docno]['BERT']]
                if docno.endswith('_Pos'):
                    insert_row.append(1)
                else:
                    insert_row.append(0)
                curr_df.loc[curr_idx] = insert_row
                curr_idx += 1

            score_list = ['MRR', 'P_5', 'P_10', 'MAP', 'MAP_50']
            summary_df = pd.DataFrame(columns=['ReMethod', 'K', 'Lambda1']+score_list )
            next_idx = 0

            for score in ['BM25', 'BERT']:
                curr_df.sort_values(score, ascending=False, inplace=True)
                curr_df[score + 'Rank'] = list(range(1, curr_idx + 1))
                curr_df['Score'] = curr_df[score]
                res_dict = sort_df_by_score_and_get_eval(curr_df[['Score', 'Relevance']])
                insert_row = [score, 0.0, 0.0]
                for eval_meth in score_list:
                    insert_row.append(res_dict[eval_meth])
                summary_df.loc[next_idx] = insert_row
                next_idx += 1

            for k in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                for lambda1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    curr_df['Score'] = curr_df.apply(lambda row: lambda1*(1.0 / float(k + row['BM25Rank'])) + (1.0 - lambda1)*(1.0 / float(k + row['BERTRank'])), axis =1)
                    res_dict = sort_df_by_score_and_get_eval(curr_df[['Score', 'Relevance']])
                    insert_row = ['Mixed', k, lambda1]
                    for eval_meth in score_list:
                        insert_row.append(res_dict[eval_meth])
                    summary_df.loc[next_idx] = insert_row
                    next_idx += 1

            summary_df.to_csv('/lv_local/home/zivvasilisky/dataset/results/' + filename.replace('.json', '.tsv'), sep = '\t', index = False)


def create_bm25_and_bert_scores_and_cls_for_test(frac):
    qid_to_q_txt_dict = get_queries_file_to_df('dev_' +str(frac), as_dict='Reverse', frac=None)
    print(frac + ": got qid_to_q_txt_dict: " + str(len(qid_to_q_txt_dict)))
    sys.stdout.flush()
    q_to_train_row_index = get_query_to_train_row_index(qid_to_q_txt_dict, test_set=True)
    print(frac + ": got q_to_train_row_index: " + str(len(q_to_train_row_index)))
    sys.stdout.flush()
    ps = nltk.stem.porter.PorterStemmer()
    for qid in qid_to_q_txt_dict:
        qid_to_q_txt_dict[qid] = create_tf_dict_bm25_ready(qid_to_q_txt_dict[qid], ps)
        del qid_to_q_txt_dict[qid]['NumWords']

    print(frac + ": fixed qid_to_q_txt_dict")
    sys.stdout.flush()
    q_txt_to_qid_dict = get_queries_file_to_df('dev', as_dict='Regular')
    print(frac + ": got q_txt_to_qid_dict")
    sys.stdout.flush()
    q_res_dict = {}

    with open('/lv_local/home/zivvasilisky/dataset/df_dict.json', 'r') as f:
        df_dict = ast.literal_eval(f.read())
    model, tokenizer = load_fine_tuned_bert_model()
    print("got df_dict, model and tokenizer")
    sys.stdout.flush()

    first_run = True
    chunk_size = 5000
    all_queries_to_handle = list(q_to_train_row_index.keys())
    for chunk_num in range(0, len(all_queries_to_handle), chunk_size):
        curr_idx = 0
        curr_chunk = all_queries_to_handle[chunk_num:chunk_num + chunk_size]
        print(frac + ': Chunk ' + str(chunk_num))
        sys.stdout.flush()
        curr_chunk_check_index = {}
        for q in curr_chunk:
            curr_chunk_check_index[q] = None
        for df in pd.read_csv('/lv_local/home/zivvasilisky/dataset/top1000.dev', sep='\t', chunksize=50000,
                              header=None):
            df.columns = ['qid', 'pid', 'query', 'par']
            for row in df.itertuples():
                try:
                    q_num = q_txt_to_qid_dict[row.query.encode('latin1').decode('utf8')]
                except Exception as e:
                    q_num = q_txt_to_qid_dict[row.query.encode('utf-8').decode('utf8')]
                if q_num not in curr_chunk_check_index:
                    curr_idx += 1
                    continue
                if q_num not in q_res_dict:
                    q_res_dict[q_num] = {}
                if first_run == True:
                    print(frac + ": Curr Idx: " + str(curr_idx))
                    print(frac + ": " + str(time.time()))
                    sys.stdout.flush()
                bm25_score = bm25_score_doc_for_query_course_proj(
                    query_stem_dict=qid_to_q_txt_dict[q_num],
                    df_dict=df_dict,
                    doc_dict=create_tf_dict_bm25_ready(row.par, ps),
                    k1=0.6,
                    b=0.62)
                try:
                    inputs = tokenizer.encode_plus(row.query, row.par, return_tensors="pt")
                    outputs = model(**inputs)
                    proba = torch.softmax(outputs[0], dim=1).tolist()[0][1]
                    cls = outputs.hidden_states[-1][0][0].tolist()
                except Exception as e:
                    print("BERT Exception " + str(e))
                    sys.stdout.flush()
                    proba = None
                    cls = None
                q_res_dict[q_num][str(row.pid)] = {'BM25': bm25_score,
                                                     'BERT': proba,
                                                     'CLS': cls
                                                             }

                if first_run == True:
                    print(frac + ": " + str(time.time()))
                    sys.stdout.flush()
                    first_run = False

                if curr_idx == int(q_to_train_row_index[q_num][-1]):
                    with open('/lv_local/home/zivvasilisky/dataset/processed_queries/test_doc_idx/' + str(q_num) + '.json',
                              'w') as f:
                        f.write(str(q_res_dict[q_num]))
                    del q_res_dict[q_num]
                    del curr_chunk_check_index[q_num]

                curr_idx += 1

            print(frac + ':' + str(curr_idx))
            sys.stdout.flush()

if __name__=="__main__":
    # create_df_dict_from_raw_passage_file()
    # create_query_to_row_idx_index_file()
    frac = sys.argv[1]
    # get initial measures
    # create_bm25_and_bert_scores_and_cls_for_train_frac(frac)
    # train scores
    # get_reciprocal_rank_and_bm25_bert_scores(frac)

    # create_query_to_row_idx_index_file_for_test_set()
    create_bm25_and_bert_scores_and_cls_for_test(frac)