import re
import torch
import nltk
import time
from random import shuffle
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

def get_reciprocal_rank_and_bm25_bert_scores_for_test():
    for filename in os.listdir('/lv_local/home/zivvasilisky/dataset/processed_queries/test_doc_idx_fixed/'):
        curr_q = int(filename.replace('.json', ''))
        print(curr_q)
        sys.stdout.flush()
        with open(os.path.join('/lv_local/home/zivvasilisky/dataset/processed_queries/test_doc_idx_fixed/', filename), 'r') as f:
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

        summary_df.to_csv('/lv_local/home/zivvasilisky/dataset/results_test/' + filename.replace('.json', '.tsv'), sep = '\t', index = False)


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
    chunk_size = 1000
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

def create_dataset_file_for_nn(is_train=True):
    folder_path = '/lv_local/home/zivvasilisky/dataset/processed_queries/doc_idx/'
    if is_train == False:
        folder_path = '/lv_local/home/zivvasilisky/dataset/processed_queries/test_doc_idx/'

    filelist = list(os.listdir(folder_path))
    shuffle(filelist)
    for filename in filelist:
        if filename.endswith('.json') and (not os.path.isfile(folder_path.replace('doc_idx', 'tsv_files') + filename.replace('.json', '.tsv'))):
            print(filename)
            sys.stdout.flush()
            df = pd.DataFrame(columns = ['Docno', 'Relevance', 'BM25'] + ['CLS_'+ str(i) for i in range(1,769)])
            next_idx =0
            with open(os.path.join(folder_path, filename), 'r') as f:
                file_dict = ast.literal_eval(f.read())

            for docno in file_dict:
                insert_row = [docno]
                if is_train == False:
                    insert_row.append(pd.np.nan)
                elif docno.endswith('_Pos'):
                    insert_row.append(1)
                else:
                    insert_row.append(0)
                insert_row.append(file_dict[docno]['BM25'])
                if file_dict[docno]['CLS'] is None:
                    continue
                insert_row.extend(file_dict[docno]['CLS'])
                df.loc[next_idx] = insert_row
                next_idx += 1
            mean_bm25 = df['BM25'].mean()
            std_bm25  = df['BM25'].std()
            if pd.np.isnan(std_bm25) or std_bm25 == 0:
                std_bm25 = 1.0

            df['BM25'] = df['BM25'].apply(lambda x: (x - mean_bm25) / float(std_bm25))
            df.to_csv(folder_path.replace('doc_idx', 'tsv_files') + filename.replace('.json', '.tsv'), sep = '\t', index = False)

def summarize_train_results_non_nn():
    folder_path = '/lv_local/home/zivvasilisky/dataset/results/'
    df_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.tsv'):
            print(filename)
            sys.stdout.flush()
            df = pd.read_csv(folder_path +filename, sep = '\t', index_col=False)
            df_list.append(df)
            # summary_df = summary_df.append(df)
    summary_df = pd.concat(df_list, axis=0, ignore_index=True)
    summary_df = summary_df.groupby(['ReMethod', 'K', 'Lambda1']).mean()
    summary_df = summary_df.reset_index()
    summary_df.to_csv('/lv_local/home/zivvasilisky/dataset/Train_Results.tsv', sep = '\t', index = False)

def create_rel_dict_for_test_set():
    rel_dict = {}
    rel_df = pd.read_csv('/lv_local/home/zivvasilisky/dataset/qrels.dev.tsv', sep ='\t', index_col=False, header=None)
    rel_df.columns = ['qid', 'not_relvant', 'pid', 'not_relvant2']
    for index, row in rel_df.iterrows():
        if row['qid'] not in rel_dict:
            rel_dict[int(row['qid'])] = {}
        rel_dict[int(row['qid'])][int(row['pid'])] = None
    return rel_dict

def append_relevance_to_test_files_nn():
    folder_path = '/lv_local/home/zivvasilisky/dataset/processed_queries/test_tsv_files/'
    res_folder = '/lv_local/home/zivvasilisky/dataset/processed_queries/test_tsv_files_fixed/'

    rel_dict = create_rel_dict_for_test_set()
    filelist = list(os.listdir(folder_path))
    shuffle(filelist)
    for filename in filelist:
        if filename.endswith('.tsv') and (not os.path.isfile(res_folder + filename)):
            print(filename)
            sys.stdout.flush()
            df = pd.read_csv(folder_path + filename, sep ='\t', index_col = False)
            q = int(filename.replace('.tsv', ''))
            df['Relevance'] = 0
            if q in rel_dict:
                for index, row in df.iterrows():
                    if int(row['Docno']) in rel_dict[q]:
                        df.at[index, 'Relevance'] = 1
                        print("Rel")
            df.to_csv(res_folder + filename, sep ='\t', index = False)

def append_relevance_to_test_files_non_nn():
    folder_path = '/lv_local/home/zivvasilisky/dataset/processed_queries/test_doc_idx/'
    res_folder = '/lv_local/home/zivvasilisky/dataset/processed_queries/test_doc_idx_fixed/'

    rel_dict = create_rel_dict_for_test_set()
    filelist = list(os.listdir(folder_path))
    shuffle(filelist)
    for filename in filelist:
        if filename.endswith('.json') and (not os.path.isfile(res_folder + filename)):
            print(filename)
            sys.stdout.flush()
            with open(folder_path + filename, 'r') as f:
                curr_dict = ast.literal_eval(f.read())
            q = int(filename.replace('.json', ''))
            new_dict = {}
            for docno in curr_dict:
                new_docno = docno + '_Neg'
                if int(docno) in rel_dict[q]:
                    new_docno = docno +'_Pos'

                new_dict[new_docno] = {}
                new_dict[new_docno]['BM25'] = curr_dict[docno]['BM25']
                new_dict[new_docno]['BERT'] = curr_dict[docno]['BERT']

            with open(res_folder + filename, 'w') as f:
                f.write(str(new_dict))

def full_res_summary(nn_epoch_num):
    nn_res_dir = '/lv_local/home/zivvasilisky/dataset/nn_res/'
    res_dir = '/lv_local/home/zivvasilisky/dataset/results_test/'

    eval_m_list =  ['MRR','P_5','P_10','MAP', 'MAP_50']
    fin_df = pd.DataFrame(columns=['Query'] + ['NN_' + eval_m for eval_m in eval_m_list])
    next_idx = 0
    print("NN calc")
    sys.stdout.flush()
    for filename in os.listdir(nn_res_dir):
        if filename.startswith('Epoch_' + str(nn_epoch_num)+'_') and filename.endswith('.tsv'):
            print(filename)
            sys.stdout.flush()
            df = pd.read_csv(nn_res_dir + filename, sep = '\t', index_col =False)
            res_dict = sort_df_by_score_and_get_eval(df.rename(columns={'RelProba' : 'Score'}))
            q_num = filename.replace('Epoch_' + str(nn_epoch_num)+'_','').replace('.tsv','')
            insert_row = [q_num]
            for eval_m in eval_m_list:
                insert_row.append(res_dict[eval_m])
            fin_df.loc[next_idx] = insert_row
            next_idx += 1

    fin_df.to_csv('/lv_local/home/zivvasilisky/dataset/Epoch_'+ str(nn_epoch_num)+'_Res.tsv', sep = '\t', index=False)
    train_summary_df = pd.read_csv('/lv_local/home/zivvasilisky/dataset/Train_Results.tsv', sep = '\t', index_col=False)
    train_summary_df['Idx'] = train_summary_df.apply(lambda row: row['ReMethod'] + '_' + str(float(row['K'])) + '_'+ str(float(row['Lambda1'])), axis =1)

    best_config_by_model_dict = {}
    relevant_mathods = []#['BERT', 'BM25']
    for eval_m in eval_m_list:
        best_config_by_model_dict[eval_m] = {'BestScore' : train_summary_df[eval_m].max()}
    
    for index, row in train_summary_df.iterrows():
        for eval_m in eval_m_list:
            if row[eval_m] == best_config_by_model_dict[eval_m]['BestScore']:
                best_config_by_model_dict[eval_m]['Method'] = row['Idx']

    for eval_m in eval_m_list:
        relevant_mathods.append(best_config_by_model_dict[eval_m]['Method'])

    relevant_mathods.extend(['BERT_0.0_0.0', 'BM25_0.0_0.0'])
    relevant_mathods = list(set(relevant_mathods))
    print("Rel Methods: " + str(relevant_mathods))
    print(best_config_by_model_dict)
    sys.stdout.flush()

    fin_2_df = pd.DataFrame({})
    for filename in os.listdir(res_dir):
        if filename.endswith('.tsv'):
            print(filename)
            sys.stdout.flush()
            df = pd.read_csv(res_dir + filename, sep='\t', index_col=False)
            df['Idx'] = df.apply(lambda row: row['ReMethod'] + '_' + str(float(row['K'])) + '_' + str(float(row['Lambda1'])), axis=1)
            first = True
            q_num = filename.replace('.tsv', '')

            for rel_m in relevant_mathods:
                t_df = df[df['Idx'] == rel_m]
                t_df['Query'] = q_num
                if 'BERT' in rel_m:
                    report_m = 'BERT'
                elif 'BM25' in rel_m:
                    report_m = 'BM25'
                else:
                    for eval_m in eval_m_list:
                        if best_config_by_model_dict[eval_m]['Method'] == rel_m:
                            report_m = 'FuseBy' + eval_m
                rename_dict = {}
                for eval_m in eval_m_list:
                    rename_dict[eval_m] = report_m + '_' + eval_m
                if first == True:
                    q_df = t_df[['Query'] + eval_m_list].rename(columns=rename_dict)
                    first = False
                else:
                    q_df = pd.merge(
                        q_df,
                        t_df[['Query'] + eval_m_list].rename(columns=rename_dict),
                        on = ['Query'],
                        how = 'inner')
            fin_2_df = fin_2_df.append(q_df, ignore_index=True)

    fin_2_df.to_csv('/lv_local/home/zivvasilisky/dataset/Other_M_Res.tsv', sep='\t', index=False)

    methods = ['NN', 'BERT', 'BM25', 'FuseByMAP_50', 'FuseByMRR', 'FuseByP_10']
    fin_df = pd.merge(
        fin_df,
        fin_2_df,
        on = ['Query'],
        how = 'inner')

    summary_df = pd.DataFrame(columns=['Method'] + eval_m_list + [eval_m +'_sign' for eval_m in eval_m_list])
    curr_idx = 0
    for method in methods:
        insert_row = [method]
        for m_eval in eval_m_list:
            insert_row.append(fin_df[method + '_' + m_eval].mean())
        for m_eval in eval_m_list:
            curr_str = ""
            for method_2 in methods:
                if method_2 != method:
                    print(method, method_2)
                    tmp_df = fin_df[[method + '_' + m_eval, method_2 + '_' + m_eval]].dropna()
                    print(len(tmp_df))
                    sys.stdout.flush()
                    t_stat, p_val = stats.ttest_rel(list(tmp_df[method + '_' + m_eval]), list(tmp_df[method_2 + '_' + m_eval]))
                    if p_val <= 0.05:
                        curr_str += method_2 +','
            insert_row.append(curr_str)
        summary_df.loc[curr_idx] = insert_row
        curr_idx += 1
    summary_df.to_csv('/lv_local/home/zivvasilisky/dataset/FinalRes_DF.tsv', sep = '\t', index = False)
if __name__=="__main__":
    # create_df_dict_from_raw_passage_file()
    # create_query_to_row_idx_index_file()
    # frac = sys.argv[1]
    # get initial measures
    # create_bm25_and_bert_scores_and_cls_for_train_frac(frac)
    # train scores
    # get_reciprocal_rank_and_bm25_bert_scores(frac)
    # get_reciprocal_rank_and_bm25_bert_scores_for_test()
    # create_query_to_row_idx_index_file_for_test_set()
    # create_bm25_and_bert_scores_and_cls_for_test(frac)

    # create_dataset_file_for_nn()
    # create_dataset_file_for_nn(False)
    # summarize_train_results_non_nn()
    # append_relevance_to_test_files_nn()
    # append_relevance_to_test_files_non_nn()
    full_res_summary('5')