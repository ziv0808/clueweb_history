import os
import ast
import math
import pandas as pd


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

def get_word_diriclet_smoothed_probability(
        tf_in_doc,
        doc_len,
        collection_count_for_word,
        collection_len,
        mue):
    return (tf_in_doc + mue*(float(collection_count_for_word)/collection_len))/(float(doc_len + mue))

def score_doc_for_query(
        query_stem_dict,
        cc_dict,
        doc_dict,
        mue):

    kl_score = 0.0
    for stem in query_stem_dict:
        stem_q_prob = get_word_diriclet_smoothed_probability(
            tf_in_doc = query_stem_dict[stem],
            doc_len = sum(list(query_stem_dict.values())),
            collection_count_for_word=cc_dict[stem],
            collection_len=cc_dict['ALL_TERMS_COUNT'],
            mue=mue)
        doc_stem_tf = 0
        for i in range(len(doc_dict['StemList'])):
            if doc_dict['StemList'][i] == stem:
                doc_stem_tf = doc_dict['TfList'][i]

        stem_d_proba = get_word_diriclet_smoothed_probability(
            tf_in_doc = doc_stem_tf,
            doc_len = doc_dict['NumWords'],
            collection_count_for_word=cc_dict[stem],
            collection_len=cc_dict['ALL_TERMS_COUNT'],
            mue=mue)

        kl_score += stem_q_prob*math.log2((stem_q_prob/stem_d_proba))

    return kl_score

def convert_query_to_tf_dict(
        query):
    query_dict = {}
    splitted_query = query.split(' ')
    for stem in splitted_query:
        if stem in query_dict:
            query_dict[stem] += 1
        else:
            query_dict[stem] = 1
    return query_dict

def get_scored_df_for_query(
        query_num,
        query,
        query_doc_df,
        interval_idx,
        interval_lookup_method,
        processed_docs_path,
        interval_list,
        cc_dict,
        mue):

    res_df= pd.DataFrame(columns = ['Query_ID','Iteration', 'Docno', 'Rank', 'Score', 'Method'])
    next_index = 0
    query_dict = convert_query_to_tf_dict(query)
    for index, row in query_doc_df.iterrows():
        docno = row['Docno']
        # retrive docno dict
        with open(os.path.join(processed_docs_path, docno + '.json'), 'r') as f:
            doc_dict = ast.literal_eval(f.read())
        # find the interval to look for
        doc_interval_dict = doc_dict[interval_list[interval_idx]]
        if doc_interval_dict is None:
            if interval_lookup_method == "Forward":
                addition = 1
                while doc_interval_dict is None:
                    doc_interval_dict = interval_list[interval_idx + addition]
                    addition += 1
            elif interval_lookup_method == "Backward":
                addition = 1
                while (doc_interval_dict is None) and ((interval_idx - addition) >= 0):
                    doc_interval_dict = interval_list[interval_idx - addition]
                    addition += 1
                if doc_interval_dict is None:
                    addition = 1
                    while doc_interval_dict is None:
                        doc_interval_dict = interval_list[interval_idx + addition]
                        addition += 1
        print(doc_interval_dict)
        doc_score = score_doc_for_query(
            query_stem_dict=query_dict,
            cc_dict=cc_dict,
            doc_dict=doc_interval_dict,
            mue=mue)
        res_df.loc[next_index] = [query_num, 'Q0', docno, 0, doc_score, 'indri']
        next_index += 1

    res_df.sort_values('Score', ascending=False, inplace=True)
    res_df['Rank'] = list(range(1, next_index + 1))
    return res_df

def convert_df_to_trec(
        df):

    trec_str = ""
    for index, row in df.iterrows():
        trec_str += str(row['Query_ID']) + " " + row['Iteration'] + " " + \
                    row['Docno'] + " " + str(row['Rank']) + " " + str(row['Score']) +\
                    " " + row['Method'] + '\n'
    return trec_str[:-1]

if __name__=='__main__':
    query_to_doc_mapping_file = '/lv_local/home/zivvasilisky/ziv/data/all_urls_no_spam_filtered.tsv'
    stemmed_query_file = '/lv_local/home/zivvasilisky/ziv/data/Stemmed_Query_Words'
    stemmed_query_collection_counts = '/lv_local/home/zivvasilisky/ziv/data/StemsCollectionCounts.tsv'
    processed_docs_folder = '/lv_local/home/zivvasilisky/ziv/data/processed_document_vectors/2008/'
    save_folder = '/lv_local/home/zivvasilisky/ziv/results/ranked_docs/'

    mue = 1000.0
    interval_lookup_method = 'Forward'
    interval_list = build_interval_list('2008', '2W')
    interval_list.append('ClueWeb09')
    # retrieve necessary dataframes
    query_to_doc_mapping_df = pd.read_csv(query_to_doc_mapping_file, sep = '\t', index_col = False)
    stemmed_queries_df = pd.read_csv(stemmed_query_file, sep = '\t', index_col = False)
    query_stems_cc_df = pd.read_csv(stemmed_query_collection_counts, sep = '\t', index_col = False)
    # create easy to use index for cc
    cc_dict = {}
    for index, row in query_stems_cc_df.iterrows():
        cc_dict[row['Stem']] = float(row['CollectionCount'])

    big_df = pd.DataFrame({})
    for index, row in stemmed_queries_df.iterrows():
        query_num = int(row['QueryNum'])
        print("Query: " + str(query_num))
        query_txt = row['QueryStems']
        relevant_df = query_to_doc_mapping_df[query_to_doc_mapping_df['QueryNum'] == query_num].copy()
        for j in range(len(interval_list)):
            print("Interval: " + str(interval_list[j]))
            res_df = get_scored_df_for_query(
                query_num=query_num,
                query=query_txt,
                query_doc_df=relevant_df,
                interval_idx=j,
                interval_list=interval_list,
                interval_lookup_method=interval_lookup_method,
                processed_docs_path=processed_docs_folder,
                cc_dict=cc_dict,
                mue=mue)

            with open(os.path.join(save_folder, str(query_num) + "_" + str(interval_list[j] + "_" + interval_lookup_method + "_Results.txt")), 'w') as f:
                f.write(convert_df_to_trec(res_df))