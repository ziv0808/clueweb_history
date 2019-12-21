import os
import sys
import pandas as pd
from utils import *

def get_relevant_docs_df(
        qurls_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc'):

    relevant_docs_df = pd.DataFrame(columns = ['Query', 'Docno', 'Relevance'])
    next_index = 0
    with open(qurls_path, 'r') as f:
        file_lines = f.readlines()

    for line in file_lines:
        line = line.strip()
        splitted_line = line.split(' ')
        relevant_docs_df.loc[next_index] = [splitted_line[0], splitted_line[2], splitted_line[3]]
        next_index += 1

    relevant_docs_df.to_csv(qurls_path.replace('.', '_') + '_relevant_docs.tsv', sep = '\t', index = False)


def merge_covered_df_with_file(
        file_path = os.path.join('dfs', 'Summay_snapshot_stats_1W.tsv')
        ):
    work_df = pd.read_csv(file_path, sep = '\t',index_col = False)
    rel_df = pd.read_csv('qrels_adhoc_relevant_docs.tsv',sep = '\t',index_col = False )
    rel_df['Relevance'] = rel_df['Relevance'].apply(lambda x: int(x))
    rel_df = rel_df[rel_df['Relevance'] > 0]
    work_df = pd.merge(
        work_df,
        rel_df[['Docno', 'Query']].rename(columns={'Query': 'QueryNum'}),
        on=['Docno', 'QueryNum'],
        how='inner')
    work_df.to_csv(file_path.replace('.tsv', '_Only_relevant.tsv'), sep = '\t',index = False)


def check_for_txt_len_problem(
        work_file = os.path.join('dfs', 'Summay_snapshot_stats_1W.tsv'),
        only_relvant = False):
    work_df = pd.read_csv(work_file, sep = '\t', index_col =False)
    if only_relvant == True:
        rel_df = pd.read_csv('qrels_adhoc_relevant_docs.tsv', sep='\t', index_col=False)
        rel_df['Relevance'] = rel_df['Relevance'].apply(lambda x: int(x))
        rel_df = rel_df[rel_df['Relevance'] > 0]
        work_df = pd.merge(
            work_df,
            rel_df[['Docno', 'Query']].rename(columns={'Query': 'QueryNum'}),
            on=['Docno', 'QueryNum'],
            how='inner')
        print("Len after relevant filter " + str(len(work_df)))
    num_with_prob = 0
    prev_row = None
    prev_doc = None
    for index, row in work_df.iterrows():
        curr_doc = row['Docno']
        if (curr_doc != prev_doc) or row['Interval'] != 'ClueWeb09':
            prev_doc = curr_doc
            prev_row = row
            continue
        else:
            if row['TextLen'] <= 0.5 * prev_row['TextLen']:
                num_with_prob += 1
            elif prev_row['TextLen'] > 2000:
                if row['TextLen'] <= 0.8 * prev_row['TextLen']:
                    num_with_prob += 1
    print (num_with_prob)

def fill_cc_dict_with_doc(
        interval_list,
        cc_dict,
        word_ref_dict,
        doc_dict,
        interval_freq,
        lookup):

    for interval_idx in range(len(interval_list)):
        curr_interval = interval_list[interval_idx]
        curr_doc_instance = get_doc_snapshot_by_lookup_method(
            doc_dict=doc_dict,
            interval_list=interval_list,
            interval_lookup_method=lookup,
            curr_interval_idx=interval_idx)
        if curr_doc_instance is not None:
            for j in range(len(curr_doc_instance['StemList'])):
                curr_stem = curr_doc_instance['StemList'][j]
                if curr_stem in word_ref_dict:
                    if curr_stem in cc_dict[interval_freq][lookup][curr_interval]:
                        cc_dict[interval_freq][lookup][curr_interval][curr_stem] += int(curr_doc_instance['TfList'][j])
                    else:
                        cc_dict[interval_freq][lookup][curr_interval][curr_stem] = int(curr_doc_instance['TfList'][j])
                cc_dict[interval_freq][lookup][curr_interval]['ALL_TERMS_COUNT'] += int(curr_doc_instance['TfList'][j])




def create_per_interval_per_lookup_cc_dict(
        work_interval_freq_list = ['1W', '2W', '1M', '2M', 'SIM', 'SIM_995'],
        lookup_method_list = ['NoLookup', 'Backward','OnlyBackward','Forward'],
        already_exists = True):

    work_df = pd.read_csv('/mnt/bi-strg3/v/zivvasilisky/ziv/data/StemsCollectionCountsAllIntervals.tsv', sep = '\t', index_col = False)
    work_df = work_df[work_df['Interval']=='ClueWeb09']

    if already_exists == False:
        res_dict = {}
        res_dict['ClueWeb09'] = {}
        for index, row in work_df.iterrows():
            res_dict['ClueWeb09'][row['Stem']] = int(row['CollectionCount'])
    else:
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/cc_per_interval_dict.json', 'r') as f:
            res_dict = ast.literal_eval(f.read())

    for interval_freq in work_interval_freq_list:
        print(interval_freq)
        sys.stdout.flush()
        if interval_freq in res_dict:
            continue
        res_dict[interval_freq] = {}
        curr_interval_list = build_interval_list(
                work_year='2008',
                frequency=interval_freq,
                add_clueweb=True)
        processed_docs_path = os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/2008/', interval_freq)
        for lookup_method in lookup_method_list:
            print(lookup_method)
            sys.stdout.flush()
            res_dict[interval_freq][lookup_method] = {}
            for interval in curr_interval_list:
                res_dict[interval_freq][lookup_method][interval] = {}
                res_dict[interval_freq][lookup_method][interval]['ALL_TERMS_COUNT'] = 0

            for file_name in os.listdir(processed_docs_path):
                if file_name.startswith('clueweb09') and file_name.endswith('.json'):
                    print(file_name)
                    sys.stdout.flush()
                    with open(os.path.join(processed_docs_path, file_name), 'r') as f:
                        doc_dict = ast.literal_eval(f.read())

                    fill_cc_dict_with_doc(
                        interval_list=curr_interval_list,
                        interval_freq=interval_freq,
                        cc_dict = res_dict,
                        word_ref_dict = res_dict['ClueWeb09'],
                        doc_dict = doc_dict,
                        lookup = lookup_method)

        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/cc_per_interval_dict.json', 'w') as f:
            f.write(str(res_dict))


def create_similarity_interval(
        from_interval_size='1W',
        sim_threshold = 0.995,
        work_year = '2008',
        sim_folder_name = "SIM_995"
        ):
    time_interval_list = build_interval_list(
        work_year=work_year,
        frequency=from_interval_size,
        add_clueweb= True)

    sim_interval_list = build_interval_list(
        work_year=work_year,
        frequency="SIM",
        add_clueweb= True)

    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/2008/'

    for file_name in os.listdir(os.path.join(processed_docs_folder, from_interval_size)):
        if file_name.startswith('clueweb09') and file_name.endswith('.json'):
            with open(os.path.join(os.path.join(processed_docs_folder, from_interval_size),file_name), 'r') as f:
                doc_dict = ast.literal_eval(f.read())
            print(file_name)
            sys.stdout.flush()
            doc_active_interval_list = list(reversed(time_interval_list[:]))
            res_doc_dict = {}
            last = ""
            for sim_interval in list(reversed(sim_interval_list)):
                for time_interval in doc_active_interval_list[:]:
                    if sim_interval == "ClueWeb09":
                        res_doc_dict[sim_interval] = doc_dict[time_interval]
                        doc_active_interval_list.remove(time_interval)
                        last = sim_interval
                        break
                    elif doc_dict[time_interval] is not None:
                        curr_sim = calc_cosine(doc_dict[time_interval]['TfIdf'], res_doc_dict[last]['TfIdf'])
                        if curr_sim < sim_threshold:
                            res_doc_dict[sim_interval] = doc_dict[time_interval]
                            doc_active_interval_list.remove(time_interval)
                            last = sim_interval
                            break
                    else:
                        doc_active_interval_list.remove(time_interval)

            for sim_interval in sim_interval_list:
                if sim_interval not in res_doc_dict:
                    res_doc_dict[sim_interval] = None

            with open(os.path.join(os.path.join(processed_docs_folder, sim_folder_name), file_name), 'w') as f:
                f.write(str(res_doc_dict))


def create_stats_data_frame_for_snapshot_changes(
        work_year='2008',
        sim_folder_name="SIM_995"):
    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/2008/'
    summary_df = pd.DataFrame(
        columns=['Docno', 'Interval', 'PrevValidInterval', 'QueryNum', 'TextLen', '#Stopword', 'QueryWords',
                 'Entropy', 'SimToClueWeb', 'SimToPrev', 'SimToClosePrev',
                 'StemDiffCluWeb', 'CluWebStemDiff'])
    next_index = 0
    df_query_stems = create_stemmed_queries_df()
    query_to_stem_mapping = {}
    for index, row in df_query_stems.iterrows():
        query_num = ('0' * (3 - len(str(row['QueryNum'])))) + str(row['QueryNum'])
        query_to_stem_mapping[query_num] = row['QueryStems'].split(' ')

    df_query_to_doc = create_query_to_doc_mapping_df()


    query_to_docno_mapping = {}
    for index, row in df_query_to_doc.iterrows():
        query_num = ('0' * (3 - len(str(row['QueryNum'])))) + str(row['QueryNum'])
        if query_num in query_to_docno_mapping:
            query_to_docno_mapping[query_num].append(row['Docno'])
        else:
            query_to_docno_mapping[query_num] = [row['Docno']]

    interval_ordered_list = build_interval_list(work_year, sim_folder_name, add_clueweb=True)
    for file_name in os.listdir(os.path.join(processed_docs_folder, sim_folder_name)):
        if file_name.startswith('clueweb09') and file_name.endswith('.json'):
            with open(os.path.join(os.path.join(processed_docs_folder, sim_folder_name), file_name), 'r') as f:
                doc_json = ast.literal_eval(f.read())
            print(file_name)
            docno = file_name.replace('.json', '')
            sys.stdout.flush()
            prev_interval = None
            for interval in interval_ordered_list:
                if doc_json[interval] is None:
                    continue
                txt_len = doc_json[interval]['NumWords']
                num_stop_words = doc_json[interval]['NumStopWords']
                entropy = doc_json[interval]['Entropy']
                sim_to_clueweb = calc_cosine(doc_json[interval]['TfIdf'], doc_json['ClueWeb09']['TfIdf'])
                sim_to_prev = None
                sim_to_close_prev = None
                if prev_interval is not None:
                    sim_to_prev = calc_cosine(doc_json[interval]['TfIdf'], doc_json[prev_interval]['TfIdf'])
                    sim_to_close_prev = sim_to_prev
                curr_document_stem_set = set(doc_json[interval]['StemList'])
                clueweb_document_stem_set = set(doc_json['ClueWeb09']['StemList'])
                document_from_clueweb_stem_diff = list(curr_document_stem_set - clueweb_document_stem_set)
                clueweb_from_document_stem_diff = list(clueweb_document_stem_set - curr_document_stem_set)
                for query_num in query_to_docno_mapping:
                    if docno in query_to_docno_mapping[query_num]:
                        query_word_num = 0
                        for j in range(len(doc_json[interval]['StemList'])):
                            if doc_json[interval]['StemList'][j] in query_to_stem_mapping[query_num]:
                                query_word_num += doc_json[interval]['TfList'][j]

                        summary_df.loc[next_index] = [docno, interval, prev_interval, query_num,
                                                      txt_len, num_stop_words, query_word_num, entropy,
                                                      sim_to_clueweb, sim_to_prev, sim_to_close_prev,
                                                      str(document_from_clueweb_stem_diff),
                                                      str(clueweb_from_document_stem_diff)]
                        next_index += 1
    summary_df.to_csv('/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/Summay_snapshot_stats_' + sim_folder_name + '.tsv',
                            sep='\t', index=False)

# create_similarity_interval()
# create_stats_data_frame_for_snapshot_changes()
# create_per_interval_per_lookup_cc_dict()

# check_for_txt_len_problem()
# merge_covered_df_with_file()
get_relevant_docs_df()

