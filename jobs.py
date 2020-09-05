import os
import sys
import pandas as pd
import re
from bs4 import BeautifulSoup
from krovetzstemmer import Stemmer
from utils import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from retrival_stats_creator import rbo, overlap

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

    return relevant_docs_df
    # relevant_docs_df.to_csv(qurls_path.replace('.', '_') + '_relevant_docs.tsv', sep = '\t', index = False)


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
        work_interval_freq_list = ['1W', '2W', '1M', '2M', 'SIM', 'SIM_995','SIM_TXT_UP_DOWN','SIM_TXT_UP','SIM_TXT_DOWN'],
        lookup_method_list      = ['NoLookup', 'Backward','OnlyBackward','Forward'],
        already_exists          = True,
        work_year               = '2008',
        inner_fold              = None):

    work_df = create_stemmed_queries_df()
    if work_year != WORK_YEAR:
        raise Exception('work_year not equal to WORK_YEAR')

    if already_exists == False:
        res_dict = {}
        res_dict['RefDict'] = {}
        for index, row in work_df.iterrows():
            query_stems = row['QueryStems'].split(' ')
            for stem in query_stems:
                if stem != '':
                    res_dict['RefDict'][stem] = 0
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
                work_year=work_year,
                frequency=interval_freq,
                add_clueweb=True)
        if inner_fold == "":
            processed_docs_path = os.path.join(
                '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + work_year + '/', interval_freq)
        else:
            processed_docs_path = os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+inner_fold+'/'+work_year+'/', interval_freq)
        for lookup_method in lookup_method_list:
            print(lookup_method)
            sys.stdout.flush()
            res_dict[interval_freq][lookup_method] = {}
            for interval in curr_interval_list:
                res_dict[interval_freq][lookup_method][interval] = {}
                res_dict[interval_freq][lookup_method][interval]['ALL_TERMS_COUNT'] = 0

            for file_name in os.listdir(processed_docs_path):
                if file_name.startswith('clueweb') and file_name.endswith('.json'):
                    print(file_name)
                    sys.stdout.flush()
                    with open(os.path.join(processed_docs_path, file_name), 'r') as f:
                        doc_dict = ast.literal_eval(f.read())

                    fill_cc_dict_with_doc(
                        interval_list=curr_interval_list,
                        interval_freq=interval_freq,
                        cc_dict = res_dict,
                        word_ref_dict = res_dict['RefDict'],
                        doc_dict = doc_dict,
                        lookup = lookup_method)

        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/cc_per_interval_dict.json', 'w') as f:
            f.write(str(res_dict))


def create_similarity_interval(
        from_interval_size='1W',
        sim_threshold = 0.995,
        work_year = '2008',
        sim_folder_name = "SIM_995",
        inner_fold  = ""):

    time_interval_list = build_interval_list(
        work_year=work_year,
        frequency=from_interval_size,
        add_clueweb= True)

    sim_interval_list = build_interval_list(
        work_year=work_year,
        frequency="SIM",
        add_clueweb= True)

    if inner_fold == "":
        processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+work_year+'/'
    else:
        processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + inner_fold + '/'+work_year+'/'

    for file_name in os.listdir(os.path.join(processed_docs_folder, from_interval_size)):
        if file_name.startswith('clueweb') and file_name.endswith('.json'):
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
        sim_folder_name="SIM",
        inner_fold = ""):

    if inner_fold == "":
        processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+work_year+'/'
    else:
        processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + inner_fold + '/'+work_year+'/'

    # processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/2008/'
    summary_df = pd.DataFrame(
        columns=['Docno', 'Interval', 'PrevValidInterval', 'QueryNum', 'TextLen', '#Stopword', 'QueryWords',
                 'Entropy', 'SimToClueWeb', 'SimToPrev', 'SimToClosePrev',
                 'StemDiffCluWeb', 'CluWebStemDiff','PerTermQueryWordRatio', 'QueryLen'])
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
        if file_name.startswith('clueweb') and file_name.endswith('.json'):
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
                        per_term_query_ratio_mean = 0.0
                        if 'TfDict' in doc_json[interval]:
                            for term_ in query_to_stem_mapping[query_num]:
                                if term_ in doc_json[interval]['TfDict']:
                                    query_word_num += doc_json[interval]['TfDict'][term_]
                                    per_term_query_ratio_mean += float(doc_json[interval]['TfDict'][term_]) / txt_len
                        else:
                            for j in range(len(doc_json[interval]['StemList'])):
                                if doc_json[interval]['StemList'][j] in query_to_stem_mapping[query_num]:
                                    query_word_num += doc_json[interval]['TfList'][j]
                                    per_term_query_ratio_mean += float(doc_json[interval]['TfList'][j])/txt_len
                        per_term_query_ratio_mean = per_term_query_ratio_mean/float(len(query_to_stem_mapping[query_num]))
                        summary_df.loc[next_index] = [docno, interval, prev_interval, query_num,
                                                      txt_len, num_stop_words, query_word_num, entropy,
                                                      sim_to_clueweb, sim_to_prev, sim_to_close_prev,
                                                      str(document_from_clueweb_stem_diff),
                                                      str(clueweb_from_document_stem_diff),per_term_query_ratio_mean,
                                                      len(query_to_stem_mapping[query_num])]
                        next_index += 1
                # prev_interval = interval
    summary_df.to_csv('/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/Summay_snapshot_stats_' + sim_folder_name + '.tsv',
                            sep='\t', index=False)


def create_tf_dict_for_processed_docs(
        work_interval_freq_list = ['2W','1W', '1M', '2M'],# 'SIM', 'SIM_995'],
        work_year = '2008'):

    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+work_year+'/'

    for interval_freq in work_interval_freq_list:
        print(interval_freq)
        sys.stdout.flush()
        for file_name in os.listdir(os.path.join(processed_docs_folder, interval_freq)):
            if file_name.startswith('clueweb') and file_name.endswith('.json'):
                with open(os.path.join(os.path.join(processed_docs_folder, interval_freq), file_name), 'r') as f:
                    doc_dict = ast.literal_eval(f.read())
                print(file_name)
                sys.stdout.flush()
                for interval in doc_dict:
                    if doc_dict[interval] is None:
                        continue
                    elif 'TfDict' not in doc_dict[interval]:
                        doc_dict[interval]['TfDict'] = {}
                        for i in range(len(doc_dict[interval]['StemList'])):
                            stem = doc_dict[interval]['StemList'][i]
                            tf = doc_dict[interval]['TfList'][i]
                            doc_dict[interval]['TfDict'][stem] = tf

                with open(os.path.join(os.path.join(processed_docs_folder, interval_freq), file_name),'w') as f:
                    f.write(str(doc_dict))


def fill_df_dict_with_doc(
        interval_list,
        df_dict,
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
            df_dict[interval_freq][lookup][curr_interval]['ALL_DOCS_COUNT'] += 1.0
            df_dict[interval_freq][lookup][curr_interval]['AVG_DOC_LEN'] += float(curr_doc_instance['NumWords'])
            for j in range(len(curr_doc_instance['StemList'])):
                curr_stem = curr_doc_instance['StemList'][j]
                if curr_stem in word_ref_dict:
                    if curr_stem in df_dict[interval_freq][lookup][curr_interval]:
                        df_dict[interval_freq][lookup][curr_interval][curr_stem] += 1.0
                    else:
                        df_dict[interval_freq][lookup][curr_interval][curr_stem] = 1.0


def create_per_interval_per_lookup_df_dict(
        work_interval_freq_list=['1W', '2W', '1M', '2M', 'SIM', 'SIM_995','SIM_TXT_UP_DOWN','SIM_TXT_UP','SIM_TXT_DOWN'],
        lookup_method_list=['NoLookup', 'Backward', 'OnlyBackward', 'Forward'],
        already_exists=True,
        work_year = '2008',
        inner_fold= None):

    work_df = create_stemmed_queries_df()
    if work_year != WORK_YEAR:
        raise Exception('work_year not equal to WORK_YEAR')

    if already_exists == False:
        res_dict = {}
        res_dict['RefDict'] = {}
        for index, row in work_df.iterrows():
            query_stems = row['QueryStems'].split(' ')
            for stem in query_stems:
                if stem != '':
                    res_dict['RefDict'][stem] = 0
    else:
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/df_per_interval_dict.json', 'r') as f:
            res_dict = ast.literal_eval(f.read())

    for interval_freq in work_interval_freq_list:
        print(interval_freq)
        sys.stdout.flush()
        if interval_freq in res_dict:
            continue
        res_dict[interval_freq] = {}
        curr_interval_list = build_interval_list(
            work_year=work_year,
            frequency=interval_freq,
            add_clueweb=True)
        if inner_fold == "" or inner_fold is None:
            processed_docs_path = os.path.join(
                '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + work_year + '/',
                interval_freq)
        else:
            processed_docs_path = os.path.join(
                '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+inner_fold+'/'+work_year+'/', interval_freq)
        for lookup_method in lookup_method_list:
            print(lookup_method)
            sys.stdout.flush()
            res_dict[interval_freq][lookup_method] = {}
            for interval in curr_interval_list:
                res_dict[interval_freq][lookup_method][interval] = {}
                res_dict[interval_freq][lookup_method][interval]['ALL_DOCS_COUNT'] = 0
                res_dict[interval_freq][lookup_method][interval]['AVG_DOC_LEN'] = 0.0

            for file_name in os.listdir(processed_docs_path):
                if file_name.startswith('clueweb') and file_name.endswith('.json'):
                    print(file_name)
                    sys.stdout.flush()
                    with open(os.path.join(processed_docs_path, file_name), 'r') as f:
                        doc_dict = ast.literal_eval(f.read())

                    fill_df_dict_with_doc(
                        interval_list=curr_interval_list,
                        interval_freq=interval_freq,
                        df_dict=res_dict,
                        word_ref_dict=res_dict['RefDict'],
                        doc_dict=doc_dict,
                        lookup=lookup_method)
            for interval in curr_interval_list:
                if float(res_dict[interval_freq][lookup_method][interval]['ALL_DOCS_COUNT']) > 0:
                    res_dict[interval_freq][lookup_method][interval]['AVG_DOC_LEN'] = res_dict[interval_freq][lookup_method][interval]['AVG_DOC_LEN'] /float(res_dict[interval_freq][lookup_method][interval]['ALL_DOCS_COUNT'])

        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/df_per_interval_dict.json', 'w') as f:
            f.write(str(res_dict))

def create_text_manipulated_interval(
        from_interval='SIM',
        work_year='2008',
        sim_folder_name="SIM_",
        limit_to_clueweb_len = True,
        fill_to_clueweb_len = True
        ):


    processed_docs_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + work_year + '/'

    processed_docs_input_path = os.path.join(
        processed_docs_folder, from_interval)
    stopword_list = get_stopword_list()

    for file_name in os.listdir(processed_docs_input_path):
        if file_name.startswith('clueweb') and file_name.endswith('.json'):
            print(file_name)
            sys.stdout.flush()
            res_dict = {}
            with open(os.path.join(processed_docs_input_path, file_name), 'r') as f:
                doc_dict = ast.literal_eval(f.read())

            for interval in doc_dict:
                if doc_dict[interval] is not None:
                    if interval == 'ClueWeb09':
                        res_dict[interval] = doc_dict[interval]
                    else:
                        build_snapshot_dict = False
                        if limit_to_clueweb_len == True:
                            if doc_dict[interval]['NumWords'] > doc_dict['ClueWeb09']['NumWords']:
                                tmp_fulltext = doc_dict[interval]['Fulltext'].split(' ')
                                tmp_fulltext = tmp_fulltext[:int(doc_dict['ClueWeb09']['NumWords'])]
                                tmp_stem_list = doc_dict[interval]['StemList']
                                tmp_cc_list = doc_dict[interval]['CCList']
                                tmp_df_list = doc_dict[interval]['DfList']
                                build_snapshot_dict = True
                        if fill_to_clueweb_len == True:
                            if doc_dict[interval]['NumWords'] < doc_dict['ClueWeb09']['NumWords']:
                                tmp_fulltext = doc_dict['ClueWeb09']['Fulltext'].split(' ')
                                tmp_fulltext = doc_dict[interval]['Fulltext'].split(' ') + tmp_fulltext[int(doc_dict[interval]['NumWords']):]
                                tmp_stem_list = doc_dict[interval]['StemList'] + doc_dict['ClueWeb09']['StemList']
                                tmp_cc_list = doc_dict[interval]['CCList'] + doc_dict['ClueWeb09']['CCList']
                                tmp_df_list = doc_dict[interval]['DfList'] + doc_dict['ClueWeb09']['DfList']
                                build_snapshot_dict = True
                        if build_snapshot_dict == True:
                            res_dict[interval] = create_full_doc_dict_from_fulltext(
                                curr_fulltext_list=tmp_fulltext,
                                concatenated_stem_list=tmp_stem_list,
                                concatenated_df_list=tmp_df_list,
                                concatenated_cc_list=tmp_cc_list,
                                stopword_list=stopword_list)
                        else:
                            res_dict[interval] = doc_dict[interval]
                else:
                    res_dict[interval] = None

            with open(os.path.join(os.path.join(processed_docs_folder, sim_folder_name), file_name), 'w') as f:
                f.write(str(res_dict))


def create_snapshot_changes_stats_plots(
        filename='Summay_snapshot_stats_1W.tsv',
        comp_queries_list=[195, 193, 188, 180, 177, 161, 144, 59, 51, 45, 36, 34, 32, 29, 11, 10,
                           9, 2, 78, 4, 167, 69, 166, 33, 164, 18, 182, 17, 98, 124, 48],
        only_relevant_docs=False,
        for_paper=True):
    filename_for_save = filename.replace('.tsv', '').replace('Summay_snapshot_stats_', '')

    work_df = pd.read_csv(os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/', filename), sep='\t', index_col=False)
    if only_relevant_docs == True:
        rel_df = pd.read_csv('qrels_adhoc_relevant_docs.tsv', sep='\t', index_col=False)
        rel_df['Relevance'] = rel_df['Relevance'].apply(lambda x: int(x))
        rel_df = rel_df[rel_df['Relevance'] > 0]
        work_df = pd.merge(
            work_df,
            rel_df[['Docno', 'Query']].rename(columns={'Query': 'QueryNum'}),
            on=['Docno', 'QueryNum'],
            how='inner')
        print("Len after relevant filter " + str(len(work_df)))
        filename = filename.replace('.tsv', '_Relvant_only.tsv')
        filename_for_save += '_Relvant_only'
    # calc ratio measures
    work_df['QueryTermsRatio'] = work_df.apply(lambda row: row['QueryWords'] / float(row['TextLen'] - row['QueryWords']), axis=1)
    work_df['StopwordsRatio'] = work_df.apply(lambda row: row['#Stopword'] / float(row['TextLen'] - row['#Stopword']), axis=1)
    no_clueweb_df = work_df[work_df['Interval'] != 'ClueWeb09'].copy()
    clueweb_df = work_df[work_df['Interval'] == 'ClueWeb09'].copy()

    print("Mean ClueWeb09 SimToPrev:")
    print(clueweb_df[clueweb_df['SimToPrev'].notnull()]['SimToPrev'].mean())
    print("Mean ClueWeb09 SimToClosePrev:")
    print(clueweb_df[clueweb_df['SimToClosePrev'].notnull()]['SimToClosePrev'].mean())
    print("Med ClueWeb09 SimToPrev:")
    print(clueweb_df[clueweb_df['SimToPrev'].notnull()]['SimToPrev'].median())
    print("Med ClueWeb09 SimToClosePrev:")
    print(clueweb_df[clueweb_df['SimToClosePrev'].notnull()]['SimToClosePrev'].median())
    print("Mean all SimToPrev:")
    print(work_df[work_df['SimToPrev'].notnull()]['SimToClosePrev'].mean())
    print("Med all SimToPrev:")
    print(work_df[work_df['SimToPrev'].notnull()]['SimToPrev'].median())
    print("Mean not ClueWeb09 SimToClueWeb:")
    print(no_clueweb_df['SimToClueWeb'].mean())
    print("Med not ClueWeb09 SimToClueWeb:")
    print(no_clueweb_df['SimToClueWeb'].median())
    sys.stdout.flush()
    all_docno_and_queries_df = work_df[['Docno', 'QueryNum']].drop_duplicates()
    only_clueweb_num = 0

    summary_df = pd.DataFrame(
        columns=['Interval', 'QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb', 'QueryNum'])
    next_index = 0
    exact_mathchs = 0
    # exact_mathch_df = pd.DataFrame(columns = ['Docno'])
    # exact_mathch_index = 0
    for index, row_ in all_docno_and_queries_df.iterrows():
        doc_df = work_df[work_df['Docno'] == row_['Docno']].copy()
        doc_df = doc_df[doc_df['QueryNum'] == row_['QueryNum']].copy()
        if len(doc_df) == 1:
            only_clueweb_num += 1
            continue
        bench_df = doc_df[doc_df['Interval'] == 'ClueWeb09']
        bench_query_term_ratio = list(bench_df['QueryTermsRatio'])[0]
        bench_stopword_ratio = list(bench_df['StopwordsRatio'])[0]
        bench_entropy_ratio = list(bench_df['Entropy'])[0]
        if bench_stopword_ratio == 0:
            bench_stopword_ratio = 1.0
        if bench_query_term_ratio == 0:
            bench_query_term_ratio = 1.0
        for index, row in doc_df.iterrows():
            if row['Interval'] != 'ClueWeb09':
                interval = row['Interval']
                query_term_ratio = (row['QueryTermsRatio'] - bench_query_term_ratio) / float(bench_query_term_ratio)
                stopword_ratio = (row['StopwordsRatio'] - bench_stopword_ratio) / float(bench_stopword_ratio)
                entropy_ratio = (row['Entropy'] - bench_entropy_ratio) / float(bench_entropy_ratio)
                sim_cluweb = row['SimToClueWeb']
                summary_df.loc[next_index] = [interval, query_term_ratio, stopword_ratio, entropy_ratio, sim_cluweb,
                                              int(row_['QueryNum'])]
                next_index += 1
            else:
                if row['SimToPrev'] == 1.0:
                    exact_mathchs += 1
                    # exact_mathch_df.loc[exact_mathch_index] = [row_['Docno']]
                    # exact_mathch_index += 1

    exact_mathch_df = work_df[work_df['Interval'] != 'ClueWeb09'].copy()
    exact_mathch_df = exact_mathch_df[exact_mathch_df['SimToClueWeb'] == 1.0][['Docno']].drop_duplicates()
    # exact_mathch_df.to_csv(filename.replace('.tsv', '_Exact_matched_docnos.tsv'), sep='\t', index=False)
    print("Num of one snapshot:")
    print(only_clueweb_num)
    print("Num of Exact Match:")
    print(len(exact_mathch_df))
    if 'SIM' in filename:
        summary_df['Interval'] = summary_df['Interval'].apply(lambda x: 0 if x == 'ClueWeb09' else int(x))
    summary_df['ExactMatchs'] = summary_df['SimClueWeb'].apply(lambda x: 1 if x == 1.0 else 0)
    plot_df = summary_df[['Interval', 'ExactMatchs']].groupby(['Interval']).sum()
    plt.cla()
    plt.clf()
    plot_df.plot(kind='bar', color='c', legend=False)
    plt.xlabel('Interval')
    plt.tick_params(axis='x', labelsize=5)
    plt.xticks(rotation=45)
    plt.ylabel("#Exact Matches")
    plt.title("#Exact Matches Per Interval")
    plt.savefig(filename_for_save + "_Exact_Matches_per_interval.png", dpi=300)
    summary_df['IsCompQuery'] = summary_df['QueryNum'].apply(lambda x: 1 if x in comp_queries_list else 0)
    interval_quantity_df = summary_df[['Interval', 'QueryNum']].groupby(['Interval']).count()
    plt.cla()
    plt.clf()
    interval_quantity_df.plot(kind='bar', color='r', legend=False)
    plt.xlabel('Interval')
    plt.tick_params(axis='x', labelsize=5)
    plt.xticks(rotation=45)
    plt.ylabel("#Snapshots")
    plt.title("#Snapshots Per Interval")
    plt.savefig(filename_for_save + "_Snapshots_per_interval.png", dpi=300)
    for measure in ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb']:
        plt.cla()
        plt.clf()
        plot_df = summary_df[['Interval', measure]].groupby(['Interval']).mean()
        plot_df.rename(columns={measure: 'ALL'}, inplace=True)
        plot_df_ = summary_df[summary_df['IsCompQuery'] == 1][['Interval', measure]].groupby(['Interval']).mean()
        plot_df_.rename(columns={measure: 'Comp Queries'}, inplace=True)
        plot_df = pd.merge(
            plot_df,
            plot_df_,
            left_index=True,
            right_index=True)
        plot_df_ = summary_df[summary_df['IsCompQuery'] == 0][['Interval', measure]].groupby(['Interval']).mean()
        plot_df_.rename(columns={measure: 'Non Comp'}, inplace=True)
        plot_df = pd.merge(
            plot_df,
            plot_df_,
            left_index=True,
            right_index=True)

        plot_df.plot(kind='bar')
        plt.xlabel('Interval')
        plt.tick_params(axis='x', labelsize=6)
        plt.xticks(rotation=45)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.75, bottom=0.15)
        if measure != 'SimClueWeb':
            plt.ylabel("%" + measure)
            plt.title("Mean " + measure + " Precentage Difference from ClueWeb09")
        else:
            plt.ylabel(measure)
            plt.title("Mean " + measure)

        plt.savefig(measure + filename + "_Precentage_Difference_per_interval.png", dpi=300)
        # plot_df.to_csv(os.path.join('plot_df', filename_for_save + '_' + measure + '.tsv'), sep='\t')
        if for_paper == True:
            plt.cla()
            plt.clf()
            if measure == 'SimClueWeb':
                tmp_df = plot_df.reset_index()
                # plot_df[['ALL']].plot(kind = 'bar', color = 'c', legend=False)
                plt.plot(pd.to_datetime(tmp_df['Interval']), tmp_df['ALL'], color='c')
                plt.xlabel('Interval')
                plt.subplots_adjust(bottom=0.15)
            else:
                plot_df[['ALL']].plot(kind='bar', color='c', legend=False)
                plt.xlabel('Snapshot Number')
            plt.tick_params(axis='x', labelsize=6)
            plt.xticks(rotation=45)
            plt.savefig(measure + filename + "_Precentage_Difference_per_interval_For_Paper.png", dpi=300)


def plot_retrival_stats(
        lookup_method,
        interval_freq,
        filename,
        comp_queries_list=[195, 193, 188, 180, 177, 161, 144, 59, 51, 45, 36, 34, 32, 29, 11, 10,
                           9, 2, 78, 4, 167, 69, 166, 33, 164, 18, 182, 17, 98, 124, 48]
        ):

    work_df = pd.read_csv(
        os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/results/retrival_stats/', filename), sep='\t',
        index_col=False)
    work_df['IsCompQuery'] = work_df['Query_Num'].apply(lambda x: 1 if x in comp_queries_list else 0)
    if interval_freq.startswith('SIM'):
        work_df['Interval'] = work_df['Interval'].apply(lambda x: 0 if x == 'ClueWeb09' else int(x))

    save_df = pd.DataFrame({})
    for measure in ['Map', 'P@5', 'P@10']:
        plt.cla()
        plt.clf()
        plot_df = work_df[['Interval', measure]].groupby(['Interval']).mean()
        plot_df.rename(columns={measure: 'ALL'}, inplace=True)
        plot_df_ = work_df[work_df['IsCompQuery'] == 1][['Interval', measure]].groupby(['Interval']).mean()
        plot_df_.rename(columns={measure: 'Comp Queries'}, inplace=True)
        plot_df = pd.merge(
            plot_df,
            plot_df_,
            left_index=True,
            right_index=True)
        plot_df_ = work_df[work_df['IsCompQuery'] == 0][['Interval', measure]].groupby(['Interval']).mean()
        plot_df_.rename(columns={measure: 'Non Comp'}, inplace=True)
        plot_df = pd.merge(
            plot_df,
            plot_df_,
            left_index=True,
            right_index=True)
        measure = measure.replace('Ext_', '')

        plot_df.plot(kind='bar')
        plt.xlabel('Interval')
        plt.tick_params(axis='x', labelsize=6)
        plt.xticks(rotation=45)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.75, bottom=0.15)

        plt.ylabel(measure)
        plt.title(lookup_method + " Mean " + measure)

        plt.savefig(filename.replace('.tsv', '') + measure + "_per_interval.png", dpi=300)
        for col in list(plot_df.columns):
            plot_df.rename(columns={col: measure + ' ' + col}, inplace=True)
        if save_df.empty == True:
            save_df = plot_df
        else:
            save_df = pd.merge(
                save_df,
                plot_df,
                left_index=True,
                right_index=True)
    # save_df.to_csv(os.path.join('plot_df', afix + interval_freq + '_' + lookup_method + addition + '_Retrival_Stats.tsv'),
    #                sep='\t')


def plot_stats_vs_winner_plots_for_wl_file(
        file_name,
        comp_queries_list=[195, 193, 188, 180, 177, 161, 144, 59, 51, 45, 36, 34, 32, 29, 11, 10,
                           9, 2, 78, 4, 167, 69, 166, 33, 164, 18, 182, 17, 98, 124, 48],
        interval_freq='2M',
        lookup_method='NoLookup',
        only_relevant_docs=False,
        for_paper=True):


    work_df = pd.read_csv(file_name, sep='\t', index_col=False)

    if only_relevant_docs == True:
        rel_df = pd.read_csv('qrels_adhoc_relevant_docs.tsv', sep='\t', index_col=False)
        rel_df['Relevance'] = rel_df['Relevance'].apply(lambda x: int(x))
        rel_df = rel_df[rel_df['Relevance'] > 0]
        work_df = pd.merge(
            work_df,
            rel_df[['Docno', 'Query']].rename(columns={'Query': 'QueryNum'}),
            on=['Docno', 'QueryNum'],
            how='inner')
        print("Len after relevant filter " + str(len(work_df)))
        file_name = file_name.replace('.tsv', '_Relvant_only.tsv')

    winner_df = work_df[work_df['Winner'] == True]
    non_winner_df = work_df[work_df['Winner'] == False]

    work_df['IsCompQuery'] = work_df['QueryNum'].apply(lambda x: 1 if x in comp_queries_list else 0)
    comp_df = work_df[work_df['IsCompQuery'] == 1]
    comp_winner_df = comp_df[comp_df['Winner'] == True]
    comp_non_winner_df = comp_df[comp_df['Winner'] == False]

    plt.cla()
    plt.clf()

    # add non/winner df docnos round zero scores
    # add won in round 0 df to plot
    round_0_winner_df = work_df[work_df['Round'] == 0]
    round_0_winner_df = round_0_winner_df[round_0_winner_df['Winner'] == True]
    for index, row in round_0_winner_df.iterrows():
        winner_doc_df = work_df[work_df['Docno'] == row['Docno']].copy()
        winner_doc_df = winner_doc_df[winner_doc_df['QueryNum'] == row['QueryNum']]
        winner_doc_df = winner_doc_df[winner_doc_df['Round'] < row['Round']]
        round_0_winner_df = round_0_winner_df.append(winner_doc_df, ignore_index=True)

    comp_round_0_winner_df = comp_df[comp_df['Round'] == 0]
    comp_round_0_winner_df = comp_round_0_winner_df[comp_round_0_winner_df['Winner'] == True]
    for index, row in comp_round_0_winner_df.iterrows():
        winner_doc_df = comp_df[comp_df['Docno'] == row['Docno']].copy()
        winner_doc_df = winner_doc_df[winner_doc_df['QueryNum'] == row['QueryNum']]
        winner_doc_df = winner_doc_df[winner_doc_df['Round'] < row['Round']]
        comp_round_0_winner_df = comp_round_0_winner_df.append(winner_doc_df, ignore_index=True)

    winner_df = work_df[work_df['Trend'] == 'W']
    winner_docs = list(winner_df['Docno'].drop_duplicates())
    round_0_w_df = work_df[work_df['Round'] == 0]
    round_0_w_df['filter'] = round_0_w_df['Docno'].apply(lambda x: True if x not in winner_docs else False)
    round_0_w_df = round_0_w_df[round_0_w_df['filter'] == False]
    del round_0_w_df['filter']
    winner_df = winner_df.append(round_0_w_df, ignore_index=True)

    non_winner_df = work_df[work_df['Trend'] == 'L']
    loser_docs = list(non_winner_df['Docno'].drop_duplicates())
    round_0_l_df = work_df[work_df['Round'] == 0]
    round_0_l_df['filter'] = round_0_l_df['Docno'].apply(lambda x: True if x not in loser_docs else False)
    round_0_l_df = round_0_l_df[round_0_l_df['filter'] == False]
    del round_0_l_df['filter']
    non_winner_df = non_winner_df.append(round_0_l_df, ignore_index=True)

    comp_winner_df = comp_df[comp_df['Trend'] == 'W']
    winner_docs = list(comp_winner_df['Docno'].drop_duplicates())
    round_0_w_df = comp_df[comp_df['Round'] == 0]
    round_0_w_df['filter'] = round_0_w_df['Docno'].apply(lambda x: True if x not in winner_docs else False)
    round_0_w_df = round_0_w_df[round_0_w_df['filter'] == False]
    del round_0_w_df['filter']
    comp_winner_df = comp_winner_df.append(round_0_w_df, ignore_index=True)

    comp_non_winner_df = comp_df[comp_df['Trend'] == 'L']
    loser_docs = list(comp_non_winner_df['Docno'].drop_duplicates())
    round_0_l_df = comp_df[comp_df['Round'] == 0]
    round_0_l_df['filter'] = round_0_l_df['Docno'].apply(lambda x: True if x not in loser_docs else False)
    round_0_l_df = round_0_l_df[round_0_l_df['filter'] == False]
    del round_0_l_df['filter']
    comp_non_winner_df = comp_non_winner_df.append(round_0_l_df, ignore_index=True)

    n_plots = 2
    f, axarr = plt.subplots(n_plots, n_plots)
    idx_r = 0
    idx_c = 0
    for measure in ['Sim_PW', 'QueryTermsRatio', 'StopwordsRatio', 'Entropy']:  # Sim_PW
        plot_df = winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='b', label='ALL_Up')
        plot_df = non_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='b', linestyle='--', label='ALL_Down')
        plot_df = round_0_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='b', linestyle=':', label='ALL_Winner',
                                 alpha=0.7)

        plot_df = comp_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='r', label='Comp_Up')
        plot_df = comp_non_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='r', linestyle='--', label='Comp_Down')
        plot_df = comp_round_0_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='r', linestyle=':', label='Comp_Winner',
                                 alpha=0.7)
        axarr[idx_r, idx_c].set_title(measure)

        if idx_r == (n_plots - 1):
            idx_r = 0
            idx_c += 1
            continue

        idx_r += 1

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75)
    print(file_name)
    plt.savefig(file_name.replace('.tsv', '_WL.png'), dpi=300)
    if for_paper == True:
        n_plots = 2
        f, axarr = plt.subplots(n_plots, n_plots)
        idx_r = 0
        idx_c = 0
        for measure in ['Sim_PW', 'QueryTermsRatio', 'StopwordsRatio', 'Entropy']:  # Sim_PW
            plot_df = winner_df[['Round', measure]].groupby(['Round']).mean()
            plot_df = plot_df.reset_index()
            axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='b', marker='o', label='H', alpha=0.7)
            plot_df = non_winner_df[['Round', measure]].groupby(['Round']).mean()
            plot_df = plot_df.reset_index()
            axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='r', marker='o', label='L', alpha=0.7)
            plot_df = round_0_winner_df[['Round', measure]].groupby(['Round']).mean()
            plot_df = plot_df.reset_index()
            axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='k', marker='o', label='W',
                                     alpha=0.7)
            axarr[idx_r, idx_c].set_title(measure.replace("_", ""), fontsize=10)
            axarr[idx_r, idx_c].tick_params(axis='x', labelsize=6)
            if idx_r == (n_plots - 1):
                idx_r = 0
                idx_c += 1
                continue

            idx_r += 1
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.75)
        plt.savefig(interval_freq + "_" + lookup_method + "_WL_STATS.png", dpi=300)

def add_stopword_stats_to_cc_dict(
        interval_freq_list,
        lookup_method_list = ['NoLookup', 'Backward','OnlyBackward','Forward'],
        inner_fold = '50_per_q',
        work_year = '2008'):

    if inner_fold == "":
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/cc_per_interval_dict.json', 'r') as f:
            res_dict = ast.literal_eval(f.read())
    else:
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + inner_fold + '/cc_per_interval_dict.json', 'r') as f:
            res_dict = ast.literal_eval(f.read())

    stopword_list = get_stopword_list()
    for interval_freq in interval_freq_list:
        print(interval_freq)
        sys.stdout.flush()
        curr_interval_list = build_interval_list(
            work_year=work_year,
            frequency=interval_freq,
            add_clueweb=True)
        if inner_fold == "":
            processed_docs_path = os.path.join(
                '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + work_year + '/',
                interval_freq)
        else:
            processed_docs_path = os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + inner_fold + '/'+work_year+'/',
                                               interval_freq)
        for lookup_method in lookup_method_list:
            print(lookup_method)
            sys.stdout.flush()
            for interval in curr_interval_list:
                res_dict[interval_freq][lookup_method][interval]['ALL_SW_COUNT'] = 0

            for file_name in os.listdir(processed_docs_path):
                if file_name.startswith('clueweb') and file_name.endswith('.json'):
                    print(file_name)
                    sys.stdout.flush()
                    with open(os.path.join(processed_docs_path, file_name), 'r') as f:
                        doc_dict = ast.literal_eval(f.read())
                    for interval_idx in range(len(curr_interval_list)):
                        curr_interval = curr_interval_list[interval_idx]
                        curr_doc_instance = get_doc_snapshot_by_lookup_method(
                            doc_dict=doc_dict,
                            interval_list=curr_interval_list,
                            interval_lookup_method=lookup_method,
                            curr_interval_idx=interval_idx)
                        if curr_doc_instance is not None:
                            for term in stopword_list:
                                if term in curr_doc_instance['TfDict']:
                                    res_dict[interval_freq][lookup_method][curr_interval]['ALL_SW_COUNT'] += curr_doc_instance['TfDict'][term]
    if inner_fold == "":
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/cc_per_interval_dict.json', 'w') as f:
            f.write(str(res_dict))
    else:
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + inner_fold + '/cc_per_interval_dict.json', 'w') as f:
            f.write(str(res_dict))


def add_stopword_stats_to_df_dict(
        interval_freq_list,
        lookup_method_list=['NoLookup', 'Backward', 'OnlyBackward', 'Forward'],
        inner_fold='50_per_q',
        work_year='2008'):

    if inner_fold == "":
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/df_per_interval_dict.json', 'r') as f:
            res_dict = ast.literal_eval(f.read())
    else:
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + inner_fold + '/df_per_interval_dict.json', 'r') as f:
            res_dict = ast.literal_eval(f.read())

    for interval_freq in interval_freq_list:
        print(interval_freq)
        sys.stdout.flush()
        curr_interval_list = build_interval_list(
            work_year=work_year,
            frequency=interval_freq,
            add_clueweb=True)
        if inner_fold == "":
            processed_docs_path = os.path.join(
                '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + work_year + '/',
                interval_freq)
        else:
            processed_docs_path = os.path.join(
                '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + inner_fold + '/'+work_year+'/',
                interval_freq)
        for lookup_method in lookup_method_list:
            print(lookup_method)
            sys.stdout.flush()
            for interval in curr_interval_list:
                res_dict[interval_freq][lookup_method][interval]['AVG_DOC_LEN_NO_SW'] = 0

            for file_name in os.listdir(processed_docs_path):
                if file_name.startswith('clueweb') and file_name.endswith('.json'):
                    print(file_name)
                    sys.stdout.flush()
                    with open(os.path.join(processed_docs_path, file_name), 'r') as f:
                        doc_dict = ast.literal_eval(f.read())
                    for interval_idx in range(len(curr_interval_list)):
                        curr_interval = curr_interval_list[interval_idx]
                        curr_doc_instance = get_doc_snapshot_by_lookup_method(
                            doc_dict=doc_dict,
                            interval_list=curr_interval_list,
                            interval_lookup_method=lookup_method,
                            curr_interval_idx=interval_idx)
                        if curr_doc_instance is not None:
                            res_dict[interval_freq][lookup_method][curr_interval]['AVG_DOC_LEN_NO_SW'] += float(curr_doc_instance['NumWords'] - curr_doc_instance['NumStopWords'])
            for interval in curr_interval_list:
                if float(res_dict[interval_freq][lookup_method][interval]['ALL_DOCS_COUNT']) > 0:
                    res_dict[interval_freq][lookup_method][interval]['AVG_DOC_LEN_NO_SW'] = \
                    res_dict[interval_freq][lookup_method][interval]['AVG_DOC_LEN_NO_SW'] / float(
                        res_dict[interval_freq][lookup_method][interval]['ALL_DOCS_COUNT'])
    if inner_fold == "":
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/df_per_interval_dict.json', 'w') as f:
            f.write(str(res_dict))
    else:
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + inner_fold + '/df_per_interval_dict.json', 'w') as f:
            f.write(str(res_dict))

def plot_interesting_stats_for_avg_model_results(
        frequency,
        retrival_model,
        interval_start_month,
        filter_params,
        sw_rmv,
        work_year):
    affix =""
    if retrival_model == 'BM25':
        affix += "BM25_"

    elif retrival_model == 'LM':
        affix += ""
    else:
        raise Exception("Unknown model")

    addition = ""
    if interval_start_month != 1:
        addition = "_" + str(interval_start_month) + "SM_"

    if filter_params is not None and len(filter_params) > 0:
        addition += create_filter_params_txt_addition(filter_params)

    if sw_rmv == True:
        addition += "_SW_RMV"

    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/avg_model_res/'
    cv_summary_df = pd.DataFrame({})
    cv_test_df = pd.DataFrame({})
    for filename in os.listdir(save_folder):
        if filename.endswith('_'+affix + frequency + '_' + addition + "_" + work_year+ "_Results.tsv"):
            if retrival_model == 'LM' and 'BM25' in filename:
                continue
            print(filename)
            cv_summary_df_tmp = pd.read_csv(os.path.join(save_folder, filename), sep='\t',
                                 index_col=False)
            last_idx = list(cv_summary_df_tmp.index)[-1]
            cv_test_df = cv_test_df.append(cv_summary_df_tmp.tail(1), ignore_index = True)
            cv_summary_df_tmp.drop(last_idx, inplace = True)
            cv_summary_df = cv_summary_df.append(cv_summary_df_tmp)

    interval_list = build_interval_list(work_year, frequency, add_clueweb=True, start_month=interval_start_month)

    corr_plot_df = pd.DataFrame(columns = ['Interval','Map','P@5','P@10'])
    next_idx = 0
    for interval in interval_list:
        insert_row = [interval.replace('ClueWeb09','ClueWeb')]
        insert_row.append(cv_summary_df[interval].corr(cv_summary_df['Map']))
        insert_row.append(cv_summary_df[interval].corr(cv_summary_df['P@5']))
        insert_row.append(cv_summary_df[interval].corr(cv_summary_df['P@10']))
        corr_plot_df.loc[next_idx] = insert_row
        next_idx += 1

    corr_plot_df.set_index('Interval', inplace = True)
    corr_plot_df.plot(kind='bar')
    plt.xlabel('Interval')
    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=45)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75, bottom=0.15)
    plt.ylabel('Correlation')
    plt.title("Interval Effectivness Corr")
    plt.savefig( affix + frequency + '_' + addition + "_" + work_year+ "_Per_Inteval_effectiness_corr.png", dpi=300)

    cw_df = cv_summary_df[['ClueWeb09','Map','P@5','P@10']].groupby(['ClueWeb09']).mean()
    cw_df[['Map']].plot(kind='bar')
    plt.xlabel('ClueWeb Weight Value')
    plt.ylim((cw_df['Map'].min() - 0.02, None))
    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=45)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75, bottom=0.15)
    plt.ylabel('Mean Map Over All Non ClueWeb Weight Vals')
    plt.savefig(affix + frequency + '_' + addition +"_" + work_year+  "_Map_Per_CW_Weight.png", dpi=300)

    cw_df[['P@5']].plot(kind='bar')
    plt.xlabel('ClueWeb Weight Value')
    plt.ylim((cw_df['P@5'].min() - 0.02, None))
    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=45)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75, bottom=0.15)
    plt.ylabel('Mean P@5 Over All Non ClueWeb Weight Vals')
    plt.savefig(affix + frequency + '_' + addition +"_" + work_year+  "_P_5_Per_CW_Weight.png", dpi=300)

    cw_df[['P@10']].plot(kind='bar')
    plt.xlabel('ClueWeb Weight Value')
    plt.ylim((cw_df['P@10'].min() - 0.02, None))
    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=45)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75, bottom=0.15)
    plt.ylabel('Mean P@10 Over All Non ClueWeb Weight Vals')
    plt.savefig(affix + frequency + '_' + addition +"_" + work_year+  "_P_10_Per_CW_Weight.png", dpi=300)

    wieght_mean_df = cv_test_df[interval_list].mean()
    wieght_std_df = cv_test_df[interval_list].std()

    wieghts_plot_df = pd.DataFrame(columns = ['Interval', 'MeanWieght', 'StdWeight'])
    next_idx = 0

    for key in list(wieght_mean_df.keys()):
        wieghts_plot_df.loc[next_idx] = [key.replace('ClueWeb09','ClueWeb'), wieght_mean_df[key], wieght_std_df[key]]
        next_idx += 1

    wieghts_plot_df.set_index('Interval', inplace = True)
    wieghts_plot_df.plot(kind='bar')
    plt.xlabel('Interval')
    plt.title('Mean and Std CV Weights Per Interval')
    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=45)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75, bottom=0.15)
    plt.savefig(affix + frequency + '_' + addition + "_" + work_year+ "_Test_Weights_per_interval.png", dpi=300)

    res_df = cv_test_df[['Map','P@5','P@10']].mean()

    fin_df = pd.DataFrame(columns = ['Map','P@5','P@10'])
    fin_df.loc[0] = [res_df['Map'], res_df['P@5'], res_df['P@10']]
    fin_df.to_csv(os.path.join(save_folder,affix + frequency + '_' + addition + "_" + work_year+ "_Test_Results.tsv"), sep = '\t', index = False)

def calc_top_50_stats(
        work_year,
        interval_freq,
        lookup_method,
        retrival_model):

    res_files_path = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/"
    interval_list = build_interval_list(work_year, interval_freq, add_clueweb=False)

    affix = ""
    if retrival_model == 'BM25':
        affix = "BM25_"
    file_end = "_" + interval_freq + "_" + lookup_method + "_Results.txt"

    if int(work_year) <= 2009:
        qrel_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc')
    else:
        qrel_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels_cw12.adhoc')

    qrel_df['Relevance'] = qrel_df['Relevance'].apply(lambda x: 1 if x > 0 else 0)
    summary_df = pd.DataFrame(columns = ['Interval', 'Query', 'P@50', 'Overlap','RBO_0.99','RBO_0.975','RBO_0.95'])
    next_index = 0
    bench_df = convert_trec_results_file_to_pandas_df(os.path.join(res_files_path, affix + 'ClueWeb09' + file_end))
    all_q_list = list(bench_df['Query_ID'].drop_duplicates())
    bench_df['Rank'] = bench_df['Rank'].apply(lambda x: int(x))
    bench_df = bench_df[bench_df['Rank'] <= 50]

    for interval in interval_list:
        print(interval)
        sys.stdout.flush()
        curr_df = convert_trec_results_file_to_pandas_df(os.path.join(res_files_path, affix + interval + file_end))
        curr_df['Rank'] = curr_df['Rank'].apply(lambda x: int(x))
        curr_df = curr_df[curr_df['Rank'] <= 50]

        for q in all_q_list:
            tmp_q_df = curr_df[curr_df['Query_ID'] == q].copy()
            tmp_b_df = bench_df[bench_df['Query_ID'] == q].copy()

            tmp_overlap = overlap(list(tmp_b_df['Docno']),list(tmp_q_df['Docno']), 50)
            rbo_0_99    = rbo(list(tmp_b_df['Docno']),list(tmp_q_df['Docno']), p=0.99)['ext']
            rbo_0_975   = rbo(list(tmp_b_df['Docno']), list(tmp_q_df['Docno']), p=0.975)['ext']
            rbo_0_95    = rbo(list(tmp_b_df['Docno']), list(tmp_q_df['Docno']), p=0.95)['ext']

            tmp_rel_df = pd.merge(
                tmp_q_df,
                qrel_df.rename(columns = {'Query' : 'Query_ID'}),
                on = ['Query_ID', 'Docno'],
                how ='inner')

            if tmp_rel_df.empty == True:
                p_50 = 0.0
            else:
                p_50 = float(tmp_rel_df['Relevance'].sum()) / 50.0
            summary_df.loc[next_index] = [interval, q, p_50, tmp_overlap,rbo_0_99,rbo_0_975,rbo_0_95 ]
            next_index += 1

    summary_df.to_csv(os.path.join(os.path.join(res_files_path,'top_50_data'), affix + 'Top_50_data' + file_end.replace('.txt','.tsv')), sep = '\t', index=False)

def create_multi_year_snapshot_file(
        year_list,
        last_interval,
        interval_freq,
        inner_fold):

    data_path = "/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/"
    if inner_fold != "":
        data_path += inner_fold + "/"

    year_df_dict = {}
    doc_query_df = pd.DataFrame({})
    for year in year_list:
        year_df_dict[year] = pd.read_csv(os.path.join(data_path, 'Summay_snapshot_stats_'+interval_freq+'_'+year+'.tsv'), sep ='\t', index_col =False)
        doc_query_df = doc_query_df.append(year_df_dict[year][['Docno','QueryNum']].drop_duplicates(), ignore_index = True)
        year_df_dict[year]['QueryTermsRatio'] = year_df_dict[year].apply(lambda row: row['QueryWords'] / float(row['TextLen'] - row['QueryWords']),
                                                   axis=1)
        year_df_dict[year]['StopwordsRatio'] = year_df_dict[year].apply(lambda row: row['#Stopword'] / float(row['TextLen'] - row['#Stopword']),
                                              axis=1)

    fin_df = pd.DataFrame(columns = ['Interval', 'QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb', 'QueryNum', 'Docno'])
    next_index = 0
    doc_query_df = doc_query_df.drop_duplicates()
    for index, row in doc_query_df.iterrows():
        docno = row['Docno']
        query = row['QueryNum']
        print(docno)
        sys.stdout.flush()
        first = True
        tmp_doc_df = pd.DataFrame({})
        for year in list(reversed(year_list)):
            tmp_doc_df_y = year_df_dict[year][year_df_dict[year]['Docno'] == docno].copy()
            tmp_doc_df_y = tmp_doc_df_y[tmp_doc_df_y['QueryNum'] == query]
            if first == True:
                first = False
                if last_interval is not None:
                    tmp_doc_df_y['Filter'] = tmp_doc_df_y['Interval'].apply(lambda x: 1 if x > last_interval and x != 'ClueWeb09' else 0)
                    tmp_doc_df_y = tmp_doc_df_y[tmp_doc_df_y['Filter'] == 0]
                    del tmp_doc_df_y['Filter']
            else:
                tmp_doc_df_y = tmp_doc_df_y[tmp_doc_df_y['Interval'] != 'ClueWeb09']

            tmp_doc_df = tmp_doc_df_y.append(tmp_doc_df, ignore_index = True)
        if len(tmp_doc_df) > 1:
            min_snap = (len(tmp_doc_df) - 1)*(-1)
            bench_df = tmp_doc_df[tmp_doc_df['Interval'] == 'ClueWeb09']
            bench_query_term_ratio = list(bench_df['QueryTermsRatio'])[0]
            bench_stopword_ratio = list(bench_df['StopwordsRatio'])[0]
            bench_entropy_ratio = list(bench_df['Entropy'])[0]
            if bench_stopword_ratio == 0:
                bench_stopword_ratio = 1.0
            if bench_query_term_ratio == 0:
                bench_query_term_ratio = 1.0
            for index, row in tmp_doc_df.iterrows():
                if row['Interval'] != 'ClueWeb09':
                    if row['TextLen'] <= 2:
                        query_term_ratio = pd.np.nan
                        stopword_ratio = pd.np.nan
                        entropy_ratio = pd.np.nan
                        sim_cluweb = pd.np.nan

                    else:
                        query_term_ratio = (row['QueryTermsRatio'] - bench_query_term_ratio) / float(
                            bench_query_term_ratio)
                        stopword_ratio = (row['StopwordsRatio'] - bench_stopword_ratio) / float(bench_stopword_ratio)
                        entropy_ratio = (row['Entropy'] - bench_entropy_ratio) / float(bench_entropy_ratio)
                        sim_cluweb = row['SimToClueWeb']

                    insert_row = [min_snap, query_term_ratio, stopword_ratio, entropy_ratio, sim_cluweb,
                                  int(query), docno]
                    min_snap += 1
                    fin_df.loc[next_index] = insert_row
                    next_index += 1
    file_end = ""
    if last_interval is not None:
        file_end += last_interval
    for year in year_list:
        file_end += "_" + year
    fin_df.to_csv(os.path.join(data_path, interval_freq + "_" +file_end + "United_summary.tsv"), sep = '\t', index = False)


def create_snapshot_changes_rev_vs_non_rel(
        filename='Summay_snapshot_stats_1M.tsv',
        work_year='2011',
        inner_fold = "",
        preprocessed = False):

    path_to_files = '/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/'
    filename_for_save = filename.replace('.tsv', '').replace('Summay_snapshot_stats_', '')
    if preprocessed == False:
        if inner_fold != "" and inner_fold is not None:
            work_df = pd.read_csv(os.path.join(path_to_files, os.path.join(inner_fold,filename)), sep='\t', index_col=False)
        else:
            work_df = pd.read_csv(os.path.join(path_to_files, filename), sep='\t', index_col=False)

        # calc ratio measures
        work_df['QueryTermsRatio'] = work_df.apply(lambda row: row['QueryWords'] / float(row['TextLen'] - row['QueryWords']),
                                                   axis=1)
        work_df['StopwordsRatio'] = work_df.apply(lambda row: row['#Stopword'] / float(row['TextLen'] - row['#Stopword']),
                                                  axis=1)

        all_docno_and_queries_df = work_df[['Docno', 'QueryNum']].drop_duplicates()
        only_clueweb_num = 0

        summary_df = pd.DataFrame(
            columns=['Interval', 'QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb', 'QueryNum','Docno'])
        next_index = 0

        for index, row_ in all_docno_and_queries_df.iterrows():
            doc_df = work_df[work_df['Docno'] == row_['Docno']].copy()
            doc_df = doc_df[doc_df['QueryNum'] == row_['QueryNum']].copy()
            if len(doc_df) == 1:
                only_clueweb_num += 1
                continue
            bench_df = doc_df[doc_df['Interval'] == 'ClueWeb09']
            bench_query_term_ratio = list(bench_df['QueryTermsRatio'])[0]
            bench_stopword_ratio = list(bench_df['StopwordsRatio'])[0]
            bench_entropy_ratio = list(bench_df['Entropy'])[0]
            if bench_stopword_ratio == 0:
                bench_stopword_ratio = 1.0
            if bench_query_term_ratio == 0:
                bench_query_term_ratio = 1.0
            for index, row in doc_df.iterrows():
                if row['Interval'] != 'ClueWeb09':
                    interval = row['Interval']
                    if row['TextLen'] <= 2:
                        query_term_ratio = pd.np.nan
                        stopword_ratio = pd.np.nan
                        entropy_ratio = pd.np.nan
                        sim_cluweb = pd.np.nan

                    else:
                        query_term_ratio = (row['QueryTermsRatio'] - bench_query_term_ratio) / float(bench_query_term_ratio)
                        stopword_ratio = (row['StopwordsRatio'] - bench_stopword_ratio) / float(bench_stopword_ratio)
                        entropy_ratio = (row['Entropy'] - bench_entropy_ratio) / float(bench_entropy_ratio)
                        sim_cluweb = row['SimToClueWeb']

                    insert_row = [interval, query_term_ratio, stopword_ratio, entropy_ratio, sim_cluweb, int(row_['QueryNum']),row_['Docno']]
                    summary_df.loc[next_index] = insert_row
                    next_index += 1
    else:
        if inner_fold != "" and inner_fold is not None:
            summary_df = pd.read_csv(os.path.join(os.path.join(path_to_files,inner_fold), filename), sep='\t', index_col=False)
        else:
            summary_df = pd.read_csv(os.path.join(path_to_files, filename), sep='\t', index_col=False)
    if int(work_year) >= 2010:
        rel_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels_cw12.adhoc')
    else:
        rel_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/qrels.adhoc')
    rel_df['Relevance'] = rel_df['Relevance'].apply(lambda x: int(x))
    summary_df['QueryNum'] = summary_df['QueryNum'].apply(lambda x: int(x))
    rel_df['Query'] = rel_df['Query'].apply(lambda x: int(x))
    summary_df = pd.merge(
        summary_df,
        rel_df[['Docno', 'Query', 'Relevance']].rename(columns={'Query': 'QueryNum'}),
        on=['Docno', 'QueryNum'],
        how='left')
    summary_df['Relevance'] = summary_df['Relevance'].apply(lambda x: 0 if (pd.np.isnan(x) or x <= 0) else 1)

    if 'SIM' in filename or 'United_summary' in filename:
        summary_df['Interval'] = summary_df['Interval'].apply(lambda x: 0 if x == 'ClueWeb09' else int(x))
    else:
        all_intervals = sorted(list(summary_df['Interval'].drop_duplicates()))
        rename_dict = {}
        i = 1
        for interval in reversed(all_intervals):
            rename_dict[interval] = (-1) * i
            i += 1
        summary_df['Interval'] = summary_df['Interval'].apply(lambda x: 0 if x == 'ClueWeb09' else rename_dict[x])

    interval_quantity_df = summary_df[['Interval', 'QueryNum']].groupby(['Interval']).count()
    interval_rel_quantity_df = summary_df[summary_df['Relevance'] == 1][['Interval', 'QueryNum']].groupby(['Interval']).count()
    interval_non_rel_quantity_df = summary_df[summary_df['Relevance'] == 0][['Interval', 'QueryNum']].groupby(['Interval']).count()
    interval_quantity_df = pd.merge(
        interval_quantity_df.rename(columns = {'QueryNum': 'All'}),
        interval_rel_quantity_df.rename(columns = {'QueryNum': 'Relevant'}),
        left_index=True,
        right_index=True)
    interval_quantity_df = pd.merge(
        interval_quantity_df.rename(columns={'QueryNum': 'All'}),
        interval_non_rel_quantity_df.rename(columns={'QueryNum': 'Non Relevant'}),
        left_index=True,
        right_index=True)
    interval_quantity_df.fillna(0, inplace = True)
    plt.cla()
    plt.clf()
    interval_quantity_df.plot(kind='bar')
    plt.legend(loc ='best')
    plt.xlabel('Interval')
    plt.tick_params(axis='x', labelsize=5)
    plt.ylabel("#Snapshots")
    plt.xticks(rotation=0)
    plt.title("#Snapshots Per Interval")
    plt.savefig(filename_for_save + "_Snapshots_per_interval.png", dpi=300)
    # for data verification

    for measure in ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb']:
        plt.cla()
        plt.clf()
        plot_df = summary_df[['Interval', measure]].groupby(['Interval']).mean()
        plot_df.rename(columns={measure: 'ALL'}, inplace=True)
        plot_df_ = summary_df[summary_df['Relevance'] == 1][['Interval', measure]].groupby(['Interval']).mean()
        plot_df_.rename(columns={measure: 'Relevant'}, inplace=True)
        plot_df = pd.merge(
            plot_df,
            plot_df_,
            left_index=True,
            right_index=True)
        plot_df_ = summary_df[summary_df['Relevance'] == 0][['Interval', measure]].groupby(['Interval']).mean()
        plot_df_.rename(columns={measure: 'Non Relevant'}, inplace=True)
        plot_df = pd.merge(
            plot_df,
            plot_df_,
            left_index=True,
            right_index=True)

        plot_df.plot(kind='bar')
        plt.xlabel('Interval')
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.75, bottom=0.15)
        if measure == 'QueryTermsRatio':
            plt.ylabel("rqtr")
        elif measure == 'StopwordsRatio':
            plt.ylabel("rswr")
        elif measure == 'Entropy':
            plt.ylabel("rent")
        plt.tick_params(axis='x', labelsize=5)
        plt.xticks(rotation=0)
        plt.savefig(measure + filename + "_Precentage_Difference_per_interval.png", dpi=300)

def top_50_data_plotter(
        retrival_model,
        interval_freq):

    files_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/top_50_data/'
    if retrival_model == 'BM25':
        filename = 'BM25_Top_50_data_' + interval_freq + '_Backward_Results.tsv'
    else:
        filename = 'Top_50_data_' + interval_freq + '_Backward_Results.tsv'

    work_df = pd.read_csv(os.path.join(files_path, filename), sep = '\t', index_col = False)

    rbo_df = work_df[['Interval', 'RBO_0.99', 'RBO_0.975','RBO_0.95']].groupby(['Interval']).mean()
    rbo_df.plot(kind='bar')
    plt.xlabel('Interval')
    plt.xlabel('Mean RBO')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75, bottom=0.15)
    plt.tick_params(axis='x', labelsize=7)
    plt.xticks(rotation=45)
    plt.savefig('RBO_' + filename.replace(".tsv",".png"), dpi=300)

    cw_df = pd.read_csv(os.path.join(files_path, 'ClueWb09_P_50__1M_Backward_Results.tsv'), sep = '\t', index_col = False)
    p_50_df = work_df[['Interval', 'P@50']].append(cw_df[['Interval', 'P@50']], ignore_index=True)
    p_50_df = p_50_df.groupby(['Interval']).mean()
    p_50_df.plot(kind='bar')
    plt.xlabel('Interval')
    plt.xlabel('P@50')
    plt.subplots_adjust( bottom=0.15)
    plt.ylim((p_50_df['P@50'].min() - 0.02, None))
    plt.tick_params(axis='x', labelsize=7)
    plt.xticks(rotation=45)
    plt.savefig('P50_' + filename.replace(".tsv", ".png"), dpi=300)

def per_query_coverage_ploter(
        filename,
        inner_fold):
    path_to_files = '/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/'
    filename_for_save = filename.replace('.tsv', '').replace('Summay_snapshot_stats_', '')
    if inner_fold != "" and inner_fold is not None:
        work_df = pd.read_csv(os.path.join(path_to_files, os.path.join(inner_fold, filename)), sep='\t',
                              index_col=False)
    else:
        work_df = pd.read_csv(os.path.join(path_to_files, filename), sep='\t', index_col=False)

    work_df = work_df[work_df['Interval'] != 'ClueWeb09']
    all_q = list(work_df['QueryNum'].drop_duplicates())
    work_df = work_df[['Docno','QueryNum']].drop_duplicates()
    work_df = work_df.groupby(['QueryNum']).count()

    uncovered_q_num = 0
    for q in all_q:
        if q not in list(work_df.index):
            work_df.loc[q] = [0]
            uncovered_q_num += 1

    work_df.plot(kind='bar')
    plt.xlabel('Query')
    plt.ylabel('#Covered Docs')
    if uncovered_q_num > 0:
        plt.title('#Covered Docs Per Query - ' + str(uncovered_q_num) + ' Uncovered')
    plt.subplots_adjust(bottom=0.15)
    plt.tick_params(axis='x', labelsize=7)
    plt.xticks(rotation=45)
    plt.savefig('Per_Query_Coverage_' + filename_for_save.replace(".tsv", ".png"), dpi=300)

def handle_rank_svm_params(
        filename):

    data_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/'
    work_df = pd.read_csv(os.path.join(data_folder, filename), sep = '\t', index_col = False)
    print(work_df[['FeatGroup','C','SnapLimit']])
    all_cols = list(work_df.columns)
    all_feat_groups = list(work_df['FeatGroup'].drop_duplicates())
    for featgroup in all_feat_groups:
        print("################  " + featgroup + "  ################")
        tmp_df = work_df[work_df['FeatGroup'] == featgroup]
        for col in all_cols:
            if col not in ['FeatGroup','C','SnapLimit','Fold']:
                addition = ""
                if abs(tmp_df[col].mean()) <= tmp_df[col].std() and pd.np.sign(tmp_df[col].min()) != pd.np.sign(tmp_df[col].max()):
                    addition = "  *** Prob"
                elif abs(tmp_df[col].mean()) >= tmp_df[col].std() and pd.np.sign(tmp_df[col].min()) == pd.np.sign(tmp_df[col].max()):
                    addition = "  *** STRONG!!"
                print(col + " -> Mean : " + str(tmp_df[col].mean()) + ", Std: " + str(tmp_df[col].std()) + ", Min: " + str(tmp_df[col].min()) + ", Max: " + str(tmp_df[col].max()) + addition)



def asrc_data_parser(
        filepath):


    with open(filepath, 'r') as f:
        soup = BeautifulSoup(f.read())

    stemmer = Stemmer()
    big_doc_index = {}
    stopword_list = get_stopword_list()
    df_query_stems = create_stemmed_queries_df(sw_rmv=True)
    query_to_stem_mapping = {}
    for index, row in df_query_stems.iterrows():
        query_num = ('0' * (3 - len(str(row['QueryNum'])))) + str(row['QueryNum'])
        query_to_stem_mapping[query_num] = convert_query_to_tf_dict(row['QueryStems'])

    all_docs = soup.find_all('doc')
    cc_dict = {'ALL_TERMS_COUNT' : 0,
               'ALL_SW_COUNT'    : 0}
    df_dict = {'ALL_DOCS_COUNT'    : 0,
               'AVG_DOC_LEN_NO_SW' : 0,
               'AVG_DOC_LEN'       : 0}
    print("Step 1...")
    sys.stdout.flush()

    for doc_ in list(all_docs):
        docno = doc_.find('docno').text
        fulltext = doc_.find('text').text
        broken_docno = docno.split('-')
        round_ = broken_docno[1]
        query_num = broken_docno[2]
        user = broken_docno[3]
        if int(round_) == 0:
            continue

        fulltext = re.sub('[^a-zA-Z0-9 ]', ' ', fulltext)

        res_dict = {}
        res_dict['StemList'] = ['[[OOV]']
        res_dict['IndexList'] = []
        res_dict['NumStopWords'] = 0
        res_dict['NumWords'] = 0
        res_dict['NumQueryWords'] = 0
        res_dict['TfList'] = [0]
        # res_dict['DfList'] = [0]
        # res_dict['CCList'] = [0]
        res_dict['Fulltext'] = ""
        res_dict['TfDict'] = {}

        curr_fulltext_list = fulltext.split(" ")
        for stem in curr_fulltext_list:
            stem = stemmer.stem(stem)
            if stem == '' or stem == '\n':
                continue
            if stem not in res_dict['TfDict']:
                res_dict['StemList'].append(stem)
                res_dict['TfDict'][stem] = 1
            else:
                res_dict['TfDict'][stem] += 1

            res_dict['IndexList'].append(res_dict['StemList'].index(stem))
            if stem in stopword_list:
                res_dict['NumStopWords'] += 1

            if stem in query_to_stem_mapping[query_num]:
                res_dict['NumQueryWords'] += 1

            res_dict['NumWords'] += 1
            res_dict['Fulltext'] += stem + " "
        res_dict['Fulltext'] = res_dict['Fulltext'][:-1]
        for stem in res_dict['StemList'][1:]:
            res_dict['TfList'].append(res_dict['TfDict'][stem])
            if stem in cc_dict:
                cc_dict[stem] += res_dict['TfDict'][stem]
                df_dict[stem] += 1
            else:
                cc_dict[stem] = res_dict['TfDict'][stem]
                df_dict[stem] = 1

        cc_dict['ALL_TERMS_COUNT'] += res_dict['NumWords']
        cc_dict['ALL_SW_COUNT'] += res_dict['NumStopWords']
        df_dict['ALL_DOCS_COUNT'] += 1
        df_dict['AVG_DOC_LEN'] += res_dict['NumWords']
        df_dict['AVG_DOC_LEN_NO_SW'] += (res_dict['NumWords']- res_dict['NumStopWords'])

        res_dict['Entropy'] = calc_shannon_entopy(res_dict['TfList'][1:])
        query_user_str = query_num +'-' +user
        if query_user_str not in big_doc_index:
            big_doc_index[query_user_str] = {}
        if round_ not in big_doc_index[query_user_str]:
            big_doc_index[query_user_str][round_] = {'json' : res_dict,
                                                    'docno': docno}
        else:
            raise Exception("double ID")


    print("Step 2...")
    sys.stdout.flush()
    df_dict['AVG_DOC_LEN'] = float(df_dict['AVG_DOC_LEN']) / df_dict['ALL_DOCS_COUNT']
    df_dict['AVG_DOC_LEN_NO_SW'] = float(df_dict['AVG_DOC_LEN_NO_SW']) / df_dict['ALL_DOCS_COUNT']
    # save cc and df dicts
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/asrc/cc_per_interval_dict.json', 'w') as f:
        f.write(str(cc_dict))
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/asrc/df_per_interval_dict.json', 'w') as f:
        f.write(str(df_dict))

    # calc cc and df list for all
    for query_user_str in big_doc_index:
        for round_ in big_doc_index[query_user_str]:
            if 'DfList' not in big_doc_index[query_user_str][round_]['json']:
                big_doc_index[query_user_str][round_]['json']['DfList'] = [0]
                big_doc_index[query_user_str][round_]['json']['CCList'] = [0]
                for stem in big_doc_index[query_user_str][round_]['json']['StemList'][1:]:
                    big_doc_index[query_user_str][round_]['json']['DfList'].append(df_dict[stem])
                    big_doc_index[query_user_str][round_]['json']['CCList'].append(cc_dict[stem])

                big_doc_index[query_user_str][round_]['json']['TfIdf'] = calc_tfidf_dict(
                    big_doc_index[query_user_str][round_]['json']['StemList'],
                    big_doc_index[query_user_str][round_]['json']['TfList'],
                    big_doc_index[query_user_str][round_]['json']['DfList'])
                big_doc_index[query_user_str][round_]['json']['LMScore'] = lm_score_doc_for_query(
                    query_stem_dict=query_to_stem_mapping[query_user_str.split('-')[0]],
                    cc_dict=cc_dict,
                    doc_dict=big_doc_index[query_user_str][round_]['json'])
                big_doc_index[query_user_str][round_]['json']['BM25Score'] = bm25_score_doc_for_query(
                    query_stem_dict=query_to_stem_mapping[query_user_str.split('-')[0]],
                    df_dict=df_dict,
                    doc_dict=big_doc_index[query_user_str][round_]['json'])
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/asrc/RawData.json' , 'w') as f:
        f.write(str(big_doc_index))


def create_base_features_for_asrc(
        rel_filepath):

    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/asrc/RawData.json', 'r') as f:
        big_doc_index = ast.literal_eval(f.read())

    dataset_name = 'asrc'
    meta_data_base_fold = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/'
    col_list = ['NumSnapshots', 'QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                 'QueryWords', 'Stopwords', 'TextLen', '-Query-SW','LMScore','BM25Score',
                 'QueryTermsRatio_M', 'StopwordsRatio_M', 'Entropy_M', 'SimClueWeb_M',
                 'QueryWords_M', 'Stopwords_M', 'TextLen_M', '-Query-SW_M','LMScore_M','BM25Score_M',
                 'QueryTermsRatio_STD', 'StopwordsRatio_STD', 'Entropy_STD', 'SimClueWeb_STD',
                 'QueryWords_STD', 'Stopwords_STD', 'TextLen_STD', '-Query-SW_STD','LMScore_STD','BM25Score_STD',
                 'QueryTermsRatio_LG', 'StopwordsRatio_LG', 'Entropy_LG', 'SimClueWeb_LG',
                 'QueryWords_LG', 'Stopwords_LG', 'TextLen_LG', '-Query-SW_LG','LMScore_LG','BM25Score_LG',
                 'QueryTermsRatio_MG', 'StopwordsRatio_MG', 'Entropy_MG', 'SimClueWeb_MG',
                 'QueryWords_MG', 'Stopwords_MG', 'TextLen_MG', '-Query-SW_MG','LMScore_MG','BM25Score_MG',
                 'QueryTermsRatio_RMG', 'StopwordsRatio_RMG', 'Entropy_RMG', 'SimClueWeb_RMG',
                 'QueryWords_RMG', 'Stopwords_RMG', 'TextLen_RMG', '-Query-SW_RMG','LMScore_RMG','BM25Score_RMG',
                 # 'Relevance',
                 'QueryNum', 'Docno']

    all_rounds = ['01', '02', '03', '04', '05', '06', '07', '08']
    fin_df_dict = {}
    for round_ in all_rounds:
        fin_df_dict[round_] = {}
        fin_df_dict[round_]['FinDF'] = pd.DataFrame(columns = col_list)
        fin_df_dict[round_]['SnapDF'] = pd.DataFrame({})
        fin_df_dict[round_]['NextIdx'] = 0
        if not os.path.exists('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+ dataset_name+ '_' + round_ + '/'):
            os.mkdir('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+ dataset_name+ '_' + round_ + '/')
            os.mkdir('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + dataset_name + '_' + round_ + '/2008/')
            os.mkdir('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + dataset_name + '_' + round_ + '/2008/SIM/')
            os.mkdir(os.path.join(meta_data_base_fold, dataset_name + '_' + round_))
    sys.stdout.flush()

    base_feature_list = ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb',
                         'QueryWords', 'Stopwords', 'TextLen', '-Query-SW','LMScore','BM25Score']
    for query_user_str in big_doc_index:
        all_rounds = list(big_doc_index[query_user_str].keys())
        query_num = query_user_str.split('-')[0]
        for round_num in all_rounds:
            docno = big_doc_index[query_user_str][round_num]['docno'].replace('ROUND','EPOCH')
            print(docno)
            sys.stdout.flush()
            res_dict = {'ClueWeb09' : big_doc_index[query_user_str][round_num]['json']}
            round_ = int(round_num)
            for additional_round in all_rounds:
                if int(additional_round) < round_:
                    diff = str(int(additional_round) - round_)
                    res_dict[diff]= big_doc_index[query_user_str][additional_round]['json']
            with open(os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+ dataset_name+ '_' + round_num + '/2008/SIM', docno +'.json'), 'w') as f:
                f.write(str(res_dict))

            curr_doc_df = pd.DataFrame(columns=['Docno', 'QueryNum', 'Interval'] + base_feature_list)
            tmp_idx = 0
            for i in list(reversed(range(round_))):
                if i == 0:
                    round_str = 'ClueWeb09'
                else:
                    round_str = str((-1)*i)
                doc_dict = res_dict[round_str]
                insert_row = [doc_dict['NumQueryWords'] / float((doc_dict['NumWords'])), doc_dict['NumStopWords'] / float(doc_dict['NumWords'] - doc_dict['NumStopWords']),
                              doc_dict['Entropy'], calc_cosine(doc_dict['TfIdf'], res_dict['ClueWeb09']['TfIdf']), doc_dict['NumQueryWords'], doc_dict['NumStopWords'],
                              doc_dict['NumWords'], doc_dict['NumWords'] - (doc_dict['NumQueryWords'] + doc_dict['NumStopWords']), doc_dict['LMScore'], doc_dict['BM25Score']]
                curr_doc_df.loc[tmp_idx] = [docno, query_num, round_str] + insert_row
                tmp_idx += 1

            bench_df = curr_doc_df[curr_doc_df['Interval'] == 'ClueWeb09']
            insert_row = [len(curr_doc_df)]
            for feature in base_feature_list:
                insert_row.append(list(bench_df[feature])[0])

            for feature in base_feature_list:
                insert_row.append(curr_doc_df[feature].mean())

            for feature in base_feature_list:
                insert_row.append(curr_doc_df[feature].std())

            if len(curr_doc_df) == 1:
                insert_row.extend([pd.np.nan] * (len(base_feature_list) * 3))
            else:
                for feature in base_feature_list:
                    curr_doc_df[feature + '_Shift'] = curr_doc_df[feature].shift(-1)
                    curr_doc_df[feature + '_Grad']  = curr_doc_df.apply(
                        lambda row_: calc_releational_measure(row_[feature + '_Shift'], row_[feature]), axis=1)
                    curr_doc_df[feature + '_RGrad'] = curr_doc_df.apply(
                        lambda row_: calc_releational_measure(row_[feature], list(bench_df[feature])[0]), axis=1)

                curr_doc_df = curr_doc_df[curr_doc_df['Interval'] != 'ClueWeb09']
                for feature in base_feature_list:
                    insert_row.append(list(curr_doc_df[feature + '_Grad'])[-1])

                for feature in base_feature_list:
                    insert_row.append(curr_doc_df[feature + '_Grad'].mean())

                for feature in base_feature_list:
                    insert_row.append(curr_doc_df[feature + '_RGrad'].mean())

            insert_row.extend([query_num, docno])
            fin_df_dict[round_num]['FinDF'].loc[fin_df_dict[round_num]['NextIdx']] = insert_row
            fin_df_dict[round_num]['NextIdx'] += 1

            curr_doc_df['NumSnapshots'] = len(curr_doc_df)
            curr_doc_df['SnapNum'] = list(range((len(curr_doc_df) - 1) * (-1), 1))
            fin_df_dict[round_num]['SnapDF'] = fin_df_dict[round_num]['SnapDF'].append(curr_doc_df, ignore_index=True)

    print("Finished features!")
    sys.stdout.flush()
    meta_data_df = get_relevant_docs_df(rel_filepath)
    filename = 'ASRC_All_features'

    # fin_df.to_csv(os.path.join(save_folder, filename + '_raw.tsv'), sep = '\t', index = False)
    # meta_data_df.to_csv(os.path.join(save_folder, filename + '_Meatdata.tsv'), sep = '\t', index = False)

    meta_data_df['Query'] = meta_data_df['Query'].apply(lambda x: int(x))
    for round_ in all_rounds:
        fin_df_dict[round_]['FinDF']['QueryNum'] = fin_df_dict[round_]['FinDF']['QueryNum'].apply(lambda x: int(x))
        fin_df_dict[round_]['FinDF'][['Docno','QueryNum']].to_csv(os.path.join(os.path.join(meta_data_base_fold, dataset_name + '_' + round_), 'all_urls_no_spam_filtered.tsv'), sep = '\t', index = False)
        fin_df_dict[round_]['FinDF'] = pd.merge(
            fin_df_dict[round_]['FinDF'],
            meta_data_df.rename(columns = {'Query' : 'QueryNum'}),
            on=['QueryNum', 'Docno'],
            how='inner')

        save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/'
        fin_df_dict[round_]['FinDF'].to_csv(os.path.join(save_folder, filename +'_Round' + round_ +'_with_meta.tsv'), sep='\t', index=False)
        fin_df_dict[round_]['SnapDF'].to_csv(os.path.join(save_folder, filename + '_Round' + round_ + '_all_snaps.tsv'), sep='\t', index=False)

def unite_asrc_data_results(
        big_model,
        snap_limit,
        ret_model,
        additional_models_to_include = {
            'F1' : {'Folder' : '/mnt/bi-strg3/v/zivvasilisky/ziv/results/avg_model_res_asrc/final_res/',
                    'FileTemplate' : 'asrc_0<RoundNum>_BM25_Results.txt'},
            'F2': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                   'FileTemplate': 'asrc_0<RoundNum>_BM25_Score_Results.txt'},
            # 'F3': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
            #        'FileTemplate': 'asrc_0<RoundNum>_BM25_Results.txt'},
            'ED KL': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/benchmark_sudomay/final_res/',
                   'FileTemplate': 'asrc_0<RoundNum>_KL_Results.txt'},
            'ED LM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/benchmark_sudomay/final_res/',
                      'FileTemplate': 'asrc_0<RoundNum>_LM_Results.txt'},

        }):
    from rank_svm_model import create_sinificance_df
    base_folder = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/"
    qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel"
    if big_model == 'SVMRank':
        base_folder = os.path.join(base_folder, 'rank_svm_res')
    else:
        base_folder = os.path.join(base_folder, 'lambdamart_res')
    base_2_folder = os.path.join(base_folder, 'ret_res')
    feature_list = ['NDCG@1', 'NDCG@3', 'MRR']
    feat_col_list = feature_list[:]
    for feat in feature_list:
        feat_col_list.append(feat + '_sign')

    big_res_dict  = {}
    round_res_dict = {}
    num_files = 0
    num_rounds = 0
    for round_ in range(2,9):
        inner_fold = 'ASRC_All_features_Round0'+str(round_)+'_with_meta.tsvSNL'+str(snap_limit)+'_'+ret_model+'_ByMonths'
        inner_fold = os.path.join(base_2_folder, inner_fold)
        round_res_dict[round_] = {}
        num_rounds += 1
        for filename in os.listdir(inner_fold):
            print(inner_fold + '/'+filename)
            num_files += 1
            sys.stdout.flush()
            tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                file_path=inner_fold,
                filename=filename,
                qrel_filepath=qrel_filepath,
                calc_ndcg_mrr=True)
            print(tmp_res_dict.keys())
            feat_group = filename.replace(inner_fold.split('/')[-1] + '_MinMax_', '').replace('.txt', '').replace('_AllByMonths', '').replace('_', '+')
            round_res_dict[round_][feat_group.replace('_', '+')] = tmp_res_dict
            print(feat_group)
            if feat_group.replace('_', '+') in big_res_dict:
                print(num_files)
                sys.stdout.flush()
                for q in tmp_res_dict:
                    for measure in tmp_res_dict[q]:
                        big_res_dict[feat_group.replace('_', '+')][q][measure] = (float(big_res_dict[feat_group][q][measure])*(num_rounds - 1) + tmp_res_dict[q][measure])/float(num_rounds)
            else:
                print ("here!")
                sys.stdout.flush()
                big_res_dict[feat_group.replace('_', '+')] = tmp_res_dict

        for model in additional_models_to_include:
            filename = additional_models_to_include[model]['FileTemplate'].replace('<RoundNum>',str(round_))
            path = additional_models_to_include[model]['Folder']
            print(filename)
            feat_group = model
            tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                file_path=path,
                filename=filename,
                qrel_filepath=qrel_filepath,
                calc_ndcg_mrr=True)
            print (tmp_res_dict.keys())
            if round_ == 6:
                del tmp_res_dict[193]
            round_res_dict[round_][feat_group.replace('_', '+')] = tmp_res_dict
            print(feat_group)
            if feat_group.replace('_', '+') in big_res_dict:
                print(num_files)
                sys.stdout.flush()
                for q in tmp_res_dict:
                    for measure in tmp_res_dict[q]:
                        big_res_dict[feat_group.replace('_', '+')][q][measure] = (float(
                            big_res_dict[feat_group][q][measure]) * (num_rounds - 1) + tmp_res_dict[q][measure]) / float(
                            num_rounds)
            else:
                print("here!")
                sys.stdout.flush()
                big_res_dict[feat_group.replace('_', '+')] = tmp_res_dict

        measure_list = ['Map', 'P@5', 'P@10', 'NDCG@1', 'NDCG@3', 'MRR', 'nMRR']
        round_summary_df = pd.DataFrame(columns=['FeatureGroup'] + measure_list)
        next_idx = 0
        for feat_group in round_res_dict[round_]:
            insert_row = [feat_group.replace('_', '+')]
            for measure in measure_list:
                if measure == 'P@5' or measure == 'P@10':
                    measure = measure.replace('@','_')
                insert_row.append(round_res_dict[round_][feat_group]['all'][measure])
            round_summary_df.loc[next_idx] = insert_row
            next_idx += 1
        significance_df = create_sinificance_df(
            round_res_dict[round_],
            calc_ndcg_mrr=True)
        round_summary_df = pd.merge(
            round_summary_df,
            significance_df,
            on=['FeatureGroup'],
            how='inner')
        round_res_dict[str(round_) + '_Sum'] = round_summary_df
        round_summary_df.to_csv('ASRC_round_' + str(round_) + '_Summary.tsv', sep = '\t', index = False)

    measure_list = ['Map', 'P@5', 'P@10', 'NDCG@1', 'NDCG@3', 'MRR', 'nMRR']
    big_summary_df = pd.DataFrame(columns=['FeatureGroup'] + measure_list)
    next_idx = 0
    for feat_group in big_res_dict:
        insert_row = [feat_group.replace('_', '+')]
        for measure in measure_list:
            if measure == 'P@5' or measure == 'P@10':
                measure = measure.replace('@', '_')
            insert_row.append(big_res_dict[feat_group]['all'][measure])
        big_summary_df.loc[next_idx] = insert_row
        next_idx += 1
    significance_df = create_sinificance_df(
        big_res_dict,
        calc_ndcg_mrr=True)
    big_summary_df = pd.merge(
        big_summary_df,
        significance_df,
        on=['FeatureGroup'],
        how='inner')

    big_summary_df.to_csv(os.path.join(base_folder, 'ASRC_All_Rounds_SNL' + str(snap_limit) + '_' + ret_model +'.tsv'), sep = '\t' ,index = False)

    for measure in ['NDCG@1', 'NDCG@3', 'MRR']:
        measure_df = pd.DataFrame({})
        for round_ in range(2, 9):
            round_df = round_res_dict[str(round_) + '_Sum'].rename(columns = {measure : str(round_)})
            round_df[str(round_)] = round_df[str(round_)].apply(lambda x: round(x, 3))
            if measure_df.empty == True:
                measure_df = round_df[['FeatureGroup', str(round_)]].copy()
            else:
                measure_df = pd.merge(
                    measure_df,
                    round_df[['FeatureGroup', str(round_)]],
                    on = ['FeatureGroup'],
                    how = 'inner')

        measure_df.set_index('FeatureGroup', inplace = True)
        measure_df = measure_df.transpose()
        print(measure_df)
        sys.stdout.flush()
        plt.cla()
        plt.clf()
        measure_df.plot()
        plt.legend(loc = 'center left', bbox_to_anchor = (1,0.5))
        plt.subplots_adjust(right=0.7)
        plt.title(measure + ' Over Rounds')
        plt.xlabel('round')
        plt.ylabel(measure)
        plt.savefig('ASRC_All_Rounds_SNL' + str(snap_limit) + '_' + ret_model +'_'+big_model+ '_' +measure + '.png', dpi =300)


def handle_rank_svm_params_asrc(
        snap_limit,
        ret_model):

    data_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/'
    num_rounds = 0
    weight_dict = {}
    for filename in os.listdir(data_folder):
        if (filename.startswith('ASRC_All_features_Round0')) and ('SNL' + str(snap_limit) in filename) and (ret_model in filename) and (filename.endswith('_Params.tsv')):
            print(filename)
            sys.stdout.flush()
            num_rounds += 1
            work_df = pd.read_csv(os.path.join(data_folder, filename), sep = '\t', index_col = False)
            # round_num = filename.replace('ASRC_All_features_Round0','')[0]
            all_cols = list(work_df.columns)
            all_feat_groups = list(work_df['FeatGroup'].drop_duplicates())
            for featgroup in all_feat_groups:
                tmp_df = work_df[work_df['FeatGroup'] == featgroup]
                featgroup = featgroup.replace('XXSnap', '').replace('_All', '').replace('_', '+')
                if featgroup not in weight_dict:
                    weight_dict[featgroup] = {}
                for col in all_cols:
                    if col not in ['FeatGroup', 'C', 'SnapLimit', 'Fold']:
                        if col in weight_dict[featgroup]:
                            weight_dict[featgroup][col] = (weight_dict[featgroup][col]*(num_rounds - 1) +  tmp_df[col].mean())/float(num_rounds)
                        else:
                            weight_dict[featgroup][col] = tmp_df[col].mean()

    for feat_group in weight_dict:
        print(feat_group)
        feat_df = pd.DataFrame(columns = ['Feature', 'Weight'])
        next_idx = 0
        for feature in weight_dict[feat_group]:
            if not pd.np.isnan(weight_dict[feat_group][feature]):
                feat_df.loc[next_idx] = [feature.replace('XXSnaps', '').replace('SimClueWeb', 'Similarity'), weight_dict[feat_group][feature]]
                next_idx += 1
        feat_df.set_index('Feature', inplace = True)
        feat_df.sort_values('Weight', inplace = True)
        plt.cla()
        plt.clf()
        feat_df.plot(legend=False, kind='bar', color='b')
        plt.ylabel('Weight')
        plt.title(feat_group +' SVM Weights')
        plt.yticks(rotation = 90)
        plt.subplots_adjust(bottom=0.35)
        plt.savefig(feat_group + '_RankSVM_Weights.png', dpi = 200)



if __name__ == '__main__':
    operation = sys.argv[1]
    if operation == 'TFDict':
        interval_freq_list = ast.literal_eval(sys.argv[2])
        work_year = sys.argv[3]
        create_tf_dict_for_processed_docs(work_interval_freq_list=interval_freq_list, work_year=work_year )

    elif operation == 'CCDict':
        interval_freq_list = ast.literal_eval(sys.argv[2])
        already_exists = ast.literal_eval(sys.argv[3])
        work_year = sys.argv[4]
        inner_fold = sys.argv[5]
        create_per_interval_per_lookup_cc_dict(work_interval_freq_list=interval_freq_list,
                                               already_exists=already_exists,
                                               work_year=work_year,
                                               inner_fold=inner_fold)

    elif operation == 'DFDict':
        interval_freq_list = ast.literal_eval(sys.argv[2])
        already_exists = ast.literal_eval(sys.argv[3])
        work_year = sys.argv[4]
        inner_fold = sys.argv[5]
        create_per_interval_per_lookup_df_dict(work_interval_freq_list=interval_freq_list,
                                               already_exists=already_exists,
                                               work_year=work_year,
                                               inner_fold=inner_fold)

    elif operation == 'SWCCDict':
        interval_freq_list = ast.literal_eval(sys.argv[2])
        inner_fold = sys.argv[3]
        work_year = sys.argv[4]
        add_stopword_stats_to_cc_dict(interval_freq_list=interval_freq_list, inner_fold=inner_fold,work_year=work_year)

    elif operation == 'SWDFDict':
        interval_freq_list = ast.literal_eval(sys.argv[2])
        inner_fold = sys.argv[3]
        work_year = sys.argv[4]
        add_stopword_stats_to_df_dict(interval_freq_list=interval_freq_list, inner_fold=inner_fold,work_year=work_year)

    elif operation == 'StatsDF':
        sim_folder_name = sys.argv[2]
        inner_fold      = sys.argv[3]
        work_year       = sys.argv[4]
        create_stats_data_frame_for_snapshot_changes(sim_folder_name=sim_folder_name, inner_fold=inner_fold, work_year=work_year)

    elif operation == 'PlotStatsDF':
        filename = sys.argv[2]
        create_snapshot_changes_stats_plots(filename=filename)

    elif operation == 'PlotTop50Data':
        retrival_model = sys.argv[2]
        interval_freq = sys.argv[3]
        top_50_data_plotter(retrival_model=retrival_model,interval_freq=interval_freq)

    elif operation == 'PlotStatsDFRelVSNot':
        filename = sys.argv[2]
        work_year = sys.argv[3]
        inner_fold = sys.argv[4]
        preprocessed = ast.literal_eval(sys.argv[5])
        create_snapshot_changes_rev_vs_non_rel(filename=filename, work_year=work_year, inner_fold=inner_fold, preprocessed=preprocessed)

    elif operation == 'PlotWLDF':
        filename = sys.argv[2]
        interval_freq = sys.argv[3]
        lookup_method = sys.argv[4]
        plot_stats_vs_winner_plots_for_wl_file(file_name=filename,interval_freq=interval_freq, lookup_method=lookup_method)

    elif operation == 'PlotQueryCoverage':
        filename = sys.argv[2]
        inner_fold = sys.argv[3]
        per_query_coverage_ploter(filename=filename, inner_fold=inner_fold)

    elif operation == 'Top50Stats':
        work_year = sys.argv[2]
        interval_freq = sys.argv[3]
        lookup_method = sys.argv[4]
        retrival_model = sys.argv[5]
        calc_top_50_stats(work_year=work_year, interval_freq=interval_freq,
                                               lookup_method=lookup_method, retrival_model=retrival_model)
    elif operation == 'MultiYearFile':
        year_list = ast.literal_eval(sys.argv[2])
        last_interval = sys.argv[3]
        interval_freq = sys.argv[4]
        inner_fold = sys.argv[5]
        create_multi_year_snapshot_file(year_list=year_list,last_interval=last_interval,interval_freq=interval_freq,inner_fold=inner_fold)

    elif operation == "SIMInterval":
        sim_thresold = float(sys.argv[2])
        sim_folder_name = sys.argv[3]
        inner_fold = sys.argv[4]
        work_year = sys.argv[5]
        create_similarity_interval(sim_threshold=sim_thresold,sim_folder_name=sim_folder_name, inner_fold=inner_fold, work_year=work_year)

    elif operation == "AvgModelStats":
        frequency = sys.argv[2]
        retrival_model = sys.argv[3]
        interval_start_month = int(sys.argv[4])
        filter_params= ast.literal_eval(sys.argv[5])
        sw_rmv = ast.literal_eval(sys.argv[6])
        work_year = sys.argv[7]

        plot_interesting_stats_for_avg_model_results(
            frequency=frequency,
            retrival_model=retrival_model,
            interval_start_month=interval_start_month,
            filter_params=filter_params,
            sw_rmv=sw_rmv,
            work_year=work_year)

    elif operation == 'PlotAllRetStats':
        for filename in os.listdir('/mnt/bi-strg3/v/zivvasilisky/ziv/results/retrival_stats/'):
            if filename.endswith('.tsv'):
                file_lookup = ""
                for lookup in ["NoLookup", "OnlyBackward", "Backward", "Forward"]:
                    if lookup in filename:
                        file_lookup = lookup
                        break
                int_freq = ""
                for freq in ["1W", "2W", "1M", "2M", "SIM"]:
                    if filename.startswith(freq) or filename.replace("BM25_","").startswith(freq):
                        int_freq = freq
                        break
                if int_freq == "" or file_lookup == "":
                    raise  Exception("Prob!")
                plot_retrival_stats(filename=filename, interval_freq=int_freq, lookup_method=file_lookup)

    elif operation == 'SVMParams':
        filename = sys.argv[2]
        handle_rank_svm_params(filename)

    elif operation == 'ASRCMeta':
        filepath = sys.argv[2]
        asrc_data_parser(filepath)

    elif operation == 'ASRCFeat':
        rel_filepath = sys.argv[2]
        create_base_features_for_asrc(rel_filepath)

    elif operation == 'ASRCFileUnite':
        big_model = sys.argv[2]
        snap_limit = int(sys.argv[3])
        ret_model = sys.argv[4]

        unite_asrc_data_results(
            big_model=big_model,
            snap_limit=snap_limit,
            ret_model=ret_model)
    elif operation == 'ASRCSVMWeights':
        snap_limit = int(sys.argv[2])
        ret_model = sys.argv[3]

        handle_rank_svm_params_asrc(snap_limit=snap_limit, ret_model=ret_model)
        # create_text_manipulated_interval(
#     sim_folder_name="SIM_TXT_UP_DOWN",
#     limit_to_clueweb_len=True,
#     fill_to_clueweb_len=True)
# create_text_manipulated_interval(
#     sim_folder_name="SIM_TXT_UP",
#     limit_to_clueweb_len=True,
#     fill_to_clueweb_len=False)
# create_text_manipulated_interval(
#     sim_folder_name="SIM_TXT_DOWN",
#     limit_to_clueweb_len=False,
#     fill_to_clueweb_len=True)


# create_similarity_interval()
# create_stats_data_frame_for_snapshot_changes(sim_folder_name="SIM_TXT_UP_DOWN")
# create_stats_data_frame_for_snapshot_changes(sim_folder_name="SIM_TXT_UP")
# create_stats_data_frame_for_snapshot_changes(sim_folder_name="SIM_TXT_DOWN")

# create_per_interval_per_lookup_cc_dict()

# check_for_txt_len_problem()
# merge_covered_df_with_file()
# get_relevant_docs_df()
#
# import ast
# with open('clueweb09-enwp01-31-11362.json', 'r') as f:
#     curr_json = ast.literal_eval(f.read())
#
#
# for interval in ['ClueWeb09', '-1', '-2']:
#     print ('Interval : ' + interval)
#     print ('DocLen : ' + str(curr_json[interval]['NumWords']))
#     print("Len Stemlist: " + str(len(curr_json[interval]['StemList'])))
#     print("Len tflist: " + str(len(curr_json[interval]['TfList'])))
#     for stem in ['obama', 'family', 'tree']:
#         for i in range(len(curr_json[interval]['StemList'])):
#             if curr_json[interval]['StemList'][i] == stem:
#                 print (stem + " " + str(curr_json[interval]['TfList'][i]))

# create_tf_dict_for_processed_docs()
# create_per_interval_per_lookup_df_dict()