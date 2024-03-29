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
        filepath,
        inner_fold,
        round_limit = None):


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
        if (int(round_) == 0) or ((round_limit is not None) and (int(round_) > int(round_limit))):
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
            raise Exception("double ID " + docno)


    print("Step 2...")
    sys.stdout.flush()
    df_dict['AVG_DOC_LEN'] = float(df_dict['AVG_DOC_LEN']) / df_dict['ALL_DOCS_COUNT']
    df_dict['AVG_DOC_LEN_NO_SW'] = float(df_dict['AVG_DOC_LEN_NO_SW']) / df_dict['ALL_DOCS_COUNT']
    # save cc and df dicts
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/'+inner_fold+'/cc_per_interval_dict.json', 'w') as f:
        f.write(str(cc_dict))
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/'+inner_fold+'/df_per_interval_dict.json', 'w') as f:
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
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/'+inner_fold+'/RawData.json' , 'w') as f:
        f.write(str(big_doc_index))


def create_base_features_for_asrc(
        rel_filepath,
        inner_fold,
        round_limit=None):

    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/'+inner_fold+'/RawData.json', 'r') as f:
        big_doc_index = ast.literal_eval(f.read())

    # dataset_name = 'asrc'
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
    if round_limit is not None:
        all_rounds = all_rounds[:int(round_limit)]
    fin_df_dict = {}
    for round_ in all_rounds:
        fin_df_dict[round_] = {}
        fin_df_dict[round_]['FinDF'] = pd.DataFrame(columns = col_list)
        fin_df_dict[round_]['SnapDF'] = pd.DataFrame({})
        fin_df_dict[round_]['NextIdx'] = 0
        if not os.path.exists('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+ inner_fold+ '_' + round_ + '/'):
            os.mkdir('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+ inner_fold+ '_' + round_ + '/')
            os.mkdir('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + inner_fold + '_' + round_ + '/2008/')
            os.mkdir('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + inner_fold + '_' + round_ + '/2008/SIM/')
            os.mkdir(os.path.join(meta_data_base_fold, inner_fold + '_' + round_))
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
            with open(os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+ inner_fold+ '_' + round_num + '/2008/SIM', docno +'.json'), 'w') as f:
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
    filename = inner_fold.upper() + '_All_features'

    # fin_df.to_csv(os.path.join(save_folder, filename + '_raw.tsv'), sep = '\t', index = False)
    # meta_data_df.to_csv(os.path.join(save_folder, filename + '_Meatdata.tsv'), sep = '\t', index = False)

    meta_data_df['Query'] = meta_data_df['Query'].apply(lambda x: int(x))
    for round_ in all_rounds:
        fin_df_dict[round_]['FinDF']['QueryNum'] = fin_df_dict[round_]['FinDF']['QueryNum'].apply(lambda x: int(x))
        fin_df_dict[round_]['FinDF'][['Docno','QueryNum']].to_csv(os.path.join(os.path.join(meta_data_base_fold, inner_fold + '_' + round_), 'all_urls_no_spam_filtered.tsv'), sep = '\t', index = False)
        fin_df_dict[round_]['FinDF'] = pd.merge(
            fin_df_dict[round_]['FinDF'],
            meta_data_df.rename(columns = {'Query' : 'QueryNum'}),
            on=['QueryNum', 'Docno'],
            how='inner')

        save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/'
        fin_df_dict[round_]['FinDF'].to_csv(os.path.join(save_folder, filename +'_Round' + round_ +'_with_meta.tsv'), sep='\t', index=False)
        fin_df_dict[round_]['SnapDF'].to_csv(os.path.join(save_folder, filename + '_Round' + round_ + '_all_snaps.tsv'), sep='\t', index=False)

def create_mixture_models_feature_dict(
        dataset_name):
    mixture_model_res_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/'
    mm_features_dict = {}
    for filename in os.listdir(mixture_model_res_path):
        if filename.startswith(dataset_name.lower()) and not filename.endswith('_Params.txt'):
            broken_name = filename.split('_')
            round = broken_name[1]
            method = broken_name[2]
            model = broken_name[3]
            if '_OnlyReservoir' in filename:
                method = 'OnlyReservoir'
            if model == 'MixtureDIR':
                model = 'DIR'
            elif model == 'LoO' or model == 'MixtureJM':
                model = 'JM'
            else:
                raise Exception("create_mixture_models_feature_dict: Unknown model")
            if '_K1_' in filename:
                method += 'K1'
            elif '_K3_' in filename:
                method += 'K3'
            elif '_Rand_' in filename:
                method += 'Rand'
            curr_df = convert_trec_results_file_to_pandas_df(os.path.join(mixture_model_res_path,filename))
            for index, row in curr_df.iterrows():
                if int(row['Query_ID']) not in mm_features_dict:
                    mm_features_dict[int(row['Query_ID'])] = {}
                if row['Docno'] not in mm_features_dict[int(row['Query_ID'])]:
                    mm_features_dict[int(row['Query_ID'])][row['Docno']] = {}
                mm_features_dict[int(row['Query_ID'])][row['Docno']][model + method] = float(row['Score'])

    return mm_features_dict


def create_domais_rhs_feature_dict(
        dataset_name,
        model_features_dict):
    sudomay_model_res_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/benchmark_sudomay/final_res/'

    for filename in os.listdir(sudomay_model_res_path):
        if filename.startswith(dataset_name.lower()) and not filename.endswith('_Params.txt') and 'LoO' in filename:
            print(filename)
            broken_name = filename.split('_')
            round = broken_name[1]
            model = broken_name[2]
            if '_KL_' in filename:
                model = 'ED_KL'
            elif '_LM_' in filename:
                model = 'ED_LM'
            else:
                raise Exception("create_domais_rhs_feature_dict: Unknown model")

            curr_df = convert_trec_results_file_to_pandas_df(os.path.join(sudomay_model_res_path, filename))
            for index, row in curr_df.iterrows():
                model_features_dict[int(row['Query_ID'])][row['Docno']][model] = float(row['Score'])

    aji_model_res_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rhs_model_asrc/final_res/'

    for filename in os.listdir(aji_model_res_path):
        if filename.startswith(dataset_name.lower()) and not filename.endswith('_Params.txt') and 'LoO' in filename:
            print(filename)
            broken_name = filename.split('_')
            round = broken_name[1]
            model = broken_name[2]
            if '_BM25_' in filename:
                model = 'RHS_BM25'
            elif '_LM_' in filename:
                model = 'RHS_LM'
            else:
                raise Exception("create_domais_rhs_feature_dict: Unknown model 2")

            curr_df = convert_trec_results_file_to_pandas_df(os.path.join(aji_model_res_path, filename))
            for index, row in curr_df.iterrows():
                model_features_dict[int(row['Query_ID'])][row['Docno']][model] = float(row['Score'])

    return model_features_dict


def create_lts_feature_dict(
        dataset_name,
        model_features_dict):
    model_res_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/final_res/'

    for filename in os.listdir(model_res_path):
        if filename.startswith(dataset_name.lower()) and not filename.endswith('_Params.txt') and 'LoO' in filename:
            print(filename)
            broken_name = filename.split('_')
            round = broken_name[1]
            model = broken_name[2]

            if  model in ['MA', 'LR', 'ARMA']:
                model = 'LTS_' + model
            else:
                raise Exception("create_lts_feature_dict: Unknown model")

            curr_df = convert_trec_results_file_to_pandas_df(os.path.join(model_res_path, filename))
            for index, row in curr_df.iterrows():
                model_features_dict[int(row['Query_ID'])][row['Docno']][model] = float(row['Score'])
    return model_features_dict


def create_fusion_feature_dict(
        dataset_name,
        model_features_dict):
    model_res_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/'


    for filename in os.listdir(model_res_path):
        if filename.startswith(dataset_name.lower()) and not filename.endswith('_Params.txt') and 'LoO' in filename:
            print(filename)
            broken_name = filename.split('_')
            round = broken_name[1]

            model = broken_name[2]
            method = broken_name[3]
            w_method = broken_name[4]

            if method != 'Rank' and w_method != 'RDecaying':
                continue

            if model in ['LM', 'BM25', 'BERT']:
                model = 'Fuse_' + model
            else:
                raise Exception("create_lts_feature_dict: Unknown model")

            curr_df = convert_trec_results_file_to_pandas_df(os.path.join(model_res_path, filename))
            for index, row in curr_df.iterrows():
                model_features_dict[int(row['Query_ID'])][row['Docno']][model] = float(row['Score'])
    return model_features_dict

def create_ltr_feature_dict(
        feature_folder):
    res_dict = {}
    for filename in os.listdir(feature_folder):
        if filename.startswith('doc') and '_' in filename:
            print(filename)
            sys.stdout.flush()
            feature = filename.split('_')[0].replace('doc', '')
            if feature == 'BM25':
                feature = 'BM25Score'
            query = filename.split('_')[1]
            if query not in res_dict:
                res_dict[query] = {}
            with open(os.path.join(feature_folder, filename), 'r') as f:
                file_str = f.read()
            file_lines = file_str.split('\n')
            first_ = True
            for line in file_lines:
                if line == '':
                    continue
                docno = line.split(' ')[0]
                if docno not in res_dict[query]:
                    res_dict[query][docno] = {}
                if len(line.split(' ')) > 2:
                    if first_ == True:
                        print ('multiple Features!')
                        first_ = False
                    feature_additions = ['Sum', 'Min', 'Max', 'Mean', 'Std']
                    for i in range(len(feature_additions)):
                        res_dict[query][docno][feature + feature_additions[i]] = float(line.split(' ')[i + 1])
                else:
                    val = float(line.split(' ')[1])
                    res_dict[query][docno][feature] = val
    return res_dict


def create_base_features_for_asrc_with_ltr_features(
        rel_filepath,
        inner_fold,
        round_limit=None):


    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + inner_fold + '/RawDataWithBERT.json', 'r') as f:
        big_doc_index = ast.literal_eval(f.read())

    meta_data_base_fold = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/'


    all_rounds = ['01', '02', '03', '04', '05', '06', '07', '08']
    if round_limit is not None:
        all_rounds = all_rounds[:int(round_limit)]


    base_feature_list = ['Boolean.AND', 'Boolean.OR', 'CoverQueryNum', 'CoverQueryRatio','Ent','FracStops',
                         'IDF', 'Len','LMIR.ABS', 'LMIR.DIR', 'LMIR.JM', 'StopCover','TFSum','TFMin','TFMax','TFMean','TFStd',
                         'TFIDFSum','TFIDFMin','TFIDFMax','TFIDFMean','TFIDFStd','TFNormSum','TFNormMin','TFNormMax',
                         'TFNormMean','TFNormStd','VSM', 'SimClueWeb','StopwordsRatio','Stopwords','-Query-SW','BERTScore','BM25Score']

    mm_feature_list = [
                       'JMPrevWinner','JMPrevBestImprove',
                       'DIRPrevWinner','DIRPrevBestImprove',
                       'JMPrev2Winners', 'JMPrev2BestImprove',
                       'DIRPrev2Winners', 'DIRPrev2BestImprove',
                       'JMPrev3Winners', 'JMPrev3BestImprove',
                       'DIRPrev3Winners', 'DIRPrev3BestImprove',
                       'JMOnlyReservoir', 'DIROnlyReservoir',
                         'ED_LM', 'RHS_BM25', 'RHS_LM', 'ED_KL',
                        'Fuse_LM', 'Fuse_BM25'
    ]
    base_feature_list.extend(mm_feature_list)
    col_list = ['NumSnapshots']
    for suffix in ["", "_M", "_STD", "_LG", "_MG", "_RMG"]:
        for feature in base_feature_list:
            col_list.append(feature + suffix)
    col_list.extend(['QueryNum', 'Docno'])
    fin_df_dict = {}
    for round_ in all_rounds:
        fin_df_dict[round_] = {}
        fin_df_dict[round_]['FinDF'] = pd.DataFrame(columns=col_list)
        fin_df_dict[round_]['SnapDF'] = pd.DataFrame({})
        fin_df_dict[round_]['NextIdx'] = 0

    feature_ref_dict = create_ltr_feature_dict(os.path.join(os.path.join(os.path.join(meta_data_base_fold,'datsets'),inner_fold),'feat_dir'))
    mm_feature_ref = create_mixture_models_feature_dict(inner_fold)
    mm_feature_ref = create_domais_rhs_feature_dict(inner_fold, mm_feature_ref)
    # mm_feature_ref = create_lts_feature_dict(inner_fold, mm_feature_ref)
    mm_feature_ref = create_fusion_feature_dict(inner_fold, mm_feature_ref)
    for query_user_str in big_doc_index:
        all_rounds = list(big_doc_index[query_user_str].keys())
        query_num = query_user_str.split('-')[0]
        for round_num in all_rounds:
            docno = big_doc_index[query_user_str][round_num]['docno'].replace('ROUND', 'EPOCH')
            print(docno)
            sys.stdout.flush()
            res_dict = {'ClueWeb09': big_doc_index[query_user_str][round_num]}
            round_ = int(round_num)
            for additional_round in all_rounds:
                if int(additional_round) < round_:
                    diff = str(int(additional_round) - round_)
                    res_dict[diff] = big_doc_index[query_user_str][additional_round]

            curr_doc_df = pd.DataFrame(columns=['Docno', 'QueryNum', 'Interval'] + base_feature_list)
            tmp_idx = 0
            for i in list(reversed(range(round_))):
                if i == 0:
                    round_str = 'ClueWeb09'
                else:
                    round_str = str((-1) * i)
                doc_dict = res_dict[round_str]
                curr_docno = doc_dict['docno'].replace('ROUND','EPOCH')
                feature_ref_dict[query_num][curr_docno]['SimClueWeb'] = calc_cosine(doc_dict['json']['TfIdf'], res_dict['ClueWeb09']['json']['TfIdf'])
                feature_ref_dict[query_num][curr_docno]['StopwordsRatio'] = doc_dict['json']['NumStopWords'] / float(doc_dict['json']['NumWords'] - doc_dict['json']['NumStopWords'])
                feature_ref_dict[query_num][curr_docno]['Stopwords'] = doc_dict['json']['NumStopWords']
                feature_ref_dict[query_num][curr_docno]['-Query-SW'] = doc_dict['json']['NumWords'] - (doc_dict['json']['NumQueryWords'] + doc_dict['json']['NumStopWords'])
                feature_ref_dict[query_num][curr_docno]['BERTScore'] = doc_dict['json']['BERTScore']
                for mm_feat in mm_feature_list:
                    if int(curr_docno.split('-')[1]) == 1:
                        if 'DIR' in mm_feat:
                            feature_ref_dict[query_num][curr_docno][mm_feat] = feature_ref_dict[query_num][curr_docno]['LMIR.DIR']
                        elif 'JM' in mm_feat :
                            feature_ref_dict[query_num][curr_docno][mm_feat] = feature_ref_dict[query_num][curr_docno]['LMIR.JM']
                        else:
                            feature_ref_dict[query_num][curr_docno][mm_feat] = pd.np.nan
                    else:
                        if mm_feat.startswith('LTS') and int(curr_docno.split('-')[1]) < 4:
                            feature_ref_dict[query_num][curr_docno][mm_feat] = pd.np.nan
                        else:
                            feature_ref_dict[query_num][curr_docno][mm_feat] = mm_feature_ref[int(query_num)][curr_docno][mm_feat]
                insert_row =[]
                for feature_name in base_feature_list:
                    insert_row.append(feature_ref_dict[query_num][curr_docno][feature_name])

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
                curr_doc_df_for_append = curr_doc_df.copy()
            else:
                for feature in base_feature_list:
                    curr_doc_df[feature + '_Shift'] = curr_doc_df[feature].shift(-1)
                    curr_doc_df[feature + '_Grad'] = curr_doc_df.apply(
                        lambda row_: calc_releational_measure(row_[feature + '_Shift'], row_[feature]), axis=1)
                    curr_doc_df[feature + '_RGrad'] = curr_doc_df.apply(
                        lambda row_: calc_releational_measure(row_[feature], list(bench_df[feature])[0]), axis=1)
                curr_doc_df_for_append = curr_doc_df.copy()
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

            curr_doc_df_for_append['NumSnapshots'] = len(curr_doc_df_for_append)
            curr_doc_df_for_append['SnapNum'] = list(range((len(curr_doc_df_for_append) - 1) * (-1), 1))
            fin_df_dict[round_num]['SnapDF'] = fin_df_dict[round_num]['SnapDF'].append(curr_doc_df_for_append, ignore_index=True)

    print("Finished features!")
    sys.stdout.flush()
    meta_data_df = get_relevant_docs_df(rel_filepath)
    filename = inner_fold.upper() + '_LTR_All_features'
    if inner_fold == 'herd_control':
        filename = inner_fold.upper() + '_LTR'
    meta_data_df['Query'] = meta_data_df['Query'].apply(lambda x: int(x))
    for round_ in all_rounds:
        fin_df_dict[round_]['FinDF']['QueryNum'] = fin_df_dict[round_]['FinDF']['QueryNum'].apply(lambda x: int(x))
        fin_df_dict[round_]['FinDF'] = pd.merge(
            fin_df_dict[round_]['FinDF'],
            meta_data_df.rename(columns={'Query': 'QueryNum'}),
            on=['QueryNum', 'Docno'],
            how='inner')

        save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/'
        fin_df_dict[round_]['FinDF'].to_csv(os.path.join(save_folder, filename + '_Round' + round_ + '_with_meta.tsv'),
                                            sep='\t', index=False)
        fin_df_dict[round_]['SnapDF'].to_csv(os.path.join(save_folder, filename + '_Round' + round_ + '_all_snaps.tsv'),
                                             sep='\t', index=False)


def unite_asrc_data_results(
        big_model,
        snap_limit,
        ret_model,
        dataset_name,
        round_limit,
        significance_type,
        leave_one_out_train,
        backward_elimination,
        snap_num_as_hyper_param,
        limited_snap_num,
        with_bert_as_feature,
        limited_features_list,
        additional_models_to_include = {
            'F1 UW' : {'Folder' : '/mnt/bi-strg3/v/zivvasilisky/ziv/results/avg_model_res_asrc/final_res/',
                    'FileTemplate' : '<DatasetName>_0<RoundNum>_BM25_Uniform_Results.txt'},
            'F2 BM25 UW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                   'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Score_Uniform_Results.txt'},
            'F2 BERT UW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                           'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Score_Uniform_Results.txt'},
            'F3 BM25 UW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                   'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Rank_Uniform_Results.txt'},
            'F3 BERT UW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                           'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Rank_Uniform_Results.txt'},
            'F1 IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/avg_model_res_asrc/final_res/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Decaying_Results.txt'},
            'F2 BM25 IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Score_Decaying_Results.txt'},
            'F2 BERT IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                           'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Score_Decaying_Results.txt'},
            'F3 BM25 IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Rank_Decaying_Results.txt'},
            'F3 BERT IW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                           'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Rank_Decaying_Results.txt'},
            'F1 DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/avg_model_res_asrc/final_res/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_RDecaying_Results.txt'},
            'F2 BM25 DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Score_RDecaying_Results.txt'},
            'F2 BERT DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                           'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Score_RDecaying_Results.txt'},
            'F3 BM25 DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Rank_RDecaying_Results.txt'},
            'F3 BERT DW': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/wieghted_list_res/final_res/',
                           'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Rank_RDecaying_Results.txt'},
            'ED KL': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/benchmark_sudomay/final_res/',
                   'FileTemplate': '<DatasetName>_0<RoundNum>_KL_Results.txt'},
            'ED LM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/benchmark_sudomay/final_res/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_LM_Results.txt'},
            'RHS BM25': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rhs_model_asrc/final_res/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_BM25_Results.txt'},
            'RHS LM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rhs_model_asrc/final_res/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_LM_Results.txt'},
            'BERT': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bert/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Results.txt'},
            'Concat BERT Dec': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bert/',
                     'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Concat_Dec_Results.txt'},
            'Concat BERT Inc': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bert/',
                     'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Concat_Inc_Results.txt'},
            # 'MM PrevWinner JM K1': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                     'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureJM_K1_Results.txt'},
            # 'MM PrevWinner JM K3': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                          'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureJM_K3_Results.txt'},
            # 'MM PrevWinner JM Rand': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                          'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureJM_Rand_Results.txt'},
            'MM PrevWinner JM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                    'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureJM_Results.txt'},
            'MM Prev2Winners JM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                   'FileTemplate': '<DatasetName>_0<RoundNum>_Prev2Winners_MixtureJM_Results.txt'},
            'MM Prev3Winners JM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3Winners_Results.txt'},
            # 'MM PrevBestImprove JM K1': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                             'FileTemplate': '<DatasetName>_0<RoundNum>_PrevBestImprove_MixtureJM_K1_Results.txt'},
            # 'MM PrevBestImprove JM K3': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                             'FileTemplate': '<DatasetName>_0<RoundNum>_PrevBestImprove_MixtureJM_K3_Results.txt'},
            # 'MM PrevBestImprove JM Rand': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                             'FileTemplate': '<DatasetName>_0<RoundNum>_PrevBestImprove_MixtureJM_Rand_Results.txt'},
            'MM PrevBestImprove JM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                              'FileTemplate': '<DatasetName>_0<RoundNum>_PrevBestImprove_MixtureJM_Results.txt'},
            'MM Prev2BestImprove JM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                            'FileTemplate': '<DatasetName>_0<RoundNum>_Prev2BestImprove_MixtureJM_Results.txt'},
            'MM Prev3BestImprove JM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                            'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3BestImprove_MixtureJM_Results.txt'},
            # 'MM PrevWinner DIR K1': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                      'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureDIR_K1_Results.txt'},
            # 'MM PrevWinner DIR K3': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                          'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureDIR_K3_Results.txt'},
            # 'MM PrevWinner DIR Rand': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                          'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureDIR_Rand_Results.txt'},
            'MM PrevWinner DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                     'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureDIR_Results.txt'},
            'MM Prev2Winners DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                    'FileTemplate': '<DatasetName>_0<RoundNum>_Prev2Winners_MixtureDIR_Results.txt'},
            'MM Prev3Winners DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                    'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3Winners_MixtureDIR_Results.txt'},

            # 'MM PrevBestImprove DIR K1': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                           'FileTemplate': '<DatasetName>_0<RoundNum>_PrevBestImprove_MixtureDIR_K1_Results.txt'},
            # 'MM PrevBestImprove DIR K3': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                           'FileTemplate': '<DatasetName>_0<RoundNum>_PrevBestImprove_MixtureDIR_K3_Results.txt'},
            # 'MM PrevBestImprove DIR Rand': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
            #                           'FileTemplate': '<DatasetName>_0<RoundNum>_PrevBestImprove_MixtureDIR_Rand_Results.txt'},
            'MM PrevBestImprove DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                          'FileTemplate': '<DatasetName>_0<RoundNum>_PrevBestImprove_MixtureDIR_Results.txt'},
            'MM Prev2BestImprove DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                        'FileTemplate': '<DatasetName>_0<RoundNum>_Prev2BestImprove_MixtureDIR_Results.txt'},
            'MM Prev3BestImprove DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                        'FileTemplate': '<DatasetName>_0<RoundNum>_Prev3BestImprove_MixtureDIR_Results.txt'},
            'MM OnlyReservoir DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                        'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureDIR_OnlyReservoir_Results.txt'},
            'MM OnlyReservoir JM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/mixture_model_res/final_res/',
                                     'FileTemplate': '<DatasetName>_0<RoundNum>_PrevWinner_MixtureJM_OnlyReservoir_Results.txt'},
            'Orig. Ranker': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/data/<DatasetName>/<DatasetName>_0<RoundNum>/RankedLists/',
                            'FileTemplate': 'LambdaMART_<DatasetName>_0<RoundNum>'},
            'LM JM': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/basic_lm/',
                             'FileTemplate': '<DatasetName>_0<RoundNum>_LMIR.JM.txt'},
            'LM DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/basic_lm/',
                      'FileTemplate': '<DatasetName>_0<RoundNum>_LMIR.DIR.txt'},
            'LTS MA': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_MA_LTS__Results.txt'},
            'LTS LR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_LR_LTS__Results.txt'},
            'LTS ARMA': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/ts_model/final_res/',
                       'FileTemplate': '<DatasetName>_0<RoundNum>_ARMA_LTS__Results.txt'},
            # 'SNAPS': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/per_snap_lambdamart_res/ret_res/<InnerFold>/',
            #            'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_MinMax_Historical.txt'},

        },
        first_round_default = 2):
    from rank_svm_model import create_sinificance_df
    base_folder = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/"
    if dataset_name == 'asrc':
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel"
    elif dataset_name == 'bot':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents_fixed.relevance'
    elif dataset_name == 'herd_control':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/control.rel'
    elif dataset_name == 'united':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel'
    elif dataset_name == 'comp2020':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/curr_comp.rel'

    if big_model == 'SVMRank':
        base_folder = os.path.join(base_folder, 'rank_svm_res')
    elif big_model == 'LambdaMART':
        base_folder = os.path.join(base_folder, 'lambdamart_res')
    else:
        raise Exception("Unknown base model")
    base_2_folder = os.path.join(base_folder, 'ret_res')
    feature_list = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'MRR']
    feat_col_list = feature_list[:]
    for feat in feature_list:
        feat_col_list.append(feat + '_sign')

    big_res_dict = {}
    round_res_dict = {}
    num_files = 0
    num_rounds = 0
    addition_to_inner_fold = ""
    if limited_snap_num != "None":
        addition_to_inner_fold += '_' + str(limited_snap_num)
    if leave_one_out_train == True:
        addition_to_inner_fold += "_LoO"
    if backward_elimination == True:
        addition_to_inner_fold += "_BElim"
    if snap_num_as_hyper_param == True:
        addition_to_inner_fold += "_SnapLim"
    if with_bert_as_feature == True:
        addition_to_inner_fold += '_Bert'
    if limited_features_list is not None:
        addition_to_inner_fold += create_feature_list_shortcut_string(limited_features_list)

    for round_ in range(first_round_default, round_limit + 1):
        if dataset_name == 'herd_control':
            inner_fold_sep = dataset_name.upper() + '_LTR_Round0'+str(round_)+'_with_meta.tsvSNL'+str(snap_limit)+'_'+ret_model+'_ByMonths' + addition_to_inner_fold
        else:
            inner_fold_sep = dataset_name.upper() + '_LTR_All_features_Round0'+str(round_)+'_with_meta.tsvSNL'+str(snap_limit)+'_'+ret_model+'_ByMonths' + addition_to_inner_fold
        inner_fold = os.path.join(base_2_folder, inner_fold_sep)
        round_res_dict[round_] = {}
        num_rounds += 1
        for filename in os.listdir(inner_fold):
            for remove_low_quality in [False , True]:
                print(inner_fold + '/'+filename)
                num_files += 1
                sys.stdout.flush()
                tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                    file_path=inner_fold,
                    filename=filename,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=True,
                    remove_low_quality=remove_low_quality)
                print(tmp_res_dict.keys())
                feat_group = filename.replace(inner_fold.split('/')[-1] + '_MinMax_', '').replace('.txt', '').replace('_AllByMonths', '').replace('_', '+')
                # if 'RMG' in feat_group:
                #     continue
                if limited_snap_num != "None":
                    feat_group = feat_group.replace('+' + str(limited_snap_num) + 'ByMonths','')
                    feat_group = feat_group.replace('+' + str(limited_snap_num),'')

                if remove_low_quality == True:
                    feat_group += " RMV LQ"

                round_res_dict[round_][feat_group.replace('_', '+')] = tmp_res_dict

                print(feat_group)
                # if (dataset_name == 'herd_control') and (59 in tmp_res_dict):
                #     del tmp_res_dict[59]
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
            if dataset_name == 'united' and 'MM P' in model:
                continue
            if model.startswith('LTS') and first_round_default < 4:
                continue
            filename = additional_models_to_include[model]['FileTemplate'].replace('<RoundNum>',str(round_)).replace('<DatasetName>', dataset_name).replace('<DatasetNameUpper>', dataset_name.upper())
            if dataset_name == 'comp2020' and ('MM P' in model and 'JM' in model):
                if 'MixtureJM' not in filename:
                    filename = filename.replace('_Results.txt', '_MixtureJM_Results.txt')
            if leave_one_out_train == True and model not in ['BERT', 'Concat BERT Dec','Concat BERT Inc']:
                filename = filename.replace('_Results.txt', '_LoO_Results.txt')
            if dataset_name == 'herd_control' and model == 'SNAPS':
                filename = filename.replace('_All_features', '')
            path = additional_models_to_include[model]['Folder'].replace('<InnerFold>', inner_fold_sep).replace('<RoundNum>',str(round_)).replace('<DatasetName>', dataset_name)
            print(filename)
            for remove_low_quality in [False, True]:
                feat_group = model
                tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                    file_path=path,
                    filename=filename,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=True,
                    remove_low_quality=remove_low_quality)
                if remove_low_quality == True:
                    feat_group += " RMV LQ"
                print (tmp_res_dict.keys())
                # if dataset_name == 'herd_control':
                #     del tmp_res_dict[59]
                if round_ in [6,7] and model != 'SNAPS':
                    del tmp_res_dict[193]
                    if round_ == 7:
                        del tmp_res_dict[195]
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

        measure_list = ['Map', 'P@5', 'P@10', 'NDCG@1', 'NDCG@3', 'NDCG@5', 'MRR', 'nMRR']
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
            calc_ndcg_mrr=True,
            sinificance_type=significance_type)
        round_summary_df = pd.merge(
            round_summary_df,
            significance_df,
            on=['FeatureGroup'],
            how='inner')
        round_res_dict[str(round_) + '_Sum'] = round_summary_df
        if big_model == 'LambdaMART':
            round_summary_df.to_csv(dataset_name.upper()+ '_round_' + str(round_) + '_'+significance_type+ addition_to_inner_fold+'_LambdaMART_Summary.tsv', sep = '\t', index = False)
        else:
            round_summary_df.to_csv(dataset_name.upper() + '_round_' + str(round_) + '_'+significance_type+ addition_to_inner_fold+'_Summary.tsv', sep='\t', index=False)
    if first_round_default == 4:
        addition_to_inner_fold += "_LTS_Files"
    measure_list = ['Map', 'P@5', 'P@10', 'NDCG@1', 'NDCG@3', 'NDCG@5', 'MRR', 'nMRR']
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
        calc_ndcg_mrr=True,
        sinificance_type=significance_type)
    big_summary_df = pd.merge(
        big_summary_df,
        significance_df,
        on=['FeatureGroup'],
        how='inner')
    if big_model == 'LambdaMART':
        ret_model += '_'+ big_model
    big_summary_df.to_csv(os.path.join(base_folder, dataset_name.upper()+ '_All_Rounds_SNL' + str(snap_limit) + '_' + ret_model +'_'+significance_type+ addition_to_inner_fold+'.tsv'), sep = '\t' ,index = False)

    for measure in ['NDCG@1', 'NDCG@3', 'MRR']:
        measure_df = pd.DataFrame({})
        for round_ in range(first_round_default, round_limit + 1):
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
        plt.savefig(dataset_name.upper()+'_All_Rounds_SNL' + str(snap_limit) + '_' + ret_model +'_'+big_model+ '_' +measure + addition_to_inner_fold+ '.png', dpi =300)

def orginize_ablation_results(
        dataset_name,
        round_limit,
        limited_snap_num,
        limited_features_list):

    from rank_svm_model import create_sinificance_df
    base_folder = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/"
    if dataset_name == 'asrc':
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel"
    elif dataset_name == 'bot':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents_fixed.relevance'
    elif dataset_name == 'herd_control':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/control.rel'
    elif dataset_name == 'united':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel'
    elif dataset_name == 'comp2020':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/curr_comp.rel'

    base_folder = os.path.join(base_folder, 'lambdamart_res')

    base_2_folder = os.path.join(base_folder, 'ret_res')
    feature_list = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'MRR']
    feat_col_list = feature_list[:]
    for feat in feature_list:
        feat_col_list.append(feat + '_sign')

    big_res_dict = {}
    num_files = 0
    num_rounds = 0
    addition_to_inner_fold = ""
    if limited_snap_num != "None":
        addition_to_inner_fold += '_' + str(limited_snap_num)

    addition_to_inner_fold += "_LoO_Ablation"
    if limited_features_list is not None:
        addition_to_inner_fold += create_feature_list_shortcut_string(limited_features_list)

    for round_ in range(2, round_limit + 1):
        inner_fold_sep = dataset_name.upper() + '_LTR_All_features_Round0' + str(round_) + '_with_meta.tsvSNL1_BM25_ByMonths' + addition_to_inner_fold
        inner_fold = os.path.join(base_2_folder, inner_fold_sep)
        num_rounds += 1
        for filename in os.listdir(inner_fold):
            print(inner_fold + '/' + filename)
            num_files += 1
            sys.stdout.flush()
            tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                file_path=inner_fold,
                filename=filename,
                qrel_filepath=qrel_filepath,
                calc_ndcg_mrr=True)
            print(tmp_res_dict.keys())
            feat_name = filename.split('_Abla_')[-1].split(create_feature_list_shortcut_string(limited_features_list))[0].replace('SimClueWeb','Similarity').replace('XXSnaps','')
            if feat_name == 'Similarity':
                continue
            print(feat_name)
            if feat_name in big_res_dict:
                print(num_files)
                sys.stdout.flush()
                for q in tmp_res_dict:
                    for measure in tmp_res_dict[q]:
                        big_res_dict[feat_name][q][measure] = (float(big_res_dict[feat_name][q][measure]) * (num_rounds - 1) + tmp_res_dict[q][measure]) / float(num_rounds)
            else:
                print("here!")
                sys.stdout.flush()
                big_res_dict[feat_name] = tmp_res_dict
        ref_model_inner_fold = os.path.join(base_2_folder, inner_fold_sep.replace('_Ablation',''))
        for filename in os.listdir(ref_model_inner_fold):
            if 'MinMax_Static_M_STD_Min_Max_MG_AllByMonths' in filename:
                print(ref_model_inner_fold + '/' + filename)
                num_files += 1
                sys.stdout.flush()
                tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                    file_path=ref_model_inner_fold,
                    filename=filename,
                    qrel_filepath=qrel_filepath,
                    calc_ndcg_mrr=True)
                print(tmp_res_dict.keys())
                feat_name = 'Full Model'
                if feat_name in big_res_dict:
                    print(num_files)
                    sys.stdout.flush()
                    for q in tmp_res_dict:
                        for measure in tmp_res_dict[q]:
                            big_res_dict[feat_name][q][measure] = (float(big_res_dict[feat_name][q][measure]) * (num_rounds - 1) +
                                                                   tmp_res_dict[q][measure]) / float(num_rounds)
                else:
                    print("here!")
                    sys.stdout.flush()
                    big_res_dict[feat_name] = tmp_res_dict

    measure_list = ['Map', 'P@5', 'P@10', 'NDCG@1', 'NDCG@3', 'NDCG@5', 'MRR', 'nMRR']
    big_summary_df = pd.DataFrame(columns=['FeatureGroup'] + measure_list)
    next_idx = 0
    sinificanse_dict = {}
    for feat_name in big_res_dict:
        insert_row = [feat_name]
        for measure in measure_list:
            if measure == 'P@5' or measure == 'P@10':
                measure = measure.replace('@', '_')
            insert_row.append(big_res_dict[feat_name]['all'][measure])
            sinificanse_dict[feat_name] = check_statistical_significance(big_res_dict[feat_name], big_res_dict['Full Model'], ndcg_mrr=True)
        big_summary_df.loc[next_idx] = insert_row
        next_idx += 1


    plot_ref_plot = big_summary_df[big_summary_df['FeatureGroup'] == 'Full Model'].reset_index()
    big_summary_df = big_summary_df[big_summary_df['FeatureGroup'] != 'Full Model']
    for measure in ['NDCG@1', 'NDCG@3', 'NDCG@5']:
        plot_df = big_summary_df[['FeatureGroup', measure]].rename(columns={'FeatureGroup' : 'Removed Feature'}).set_index('Removed Feature')
        plot_df.sort_values(measure, inplace = True)
        plt.cla()
        plt.clf()
        colors = []
        x_labels = list(plot_df.index)
        for x_label in x_labels:
            if sinificanse_dict[x_label][measure]['Significant'] == True:
                colors.append('b')
            else:
                colors.append('k')
        print(colors)
        plot_df['Color'] = colors
        plot_df[measure].plot(legend=False, kind='bar', color=plot_df['Color'])
        plt.axhline(y=plot_ref_plot.loc[0][measure], color='r', linestyle='--', label='Full Model')
        # plt.legend(loc='best')
        plt.ylabel(measure)
        min_val = big_summary_df[measure].min()
        max_val = big_summary_df[measure].max()
        plt.ylim((min_val - 0.01, max_val + 0.01))
        plt.title(measure + ' Ablation Results')
        plt.xticks(rotation=90, fontsize=5)
        plt.subplots_adjust(bottom=0.35)
        plt.savefig('Ablation_Res_' + measure + '_' + dataset_name.upper() + addition_to_inner_fold + '.png',
                    dpi=300)


def orginize_mm_features_data_compare(
        dataset_name,
        round_limit,
        limited_snap_num,
        limited_features_list,
        first_round_default=2):


    from rank_svm_model import create_sinificance_df

    base_folder = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/"
    if dataset_name == 'asrc':
        qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel"
    elif dataset_name == 'bot':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents_fixed.relevance'
    elif dataset_name == 'herd_control':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/control.rel'
    elif dataset_name == 'united':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel'
    elif dataset_name == 'comp2020':
        qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/curr_comp.rel'

    base_folder = os.path.join(base_folder, 'lambdamart_res')

    base_2_folder = os.path.join(base_folder, 'ret_res')
    feature_list = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'MRR']
    feat_col_list = feature_list[:]
    for feat in feature_list:
        feat_col_list.append(feat + '_sign')

    big_res_dict = {}
    num_files = 0
    num_rounds = 0
    addition_to_inner_fold = ""
    if limited_snap_num != "None":
        addition_to_inner_fold += '_' + str(limited_snap_num)

    addition_to_inner_fold += "_LoO"
    if limited_features_list is not None:
        addition_to_inner_fold += create_feature_list_shortcut_string(limited_features_list)

    addition_to_inner_fold_bkp = addition_to_inner_fold
    feature_inner_fold_list = [('None', addition_to_inner_fold)]
    mm_feature_list = ['JMPrevWinner', 'JMPrevBestImprove',
                         'DIRPrevWinner', 'DIRPrevBestImprove',
                         'JMPrev2Winners', 'JMPrev2BestImprove',
                         'DIRPrev2Winners', 'DIRPrev2BestImprove',
                         'JMPrev3Winners', 'JMPrev3BestImprove',
                         'DIRPrev3Winners', 'DIRPrev3BestImprove',
                         'JMOnlyReservoir', 'DIROnlyReservoir',
                         'ED_KL', 'ED_LM', 'RHS_BM25', 'RHS_LM',
                        'LTS_MA', 'LTS_LR','LTS_ARMA']

    for mm_feture in mm_feature_list:
        feature_inner_fold_list.append((mm_feture, addition_to_inner_fold + create_feature_list_shortcut_string([mm_feture])))

    for round_ in range(first_round_default, round_limit + 1):
        num_rounds += 1
        for mm_fetue_config in feature_inner_fold_list:
            mm_feture = mm_fetue_config[0]
            if mm_feture.startswith('LTS') and first_round_default < 4:
                continue
            addition_to_inner_fold = mm_fetue_config[1]
            inner_fold_sep = dataset_name.upper() + '_LTR_All_features_Round0' + str(round_) + '_with_meta.tsvSNL1_BM25_ByMonths' + addition_to_inner_fold
            inner_fold = os.path.join(base_2_folder, inner_fold_sep)
            for filename in os.listdir(inner_fold):
                if 'MinMax_Static_All' in filename:
                    method = 'S'
                elif 'MinMax_Static_M_STD_Min_Max_MG_All' in filename:
                    method = 'S+MSMM+MG'
                else:
                    continue
                for remove_low_quality in [False, True]:
                    print(inner_fold + '/' + filename)
                    num_files += 1
                    sys.stdout.flush()
                    tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                        file_path=inner_fold,
                        filename=filename,
                        qrel_filepath=qrel_filepath,
                        calc_ndcg_mrr=True,
                        remove_low_quality=remove_low_quality)
                    print(tmp_res_dict.keys())
                    feat_name = mm_feture + '_' + method
                    print(feat_name)
                    if remove_low_quality == True:
                        feat_name += " RMV LQ"
                    if feat_name in big_res_dict:
                        print(num_files)
                        sys.stdout.flush()
                        for q in tmp_res_dict:
                            for measure in tmp_res_dict[q]:
                                big_res_dict[feat_name][q][measure] = (float(big_res_dict[feat_name][q][measure]) * (
                                num_rounds - 1) + tmp_res_dict[q][measure]) / float(num_rounds)
                    else:
                        print("here!")
                        sys.stdout.flush()
                        big_res_dict[feat_name] = tmp_res_dict

    measure_list = ['Map', 'P@5', 'P@10', 'NDCG@1', 'NDCG@3', 'NDCG@5', 'MRR', 'nMRR']
    big_summary_df = pd.DataFrame(columns=['FeatureGroup'] + measure_list)
    next_idx = 0
    for feat_name in big_res_dict:
        insert_row = [feat_name]
        for measure in measure_list:
            if measure == 'P@5' or measure == 'P@10':
                measure = measure.replace('@', '_')
            insert_row.append(big_res_dict[feat_name]['all'][measure])
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

    if first_round_default == 4:
        addition_to_inner_fold += "_LTS_Files"

    big_summary_df.to_csv('MM_Features_Summary_' + dataset_name.upper() + addition_to_inner_fold + '.tsv', sep='\t',
                          index=False)


def handle_rank_svm_params_asrc(
        dataset_name,
        round_limit,
        limited_features_list):

    data_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/rank_svm_res/'
    num_rounds = 0
    weight_dict = {}
    addition_to_filename = ""
    addition_to_filename += '_All'
    addition_to_filename += "_LoO"

    if limited_features_list is not None:
        addition_to_filename += create_feature_list_shortcut_string(limited_features_list)

    for round_ in range(2, round_limit + 1):
        for filename in os.listdir(data_folder):
            if filename.startswith(dataset_name.upper()+'_LTR_All_features_Round0'+str(round_)+'_with_meta.tsvSNL1_BM25_ByMonths'+addition_to_filename+'_MinMax') and (filename.endswith('_Params.tsv')):
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
                if feature.replace('XXSnaps', '').replace('SimClueWeb', 'Similarity') == 'Similarity':
                    continue
                feat_df.loc[next_idx] = [feature.replace('XXSnaps', '').replace('SimClueWeb', 'Similarity'), weight_dict[feat_group][feature]]
                next_idx += 1
        feat_df.set_index('Feature', inplace = True)
        feat_df.sort_values('Weight', inplace = True)
        if len(feat_df) > 70:
            bottom_df = feat_df.head(50)
            plt.cla()
            plt.clf()
            bottom_df.plot(legend=False, kind='bar', color='b')
            plt.ylabel('Weight')
            plt.title(feat_group + ' Bottom SVM Weights')
            plt.xticks(rotation=90, fontsize =7)
            plt.subplots_adjust(bottom=0.35)
            plt.savefig(feat_group + dataset_name.upper() + addition_to_filename + '_RankSVM_Weights_Bottom_Weights.png', dpi=200)

            top_df = feat_df.tail(50)
            plt.cla()
            plt.clf()
            top_df.plot(legend=False, kind='bar', color='b')
            plt.ylabel('Weight')
            plt.title(feat_group + ' Top SVM Weights')
            plt.xticks(rotation=90, fontsize =7)
            plt.subplots_adjust(bottom=0.35)
            plt.savefig(feat_group + dataset_name.upper() + addition_to_filename + '_RankSVM_Weights_Top_Weights.png',
                        dpi=200)
            if len(feat_df) > 100:
                feat_df.drop(list(top_df.index), inplace=True)
                feat_df.drop(list(bottom_df.index), inplace=True)
                plt.cla()
                plt.clf()
                feat_df.plot(legend=False, kind='bar', color='b')
                plt.ylabel('Weight')
                plt.title(feat_group + ' Middle SVM Weights')
                plt.xticks(rotation=90, fontsize =7)
                plt.subplots_adjust(bottom=0.35)
                plt.savefig(feat_group + dataset_name.upper() + addition_to_filename + '_RankSVM_Weights_Mid_Weights.png', dpi=200)
        else:
            plt.cla()
            plt.clf()
            feat_df.plot(legend=False, kind='bar', color='b')
            plt.ylabel('Weight')
            plt.title(feat_group +' SVM Weights')
            plt.xticks(rotation=90, fontsize=6)
            plt.subplots_adjust(bottom=0.35)
            plt.savefig(feat_group + dataset_name.upper() + addition_to_filename + '_RankSVM_Weights.png', dpi = 200)

def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)
    out, err = p.communicate()
    return out

def create_all_files_for_competition_features(
        inner_fold,
        round_limit,
        doc_file_path,
        only_feats):

    if only_feats == False:
        q_list ,fold_list  = get_asrc_q_list_and_fold_list(inner_fold=inner_fold, train_leave_one_out=False)
        with open("/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/query/query_num_to_text.txt", 'r') as f:
            query_texts = f.read()
        query_xml_str = "<parameters>" + '\n'
        query_texts = query_texts.split('\n')
        num_qs = 0
        for row_ in query_texts:
            splitted_row = row_.split(':')
            if len(splitted_row) > 1:
                if int(splitted_row[0]) in q_list:
                    query_xml_str += "<query>" + '\n' + '<number>' + splitted_row[0] + '</number>' + '\n' + \
                                    "<text>#combine( " + splitted_row[1] + ' )</text>' + '\n' + '</query>' + '\n'
                    num_qs += 1
        print('#Queries: ' + str(num_qs))
        queries_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/' + inner_fold + '/QueriesFile.xml'
        with open(queries_file, 'w') as f:
            f.write(query_xml_str + '</parameters>')

        doc_xml_str = ""
        workingset_str = ""
        with open(doc_file_path, 'r') as f:
            soup = BeautifulSoup(f.read())

        all_docs = soup.find_all('doc')
        for doc_ in list(all_docs):
            docno = (doc_.find('docno').text).replace('ROUND','EPOCH')
            fulltext = doc_.find('text').text
            broken_docno = docno.split('-')
            round_ = broken_docno[1]
            query_num = broken_docno[2]
            if (int(round_) == 0) or ((round_limit is not None) and (int(round_) > int(round_limit))):
                continue
            doc_xml_str += "<DOC>" +'\n' + "<DOCNO>" + docno + "</DOCNO>" + '\n' + "<TEXT>" + fulltext +\
                           "</TEXT>" + '\n' + "</DOC>" + '\n'
            workingset_str += query_num + ' Q0 ' + docno + " 0 0 indri" + '\n'

        doc_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/' + inner_fold + '/DocumentsFile.trectext'
        with open(doc_filepath, 'w') as f:
            f.write(re.sub(r'[^\x00-\x7F]+',' ', doc_xml_str))

        working_set_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/' + inner_fold + '/workingset.trec'
        with open(working_set_file, 'w') as f:
            f.write(workingset_str)

        index_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/' + inner_fold +'/IndexFile'
        run_bash_command("rm -r " + index_path)
        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/IndriBuildIndex_ForASRC.xml', 'r') as f:
            params_text = f.read()

        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/Index_params/IndriBuildIndex_' + str(inner_fold) + '.xml',
                  'w') as f:
            f.write(params_text.replace('###', index_path).replace('%%%',doc_filepath))
        print('Params fixed...')
        sys.stdout.flush()
        res = subprocess.check_call(['/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/indri/bin/IndriBuildIndex',
                                     '/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/Index_params/IndriBuildIndex_' + str(inner_fold) + '.xml'])
        print('Index built...')
        bash_command = '/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/indri/bin/dumpindex ' + index_path.replace('IndexFile', 'MergedIndexFile')
        bash_command += ' merge ' + '/mnt/bi-strg3/v/zivvasilisky/index/CW09ForMerge' + ' ' + index_path
        out = run_bash_command(bash_command)
        print(out)
        index_path = index_path.replace('IndexFile', 'MergedIndexFile')
        print('Index merged...')
    else:
        index_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/' + inner_fold + '/IndexFile'
        index_path = index_path.replace('IndexFile', 'MergedIndexFile')
        queries_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/' + inner_fold + '/QueriesFile.xml'
        working_set_file = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/' + inner_fold + '/workingset.trec'

    features_dir = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/' + inner_fold + '/feat_dir/'
    run_bash_command("rm -r " + features_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    scripts_path = '/lv_local/home/zivvasilisky/ziv/content_modification_code/scripts/'
    command = scripts_path + "LTRFeatures " + queries_file + ' -stream=doc -index=' + index_path + ' -repository=' + index_path + ' -useWorkingSet=true -workingSetFile=' + working_set_file + ' -workingSetFormat=trec'
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command("mv doc*_* " + features_dir)
    run_bash_command("mv " + features_dir +'doc_snapshot_stats_compare.py .')

def compare_rel_files_to_curr_comp():
    curr_rel_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/curr_comp.rel')
    curr_rel_df['Round'] = curr_rel_df['Docno'].apply(lambda x: x.split('-')[1])
    asrc_rel_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel')
    asrc_rel_df['Round'] = asrc_rel_df['Docno'].apply(lambda x: x.split('-')[1])
    united_rel_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel')
    united_rel_df['Round'] = united_rel_df['Docno'].apply(lambda x: x.split('-')[1])

    summary_df = pd.DataFrame(columns=['Dataset', '%Query-Round Withount non-relevant', '%Query-Round All Most Relevant'])
    next_idx = 0
    df_list = [('ASRC', asrc_rel_df), ('UNITED', united_rel_df), ('COMP2020',curr_rel_df)]
    big_hist_df = pd.DataFrame({})
    for elem in df_list:
        dataset = elem[0]
        df = elem[1]
        print(dataset)
        df['Relevance'] = df['Relevance'].apply(lambda x: float(x))
        df['IsRel'] = df['Relevance'].apply(lambda x: 0 if x==0 else 1.0)
        print("Num Docs :" +str(len(df)))
        print("Num rel :" + str(len(df[df['IsRel'] == 1])))
        print("Num non-rel :" + str(len(df[df['Relevance'] == 0])))
        print("Num rel 1 :" + str(len(df[df['Relevance'] == 1])))
        print("Num rel 2 :" + str(len(df[df['Relevance'] == 2])))
        print("Num rel 3 :" + str(len(df[df['Relevance'] == 3])))
        mdf = df[['Query', 'Round', 'Relevance', 'IsRel']].groupby(['Query', 'Round']).mean()
        mdf2 = df[['Round', 'Relevance']].groupby(['Round']).mean()
        print(mdf2)
        mdf2 = df[['Round', 'Relevance']].groupby(['Round']).median()
        print(mdf2)
        print('Mean Query Round relevance: ' + str(mdf['Relevance'].mean()))
        print('Mean Query Round relevant doc %: ' + str(mdf['IsRel'].mean()))
        print('Num Query Round all relevant docs: ' + str(len(mdf[mdf['IsRel'] == 1])))
        print('Num Query Round all most relevant docs: ' + str(len(mdf[mdf['Relevance'] == 3])))
        mdf = df[['Query', 'Round', 'Relevance', 'IsRel']].groupby(['Query', 'Round']).min()
        summary_df.loc[next_idx] = [dataset, len(mdf[mdf['Relevance'] >= 1]) / float(len(mdf)), len(mdf[mdf['Relevance'] == 3]) / float(len(mdf))]
        next_idx += 1
        hist_df = df[['Relevance', 'Round']].groupby(['Relevance']).count()
        hist_df.rename(columns = {'Round' : dataset}, inplace= True)
        if big_hist_df.empty == True:
            big_hist_df = hist_df
        else:
            big_hist_df = pd.merge(
                big_hist_df,
                hist_df,
                right_index=True,
                left_index=True,
                how = 'inner')
    big_hist_df.plot(kind='bar')
    plt.legend(loc='best')
    plt.title('Relevace Score Distribution Per Datset')
    plt.ylabel('#Docs')
    plt.xlabel('Relevance Score')
    plt.savefig('Dataset_Relevance_Compare.png', dpi=200)

    info_list = [('ASRC', 'asrc_08'), ('UNITED', 'united_05'), ('COMP2020','comp2020_03')]
    summary_sim_df = pd.DataFrame(columns=['Dataset', '%Not-Changed', 'Avg Cosine'])
    next_idx = 0
    for elem in info_list:
        dataset = elem[0]
        path_part = elem[1]
        stats_dict = {'equal_count' : 0,
                      'num_docs' : 0,
                      'sim_list' : []}
        for filename in os.listdir('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + path_part + '/2008/SIM/'):
            if filename.endswith('.json'):
                with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/' + path_part + '/2008/SIM/' + filename, 'r') as f:
                    curr_dict = ast.literal_eval(f.read())
                inteval_list = sorted(list(curr_dict.keys()))
                inteval_list.remove('ClueWeb09')
                inteval_list = ['ClueWeb09'] + inteval_list
                stats_dict['num_docs'] += len(inteval_list)
                for i in range(1, len(inteval_list)):
                    sim = calc_cosine(curr_dict[inteval_list[i-1]]['TfIdf'],curr_dict[inteval_list[i]]['TfIdf'])
                    if sim == 1.0:
                        stats_dict['equal_count'] += 1
                    stats_dict['sim_list'].append(sim)

        summary_sim_df.loc[next_idx] = [dataset, stats_dict['equal_count'] / float(stats_dict['num_docs']), np.mean(stats_dict['sim_list'])]
        next_idx += 1
    summary_sim_df = pd.merge(
        summary_df,
        summary_sim_df,
        on = ['Dataset'],
        how = 'inner')
    summary_sim_df.to_csv('DatasetCompare.tsv', sep = '\t', index=False)




def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')

def mixture_model_pre_process(
        dataset_name,
        num_rounds,
        trectext_file):

    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + dataset_name + '/cc_per_interval_dict.json', 'r') as f:
        cc_dict = ast.literal_eval(f.read())
    stem_str = ""
    for key in cc_dict:
        if key not in ['ALL_TERMS_COUNT', 'ALL_SW_COUNT']:
            stem_str += key + '\n'
    stem_str = stem_str[:-1]
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/' + dataset_name + '/AllStems.txt', 'w') as f:
        f.write(stem_str)

    dataset_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/' + dataset_name + '/'
    save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + dataset_name + '/'

    with open(dataset_folder + trectext_file, 'r') as f:
        soup = BeautifulSoup(f.read())
    round_files_dict = {}
    for round_ in range(1, num_rounds + 1):
        round_files_dict[str(round_).zfill(2)] = {'XML' : "", 'WorkSet' : ""}

    all_docs = soup.find_all('doc')
    for doc_ in list(all_docs):
        docno = (doc_.find('docno').text).replace('ROUND', 'EPOCH')
        fulltext = doc_.find('text').text
        broken_docno = docno.split('-')
        round_ = broken_docno[1]
        query_num = broken_docno[2]
        if round_ != '00':
            round_files_dict[round_]['XML'] +=  "<DOC>" + '\n' + "<DOCNO>" + docno + "</DOCNO>" + '\n' + "<TEXT>" + fulltext + \
                                                "</TEXT>" + '\n' + "</DOC>" + '\n'
            round_files_dict[round_]['WorkSet'] += query_num + ' Q0 ' + docno + " 0 0 indri" + '\n'

    QUERIES_FILE = dataset_folder + 'QueriesFile.xml'
    INDEX = dataset_folder + 'MergedIndexFile'
    scriptDir = '/lv_local/home/zivvasilisky/ziv/content_modification_code/scripts/'
    MODEL_FILE = '/lv_local/home/zivvasilisky/ASR20/epoch_run/content_modification_code/rank_models/model_lambdatamart'
    for round_ in round_files_dict:
        curr_save_folder = save_folder + dataset_name +'_' + round_ +'/'
        if not os.path.exists(curr_save_folder):
            os.makedirs(curr_save_folder)
        # trectext_file = curr_save_folder + dataset_name +'_' + round_ +'.trectext'
        # with open(trectext_file, 'w') as f:
        #     f.write(round_files_dict[round_]['XML'])
        wset_file = curr_save_folder + dataset_name + '_' + round_ + '.workingset'
        with open(wset_file, 'w') as f:
            f.write(round_files_dict[round_]['WorkSet'])

        FEATURES_DIR = curr_save_folder + 'Features/'
        if not os.path.exists(FEATURES_DIR):
            os.makedirs(FEATURES_DIR)
        WORKING_SET_FILE = wset_file
        ORIGINAL_FEATURES_FILE = 'features'
        command = scriptDir + 'LTRFeatures ' + QUERIES_FILE + ' -stream=doc -index=' + INDEX + ' -repository=' + INDEX + ' -useWorkingSet=true -workingSetFile=' + WORKING_SET_FILE + ' -workingSetFormat=trec'
        print(command)
        out = run_bash_command(command)
        print(out)
        run_command('mv doc*_* ' + FEATURES_DIR)
        command = 'perl ' + scriptDir + 'generate.pl ' + FEATURES_DIR + ' ' + WORKING_SET_FILE
        print(command)
        out = run_bash_command(command)
        print(out)
        FEATURES_FILE = ORIGINAL_FEATURES_FILE
        command = 'java -jar ' + scriptDir + 'RankLib.jar -load ' + MODEL_FILE + ' -rank ' + FEATURES_FILE + ' -score predictions.tmp'
        print(command)
        out = run_bash_command(command)
        print(out)
        command = 'cut -f3 predictions.tmp > predictions'
        print(command)
        out = run_bash_command(command)
        print(out)
        run_bash_command('rm predictions.tmp')
        RANKED_LIST_DIR = curr_save_folder + 'RankedLists/'
        if not os.path.exists(RANKED_LIST_DIR):
            os.makedirs(RANKED_LIST_DIR)
        PREDICTIONS_FILE = 'predictions'
        command = 'perl ' + scriptDir + 'order.pl ' + RANKED_LIST_DIR + 'LambdaMART_' + dataset_name +'_' + round_ + ' ' + FEATURES_FILE + ' ' + PREDICTIONS_FILE
        print(command)
        out = run_bash_command(command)
        print(out)

def add_lm_score_files(
        datasets =['ASRC', 'UNITED']):
    savepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/basic_lm/'
    for datset in datasets:
        if datset == 'ASRC':
            rounds = list(range(1,9))
        elif datset == 'UNITED':
            rounds = list(range(1,6))
        for round_ in rounds:
            path = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/base_features_for_svm_rank/'
            filename = datset + '_LTR_All_features_Round0' + str(round_) + '_with_meta.tsv'
            work_df = pd.read_csv(path +filename, sep = '\t', index_col = False)
            work_df = work_df[['QueryNum', 'Docno', 'LMIR.DIR', 'LMIR.JM']]
            q_list = list(work_df['QueryNum'].drop_duplicates())
            for model in ['LMIR.DIR', 'LMIR.JM']:
                big_df = pd.DataFrame({})
                for query in q_list:
                    res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])
                    next_index = 0
                    q_df = work_df[work_df['QueryNum'] == query].copy()
                    for index, row in q_df.iterrows():
                        docno = row['Docno']
                        query_num = query
                        doc_score = float(row[model])
                        res_df.loc[next_index] = ["0" * (3 - len(str(query_num))) + str(query_num), 'Q0', docno, 0,
                                                  doc_score, 'indri']
                        next_index += 1
                    if res_df.empty == False:
                        res_df.sort_values('Score', ascending=False, inplace=True)
                        res_df['Rank'] = list(range(1, next_index + 1))
                    big_df = big_df.append(res_df, ignore_index = False)
                with open(savepath + datset.lower() + '_0' + str(round_) + '_' + model + '.txt', 'w') as f:
                    f.write(convert_df_to_trec(big_df))

def fix_ks_files_and_produce_stats():
    ks_asrc_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/data/keyword_stuffed/asrc.ks')
    ks_herd_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/data/keyword_stuffed/herd.ks')

    ks_asrc_df['Docno'] = ks_asrc_df['Docno'].apply(lambda x: x.replace('ROUND', 'EPOCH'))
    ks_herd_df['Docno'] = ks_herd_df['Docno'].apply(lambda x: x.replace('ROUND', 'EPOCH'))

    ks_asrc_df['Round'] = ks_asrc_df['Docno'].apply(lambda x: int(x.split('-')[1]))
    ks_herd_df['Round'] = ks_herd_df['Docno'].apply(lambda x: int(x.split('-')[1]))

    ks_asrc_df['IsKS'] = ks_asrc_df['Relevance'].apply(lambda x: 1 if int(x) >= 2 else 0)
    ks_herd_df['IsKS'] = ks_herd_df['Relevance'].apply(lambda x: 1 if int(x) == 0 else 0)

    ks_united_df = ks_asrc_df[ks_asrc_df['Round'] <= 5].append(ks_herd_df, ignore_index = True)
    ks_list = list(set(list(ks_united_df[ks_united_df['IsKS'] == 1]['Docno'])  + list(ks_asrc_df[ks_asrc_df['IsKS'] == 1]['Docno'])))
    print(ks_list)
    df_list = [('ASRC', ks_asrc_df) , ('UNITED', ks_united_df)]
    for dataset, df in df_list:
        print(dataset)
        df['num'] = 1
        mdf = df[['Query', 'Round', 'IsKS', 'num']].groupby(['Query', 'Round']).sum()
        mdf['%ks'] = mdf.apply(lambda row: row['IsKS'] / float(row['num']), axis = 1)
        print("Num KS: " + str(len(df[df['IsKS'] == 1])))
        print("Mean % KS Per Round: " + str(mdf['%ks'].mean()))
        print("% Rounds KS > 0 : " + str(len(mdf[mdf['%ks']>0])/ float(len(mdf))))


def low_quality_satats():
    low_qulity_docs_list = ['EPOCH-05-069-PLB6ZI', 'EPOCH-02-004-02', 'EPOCH-03-195-3RXCQH', 'EPOCH-04-011-GUMSZY',
                            'EPOCH-02-098-UONKT7',
                            'EPOCH-08-004-48', 'EPOCH-05-177-YA7EE4', 'EPOCH-02-029-15', 'EPOCH-05-011-U4XHBD',
                            'EPOCH-01-144-DX0IY8',
                            'EPOCH-06-029-41', 'EPOCH-03-017-22', 'EPOCH-05-180-MEP5Y4', 'EPOCH-05-144-06',
                            'EPOCH-03-124-34PBG2',
                            'EPOCH-02-048-RXK8H3', 'EPOCH-02-045-71QP3V', 'EPOCH-06-017-01', 'EPOCH-03-098-UONKT7',
                            'EPOCH-08-029-16',
                            'EPOCH-02-017-22', 'EPOCH-05-013-SBHV4U', 'EPOCH-05-167-H4PVOF', 'EPOCH-05-010-9DWMMZ',
                            'EPOCH-02-011-GUMSZY',
                            'EPOCH-02-182-01', 'EPOCH-01-011-GUMSZY', 'EPOCH-08-017-01', 'EPOCH-08-002-43',
                            'EPOCH-02-011-09', 'EPOCH-07-069-43',
                            'EPOCH-01-124-34PBG2', 'EPOCH-02-004-BL2KLC', 'EPOCH-07-029-41', 'EPOCH-06-069-43',
                            'EPOCH-03-011-GUMSZY',
                            'EPOCH-05-144-15', 'EPOCH-07-002-13', 'EPOCH-05-124-S4SDDG', 'EPOCH-04-124-34PBG2',
                            'EPOCH-08-144-06',
                            'EPOCH-03-144-15', 'EPOCH-05-013-T7KWL2', 'EPOCH-01-180-37', 'EPOCH-08-048-27',
                            'EPOCH-06-193-01',
                            'EPOCH-01-048-RXK8H3', 'EPOCH-03-069-41', 'EPOCH-03-124-S4SDDG', 'EPOCH-05-033-SA9WFV',
                            'EPOCH-02-059-14T9OZ',
                            'EPOCH-01-034-6LWZ77', 'EPOCH-08-045-38', 'EPOCH-02-124-39', 'EPOCH-08-032-16',
                            'EPOCH-01-098-15', 'EPOCH-05-182-W129AB',
                            'EPOCH-07-193-01', 'EPOCH-05-032-7XUGJ0', 'EPOCH-02-195-3RXCQH', 'EPOCH-05-048-RXK8H3',
                            'EPOCH-07-048-27', 'EPOCH-02-078-WVSJJH',
                            'EPOCH-05-051-6QL968', 'EPOCH-05-033-KP5G43', 'EPOCH-01-034-20', 'EPOCH-06-018-03',
                            'EPOCH-02-144-T14SSS', 'EPOCH-02-124-S4SDDG',
                            'EPOCH-04-018-609XQD', 'EPOCH-07-144-06', 'EPOCH-01-144-T14SSS', 'EPOCH-03-004-48',
                            'EPOCH-01-098-UONKT7', 'EPOCH-04-144-06',
                            'EPOCH-08-069-45', 'EPOCH-08-069-43', 'EPOCH-08-144-24', 'EPOCH-05-161-FCT04D',
                            'EPOCH-03-048-RXK8H3', 'EPOCH-05-195-3RXCQH',
                            'EPOCH-05-098-UONKT7', 'EPOCH-06-029-16', 'EPOCH-06-029-15', 'EPOCH-07-017-01',
                            'EPOCH-06-195-43', 'EPOCH-01-059-14T9OZ',
                            'EPOCH-04-124-S4SDDG', 'EPOCH-01-011-U4XHBD', 'EPOCH-07-059-45', 'EPOCH-02-144-24',
                            'EPOCH-07-059-41', 'EPOCH-04-144-T14SSS',
                            'EPOCH-04-032-7XUGJ0', 'EPOCH-04-098-UONKT7', 'EPOCH-07-002-43', 'EPOCH-06-166-03',
                            'EPOCH-04-048-RXK8H3', 'EPOCH-04-195-35',
                            'EPOCH-02-144-15', 'EPOCH-03-029-15', 'EPOCH-04-069-41', 'EPOCH-01-033-KP5G43',
                            'EPOCH-03-098-SPSHA3', 'EPOCH-04-009-19',
                            'EPOCH-08-002-13', 'EPOCH-03-144-52', 'EPOCH-03-144-T14SSS', 'EPOCH-05-018-9PFCP4',
                            'EPOCH-06-059-41', 'EPOCH-02-144-09',
                            'EPOCH-06-144-06', 'EPOCH-01-017-VKBYGU', 'EPOCH-04-045-71QP3V', 'EPOCH-01-029-15',
                            'EPOCH-01-124-S4SDDG', 'EPOCH-03-180-19',
                            'EPOCH-08-029-15', 'EPOCH-08-009-33', 'EPOCH-05-144-T14SSS', 'EPOCH-08-078-04',
                            'EPOCH-07-029-16', 'EPOCH-01-004-0YGZO0',
                            'EPOCH-05-018-609XQD', 'EPOCH-06-009-02', 'EPOCH-03-011-U4XHBD', 'EPOCH-06-048-27',
                            'EPOCH-01-195-3RXCQH', 'EPOCH-03-018-609XQD',
                            'EPOCH-02-078-14', 'EPOCH-04-195-3RXCQH', 'EPOCH-08-010-40', 'EPOCH-04-004-48',
                            'EPOCH-05-011-GUMSZY', 'EPOCH-03-144-24',
                            'EPOCH-04-029-15', 'EPOCH-08-164-31', 'EPOCH-06-002-43', 'EPOCH-04-059-14T9OZ',
                            'EPOCH-03-017-32', 'EPOCH-05-048-V0HAX1',
                            'EPOCH-04-144-24', 'EPOCH-07-029-25', 'EPOCH-03-078-WVSJJH']
    # low_qulity_docs_list = ['EPOCH-05-069-PLB6ZI', 'EPOCH-03-195-3RXCQH', 'EPOCH-04-011-GUMSZY', 'EPOCH-02-098-UONKT7',
    #                         'EPOCH-08-004-48', 'EPOCH-05-177-YA7EE4', 'EPOCH-02-029-15', 'EPOCH-05-011-U4XHBD',
    #                         'EPOCH-01-144-DX0IY8', 'EPOCH-03-017-22', 'EPOCH-07-002-43', 'EPOCH-03-124-34PBG2',
    #                         'EPOCH-02-048-RXK8H3', 'EPOCH-02-045-71QP3V', 'EPOCH-03-098-UONKT7', 'EPOCH-02-017-22',
    #                         'EPOCH-01-034-6LWZ77', 'EPOCH-05-167-H4PVOF', 'EPOCH-05-010-9DWMMZ', 'EPOCH-02-011-GUMSZY',
    #                         'EPOCH-06-029-41', 'EPOCH-01-011-GUMSZY', 'EPOCH-08-002-43', 'EPOCH-07-069-43',
    #                         'EPOCH-01-124-34PBG2', 'EPOCH-02-004-BL2KLC', 'EPOCH-07-029-41', 'EPOCH-06-069-43',
    #                         'EPOCH-03-011-GUMSZY', 'EPOCH-05-144-15', 'EPOCH-05-124-S4SDDG', 'EPOCH-04-124-34PBG2',
    #                         'EPOCH-03-144-15', 'EPOCH-05-013-T7KWL2', 'EPOCH-06-059-41', 'EPOCH-01-048-RXK8H3',
    #                         'EPOCH-03-124-S4SDDG', 'EPOCH-05-033-SA9WFV', 'EPOCH-02-059-14T9OZ', 'EPOCH-05-013-SBHV4U',
    #                         'EPOCH-02-124-39', 'EPOCH-05-182-W129AB', 'EPOCH-05-032-7XUGJ0', 'EPOCH-02-195-3RXCQH',
    #                         'EPOCH-05-048-RXK8H3', 'EPOCH-02-078-WVSJJH', 'EPOCH-05-051-6QL968', 'EPOCH-05-033-KP5G43',
    #                         'EPOCH-02-144-T14SSS', 'EPOCH-02-124-S4SDDG', 'EPOCH-04-018-609XQD', 'EPOCH-01-144-T14SSS',
    #                         'EPOCH-05-048-V0HAX1', 'EPOCH-01-098-UONKT7', 'EPOCH-04-144-06', 'EPOCH-08-069-45',
    #                         'EPOCH-08-069-43', 'EPOCH-08-144-24', 'EPOCH-05-161-FCT04D', 'EPOCH-03-048-RXK8H3',
    #                         'EPOCH-05-195-3RXCQH', 'EPOCH-05-098-UONKT7', 'EPOCH-06-029-16', 'EPOCH-06-195-43',
    #                         'EPOCH-01-059-14T9OZ', 'EPOCH-04-124-S4SDDG', 'EPOCH-01-011-U4XHBD', 'EPOCH-07-059-45',
    #                         'EPOCH-07-059-41', 'EPOCH-04-144-T14SSS', 'EPOCH-04-032-7XUGJ0', 'EPOCH-04-098-UONKT7',
    #                         'EPOCH-04-048-RXK8H3', 'EPOCH-04-195-35', 'EPOCH-02-144-15', 'EPOCH-01-033-KP5G43',
    #                         'EPOCH-03-098-SPSHA3', 'EPOCH-03-144-52', 'EPOCH-03-144-T14SSS', 'EPOCH-05-018-9PFCP4',
    #                         'EPOCH-04-004-48', 'EPOCH-01-017-VKBYGU', 'EPOCH-04-045-71QP3V', 'EPOCH-01-124-S4SDDG',
    #                         'EPOCH-05-180-MEP5Y4', 'EPOCH-05-144-T14SSS', 'EPOCH-07-029-16', 'EPOCH-01-004-0YGZO0',
    #                         'EPOCH-05-018-609XQD', 'EPOCH-03-011-U4XHBD', 'EPOCH-01-195-3RXCQH', 'EPOCH-03-018-609XQD',
    #                         'EPOCH-02-078-14', 'EPOCH-04-195-3RXCQH', 'EPOCH-05-011-GUMSZY', 'EPOCH-04-029-15',
    #                         'EPOCH-08-164-31', 'EPOCH-06-002-43', 'EPOCH-04-059-14T9OZ', 'EPOCH-03-017-32',
    #                         'EPOCH-03-004-48',
    #                         'EPOCH-04-144-24', 'EPOCH-03-078-WVSJJH']
    rel_asrc_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel')
    rel_united_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel')
    ks_asrc_df = get_relevant_docs_df('/mnt/bi-strg3/v/zivvasilisky/ziv/data/keyword_stuffed/asrc.ks')

    ks_asrc_df['KS_Score'] = ks_asrc_df['Relevance'].apply(lambda x: float(x))
    ks_asrc_df['Docno'] = ks_asrc_df['Docno'].apply(lambda x: x.replace('ROUND','EPOCH'))
    ks_asrc_df = pd.merge(
        ks_asrc_df[['Docno','KS_Score']],
        rel_asrc_df,
        on = ['Docno'],
        how = 'inner')
    ks_asrc_df['Relevance'] = ks_asrc_df['Relevance'].apply(lambda x: float(x))
    ks_asrc_df = ks_asrc_df[['Docno', 'Relevance', 'KS_Score']].groupby(['Relevance', 'KS_Score']).count()
    print(ks_asrc_df.reset_index().pivot(index= 'Relevance', columns= 'KS_Score', values='Docno'))

    rel_asrc_df['IsKS'] = rel_asrc_df['Docno'].apply(lambda x: 1 if x in low_qulity_docs_list else 0)
    rel_united_df['IsKS'] = rel_united_df['Docno'].apply(lambda x: 1 if x in low_qulity_docs_list else 0)

    rel_asrc_df['Q-R'] = rel_asrc_df['Docno'].apply(lambda x: x.split('-')[1] + '-'+ x.split('-')[2])
    rel_united_df['Q-R'] = rel_united_df['Docno'].apply(lambda x: x.split('-')[1] + '-' + x.split('-')[2])

    l = []
    for datset, df in [('ASRC', rel_asrc_df), ('UNITED', rel_united_df)]:
        print(datset)
        df['Relevance'] = df['Relevance'].apply(lambda x: float(x))
        print("Mean LQ Relevance : " + str(df[df['IsKS'] == 1]['Relevance'].mean()))
        print("Min LQ Relevance : " + str(df[df['IsKS'] == 1]['Relevance'].min()))
        print("Max LQ Relevance : " + str(df[df['IsKS'] == 1]['Relevance'].max()))
        print("Num LQ Relevance : " + str(len(df[df['IsKS'] == 1])))
        print("SUM LQ Relevance : " + str(df[df['IsKS'] == 1]['Relevance'].sum()))

        print("Max LQ In Q-R : " + str(rel_united_df[['Q-R','IsKS']].groupby(['Q-R']).sum()['IsKS'].max()))
        # print("Max KS In Q-R : " + str(rel_united_df[['Q-R', 'IsKS']].groupby(['Q-R']).sum()['IsKS'].max()))
        df = df[df['IsKS'] == 1]
        print("Num LQ and Relevant: " + str(len(df[df['Relevance'] >= 1])))
        print("Num LQ and Relevant = 0: " + str(len(df[df['Relevance'] == 0])))
        print("Num LQ and Relevant = 1: " + str(len(df[df['Relevance'] == 1])))
        print("Num LQ and Relevant = 2: " + str(len(df[df['Relevance'] == 2])))
        print("Num LQ and Relevant = 3: " + str(len(df[df['Relevance'] == 3])))

        mdf = df

        l.extend(list(df[df['Relevance'] <= 1]['Docno']))

    print(list(set(l)))


def bert_checker():
    model_files_dict = {
    'BERT': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bert/',
             'FileTemplate': '<DatasetName>_0<RoundNum>_BERT_Results.txt'},
    'LM DIR': {'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/basic_lm/',
               'FileTemplate': '<DatasetName>_0<RoundNum>_LMIR.DIR.txt'},

    'BM25': {
            'Folder': '/mnt/bi-strg3/v/zivvasilisky/ziv/results/lambdamart_res/ret_res/<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT/',
            'FileTemplate': '<DatasetNameUpper>_LTR_All_features_Round0<RoundNum>_with_meta.tsvSNL1_BM25_ByMonths_All_LoO_E_FS_L_LMD_SC_TFSm_TFNSm_SCw_BM25_IDF_BRT_MinMax_BM25.txt',
            'AlsoLQRmv': True},
    }
    dataset = 'asrc'
    qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/documents.rel"
    measures = ['NDCG@1','NDCG@3','NDCG@5']
    summary_df = pd.DataFrame(columns = ['Dataset', 'Model','Round']+measures)
    next_idx = 0
    for round_ in range(1,9):
        for model in model_files_dict:
            file_path = model_files_dict[model]['Folder'].replace('<DatasetNameUpper>', dataset.upper()).replace(
                '<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            filename = model_files_dict[model]['FileTemplate'].replace('<DatasetNameUpper>', dataset.upper()).replace(
                '<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            if round_ == 1 and model == 'BM25':
                file_path= '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bm25_model_res/final_res/'
                filename= '<DatasetName>_01_BM25_LoO_Results.txt'.replace('<DatasetName>', dataset)

            print(model, filename)
            sys.stdout.flush()
            tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                file_path=file_path,
                filename=filename,
                qrel_filepath=qrel_filepath,
                calc_ndcg_mrr=True,
                remove_low_quality=False)

            insert_row = [dataset.upper(), model, round_]
            for measure in measures:
                insert_row.append(tmp_res_dict['all'][measure])
            summary_df.loc[next_idx] = insert_row
            next_idx += 1
    dataset = 'united'
    qrel_filepath = "/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/united.rel"
    for round_ in range(1, 6):
        for model in model_files_dict:

            file_path = model_files_dict[model]['Folder'].replace('<DatasetNameUpper>', dataset.upper()).replace(
                '<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            filename = model_files_dict[model]['FileTemplate'].replace('<DatasetNameUpper>', dataset.upper()).replace(
                '<RoundNum>', str(round_)).replace('<DatasetName>', dataset)
            if round_ == 1 and model == 'BM25':
                file_path = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bm25_model_res/final_res/'
                filename = '<DatasetName>_01_BM25_LoO_Results.txt'.replace('<DatasetName>', dataset)
            print(model, filename)

            sys.stdout.flush()
            tmp_res_dict = get_ranking_effectiveness_for_res_file_per_query(
                file_path=file_path,
                filename=filename,
                qrel_filepath=qrel_filepath,
                calc_ndcg_mrr=True,
                remove_low_quality=False)

            insert_row = [dataset.upper(), model, round_]
            for measure in measures:
                insert_row.append(tmp_res_dict['all'][measure])
            summary_df.loc[next_idx] = insert_row
            next_idx += 1
    summary_df.to_csv('BERT_CHECKS.tsv', sep = '\t',index = False)



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
        inner_fold = sys.argv[3]
        round_limit = sys.argv[4]
        asrc_data_parser(filepath, inner_fold, round_limit)

    elif operation == 'ASRCFeat':
        rel_filepath = sys.argv[2]
        inner_fold = sys.argv[3]
        round_limit = sys.argv[4]
        create_base_features_for_asrc(rel_filepath, inner_fold, round_limit)

    elif operation == 'ASRCFeatLTR':
        rel_filepath = sys.argv[2]
        inner_fold = sys.argv[3]
        round_limit = sys.argv[4]
        create_base_features_for_asrc_with_ltr_features(rel_filepath, inner_fold, round_limit)

    elif operation == 'ASRCFileUnite':
        big_model = sys.argv[2]
        snap_limit = int(sys.argv[3])
        ret_model = sys.argv[4]
        dataset_name = sys.argv[5]
        round_limit = int(sys.argv[6])
        significance_type = sys.argv[7]
        leave_one_out_train = ast.literal_eval(sys.argv[8])
        backward_elimination = ast.literal_eval(sys.argv[9])
        snap_num_as_hyper_param = ast.literal_eval(sys.argv[10])
        limited_snap_num = sys.argv[11]
        with_bert_as_feature = ast.literal_eval(sys.argv[12])
        limited_features_list = ast.literal_eval(sys.argv[13])

        unite_asrc_data_results(
            big_model=big_model,
            snap_limit=snap_limit,
            ret_model=ret_model,
            dataset_name=dataset_name,
            round_limit=round_limit,
            significance_type=significance_type,
            leave_one_out_train=leave_one_out_train,
            backward_elimination=backward_elimination,
            snap_num_as_hyper_param=snap_num_as_hyper_param,
            limited_snap_num=limited_snap_num,
            with_bert_as_feature=with_bert_as_feature,
            limited_features_list=limited_features_list)

    elif operation == 'LTSASRCFileUnite':
        big_model = sys.argv[2]
        snap_limit = int(1)
        ret_model = 'BM25'
        dataset_name = sys.argv[3]
        round_limit = int(sys.argv[4])
        significance_type = 'TTest'
        leave_one_out_train = True
        backward_elimination = False
        snap_num_as_hyper_param = False
        limited_snap_num = 'All'
        with_bert_as_feature = False
        limited_features_list = ast.literal_eval(sys.argv[5])

        unite_asrc_data_results(
            big_model=big_model,
            snap_limit=snap_limit,
            ret_model=ret_model,
            dataset_name=dataset_name,
            round_limit=round_limit,
            significance_type=significance_type,
            leave_one_out_train=leave_one_out_train,
            backward_elimination=backward_elimination,
            snap_num_as_hyper_param=snap_num_as_hyper_param,
            limited_snap_num=limited_snap_num,
            with_bert_as_feature=with_bert_as_feature,
            limited_features_list=limited_features_list,
            first_round_default=4)

    elif operation == 'ASRCSVMWeights':
        dataset_name = sys.argv[2]
        round_limit = int(sys.argv[3])
        limited_features_list = ast.literal_eval(sys.argv[4])

        handle_rank_svm_params_asrc(
            dataset_name=dataset_name,
            round_limit=round_limit,
            limited_features_list=limited_features_list)

    elif operation == 'ASRCFeatLTRFiles':
        doc_file_path = sys.argv[2]
        inner_fold = sys.argv[3]
        round_limit = sys.argv[4]
        only_feats = ast.literal_eval(sys.argv[5])
        create_all_files_for_competition_features(
            inner_fold=inner_fold,
            round_limit=round_limit,
            doc_file_path=doc_file_path,
            only_feats=only_feats)

    elif operation == 'RelCompare':
        compare_rel_files_to_curr_comp()

    elif operation == 'MixtureModelPreprocess':
        dataset_name = sys.argv[2]
        num_rounds = int(sys.argv[3])
        trectext_file = sys.argv[4]
        mixture_model_pre_process(
            dataset_name = dataset_name,
            num_rounds=num_rounds,
            trectext_file=trectext_file)
    elif operation == 'LMScores':
        add_lm_score_files()

    elif operation == 'AblationSummary':
        dataset_name = sys.argv[2]
        round_limit = int(sys.argv[3])
        limited_snap_num = sys.argv[4]
        limited_features_list= ast.literal_eval(sys.argv[5])
        orginize_ablation_results(
            dataset_name,
            round_limit,
            limited_snap_num,
            limited_features_list)
    elif operation == 'MMFeatureCompare':
        dataset_name = sys.argv[2]
        round_limit = int(sys.argv[3])
        limited_snap_num = sys.argv[4]
        limited_features_list = ast.literal_eval(sys.argv[5])
        orginize_mm_features_data_compare(
            dataset_name,
            round_limit,
            limited_snap_num,
            limited_features_list)
    elif operation == 'LTSMMFeatureCompare':
        dataset_name = sys.argv[2]
        round_limit = int(sys.argv[3])
        limited_snap_num = sys.argv[4]
        limited_features_list = ast.literal_eval(sys.argv[5])
        first_round_default = 4
        orginize_mm_features_data_compare(
            dataset_name,
            round_limit,
            limited_snap_num,
            limited_features_list,
            first_round_default=first_round_default)

    elif operation == 'FixKS':
        fix_ks_files_and_produce_stats()

    elif operation == 'LQStats':
        low_quality_satats()

    elif operation == 'BERTStats':
        bert_checker()
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