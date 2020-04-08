import os
import sys
import pandas as pd
from utils import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
        work_interval_freq_list = ['1W', '2W', '1M', '2M', 'SIM', 'SIM_995','SIM_TXT_UP_DOWN','SIM_TXT_UP','SIM_TXT_DOWN'],
        lookup_method_list      = ['NoLookup', 'Backward','OnlyBackward','Forward'],
        already_exists          = True,
        work_year               = '2008'):

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
        processed_docs_path = os.path.join('/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+work_year+'/', interval_freq)
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
        work_year = '2008'):

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
        processed_docs_path = os.path.join(
            '/mnt/bi-strg3/v/zivvasilisky/ziv/data/processed_document_vectors/'+work_year+'/', interval_freq)
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
        create_per_interval_per_lookup_cc_dict(work_interval_freq_list=interval_freq_list,
                                               already_exists=already_exists,
                                               work_year=work_year)

    elif operation == 'DFDict':
        interval_freq_list = ast.literal_eval(sys.argv[2])
        already_exists = ast.literal_eval(sys.argv[3])
        work_year = sys.argv[4]
        create_per_interval_per_lookup_df_dict(work_interval_freq_list=interval_freq_list,
                                               already_exists=already_exists,
                                               work_year=work_year)

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
        create_stats_data_frame_for_snapshot_changes(sim_folder_name=sim_folder_name, inner_fold=inner_fold)

    # elif operation == 'SimInterva':
    #     from_int_size = sys.argv[2]
    #     sim_threshold = float(sys.argv[3])
    #     sim_folder_name = sys.argv[4]
    #     create_similarity_interval(from_interval_size=from_int_size,sim_threshold=sim_threshold,sim_folder_name=sim_folder_name)

    elif operation == 'PlotStatsDF':
        filename = sys.argv[2]
        create_snapshot_changes_stats_plots(filename=filename)

    elif operation == 'PlotWLDF':
        filename = sys.argv[2]
        interval_freq = sys.argv[3]
        lookup_method = sys.argv[4]
        plot_stats_vs_winner_plots_for_wl_file(file_name=filename,interval_freq=interval_freq, lookup_method=lookup_method)

    elif operation == "SIMInterval":
        sim_thresold = float(sys.argv[2])
        sim_folder_name = sys.argv[3]
        inner_fold = sys.argv[4]
        work_year = sys.argv[5]
        create_similarity_interval(sim_threshold=sim_thresold,sim_folder_name=sim_folder_name, inner_fold=inner_fold, work_year=work_year)

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