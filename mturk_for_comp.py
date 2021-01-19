import sys
import boto3
import pandas as pd
from bs4 import BeautifulSoup
from jobs import get_relevant_docs_df

def read_current_doc_file(doc_filepath):
    stats = {}
    with open(doc_filepath, 'r') as f:
        soup = BeautifulSoup(f.read())
    all_docs = soup.find_all('doc')

    for doc_ in list(all_docs):
        docno = doc_.find('docno').text
        fulltext = doc_.find('text').text
        query = docno.split('-')[0]
        user =docno.split('-')[1]
        if query not in stats:
            stats[query] = {}

        stats[query][user] = fulltext

    return stats

def read_initial_data(docs_path, query_meta_path):
    stats = {}
    with open(docs_path, 'r') as f:
        soup = BeautifulSoup(f.read())
    all_docs = soup.find_all('doc')

    for doc_ in list(all_docs):
        docno = doc_.find('docno').text
        fulltext = doc_.find('text').text
        query = docno.split('-')[2]
        stats[query] = {}
        stats[query]['document'] = fulltext

    with open(query_meta_path, 'r') as f:
        soup = BeautifulSoup(f.read())
    all_q = soup.find_all('topic')
    for query in all_q:
        q_num = query['number'].zfill(3)
        if q_num in stats:
            q_text = query.find('query').text
            q_describe = query.find('description').text
            stats[q_num]['query_text'] = q_text
            stats[q_num]['description'] = q_describe
    return stats


def create_boto_client(param_patt):
    with open(param_patt, 'r') as f:
        params = f.read().split('\n')
    region_name = 'us-east-1'
    aws_id = params[0]
    aws_sa_key = params[1]

    endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

    client = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_sa_key,
    )
    print("Curr Balance in Account:")
    print(client.get_account_balance()['AvailableBalance'])
    return client

def create_hits_in_mturk(
        curr_round_data,
        query_and_init_doc_data,
        curr_round_file):

    summary_df = pd.DataFrame(columns = ['query', 'description', 'current_document', 'user'])
    hit_df = pd.DataFrame(columns=['query', 'description', 'current_document'])
    next_idx = 0
    for query_num in query_and_init_doc_data:
        query = query_and_init_doc_data[query_num]['query_text']
        description = query_and_init_doc_data[query_num]['description']
        for group in ['', '_0', '_1', '_2']:
            query_num_ = query_num + group
            if query_num_ in curr_round_data:
                for user in curr_round_data[query_num_]:
                    current_document = curr_round_data[query_num_][user]
                    insert_row = [query, description, current_document, user]
                    summary_df.loc[next_idx] = insert_row
                    insert_row = [query, description, current_document]
                    hit_df.loc[next_idx] = insert_row
                    next_idx += 1
    summary_df.to_csv('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/HITs/Summary_' + curr_round_file.split('/')[-1] + '.tsv', sep ='\t', index = False)
    hit_df.to_csv('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/HITs/HITs_' + curr_round_file.split('/')[-1] + '.csv', index = False)

def create_qrel_string_for_round(
        round_num,
        query_and_init_doc_data,
        round_ts,
        batch_file):

    qrel_str = ""
    hit_score_df = pd.read_csv('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/HITs/' + batch_file, index_col = False)
    hit_summary_df = pd.read_csv('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/HITs/Summary_' + round_ts +'.tsv', sep = '\t',index_col = False)
    score_df_len = len(hit_score_df)
    hit_df_len = len(hit_summary_df)
    hit_score_df.rename(columns = {'Input.current_document' : 'current_document',
                                   'Input.description'      : 'description',
                                   'Input.query'            : 'query'}, inplace = True)

    merged_df = pd.merge(
        hit_score_df[['current_document', 'description', 'query', 'WorkTimeInSeconds', 'Answer.this_document_is']],
        hit_summary_df,
        on = ['current_document', 'description', 'query'],
        how = 'inner')

    if len(merged_df) < score_df_len:
        raise Exception("create_qrel_string_for_round: Merge Prob! hit df len = " +str(score_df_len) +' merged len = ' + str(len(merged_df)))

    users = list(merged_df['user'].drop_duplicates())
    hit_count = 0
    for user in users:
        curr_user_df = merged_df[merged_df['user'] == user].copy()
        q_list = list(curr_user_df['query'].drop_duplicates())
        for q in q_list:
            curr_user_query_df = curr_user_df[curr_user_df['query'] == q].copy()
            for q_num in query_and_init_doc_data:
                if query_and_init_doc_data[q_num]['query_text'] == q:
                    curr_q_num = q_num
                    break
            docno = 'EPOCH-' + round_num + '-' + curr_q_num + '-' + user
            rel_score = clac_rel_for_doc(curr_user_query_df)
            qrel_str += curr_q_num + ' 0 ' + docno + ' ' + str(rel_score) + '\n'
            hit_count += 1
    if hit_count != hit_df_len:
        print("Somthings wrong...")
    return qrel_str

def clac_rel_for_doc(
        doc_df):

    if len(doc_df) < 5:
        raise Exception("Not enough rel judgments")
    if len(doc_df) > 5:
        work_time_list = sorted(list(doc_df['WorkTimeInSeconds']))
        bench = work_time_list[-5]
        doc_df = doc_df[doc_df['WorkTimeInSeconds'] >= bench]
        if bench == work_time_list[-6]:
            doc_df = doc_df.sort_values('WorkTimeInSeconds')
            doc_df = doc_df.head(5)
    if len(doc_df) != 5:
        raise Exception("clac_rel_for_doc: Prob")
    score = 0
    for index, row in doc_df.iterrows():
        if row['Answer.this_document_is'] == 'relevant':
            score += 1
    if score >= 3:
        score = score - 2
    else:
        score = 0
    return score

def create_qrel_file(file_mapping_dict, query_and_init_doc_data):
    qrel_string = ""
    for round_ in file_mapping_dict:
        qrel_string += create_qrel_string_for_round(
                            round_num=round_,
                            query_and_init_doc_data=query_and_init_doc_data,
                            round_ts=file_mapping_dict[round_]['TS'],
                            batch_file=file_mapping_dict[round_]['Hit'])

    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/curr_comp.rel', 'w') as f:
        f.write(qrel_string)

def creat_trectext_for_all_rounds(file_mapping_dict):
    trectext_str = ""
    for round_ in file_mapping_dict:
        round_ts = file_mapping_dict[round_]['TS']
        with open('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/' + round_ts, 'r') as f:
            curr_file = f.read()
        curr_file = curr_file.replace('<DOCNO>', '<DOCNO>EPOCH-' + round_ + '-')
        trectext_str += curr_file + '\n'
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/comp2020/comp2020.terctext', 'w') as f:
        f.write(trectext_str)

def split_res_files_to_groups(file_mapping_dict):
    group_ref_dict = read_current_doc_file("/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/" + file_mapping_dict['01']['TS'])
    rel_groups = ['_0', '_1']
    terctext_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/data/datsets/comp2020/comp2020.terctext'
    qrel_filepath = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/qrels/curr_comp.rel'

    group_file_dict = {}
    for grp in rel_groups:
        group_file_dict[grp] = {}
        group_file_dict[grp]['TrecText'] = ""
        group_file_dict[grp]['Qrel'] = ""
        group_file_dict[grp]['TrecTextCount'] = 0
        group_file_dict[grp]['QrelCount'] = 0

    with open(terctext_filepath, 'r') as f:
        soup = BeautifulSoup(f.read())
    all_docs = soup.find_all('doc')

    for doc_ in list(all_docs):
        docno = doc_.find('docno').text
        fulltext = doc_.find('text').text
        query = docno.split('-')[2]
        grp = '_' + query.split('_')[1]
        user = docno.split('-')[3]
        if grp in rel_groups:
            group_file_dict[grp]['TrecText'] += '<DOC>\n'
            group_file_dict[grp]['TrecText'] += '<DOCNO>' + docno.replace(grp + '-', '-') + '</DOCNO>\n'
            group_file_dict[grp]['TrecText'] += '<TEXT>\n'
            group_file_dict[grp]['TrecText'] += fulltext
            group_file_dict[grp]['TrecText'] += '\n</TEXT>\n'
            group_file_dict[grp]['TrecText'] += '</DOC>\n'
            group_file_dict[grp]['TrecTextCount'] += 1

    qrel_df = get_relevant_docs_df(qrel_filepath)
    for index, row in qrel_df.iterrows():
        docno = row['Docno']
        query = docno.split('-')[2]
        user = docno.split('-')[3]
        for grp in rel_groups:
            if user in group_ref_dict[query + grp]:
                group_file_dict[grp]['Qrel'] += query + ' 0 ' + docno + ' ' + str(row['Relevance']) + '\n'
                group_file_dict[grp]['QrelCount'] += 1

    for grp in rel_groups:
        file_addition = grp.replace('_0','')
        print("Group: " + grp + ' #TrecDocs : ' + str(group_file_dict[grp]['TrecTextCount']) + "  #QrelDocs : "  + str(group_file_dict[grp]['QrelCount']))
        with open(terctext_filepath + file_addition, 'w') as f:
            f.write(group_file_dict[grp]['TrecText'])
        with open(qrel_filepath + file_addition, 'w') as f:
            f.write(group_file_dict[grp]['Qrel'])







if __name__ == '__main__':
    operation = sys.argv[1]
    query_and_init_doc_data = read_initial_data("documents.trectext", "topics.full.xml")
    if operation == 'HIT':
        round_list = ['2021-01-19-01-59-48-181622']
        for curr_round_file in round_list:
            print(curr_round_file)
            curr_round_file = "/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/" + curr_round_file
            curr_round_data = read_current_doc_file(curr_round_file)
            create_hits_in_mturk(curr_round_file=curr_round_file,
                                 curr_round_data=curr_round_data,
                                 query_and_init_doc_data=query_and_init_doc_data)
    elif operation == 'CreateFiles':
        file_mapping_dict = {
            '01' : {'TS' : '2021-01-10-22-42-25-566245',
                    'Hit': 'Batch_4306887_batch_results.csv'},
            '02': {'TS': '2021-01-14-22-37-00-218428',
                   'Hit': 'Batch_4306972_batch_results.csv'},
            '03': {'TS': '2021-01-19-01-59-48-181622',
                   'Hit': 'Batch_4307491_batch_results.csv'},
            # '04': {'TS': '2020-12-02-22-02-13-998936',
                   # 'Hit': 'Batch_4274149_batch_results.csv'},
            # '05': {'TS': '2020-12-09-22-44-03-416874',
                   # 'Hit': 'Batch_4275854_batch_results.csv'}
            }
        create_qrel_file(file_mapping_dict, query_and_init_doc_data)
        creat_trectext_for_all_rounds(file_mapping_dict)
        split_res_files_to_groups(file_mapping_dict)



