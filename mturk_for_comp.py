import sys
import boto3
import xmltodict
import pandas as pd
from bs4 import BeautifulSoup

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
        boto_client,
        curr_round_data,
        query_and_init_doc_data,
        questions_html,
        curr_round_file):

    summary_df = pd.DataFrame(columns = ['Query', 'User', 'HitID','HitURL'])
    next_idx = 0
    for query in query_and_init_doc_data:
        q_text = query_and_init_doc_data[query]['query_text']
        q_descrip = query_and_init_doc_data[query]['description']
        for user in curr_round_data[query]:
            fulltext = curr_round_data[query][user]
            new_hit = boto_client.create_hit(
                query=q_text,
                description=q_descrip,
                current_document=fulltext,
                Keywords='search, relevance',
                Reward='0.05',
                MaxAssignments=5,
                LifetimeInSeconds=259200,
                AssignmentDurationInSeconds=600,
                AutoApprovalDelayInSeconds=14400,
                Question=questions_html)
            insert_row = [query, user, new_hit['HIT']['HITId'], "https://workersandbox.mturk.com/mturk/preview?groupId=" +new_hit['HIT']['HITGroupId']]
            print("https://workersandbox.mturk.com/mturk/preview?groupId=" +new_hit['HIT']['HITGroupId'])
            summary_df.loc[next_idx] = insert_row
            next_idx += 1
    summary_df.to_csv('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/HITs/' + curr_round_file.split('/')[-1])

if __name__ == '__main__':
    param_path = sys.argv[1]
    curr_round_file = sys.argv[2]
    curr_round_file = "/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/" + curr_round_file
    query_and_init_doc_data = read_initial_data("documents.trectext", "topics.full.xml")
    curr_round_data = read_current_doc_file(curr_round_file)
    client = create_boto_client(param_path)

    with open('questions.xml','r') as f:
        questions_html = f.read()

    create_hits_in_mturk(
        boto_client=client,
        curr_round_data=curr_round_data,
        query_and_init_doc_data=query_and_init_doc_data,
        questions_html=query_and_init_doc_data,
        curr_round_file=curr_round_file)


