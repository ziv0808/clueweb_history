from bs4 import BeautifulSoup
from pymongo import MongoClient
from copy import copy,deepcopy
from random import shuffle,seed
import csv
import ast
import unicodedata

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

def retrieve_users():
    return ["K5W94I","FQ5ADE","ZZ9G1J","55BITZ","0H1M36","AS64YR","MPNM2K","B8F6E9","632S70","YZ797R","R63H9T","1HAIVS","IL2Z4J","OAR3JM","SQK6UG","6WBO8B"
        , "4IB6XC","P3VUZZ","U3BWFS","ECXXPC","YF1WGX","T04YHN","GVWD05","KN0P86","IR1LI3","TXOH1U","USDW0H","YR85MD","W0N7XO","E4XDKY","WZZIQQ","PW656R","UAYNLZ","319WBN","S8C22R"
        , "37ZSCA","PBXOHX","S33OJM","TX5GTH","8G7EXD","5PFRI1","H8MTZJ","P4XXKM","B4UN6O","GF7KCF","R3CHA3","ELOCB2","5RJQ5W","8QHNXA","0H4T7V","93TF3E","FBZ7PM","J2O460"
        ,"XK1O1C","SYK2VV","EHOVYP","WPGM77","6OJZI1","5440N4","5KBCIG","G3UP0M","XW13YM","678QUM","JUKQRU","BCI0MP","J5YEUV","ITIHJZ","C07X4Q","MR6B6Q","Y150M4","7DSLJS","KKS1QA"
        ,"MRCU75","IM4W6P","VS0C4F","6QRS72","JJN2DN","E9NUZ6","AQ5V72","F30TB1","BNFNVS","8ZJ42Z","4EE20G","4BJPNQ"]

def expand_queries_to_participates(queries,number_of_rankers,number_of_participants):
    expanded_version=[]
    for query in queries:
        for i in range(number_of_rankers):
            for j in range(number_of_participants):
                expanded_version.append(query+"_"+str(i)+"_"+str(j))
    return expanded_version

def expand_working_qeuries(queries, number_of_groups):
    expanded_version = []
    for query in queries:
        for i in range(number_of_groups):
            expanded_version.append(str(query) + "_" + str(i))
    return expanded_version



def get_query_to_user(user_query_map,queries):
    qtu = {q:[] for q in  queries}
    for user in user_query_map:
        for query in user_query_map[user]:
            qtu[query].append(user)
    return qtu




def test_mapping(new_map,old_map):
    for user in new_map:
        for query in new_map[user]:
            prefix = query.split("_")[0]
            if prefix in old_map[user]:
                return False
    return True


def update_user_banned_queries(user_banned_queries,user_groups,user,queries):
    result=[q for q in queries if q.split("_")[1] in user_groups]
    user_banned_queries[user].extend(result)
    user_banned_queries[user]=list(set(user_banned_queries[user]))
    return user_banned_queries

def user_query_mapping(users, expanded_queries, number_of_user_per_query):
    user_query_map = {user: [] for user in users}
    user_groups = {user: [] for user in users}
    user_banned_query_map = {user: [] for user in users}
    groups = ["0","1","2","3","4"]
    user_ranker_index = {}
    working_queries = copy(expanded_queries)
    queries_competitor_number = {query: 0 for query in expanded_queries}
    while working_queries:
        for user in users:
            tmp = list(set(working_queries) - set(user_banned_query_map[user]))
            shuffle(tmp)
            if not tmp:continue
            query_to_user = get_query_to_user(user_query_map,expanded_queries)
            query = get_query_for_user(user_query_map,query_to_user,user,tmp)
            user_groups[user].append(query.split("_")[1])
            user_query_map[user].append(query)
            if not user_ranker_index.get(user,False):
                user_ranker_index[user]=query.split("_")[1]
            user_banned_query_map[user].append(query)
            more_groups = [i for i in groups if i != query.split("_")[1]]
            for group in more_groups:
                user_banned_query_map[user].append(query.split("_")[0]+"_"+group)
            user_banned_query_map=update_user_banned_queries(user_banned_query_map,user_groups[user],user,tmp)
            queries_competitor_number[query] += 1
            working_queries = [q for q in working_queries if queries_competitor_number[q] < number_of_user_per_query]
            if not working_queries:
                break
    return user_query_map

def user_query_mapping_z(users, queries, number_of_user_per_query, num_of_queries_with_additional_user, max_allowed_ovelap):
    user_query_map = {user: [] for user in users}
    query_user_map = {query: [] for query in queries}
    queries_competitor_number = {query: 0 for query in queries}
    working_queries = copy(queries)
    shuffle(working_queries)
    finished_users = []
    finished_queries = []
    for query in working_queries:
        possible_users = list(set(users) - set(finished_users))
        shuffle(possible_users)
        for user in possible_users:
            if num_of_queries_with_additional_user > 0:
                if queries_competitor_number[query] == number_of_user_per_query + 1:
                    num_of_queries_with_additional_user -= 1
                    print("Query: " + query)
                    finished_queries.append(query)
                    break
            else:
                if queries_competitor_number[query] == number_of_user_per_query:
                    print("Query: " +query)
                    finished_queries.append(query)
                    break
            curr_max_overlap, user_overlap_dict = find_user_query_overlaps(user_query_map, query_user_map, user, query)
            if curr_max_overlap < max_allowed_ovelap:
                query_user_map[query].append(user)
                user_query_map[user].append(query)
                queries_competitor_number[query] += 1
                if len(user_query_map[user]) == 3:
                    finished_users.append(user)

    if queries_competitor_number[query] == number_of_user_per_query:
        print("Query: " + query)
        finished_queries.append(query)

    if len(finished_users) != len(users):
        print('Not all users assined: ' +str(len(finished_users)))
    if len(finished_queries) != len(queries):
        print('Not all queries assined: ' + str(len(finished_queries)))
    for user in users:
        curr_max_overlap, user_overlap_dict = find_user_query_overlaps(user_query_map, query_user_map, user, query)
        if len(user_query_map[user]) < 3:
            print(user, user_query_map[user])
    return user_query_map, query_user_map


def find_user_query_overlaps(user_query_map, query_user_map, user, curr_query):
    num_overlap = {}
    for query in user_query_map[user]:
        for inner_user in query_user_map[query]:
            if inner_user != user:
                if inner_user in num_overlap:
                    num_overlap[inner_user] += 1
                else:
                    num_overlap[inner_user] = 1

    max_overlap = 0
    for inner_user in query_user_map[curr_query]:
        if inner_user in num_overlap:
            if num_overlap[inner_user] > max_overlap:
                max_overlap = num_overlap[inner_user]
    return max_overlap, num_overlap


# def get_query_for_user(user_to_query,query_to_user,user,working_set):
#     query_overlap_count={q:0 for q in working_set}
#     for query in working_set:
#         users = set(deepcopy(query_to_user[query]))
#         for q in user_to_query[user]:
#             users_of_query= set(deepcopy(query_to_user[q]))
#             query_overlap_count[query]+=len(users.intersection(users_of_query))
#     query_result = sorted(working_set,key = lambda x: query_overlap_count[x])[0]
#     return query_result



def upload_data_to_mongo(data,user_query_map):
    client = MongoClient('asr2.iem.technion.ac.il', 27017)
    db = client.asr16

    for user in user_query_map:
        for query in user_query_map[user]:
            object={}
            object["username"]=user
            object["current_document"]=data[query]["document"]
            object["posted_document"]=data[query]["document"]
            object["query_id"] = query
            object["query"]=data[query]["query_text"]
            object["description"]=data[query]["description"]
            db.documents.insert(object)

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

def compare_doc_files(curr_stats, old_stats, is_first, rank_dict = None):
    unchanged_docs = 0
    unchanged_users_dict = {}
    if rank_dict is not None:
        winner_unchanged = 0
        loser_unchanged = 0
        under_4_unchanged = 0
    for query in curr_stats:
        for user in curr_stats[query]:
            if is_first == True:
                if unicodedata.normalize('NFKD', old_stats[query]['document']).encode('cp1252', "ignore").decode('utf-8', 'replace').replace(u'\uFFFD', ' ').strip() == unicodedata.normalize('NFKD',curr_stats[query][user]).encode('cp1252', "ignore").decode('utf-8', 'replace').replace(u'\uFFFD', ' ').strip():
                    unchanged_docs += 1
                    if user not in unchanged_users_dict:
                        unchanged_users_dict[user] = 1
                    else:
                        unchanged_users_dict[user] += 1
            else:
                if unicodedata.normalize('NFKD', old_stats[query][user]).encode('cp1252', "ignore").decode('utf-8',
                                                                                                                 'replace').replace(
                        u'\uFFFD', ' ').strip() == unicodedata.normalize('NFKD', curr_stats[query][user]).encode('cp1252',
                                                                                                                  "ignore").decode(
                        'utf-8', 'replace').replace(u'\uFFFD', ' ').strip():
                    unchanged_docs += 1
                    if user not in unchanged_users_dict:
                        unchanged_users_dict[user] = 1
                    else:
                        unchanged_users_dict[user] += 1
                    if rank_dict is not None:
                        if rank_dict[query + '-' + user] == 1:
                            winner_unchanged +=1
                        else:
                            loser_unchanged += 1
                        if rank_dict[query + '-' + user] > 4:
                            under_4_unchanged += 1
    print('Unchnged Docs: ' + str(unchanged_docs))
    if rank_dict is not None:
        print('Unchnged Docs Winner: ' + str(winner_unchanged))
        print('Unchnged Docs Loser: ' + str(loser_unchanged))
        print('Unchnged Docs Under 4: ' + str(under_4_unchanged))

    unchanged_users = 0
    for user in unchanged_users_dict:
        if unchanged_users_dict[user] >= 3:
            unchanged_users += 1

    print('Unchnged Users: ' + str(unchanged_users))

def compare_doc_files_vs_db(curr_stats, is_first, rank_dict = None):
    unchanged_docs = 0
    unchanged_users_dict = {}
    if rank_dict is not None:
        winner_unchanged = 0
        loser_unchanged = 0
        under_4_unchanged = 0

    client = MongoClient('asr2.iem.technion.ac.il', 27017)
    db = client.asr16
    documents = db.documents.find({}).sort('query_id', 1)
    for document in documents:
        query = str(document['query_id']).zfill(3)
        user =  str(document['username'])
        if is_first == True:
            if unicodedata.normalize('NFKD', curr_stats[query]['document']).encode('cp1252', "ignore").decode('utf-8', 'replace').replace(u'\uFFFD', ' ').strip() == unicodedata.normalize('NFKD',document['current_document']).encode('cp1252', "ignore").decode('utf-8', 'replace').replace(u'\uFFFD', ' ').strip():
                unchanged_docs += 1
                if user not in unchanged_users_dict:
                    unchanged_users_dict[user] = 1
                else:
                    unchanged_users_dict[user] += 1
        else:
            if unicodedata.normalize('NFKD', document['current_document']).encode('cp1252', "ignore").decode('utf-8',
                                                                                                             'replace').replace(
                    u'\uFFFD', ' ').strip() == unicodedata.normalize('NFKD', curr_stats[query][user]).encode('cp1252',
                                                                                                              "ignore").decode(
                    'utf-8', 'replace').replace(u'\uFFFD', ' ').strip():
                unchanged_docs += 1
                if user not in unchanged_users_dict:
                    unchanged_users_dict[user] = 1
                else:
                    unchanged_users_dict[user] += 1
                if rank_dict is not None:
                    if rank_dict[query + '-' + user] == 1:
                        winner_unchanged +=1
                    else:
                        loser_unchanged += 1
                    if rank_dict[query + '-' + user] > 4:
                        under_4_unchanged += 1
    print('Unchnged Docs: ' + str(unchanged_docs))
    if rank_dict is not None:
        print('Unchnged Docs Winner: ' + str(winner_unchanged))
        print('Unchnged Docs Loser: ' + str(loser_unchanged))
        print('Unchnged Docs Under 4: ' + str(under_4_unchanged))

    unchanged_users = 0
    for user in unchanged_users_dict:
        if unchanged_users_dict[user] >= 3:
            unchanged_users += 1

    print('Unchnged Users: ' + str(unchanged_users))


def parse_res_file(filepath):
    with open(filepath , 'r') as f:
        file_content = f.read()
    file_content = file_content.split('\n')
    res_dict = {}
    for line_ in file_content:
        if len(line_.split(' ')) > 1:
            q = line_.split(' ')[0]
            docno = line_.split(' ')[2]
            rank = int(line_.split(' ')[3])
            res_dict[docno] = rank
    return res_dict


def test_number_of_queries(mapping,number_of_queries):
    for user in mapping:
        if len(mapping[user])!=number_of_queries:
            return False
    return True

def changeStatus():
    client = MongoClient('asr2.iem.technion.ac.il',27017)
    db = client.asr16
    status = db.status.find_one({})
    indeterminate = status['indeterminate']
    status['indeterminate'] = not(indeterminate)
    db.status.save(status)

def get_curr_user_query_mapping_and_backup_doc_collection():
    client = MongoClient('asr2.iem.technion.ac.il', 27017)
    db = client.asr16
    documents = db.documents.find({}).sort('query_id', 1)
    backup_list = []
    user_to_query_dict = {}
    for document in documents:
        backup_list.append(document)
        if document['username'] not in user_to_query_dict:
            user_to_query_dict[document['username']] = [document['query_id']]
        else:
            user_to_query_dict[document['username']].append(document['query_id'])
    with open('/lv_local/home/zivvasilisky/ASR20/bkup/BackupDocs.txt', 'w') as f:
        f.write(str(backup_list))
    with open('/lv_local/home/zivvasilisky/ASR20/bkup/UserQueryMapping.txt', 'w') as f:
        f.write(str(user_to_query_dict))

def get_curr_user_query_mapping():
    with open('/lv_local/home/zivvasilisky/ASR20/bkup/UserQueryMapping.txt', 'r') as f:
        user_q_curr_map = ast.literal_eval(f.read())
    return user_q_curr_map


seed(9001)
users = retrieve_users()
data = read_initial_data("documents.trectext", "topics.full.xml")
queries = list(data.keys())
user_q_curr_map = get_curr_user_query_mapping()

# data_round_1 = read_current_doc_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/2020-11-09-23-55-23-857656')
# print("First Round VS Zero:")
# compare_doc_files(data_round_1, data, is_first=True)
#
# data_round_2 = read_current_doc_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/2020-11-17-10-30-21-396460')
# print("Second Round VS Zero:")
# compare_doc_files(data_round_2, data, is_first=True)

# data_round_3 = read_current_doc_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/2020-11-23-23-12-59-474081')
# print("Third Round VS Zero:")
# compare_doc_files(data_round_3, data, is_first=True)
#
# rank_round_1 = parse_res_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/RankedLists/LambdaMART2020-11-09-23-55-23-857656')
# rank_round_2 = parse_res_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/RankedLists/LambdaMART2020-11-17-10-30-21-396460')

# print(data)
# print(data_round_1)

# print("First Round VS Second:")
# compare_doc_files(data_round_1, data_round_2, is_first=False, rank_dict=rank_round_1)
# print("Second Round VS Third:")
# compare_doc_files(data_round_2, data_round_3, is_first=False, rank_dict=rank_round_2)

# data_round_3 = read_current_doc_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/2020-11-23-23-12-59-474081')
# rank_round_3 = parse_res_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/RankedLists/LambdaMART2020-11-23-23-12-59-474081')
#
# print("Curr Round VS Third:")
# compare_doc_files_vs_db(data_round_3,is_first=False,rank_dict=rank_round_3)
# print("Curr Round VS Zero:")
# compare_doc_files_vs_db(data,is_first=True)

# data_round_4 = read_current_doc_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/2020-12-02-22-02-13-998936')
# rank_round_4 = parse_res_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/RankedLists/LambdaMART2020-12-02-22-02-13-998936')
#
# print("Curr Round VS 4TH:")
# compare_doc_files_vs_db(data_round_4,is_first=False,rank_dict=rank_round_4)
# print("Curr Round VS Zero:")
# compare_doc_files_vs_db(data,is_first=True)

# data_round_5 = read_current_doc_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/2020-12-09-22-44-03-416874')
# rank_round_5 = parse_res_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/RankedLists/LambdaMART2020-12-09-22-44-03-416874')
#
# print("Curr Round VS 5TH:")
# compare_doc_files_vs_db(data_round_5,is_first=False,rank_dict=rank_round_5)
# print("Curr Round VS Zero:")
# compare_doc_files_vs_db(data,is_first=True)

# data_round_6 = read_current_doc_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/2020-12-21-09-38-10-759298')
# rank_round_6 = parse_res_file('/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/RankedLists/LambdaMART2020-12-21-09-38-10-759298')
#
# print("Curr Round VS 6TH:")
# compare_doc_files_vs_db(data_round_6,is_first=False,rank_dict=rank_round_6)
# print("Curr Round VS Zero:")
# compare_doc_files_vs_db(data,is_first=True)
# while True:
#     mapping, query_user_map = user_query_mapping_z(
#         users=users,
#         queries=queries,
#         number_of_user_per_query=8,
#         num_of_queries_with_additional_user=1,
#         max_allowed_ovelap=1)
#     if test_number_of_queries(mapping,3):
#         break

# for user in users:
#     curr_max_overlap, user_overlap_dict = find_user_query_overlaps(mapping, query_user_map, user, '002')
#     print(user_overlap_dict)
#
# for q in query_user_map:
#     print(len(query_user_map[q]))
# print(mapping)
# print(query_user_map)
# with open('UserQueryMapping.txt', 'w') as f:
#     f.write(str(mapping))
# with open('QueryUserMapping.txt', 'w') as f:
#     f.write(str(query_user_map))
# mapping = {'4EE20G' : ['048', '010', '002']}
# mapping = {'4BJPNQ' : ['048', '098', '004']}
#
# upload_data_to_mongo(data,mapping)


# changeStatus()