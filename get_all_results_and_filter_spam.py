
import os
import sys
import ast
import subprocess
import pandas as pd
from utils import *


def create_spam_score_index(
        path = '/lv_local/home/zivvasilisky/ziv/env/indri/query/spamer',
        file_name = 'clueweb09spam.Fusion'):
    index_dic = {}
    with open(os.path.join(path, file_name), 'r') as f:
        file_lines = f.readlines()

    for line in file_lines:
        line = line.strip()
        if len(line)> 0:
            splitted_line = line.split(' ')
            index_dic[splitted_line[1]] = splitted_line[0]
    with open(os.path.join(path, file_name + '.json'), 'w') as f:
        f.write(str(index_dic))

    return index_dic







if __name__=='__main__':

    run_retrival = convert_str_to_bool(sys.argv[1])
    run_create_spam_index = convert_str_to_bool(sys.argv[2])
    specific_queries_list = ast.literal_eval(sys.argv[3])

    num_of_required_results_per_query = 1000
    dir_path = '/lv_local/home/zivvasilisky/ziv/env/indri/query/query_url/'
    query_list = list(range(1,201))
    if specific_queries_list is not None:
        query_list = specific_queries_list
    # query_num_list = ['195','193','188','180','177','011','018','051','167','144','032','002','161',
    #               '059','036','124','098','069','009','004','048','029','166','182','164','017','033',
    #               '045','010','078','034']
    query_num_list = []
    if run_retrival == True:
        print ("Running query retrieval...")
        for i in query_list:
            query_num = (3 - len(str(i))) * '0' + str(i)
            if i not in query_num_list:
                res = subprocess.check_call(['/lv_local/home/zivvasilisky/ziv/env/indri/query/url_retrival.sh', query_num])
                query_num_list.append(query_num)
    else:
        for i in query_list:
            query_num = (3 - len(str(i))) * '0' + str(i)
            query_num_list.append(query_num)

    if run_create_spam_index == True:
        print("Creating spam score index...")
        spam_index_dict = create_spam_score_index()
    else:
        print("Loading spam score index...")
        with open('/lv_local/home/zivvasilisky/ziv/env/indri/query/spamer/clueweb09spam.Fusion.json', 'r') as f:
            spam_index_dict = ast.literal_eval(f.read())


    print ("Running spam filter...")
    big_df = pd.DataFrame(columns = ['Docno','QueryNum','Url'])
    next_index = 0

    for filename in os.listdir(dir_path):
        if filename.endswith('urls.txt'):
            query_num = filename.split('_')[1]
            if query_num not in query_num_list:
                continue
            with open(dir_path + filename, 'r') as f:
                cur_file_txt = f.read()
            print("Working on :" + query_num)
            all_lines = cur_file_txt.split('\n')
            good_docs = 0
            for line in all_lines:
                splitted_line = line.split(' ')
                if len(splitted_line) == 1:
                    continue
                else:
                    big_df.loc[next_index] = [splitted_line[0], query_num, splitted_line[1]]
                    next_index += 1

    big_df['Spam_score'] = big_df['Docno'].apply(lambda x: int(spam_index_dict[x]))
    index_drop_list = []
    query_num = ""
    for index, row in big_df.iterrows():
        if row['QueryNum'] != query_num:
            if query_num != '' and good_docs < num_of_required_results_per_query:
                print("Query " + str(query_num) + " Not enough docs!")
            good_docs = 0
            query_num = row['QueryNum']

        if good_docs < num_of_required_results_per_query:
            if row['Spam_score'] < 50:
                index_drop_list.append(index)
            else:
                good_docs += 1
        else:
            index_drop_list.append(index)

    big_df.to_csv(dir_path + 'all_urls_no_spam.tsv', sep='\t', index=False)
    big_df = big_df.drop(index_drop_list)
    if specific_queries_list is not None:
        former_df = pd.read_csv(dir_path + 'all_urls_no_spam_filtered.tsv', sep='\t', index_col=False)
        big_df = former_df.append(big_df, ignore_index=True)
        big_df.drop_duplicates(inplace=True)
    big_df.to_csv(dir_path + 'all_urls_no_spam_filtered.tsv', sep='\t', index=False)




