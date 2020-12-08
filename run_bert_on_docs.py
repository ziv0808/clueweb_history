import os
import sys
import ast
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import convert_df_to_trec


def get_query_doc_rel_proba(
        tokenizer,
        model,
        query,
        document):
    inputs = tokenizer.encode_plus(query, document, return_tensors="pt")
    return torch.softmax(model(**inputs)[0], dim=1).tolist()[0][1]


def get_trec_prepared_df_form_res_df(
        scored_docs_df,
        score_colname):


    all_q = sorted(list(scored_docs_df['QueryNum'].drop_duplicates()))
    big_df = pd.DataFrame({})
    for query_num in all_q:
        res_df = pd.DataFrame(columns=['Query_ID', 'Iteration', 'Docno', 'Rank', 'Score', 'Method'])
        next_index = 0
        query_df = scored_docs_df[scored_docs_df['QueryNum'] == query_num].copy()
        for index, row in query_df.iterrows():
            res_df.loc[next_index] = ["0" * (3 - len(str(query_num))) + str(query_num), 'Q0', row['Docno'], 0,
                                      row[score_colname],
                                      'indri']
            next_index += 1

        if res_df.empty == False:
            res_df.sort_values('Score', ascending=False, inplace=True)
            res_df['Rank'] = list(range(1, next_index + 1))
            big_df = big_df.append(res_df, ignore_index=True)

    return big_df

if __name__=="__main__":
    inner_fold = sys.argv[1]
    operation = sys.argv[2]

    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + inner_fold + '/RawData.json', 'r') as f:
        big_doc_index = ast.literal_eval(f.read())

    query_num_to_text = {}
    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/query/query_num_to_text.txt', 'r') as f:
        file_content = f.read()

    file_content = file_content.split('\n')
    for line in file_content:
        if ':' in line:
            query_num_to_text[line.split(':')[0]] = line.split(':')[1].strip()

    tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
    model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

    per_round_res_df_dict = {}
    if operation == 'BaseModel':
        for query_user in big_doc_index:
            query = query_user.split('-')[0]
            print(query_user)
            sys.stdout.flush()
            for round_ in big_doc_index[query_user]:
                if round_ not in per_round_res_df_dict:
                    per_round_res_df_dict[round_] = {}
                    per_round_res_df_dict[round_]['Df'] = pd.DataFrame(columns = ['QueryNum','Docno', 'Score'])
                    per_round_res_df_dict[round_]['Idx'] = 0
                big_doc_index[query_user][round_]['json']['BERTScore'] = get_query_doc_rel_proba(
                                                            tokenizer=tokenizer,
                                                            model=model,
                                                            query=query_num_to_text[query],
                                                            document=big_doc_index[query_user][round_]['json']['Fulltext'])
                docno = "EPOCH-" + str(round_).zfill(2) + '-' + query_user
                print(docno)
                sys.stdout.flush()
                per_round_res_df_dict[round_]['Df'].loc[per_round_res_df_dict[round_]['Idx']] = [query, docno, big_doc_index[query_user][round_]['json']['BERTScore']]
                per_round_res_df_dict[round_]['Idx'] += 1

        with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + inner_fold + '/RawDataWithBERT.json', 'w') as f:
            big_doc_index = f.write(str(big_doc_index))

        save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bert/'
        for round_ in per_round_res_df_dict:
            df = per_round_res_df_dict[round_]['Df']
            score_df = get_trec_prepared_df_form_res_df(df, 'Score')
            save_filename = inner_fold + '_' + str(round_).zfill(2)+'_BERT_Results.txt'
            with open(os.path.join(save_folder, save_filename), 'w') as f:
                big_doc_index = f.write(convert_df_to_trec(score_df))

    if operation == 'ConcatModel':
        for query_user in big_doc_index:
            query = query_user.split('-')[0]
            print(query_user)
            sys.stdout.flush()
            for round_ in big_doc_index[query_user]:
                if int(round_) > 1:
                    if round_ not in per_round_res_df_dict:
                        per_round_res_df_dict[round_] = {'Inc' : {}, 'Dec' :{}}
                        per_round_res_df_dict[round_]['Inc']['Df'] = pd.DataFrame(columns=['QueryNum', 'Docno', 'Score'])
                        per_round_res_df_dict[round_]['Dec']['Df'] = pd.DataFrame(columns=['QueryNum', 'Docno', 'Score'])
                        per_round_res_df_dict[round_]['Idx'] = 0

                    if round_ == '02':
                        fulltext_inc = big_doc_index[query_user]['01']['json']['Fulltext'] + '\n' + big_doc_index[query_user][round_]['json']['Fulltext']
                        fulltext_dec = big_doc_index[query_user][round_]['json']['Fulltext'] + '\n' + big_doc_index[query_user]['01']['json']['Fulltext']
                    else:
                        fulltext_inc = big_doc_index[query_user][str(int(round_) - 2).zfill(2)]['json']['Fulltext'] + '\n' + \
                                       big_doc_index[query_user][str(int(round_) - 1).zfill(2)]['json']['Fulltext'] + '\n' + \
                                       big_doc_index[query_user][round_]['json']['Fulltext']
                        fulltext_dec = big_doc_index[query_user][round_]['json']['Fulltext'] + '\n' + \
                                       big_doc_index[query_user][str(int(round_) - 1).zfill(2)]['json']['Fulltext'] + '\n' + \
                                       big_doc_index[query_user][str(int(round_) - 2).zfill(2)]['json']['Fulltext']
                    try:
                        score_inc = get_query_doc_rel_proba(
                            tokenizer=tokenizer,
                            model=model,
                            query=query_num_to_text[query],
                            document=fulltext_inc)
                    except Exception as e:
                        score_inc = get_query_doc_rel_proba(
                            tokenizer=tokenizer,
                            model=model,
                            query=query_num_to_text[query],
                            document=fulltext_inc.rsplit(' ', 5)[0])
                        print('Worked!')
                    score_dec = get_query_doc_rel_proba(
                        tokenizer=tokenizer,
                        model=model,
                        query=query_num_to_text[query],
                        document=fulltext_dec)
                    docno = "EPOCH-" + str(round_).zfill(2) + '-' + query_user
                    print(docno)
                    sys.stdout.flush()
                    per_round_res_df_dict[round_]['Inc']['Df'].loc[per_round_res_df_dict[round_]['Idx']] = [query, docno, score_inc]
                    per_round_res_df_dict[round_]['Dec']['Df'].loc[per_round_res_df_dict[round_]['Idx']] = [query, docno, score_dec]
                    per_round_res_df_dict[round_]['Idx'] += 1

        save_folder = '/mnt/bi-strg3/v/zivvasilisky/ziv/results/bert/'
        for round_ in per_round_res_df_dict:
            df = per_round_res_df_dict[round_]['Inc']['Df']
            score_df = get_trec_prepared_df_form_res_df(df, 'Score')
            save_filename = inner_fold + '_' + str(round_).zfill(2) + '_BERT_Concat_Inc_Results.txt'
            with open(os.path.join(save_folder, save_filename), 'w') as f:
                big_doc_index = f.write(convert_df_to_trec(score_df))

            df = per_round_res_df_dict[round_]['Dec']['Df']
            score_df = get_trec_prepared_df_form_res_df(df, 'Score')
            save_filename = inner_fold + '_' + str(round_).zfill(2) + '_BERT_Concat_Dec_Results.txt'
            with open(os.path.join(save_folder, save_filename), 'w') as f:
                big_doc_index = f.write(convert_df_to_trec(score_df))





