import sys
import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def get_query_doc_rel_proba(
        tokenizer,
        model,
        query,
        document):
    inputs = tokenizer.encode_plus(query, document, return_tensors="pt")
    return torch.softmax(model(**inputs)[0], dim=1).tolist()[0][1]

if __name__=="__main__":
    inner_fold = sys.argv[1]

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

    for query_user in big_doc_index:
        query = big_doc_index.split('-')[0]
        print(query_user)
        sys.stdout.flush()
        for round_ in big_doc_index[query_user]:
            big_doc_index[query_user][round_]['json']['BERTScore'] = get_query_doc_rel_proba(
                                                        tokenizer=tokenizer,
                                                        model=model,
                                                        query=query_num_to_text[query],
                                                        document=big_doc_index[query_user][round_]['json']['Fulltext'])

    with open('/mnt/bi-strg3/v/zivvasilisky/ziv/data/' + inner_fold + '/RawDataWithBERT.json', 'w') as f:
        big_doc_index = f.write(str(big_doc_index))




