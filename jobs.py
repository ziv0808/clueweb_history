
import pandas as pd


def get_relevant_docs_df(
        qurls_path = '/lv_local/home/zivvasilisky/ziv/results/qrels/qrels.adhoc'):

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


get_relevant_docs_df()