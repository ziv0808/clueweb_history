import os
import sys
import time
import json
import random
import requests
import pandas as pd


def get_history_links_for_url_web_api(
        docno,
        url,
        run_year):
    web_archive_main_url = "http://web.archive.org/cdx/search/cdx?url=" + url + "&output=txt&from=" + str(run_year) + "&to=" + str(run_year)
    for i in range(10):
        try:
            response = requests.get(web_archive_main_url)
            break
        except Exception as e:
            time.sleep(10)

    response = str(response._content).split('\n')
    res_dict = {}
    for elem in response:
        elem = elem.split(' ')
        if len(elem) < 2 and len(res_dict.keys()) > 0:
            break
        if len(elem) < 2:
            print("Not Extracted Doc well: " + docno)
            print("### URL: " + web_archive_main_url)
            break
        time_stamp = elem[1]
        link = "https://web.archive.org/web/" + str(time_stamp) + "id_/" + url
        response_code = elem[4]
        res_dict[str(pd.to_datetime(time_stamp))] = {'Url': link,
                                                'ResponeCode': response_code}
    return res_dict



if __name__=='__main__':
    run_year = int(sys.argv[1])
    url_file_sufix = sys.argv[2]
    init_query = int(sys.argv[3])
    end_query = int(sys.argv[4])
    print("Run Year: " + str(run_year))
    print("From Query : " + str(init_query))
    print("Till Query : " + str(end_query))
    sys.stdout.flush()
    LOAD_FILE = None#"/lv_local/home/zivvasilisky/ziv/data/history_snapshots_2008.json"
    save_path = "/mnt/bi-strg3/v/zivvasilisky/ziv/data/history_snapshots/" + str(run_year) + "/"


    work_df = pd.read_csv('/mnt/bi-strg3/v/zivvasilisky/ziv/data/all_urls_no_spam_filtered'+url_file_sufix+'.tsv', sep = '\t', index_col = False)
    if LOAD_FILE is not None:
        with open(LOAD_FILE, 'r') as f:
            ref_json = json.load(f)

    processed_json = {}
    work_df['QueryInt'] = work_df['QueryNum'].apply(lambda x: int(x))
    work_df = work_df[work_df['QueryInt'] >= init_query]
    work_df = work_df[work_df['QueryInt'] <= end_query]
    del work_df['QueryInt']
    work_df['#Snapshots'] = 0
    work_df['#Snapshots_without_redirect'] = 0
    work_df['Remark'] = None

    curr_q_num = '000'
    num_processed = 0
    num_new_processed = 0
    for index, row in work_df.iterrows():
        if row['QueryNum'] != curr_q_num:
            curr_q_num = row['QueryNum']
            print ("Curr Query: "+ str(curr_q_num))

        if (LOAD_FILE is not None) and (row['Docno'] in ref_json):
            processed_json[row['Docno']] = True
            if not os.path.isfile(os.path.join(save_path, row['Docno'] + '.json')):
                with open(os.path.join(save_path, row['Docno'] + '.json', 'w')) as f:
                    json.dump(ref_json[row['Docno']], f)

        if os.path.isfile(os.path.join(save_path, row['Docno'] + '.json')):
            print('Docno: ' + row['Docno'] + " already processed")
            continue

        if row['Docno'] not in processed_json:
            try:
                num_new_processed += 1
                curr_json = get_history_links_for_url_web_api(row['Docno'] ,row['Url'],run_year)
            except Exception as e:
                try:
                    print(e)
                    curr_json = get_history_links_for_url_web_api(row['Docno'], row['Url'],run_year)
                except Exception as e:
                    print('Url:' + str(row['Url']) + " Needs retry")
                    work_df.set_value(index, 'Remark', "Needs Retry")
                    continue
            time.sleep(2)
            if num_new_processed % 3 == 0:
                time.sleep(random.randint(5, 10))

        if not os.path.isfile(os.path.join(save_path, row['Docno'] + '.json')):
            with open(os.path.join(save_path, row['Docno'] + '.json'), 'w') as f:
                json.dump(curr_json, f)
            processed_json[row['Docno']] = True

        work_df.set_value(index, '#Snapshots', len(curr_json.keys()))
        non_redirect_snapshots = 0
        for snapshot in curr_json:
            if curr_json[snapshot]['ResponeCode'] == "200":
                non_redirect_snapshots += 1
        work_df.set_value(index, '#Snapshots_without_redirect', non_redirect_snapshots)
        num_processed += 1
        if num_processed % 3 == 0:
            print("Num Processed: " + str(num_processed))
            sys.stdout.flush()
    work_df.to_csv(save_path + "all_urls_exrtacted_" + str(run_year) + '_' + str(init_query) + '_' + str(end_query) + '.tsv', sep = '\t', index= False)



