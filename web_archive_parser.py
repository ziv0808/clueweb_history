import time
import json
import random
import requests
import pandas as pd



def get_history_links_for_url_web_api(
        docno,
        url):
    web_archive_main_url = "http://web.archive.org/cdx/search/cdx?url=" + url + "*&output=txt&from=" + str(RUN_YEAR) + "&to=" + str(RUN_YEAR)
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
        link = "https://web.archive.org/web/" + str(time_stamp) + "*/" + url
        response_code = elem[4]
        res_dict[str(pd.to_datetime(time_stamp))] = {'Url': link,
                                                'ResponeCode': response_code}
    return res_dict



if __name__=='__main__':
    RUN_YEAR = 2009
    LOAD_FILE = "/lv_local/home/zivvasilisky/ziv/data/bkup/history_snapshots_2009.json"
    save_path = "/lv_local/home/zivvasilisky/ziv/data/"

    work_df = pd.read_csv('/lv_local/home/zivvasilisky/ziv/data/all_urls_no_spam_filtered.tsv', sep = '\t', index_col = False)
    if LOAD_FILE is not None:
        with open(LOAD_FILE, 'r') as f:
            ref_json = json.load(f)

    res_json = {}

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
            res_json[row['Docno']] = ref_json[row['Docno']]

        if row['Docno'] not in res_json:
            try:
                num_new_processed += 1
                res_json[row['Docno']] = get_history_links_for_url_web_api(row['Docno'] ,row['Url'])
            except Exception as e:
                try:
                    print(e)
                    res_json[row['Docno']] = get_history_links_for_url_web_api(row['Docno'], row['Url'])
                except Exception as e:
                    print('Url:' + str(row['Url']) + " Needs retry")
                    work_df.set_value(index, 'Remark', "Needs Retry")
                    continue
            time.sleep(3)
            if num_new_processed % 3 == 0:
                time.sleep(random.randint(5, 15))

        work_df.set_value(index, '#Snapshots', len(res_json[row['Docno']].keys()))
        non_redirect_snapshots = 0
        for snapshot in res_json[row['Docno']]:
            if res_json[row['Docno']][snapshot]['ResponeCode'] == "200":
                non_redirect_snapshots += 1
        work_df.set_value(index, '#Snapshots_without_redirect', non_redirect_snapshots)
        num_processed += 1
        if num_processed % 3 == 0:
            print("Num Processed: " + str(num_processed))

        if num_processed % 50 == 0:
            work_df.to_csv(save_path + "all_urls_exrtacted_" + str(RUN_YEAR) + '.tsv', sep='\t', index=False)
            with open(save_path + "history_snapshots_" + str(RUN_YEAR) + '.json', 'w') as f:
                json.dump(res_json, f)


    work_df.to_csv(save_path + "all_urls_exrtacted_" + str(RUN_YEAR) + '.tsv', sep = '\t', index= False)

    with open(save_path + "history_snapshots_" + str(RUN_YEAR)+ '.json', 'w') as f:
        json.dump(res_json, f)



