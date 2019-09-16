import os
import time
import random
import requests
import ast
import pandas as pd


def get_html_for_url(
        url):
    request_url = "http" + url[5:].replace('*/', 'id_/')
    for i in range(10):
        try:
            response_ = requests.get(request_url)
            break
        except Exception as e:
            time.sleep(random.randint(3,10))
    decoded_content = response_.content
    response_code = response_.status_code

    if response_code != 200:
        print('Url: ' + request_url + " - Response code: " + str(response_code))
    if response_ is not None:
        response_.close()
    return response_code, decoded_content

def build_interval_dict(
        work_year,
        frequency):

    interval_dict = {}
    for i in range(1, 13):
        interval_dict[work_year + '-' + (2 - len(str(i)))*'0' + str(i)] = []
        if frequency == '2W':
            interval_dict[work_year + '-' + (2 - len(str(i)))*'0' + str(i)].extend([work_year + "-" + (2 - len(str(i)))*'0' + str(i) + '-01',
                                                                                    work_year + "-" + (2 - len(str(i)))*'0' + str(i) + '-16'])
        else:
            raise Exception('build_interval_dict: Unknoen frequency...')

    return interval_dict

def handle_docno_retrival(
        dest_folder,
        docno,
        docno_snapshots_json,
        interval_dict,
        docno_ref_json = None):
    # the output to be saved in file
    res_json = {}
    # the defined time intervals that are already covered to this doc
    covered_intervals = []
    snapshot_list = sorted(list(docno_snapshots_json.keys()))
    for snapshot in snapshot_list:
        # find relevant interval
        relvent_month_interval_list = interval_dict[snapshot[:7]]
        relvent_interval = relvent_month_interval_list[0]
        snapshot_day_in_month = int(snapshot[8:10])
        for j in range(len(relvent_month_interval_list) - 1):
            if snapshot_day_in_month < int(relvent_month_interval_list[j + 1][8:10]):
                break
            else:
                relvent_interval = relvent_month_interval_list[j + 1]
        # if interval already covered continue
        if relvent_interval in covered_intervals:
            continue
        # if snapshot was already retrieved in previos iterations
        if (docno_ref_json is not None) and (snapshot in docno_ref_json) and ('html' in docno_ref_json[snapshot]) and (docno_ref_json[snapshot]['htmlResponseCode'] == '200'):
            res_json[snapshot] = docno_ref_json[snapshot]
            res_json[snapshot]['RelevantInterval'] = relvent_interval
            covered_intervals.append(relvent_interval)
        else:
            # retrieve html
            try:
                html_response_code, html_str = get_html_for_url(docno_snapshots_json[snapshot]['Url'])
            except Exception as e:
                try:
                    html_response_code, html_str = get_html_for_url(docno_snapshots_json[snapshot]['Url'])
                except Exception as e:
                    print("Didnt work - " + work_json[docno][snapshot]['Url'])
                    continue
            if str(html_response_code) == '200':
                res_json[snapshot] = {}
                res_json[snapshot]['Url']              = docno_snapshots_json[snapshot]['Url']
                res_json[snapshot]['html']             = html_str
                res_json[snapshot]['htmlResponseCode'] = str(html_response_code)
                res_json[snapshot]['RelevantInterval'] = relvent_interval
                covered_intervals.append(relvent_interval)

    # save output for docno
    if len(covered_intervals) > 0:
        with open(os.path.join(dest_folder, docno + '.json'), 'w') as f:
            f.write(str(res_json))

    return len(covered_intervals)





if __name__=='__main__':

    work_year = '2008'
    interval_dict = build_interval_dict(
        work_year=work_year,
        frequency='2W')

    save_folder = "/lv_local/home/zivvasilisky/ziv/data/retrived_htmls/2008/"
    resource_path = "/lv_local/home/zivvasilisky/ziv/data/bkup/2008"
    filename = os.path.join(resource_path, "history_snapshots_2008.json")
    reference_filename = os.path.join(resource_path, "history_snapshots_2008_with_html_with_html.json")

    summary_df = pd.DataFrame(columns=['Docno', 'NumOfCoveredIntervals'])
    next_index = 0
    # get all possible snapshots dict
    with open(filename, 'r') as f:
        work_json = f.read()
        work_json = ast.literal_eval(work_json)
    # get all reference html dict
    if reference_filename is not None:
        with open(reference_filename, 'r') as f:
            ref_json = f.read()
            ref_json = ast.literal_eval(ref_json)
    else:
        ref_json = None

    docnos_processed            = 0
    covered_intervals_processed = 0
    for docno in work_json:
        if (ref_json is None) or (docno not in ref_json):
            curr_ref_json = None
        else:
            curr_ref_json = ref_json[docno]
        covered_intervals = handle_docno_retrival(
            dest_folder=save_folder,
            docno=docno,
            docno_snapshots_json=work_json[docno],
            interval_dict=interval_dict,
            docno_ref_json=curr_ref_json)
        docnos_processed += 1
        covered_intervals_processed += covered_intervals
        print("Docno : " + str(docno) + ' , Covered: ' + str(covered_intervals))
        print("Docnos Processed: " + str(docnos_processed))
        if covered_intervals_processed % 10 == 0:
            print("HTMLs Processed: " + str(covered_intervals_processed))

        summary_df.loc[next_index] = [docno, covered_intervals]
        next_index += 1

    summary_df.to_csv(os.path.join(save_folder, filename.replace('.json', '_Html_retrival_summary.tsv')), sep = '\t', index = False)
