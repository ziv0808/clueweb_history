import os
import re
import sys
import ast
import uuid
import pandas as pd
from bs4 import BeautifulSoup
from utils import *



def parse_timestamp(timestamp_str):
    month_abrev_dict = {
        '01' : 'Jan',
        '02' : 'Feb',
        '03' : 'Mar',
        '04' : 'Apr',
        '05' : 'May',
        '06' : 'Jun',
        '07' : 'Jul',
        '08' : 'Aug',
        '09' : 'Sep',
        '10' : 'Oct',
        '11' : 'Nov',
        '12' : 'Dec'}
    weekday_abrev_dict = {
        0: 'Mon',
        1: 'Tue',
        2: 'Wed',
        3: 'Thu',
        4: 'Fri',
        5: 'Sat',
        6: 'Sun'}

    date_datetime = pd.to_datetime(timestamp_str)
    date_splitted = timestamp_str.split(' ')[0].split('-')
    date_parsed = weekday_abrev_dict[date_datetime.weekday()] + ', ' + date_splitted[2] + ' ' + month_abrev_dict[date_splitted[1]] + ' ' +\
                    date_splitted[0] + " " + timestamp_str.split(' ')[1] + ' GMT'
    return date_parsed

def find_html_content_type(html):
    soup = BeautifulSoup(html)
    meta = soup.find('meta', attrs={'http-equiv' : 'Content-Type'})
    if meta is not None:
        try:
            return_str = str(meta['content'])
        except Exception as e:
            return "text/html"
        if return_str == "None":
            return "text/html"
        else:
            # print(return_str)
            return return_str
    else:
        return "text/html"


def create_warc_head(
        warc_date,
        warc_info_id):
    warc_info_str = "WARC/0.18\n" +\
                    "WARC-Type: warcinfo\n" +\
                    "WARC-Date: " + warc_date + "\n" +\
                    "WARC-Record-ID: <urn:uuid:" + warc_info_id + ">" + "\n" + \
                    "Content-Type: application/warc-fields\n" +\
                    "Content-Length: "
    next_str      = "\n\nsoftware: Nutch 1.0-dev (modified for clueweb09)\n" +\
                    "isPartOf: clueweb09-en\n" +\
                    "description: clueweb09 crawl with WARC output\n" +\
                    "format: WARC file version 0.18\n" + \
                    "conformsTo: http://www.archive.org/documents/WarcFileFormat-0.18.html\n\n"
    return warc_info_str + str(len((next_str).encode('utf-8')) - 3) + next_str


def create_warc_record(
        docno,
        timstamp,
        url,
        html,
        warc_info_id,
        warc_date):


    html =  re.sub(r'[^\x00-\x7F]+',' ', html)

    record_str = "WARC/0.18\n" +\
                 "WARC-Type: response\n" +\
                 "WARC-Target-URI: " + url + "\n" +\
                 "WARC-Warcinfo-ID: " + warc_info_id + "\n" + \
                 "WARC-Date: " + warc_date + "\n" + \
                 "WARC-Record-ID: <urn:uuid:" + str(uuid.uuid1())  + ">" + "\n" + \
                 "WARC-TREC-ID: " + docno + "\n" + \
                 "Content-Type: application/http;msgtype=response" + "\n" + \
                 "WARC-Identified-Payload-Type: " + "\n" + \
                 "Content-Length: "
    try:
        next_str   = "HTTP/1.1 200 OK" + "\n" + \
                     "Content-Type: " + find_html_content_type(html) + "\n" + \
                     "Date: " + parse_timestamp(timstamp) + "\n" + \
                     "Pragma: no-cache"+ "\n" + \
                     "Cache-Control: no-cache, must-revalidate" + "\n" + \
                     "X-Powered-By: PHP/4.4.8" + "\n" + \
                     "Server: WebServerX" + "\n" + \
                     "Connection: close" + "\n" + \
                     "Last-Modified: " + parse_timestamp(timstamp) + "\n" + \
                     "Expires: Mon, 20 Dec 1998 01:00:00 GMT" + "\n"

        next_str += "Content-Length: " + str(len((html))) + "\n\n" + html + "\n\n"
        record_str += str(len((next_str))) + "\n\n" + next_str

    except Exception as e:
        print("Warcer Prob - Docno:" + docno)
        print(e)
        with open('Prob.txt', 'w') as f:
            f.write(html)
        record_str = ''


    return record_str


def create_warc_files_for_time_interval(
        destination_folder,
        time_interval,
        data_folder,
        work_year,
        universe_list):

    num_of_records_in_interval = 0
    folder_files_hirarcy_dict = {}
    for filename in os.listdir(data_folder):
        if filename.endswith('.json') and (filename.replace('.json', '') in universe_list):
            with open(os.path.join(data_folder, filename), 'r') as f:
                curr_json = f.read()
            curr_json = ast.literal_eval(curr_json)
            interval_found = False
            for snapshot in curr_json:
                if curr_json[snapshot]['RelevantInterval'] == time_interval:
                    interval_found = True
                    docno = filename.replace('.json','')
                    inner_folder_name = docno.split('-')[1]
                    inner_file_name = docno.split('-')[2]
                    if inner_folder_name not in folder_files_hirarcy_dict:
                        folder_files_hirarcy_dict[inner_folder_name] = {}
                    if inner_file_name not in folder_files_hirarcy_dict[inner_folder_name]:
                        folder_files_hirarcy_dict[inner_folder_name][inner_file_name] = []

                    if len(curr_json[snapshot]['Url'].split('id_/')) > 2:
                        raise Exception("create_warc_files_for_time_interval: Url has id_/ inside " +  str(curr_json[snapshot]['Url']))
                    else:
                        curr_url = curr_json[snapshot]['Url'].split('id_/')[1]
                    curr_doc_representation = {'Docno'      : docno,
                                               'Url'        : curr_url,
                                               'TimeStamp'  : snapshot,
                                               'HTML'       : curr_json[snapshot]['html']}
                    folder_files_hirarcy_dict[inner_folder_name][inner_file_name].append(curr_doc_representation)

            if interval_found == False:
                optional_ref_list = [time_interval[:8] + '08', time_interval[:8] + '16', time_interval[:8] + '23']
                for optional_ref_interval in optional_ref_list:
                    if interval_found == False:
                        for snapshot in curr_json:
                            if curr_json[snapshot]['RelevantInterval'] == optional_ref_interval:
                                interval_found = True
                                docno = filename.replace('.json', '')
                                inner_folder_name = docno.split('-')[1]
                                inner_file_name = docno.split('-')[2]
                                if inner_folder_name not in folder_files_hirarcy_dict:
                                    folder_files_hirarcy_dict[inner_folder_name] = {}
                                if inner_file_name not in folder_files_hirarcy_dict[inner_folder_name]:
                                    folder_files_hirarcy_dict[inner_folder_name][inner_file_name] = []

                                if len(curr_json[snapshot]['Url'].split('id_/')) > 2:
                                    raise Exception("create_warc_files_for_time_interval: Url has id_/ inside " + str(
                                        curr_json[snapshot]['Url']))
                                else:
                                    curr_url = curr_json[snapshot]['Url'].split('id_/')[1]
                                curr_doc_representation = {'Docno': docno,
                                                           'Url': curr_url,
                                                           'TimeStamp': snapshot,
                                                           'HTML': curr_json[snapshot]['html']}

                                folder_files_hirarcy_dict[inner_folder_name][inner_file_name].append(curr_doc_representation)
    print('Hirarchy Dict Finished...')
    lost_records = 0
    for folder_name in folder_files_hirarcy_dict:
        for file_name in folder_files_hirarcy_dict[folder_name]:
            print("Creating Warc For " +folder_name + '-' + file_name)
            warc_str = ""
            warc_info_id = str(uuid.uuid1())
            warc_date = str(int(work_year)+1) + "-03-65T08:43:19-0800"
            warc_str += create_warc_head(
                    warc_date=warc_date,
                    warc_info_id=warc_info_id)
            for doc in folder_files_hirarcy_dict[folder_name][file_name]:
                curr_str = create_warc_record(
                    docno=doc['Docno'],
                    url=doc['Url'],
                    timstamp=doc['TimeStamp'],
                    html=doc['HTML'],
                    warc_date=warc_date,
                    warc_info_id=warc_info_id)
                if curr_str != '':
                    warc_str += curr_str
                    num_of_records_in_interval += 1
                else:
                    lost_records += 1
                    print('Docno: '  + doc['Docno'] + " Problematic")
                    raise Exception('Docno: '  + doc['Docno'] + " Problematic")

            if not os.path.exists(os.path.join(destination_folder, time_interval, folder_name)):
                os.mkdir(os.path.join(destination_folder, time_interval, folder_name))
            with open(os.path.join(destination_folder, time_interval, folder_name, file_name + '.warc'), 'w') as f:
                f.write(warc_str)

    return num_of_records_in_interval, lost_records

def gzip_all_folder(
        folder_name):

    for inner_folder in os.listdir(folder_name):
        print (inner_folder)
        for file_name in os.listdir(os.path.join(folder_name, inner_folder)):
            if not file_name.endswith('.gz'):
                print(os.path.join(os.path.join(folder_name, inner_folder), file_name))
                res = subprocess.check_call(['gzip', os.path.join(os.path.join(folder_name, inner_folder), file_name)])

if __name__ == '__main__':
    work_year = "2008"
    init_interval_month = 1
    end_interval_month = 12
    action = sys.argv[1]
    if action == "Warcer":
        print("Run Year: " + str(work_year))
        print("From interval month: " + str(init_interval_month))
        print("Till interval month: " + str(end_interval_month))
        sys.stdout.flush()
        interval_list = build_interval_list(
            work_year=work_year,
            frequency='1M',
            start_month=int(init_interval_month),
            end_month=int(end_interval_month))

        destination_folder = "/mnt/bi-strg3/v/zivvasilisky/data/HSCW09/"
        data_folder ="/mnt/bi-strg3/v/zivvasilisky/ziv/data/retrived_htmls/2008/"
        relevant_docnos_file = "/mnt/bi-strg3/v/zivvasilisky/ziv/data/50_per_q/all_urls_no_spam_filtered.tsv"
        universe_df = pd.read_csv(relevant_docnos_file, sep = '\t', index_col = False)
        universe_list = list(universe_df['Docno'].drop_duplicates())

        summary_df = pd.DataFrame(columns = ['Interval', 'NumOfDocs', 'LostRecords'])
        next_index = 0
        for interval in interval_list:
            print ("Curr interval: " + str(interval))
            sys.stdout.flush()
            if not os.path.exists(os.path.join(destination_folder, interval)):
                os.mkdir(os.path.join(destination_folder, interval))

            num_records, lost_records = create_warc_files_for_time_interval(
                destination_folder=destination_folder,
                time_interval=interval,
                data_folder=data_folder,
                work_year=work_year,
                universe_list=universe_list)

            summary_df.loc[next_index] = [interval, num_records, lost_records]
            next_index += 1

        summary_df.to_csv(os.path.join(data_folder, 'Summry_warcer.tsv'), sep = '\t', index = False)
    elif action == "Indexer":
        print("Run Year: " + str(work_year))
        print("From interval month: " + str(init_interval_month))
        print("Till interval month: " + str(end_interval_month))
        sys.stdout.flush()
        interval_list = build_interval_list(
            work_year=work_year,
            frequency='1M',
            start_month=int(init_interval_month),
            end_month=int(end_interval_month))

        destination_folder = "/mnt/bi-strg3/v/zivvasilisky/index/HSCW09/"
        data_folder = "/mnt/bi-strg3/v/zivvasilisky/data/HSCW09/"

        for interval in interval_list:
            print("Curr interval: " + str(interval))
            sys.stdout.flush()
            gzip_all_folder(os.path.join(data_folder, interval))
            print('Gziped all...')
            sys.stdout.flush()
            with open('/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/IndriBuildIndex.xml', 'r') as f:
                params_text = f.read()

            with open('/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/Index_params/IndriBuildIndex_' + str(
                    interval) + '.xml',
                      'w') as f:
                f.write(params_text.replace('###', str(os.path.join(destination_folder, interval))).replace('%%%', str(
                    os.path.join(data_folder, interval))))
            print('Params fixed...')
            sys.stdout.flush()
            res = subprocess.check_call(['/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/indri/bin/IndriBuildIndex',
                                         '/mnt/bi-strg3/v/zivvasilisky/ziv/clueweb_history/Index_params/IndriBuildIndex_' + str(
                                             interval) + '.xml'])
            print('Index built...')
            sys.stdout.flush()
