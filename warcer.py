import os
import ast
import uuid
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup

import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


def build_interval_list(
        work_year,
        frequency):

    interval_list = []
    for i in range(1, 13):
        if frequency == '2W':
            interval_list.extend(
                [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01',
                 work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-16'])
        else:
            raise Exception('build_interval_dict: Unknoen frequency...')

    return interval_list

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
        return_str = str(meta['content'])
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
        warc_date,
        normalize = False):
    # normalize unicode form
    if normalize == True:
        print("Normalizing ... ")
        html = unicodedata.normalize("NFKD", html.decode('utf-8', 'ignore')).encode('ascii', 'ignore').encode(encoding='UTF-8',     errors='strict')

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

        if normalize == False:
            next_str += "Content-Length: " + str(len((html)) + 1) + "\n\n" + html + "\n\n"
            record_str += str(len((next_str)) + 1) + "\n\n" + next_str
        else:
            next_str += "Content-Length: " + str(len((html.encode('utf-8'))) + 1) + "\n\n" + html + "\n\n"
            record_str += str(len((next_str.encode('utf-8'))) + 1) + "\n\n" + next_str

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
        data_folder):

    num_of_records_in_interval = 0
    folder_files_hirarcy_dict = {}
    for filename in os.listdir(data_folder):
        if filename.endswith('.json'):
            with open(os.path.join(data_folder, filename), 'r') as f:
                curr_json = f.read()
            curr_json = ast.literal_eval(curr_json)
            for snapshot in curr_json:
                if curr_json[snapshot]['RelevantInterval'] == time_interval:
                    docno = filename.replace('.json','')
                    inner_folder_name = docno.split('-')[1]
                    inner_file_name = docno.split('-')[2]
                    if inner_folder_name not in folder_files_hirarcy_dict:
                        folder_files_hirarcy_dict[inner_folder_name] = {}
                    if inner_file_name not in folder_files_hirarcy_dict[inner_folder_name]:
                        folder_files_hirarcy_dict[inner_folder_name][inner_file_name] = []

                    if len(curr_json[snapshot]['Url'].split('*/')) > 2:
                        raise Exception("create_warc_files_for_time_interval: Url has */ inside " +  str(curr_json[snapshot]['Url']))
                    else:
                        curr_url = curr_json[snapshot]['Url'].split('*/')[1]
                    curr_doc_representation = {'Docno'      : docno,
                                               'Url'        : curr_url,
                                               'TimeStamp'  : snapshot,
                                               'HTML'       : curr_json[snapshot]['html']}

                    folder_files_hirarcy_dict[inner_folder_name][inner_file_name].append(curr_doc_representation)
    print('Hirarchy Dict Finished...')
    lost_records = 0
    for folder_name in folder_files_hirarcy_dict:
        for file_name in folder_files_hirarcy_dict[folder_name]:
            print("Creating Warc For " +folder_name + '-' + file_name)
            warc_str = ""
            warc_info_id = str(uuid.uuid1())
            warc_date = "2009-03-65T08:43:19-0800"
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
                    curr_str = create_warc_record(
                        docno=doc['Docno'],
                        url=doc['Url'],
                        timstamp=doc['TimeStamp'],
                        html=doc['HTML'],
                        warc_date=warc_date,
                        warc_info_id=warc_info_id,
                        normalize=True)
                    if curr_str != '':
                        try:
                            warc_str += curr_str
                        except Exception as e:
                            print (curr_str[36450:36470])
                        num_of_records_in_interval += 1
                    else:
                        lost_records += 1
                        print('Docno: '  + doc['Docno'] + " Problematic")
                        raise Exception('Docno: '  + doc['Docno'] + " Problematic")
            # add last record double time because last record does not get indexed by indri
            warc_str += curr_str

            if not os.path.exists(os.path.join(destination_folder, time_interval, folder_name)):
                os.mkdir(os.path.join(destination_folder, time_interval, folder_name))
            with open(os.path.join(destination_folder, time_interval, folder_name, file_name + '.warc'), 'w') as f:
                f.write(warc_str)

    return num_of_records_in_interval, lost_records

if __name__ == '__main__':
    test = False
    if test != True:
        work_year = '2008'
        interval_list = build_interval_list(
            work_year=work_year,
            frequency='2W')

        destination_folder = "/mnt/bi-strg3/v/zivvasilisky/data/2008/"
        data_folder ="/lv_local/home/zivvasilisky/ziv/data/retrived_htmls/2008/"

        summary_df = pd.DataFrame(columns = ['Interval', 'NumOfDocs', 'LostRecords'])
        next_index = 0
        for interval in interval_list:
            print ("Curr interval: " + str(interval))
            if not os.path.exists(os.path.join(destination_folder, interval)):
                os.mkdir(os.path.join(destination_folder, interval))

            num_records, lost_records = create_warc_files_for_time_interval(
                destination_folder=destination_folder,
                time_interval=interval,
                data_folder=data_folder)

            summary_df.loc[next_index] = [interval, num_records, lost_records]
            next_index += 1

        summary_df.to_csv(os.path.join(data_folder, 'Summry_warcer.tsv'), sep = '\t', index = False)

    else:
        docno = "clueweb09-en0000-52-32023"
        with open(docno + ".txt", 'r') as f:
            html = f.read()
        # test ####
        print (create_warc_record(
            docno=docno,
            timstamp="2009-01-13 18:05:10",
            url = 'http://00000-nrt-realestate.homepagestartup.com/',
            html=html,
            warc_info_id ='993d3969-9643-4934-b1c6-68d4dbe55b83',
            warc_date='2009-03-65T08:43:19-0800'))

        # print (create_warc_head(warc_date="2009-03-65T08:43:19-0800",
        #                         warc_info_id='993d3969-9643-4934-b1c6-68d4dbe55b83'))


