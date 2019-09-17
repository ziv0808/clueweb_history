import os
import subprocess

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

def gzip_all_folder(
        folder_name):

    for inner_folder in os.listdir(folder_name):
        print (inner_folder)
        for file_name in os.listdir(os.path.join(folder_name, inner_folder)):
            if not file_name.endswith('.gz'):
                print(os.path.join(os.path.join(folder_name, inner_folder), file_name))
                res = subprocess.check_call(['gzip ' + os.path.join(os.path.join(folder_name, inner_folder), file_name)])

if __name__=='__main__':
    work_year = '2008'
    interval_list = build_interval_list(
        work_year=work_year,
        frequency='2W')

    destination_folder = "/mnt/bi-strg3/v/zivvasilisky/index/2008/"
    data_folder = "/mnt/bi-strg3/v/zivvasilisky/data/2008/"

    for interval in interval_list[:1]:
        print("Curr interval: " + str(interval))
        gzip_all_folder(os.path.join(data_folder, interval))
        print('Gziped all...')
        with open('/lv_local/home/zivvasilisky/ziv/clueweb_history/IndriBuildIndex.xml', 'r') as f:
            params_text = f.read()

        with open('/lv_local/home/zivvasilisky/ziv/clueweb_history/Index_params/IndriBuildIndex_' + str(interval) + '.xml',
                  'w') as f:
            f.write(params_text.replace('###', str(os.path.join(destination_folder,interval))).replace('%%%', str(os.path.join(data_folder,interval))))
        print('Params fixed...')
        res = subprocess.check_call(['/lv_local/home/zivvasilisky/ziv/env/indri/indri/bin/IndriBuildIndex /lv_local/home/zivvasilisky/ziv/clueweb_history/Index_params/IndriBuildIndex_' + str(interval) + '.xml'])
        print('Index built...')


