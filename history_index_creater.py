import os
import sys
import subprocess
from utils import *

def gzip_all_folder(
        folder_name):

    for inner_folder in os.listdir(folder_name):
        print (inner_folder)
        for file_name in os.listdir(os.path.join(folder_name, inner_folder)):
            if not file_name.endswith('.gz'):
                print(os.path.join(os.path.join(folder_name, inner_folder), file_name))
                res = subprocess.check_call(['gzip', os.path.join(os.path.join(folder_name, inner_folder), file_name)])

if __name__=='__main__':
    work_year = sys.argv[1]
    init_interval_month = sys.argv[2]
    end_interval_month = sys.argv[3]
    print("Run Year: " + str(work_year))
    print("From interval month: " + str(init_interval_month))
    print("Till interval month: " + str(end_interval_month))
    sys.stdout.flush()
    interval_list = build_interval_list(
        work_year=work_year,
        frequency='1W',
        start_month=int(init_interval_month),
        end_month=int(end_interval_month))

    destination_folder = "/mnt/bi-strg3/v/zivvasilisky/index/" + work_year + "/"
    data_folder = "/mnt/bi-strg3/v/zivvasilisky/data/" + work_year + "/"

    for interval in interval_list:
        print("Curr interval: " + str(interval))
        sys.stdout.flush()
        gzip_all_folder(os.path.join(data_folder, interval))
        print('Gziped all...')
        sys.stdout.flush()
        with open('/lv_local/home/zivvasilisky/ziv/clueweb_history/IndriBuildIndex.xml', 'r') as f:
            params_text = f.read()

        with open('/lv_local/home/zivvasilisky/ziv/clueweb_history/Index_params/IndriBuildIndex_' + str(interval) + '.xml',
                  'w') as f:
            f.write(params_text.replace('###', str(os.path.join(destination_folder,interval))).replace('%%%', str(os.path.join(data_folder,interval))))
        print('Params fixed...')
        sys.stdout.flush()
        res = subprocess.check_call(['/lv_local/home/zivvasilisky/ziv/env/indri/indri/bin/IndriBuildIndex', '/lv_local/home/zivvasilisky/ziv/clueweb_history/Index_params/IndriBuildIndex_' + str(interval) + '.xml'])
        print('Index built...')
        sys.stdout.flush()


