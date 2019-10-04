import os
import ast
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



plt.style.use('seaborn-whitegrid')
def plot_retrival_stats(
        file_name  = "all_urls_exrtacted_2008.tsv",
        merge_file = "history_snapshots_2008_Html_retrival_summary.tsv"):

    work_df = pd.read_csv(file_name, sep = '\t', index_col = False)
    if merge_file is not None:
        merge_df = pd.read_csv(merge_file, sep = '\t', index_col = False)
        print('Len work_df: ' +str(len(work_df)) )
        work_df  = pd.merge(
            work_df,
            merge_df,
            on = ['Docno'],
            how = 'left')
        print('Len work_df after merge: ' + str(len(work_df)))
        work_df.fillna(0, inplace=True)
        print ((work_df[work_df['#Snapshots_without_redirect'] > 0]['#Snapshots_without_redirect'] - work_df[work_df['#Snapshots_without_redirect'] > 0]['NumOfCoveredIntervals']).median())
        del work_df['#Snapshots_without_redirect']
        work_df.rename(columns = {'NumOfCoveredIntervals':'#Snapshots_without_redirect'}, inplace = True)
    work_df['Covered'] = work_df['#Snapshots'].apply(lambda x: 1.0 if x > 0 else 0.0)
    work_df['Covered_200'] = work_df['#Snapshots_without_redirect'].apply(lambda x: 1.0 if x > 0 else 0.0)
    work_df['Covered_200_more_than_1'] = work_df['#Snapshots_without_redirect'].apply(lambda x: 1.0 if x > 1 else 0.0)
    covered_precentage = float(work_df['Covered'].sum())/len(work_df)
    covered_200_precentage = float(work_df['Covered_200'].sum())/len(work_df)
    print("Covered Precentage: " + str(covered_precentage))
    print("Covered 200 Precentage: " + str(covered_200_precentage))

    covered_df = work_df[work_df['Covered'] == 1]
    print("Covered Snapshot Mean: " + str(covered_df['#Snapshots'].mean()))
    covered_df = work_df[work_df['Covered_200'] == 1]
    print("Covered 200 Snapshot Mean: " + str(covered_df['#Snapshots_without_redirect'].mean()))
    print("Covered 200 Snapshot Med: " + str(covered_df['#Snapshots_without_redirect'].median()))

    print_df = work_df[['QueryNum','Covered']].groupby(['QueryNum']).sum()
    print_df = print_df.reset_index()
    print_df = print_df.sort_values('Covered', ascending=False)
    print_df = print_df.set_index('QueryNum')

    plt.style.use('seaborn-whitegrid')
    print_df.plot(legend = False, kind = 'bar', color = 'c')
    plt.xlabel('Query')
    plt.xticks(visible= False)
    plt.ylabel('#Covered Documents')
    plt.savefig("Covered_per_query.png", dpi=300)


def create_interval_coverage_plot(file_name = "Summry_warcer.tsv"):
    work_df = pd.read_csv(file_name, sep='\t', index_col=False)
    print_df = work_df.set_index('Interval')

    print_df.plot(legend=False, kind='bar', color='r')
    plt.xlabel('Interval')
    plt.tick_params(axis='x', labelsize=5)
    plt.xticks(rotation=45)
    plt.ylabel('#Documents')
    plt.savefig("doc_per_interval.png", dpi=300)


def build_interval_dict(
        work_year,
        frequency):


    interval_dict = {}
    for i in range(1, 13):
        interval_dict[work_year + '-' + (2 - len(str(i))) * '0' + str(i)] = []
        if frequency == '2W':
            interval_dict[work_year + '-' + (2 - len(str(i))) * '0' + str(i)].extend(
                [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01',
                 work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-16'])
        else:
            raise Exception('build_interval_dict: Unknoen frequency...')

    return interval_dict

def find_relevant_interval(interval_dict, snapshot):
    relvent_month_interval_list = interval_dict[snapshot[:7]]
    relvent_interval = relvent_month_interval_list[0]
    snapshot_day_in_month = int(snapshot[8:10])
    for j in range(len(relvent_month_interval_list) - 1):
        if snapshot_day_in_month < int(relvent_month_interval_list[j + 1][8:10]):
            break
        else:
            relvent_interval = relvent_month_interval_list[j + 1]
    return  relvent_interval

def create_lost_snapshot_stats(
        snapshot_potential_file = "/lv_local/home/zivvasilisky/ziv/data/history_snapshots_2008.json",
        warcer_summary_file = "/lv_local/home/zivvasilisky/ziv/data/retrived_htmls/2008/Summry_warcer.tsv",
        doc_retrived_folder = "/lv_local/home/zivvasilisky/ziv/data/retrived_htmls/2008/",
        work_year = "2008"):

    with open(snapshot_potential_file, 'r') as f:
        work_json = f.read()
        work_json = ast.literal_eval(work_json)

    warcer_stats = pd.read_csv(warcer_summary_file, sep ='\t', index_col = False)

    interval_dict = build_interval_dict(work_year, '2W')
    doc_dict = {}
    interval_covered_dict = {}
    for docno in work_json:
        doc_dict[docno] = 0

        for snapshot in work_json[docno]:
            if work_json[docno][snapshot]['ResponeCode'] == "200":
                relevant_interval = find_relevant_interval(interval_dict, snapshot)
                if relevant_interval not in interval_covered_dict:
                    interval_covered_dict[relevant_interval] = 1
                else:
                    interval_covered_dict[relevant_interval] += 1

                doc_dict[docno] += 1
        if os._exists(os.path.join(doc_retrived_folder, docno + '.json')):
            with open(os.path.join(doc_retrived_folder, docno + '.json'), 'r') as f:
                covered_json = f.read()
                covered_json = ast.literal_eval(covered_json)
            doc_dict[docno] = doc_dict[docno] - len(covered_json)

    for index, row in warcer_stats.iterrows():
        for interval in interval_covered_dict:
            if pd.to_datetime(interval) == pd.to_datetime(row['Interval']):
                interval_covered_dict[interval] = interval_covered_dict[interval] - int(row['NumOfDocs'])

    summary_df = pd.DataFrame(columns=['Interval', '#Documents'])
    next_index = 0
    for interval in interval_covered_dict:
        summary_df.loc[next_index] = [interval, interval_covered_dict[interval]]
        next_index += 1

    print_df = summary_df.set_index('Interval')
    print_df.plot(legend=False, kind='bar', color='c')
    plt.xlabel('Interval')
    plt.tick_params(axis='x', labelsize=5)
    plt.xticks(rotation=45)
    plt.ylabel('#Documents')
    plt.savefig("lost_doc_per_interval.png", dpi=300)

    summary_df = pd.DataFrame(columns=['Docno', '#Documents'])
    next_index = 0
    for docno in doc_dict:
        summary_df.loc[next_index] = [docno, doc_dict[docno]]
        next_index += 1

    print_df = summary_df.sort_values('#Documents', ascending=False)
    print_df = print_df.set_index('Docno')

    print_df.plot(legend=False,  color='r')
    plt.xlabel('Doc')
    plt.xticks(visible=False)
    plt.ylabel('#Documents')
    plt.savefig("lost_snapshots_per_doc.png", dpi=300)

create_lost_snapshot_stats()

# plot_retrival_stats()
# create_interval_coverage_plot()
