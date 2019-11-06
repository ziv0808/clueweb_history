import os
import ast
# import tikzplotlib
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



plt.style.use('seaborn-whitegrid')
def plot_retrival_stats(
        file_name  = "all_urls_exrtacted_2009.tsv",
        merge_file =None):

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
    plt.ylabel('#Losts Snapshots')
    plt.savefig("lost_doc_per_interval.png", dpi=300)

    summary_df = pd.DataFrame(columns=['Docno', '#Documents'])
    next_index = 0
    for docno in doc_dict:
        summary_df.loc[next_index] = [docno, doc_dict[docno]]
        next_index += 1

    print_df = summary_df.sort_values('#Documents', ascending=False)
    print_df = print_df.set_index('Docno')

    print_df.plot(legend=False,  color='r')
    plt.xlabel('Document')
    plt.xticks(visible=False)
    plt.ylabel('#Losts Snapshots')
    plt.savefig("lost_snapshots_per_doc.png", dpi=300)

def create_snapshot_changes_stats(
        filename = 'Summay_snapshot_stats_1M.tsv',
        comp_queries_list = [195,193,188,180,177,161,144,59,51,45,36,34,32,29,11,10,
                     9,2,78,4,167,69,166,33,164,18,182,17,98,124,48]):
    filename_for_save = filename.replace('.tsv', '').replace('Summay_snapshot_stats_', '')
    work_df = pd.read_csv(filename, sep = '\t', index_col = False)
    # calc ratio measures
    work_df['QueryTermsRatio'] = work_df.apply(lambda row: row['QueryWords'] / (row['TextLen'] - row['QueryWords']),axis =1)
    work_df['StopwordsRatio'] = work_df.apply(lambda row: row['#Stopword'] / (row['TextLen'] - row['#Stopword']),axis=1)
    no_clueweb_df = work_df[work_df['Interval'] != 'ClueWeb09'].copy()
    clueweb_df    = work_df[work_df['Interval'] == 'ClueWeb09'].copy()
    print("Mean ClueWeb09 SimToPrev:")
    print(clueweb_df[clueweb_df['SimToPrev'].notnull()]['SimToPrev'].mean())
    print("Mean ClueWeb09 SimToClosePrev:")
    print(clueweb_df[clueweb_df['SimToClosePrev'].notnull()]['SimToClosePrev'].mean())
    print("Med ClueWeb09 SimToPrev:")
    print(clueweb_df[clueweb_df['SimToPrev'].notnull()]['SimToPrev'].median())
    print("Med ClueWeb09 SimToClosePrev:")
    print(clueweb_df[clueweb_df['SimToClosePrev'].notnull()]['SimToClosePrev'].median())
    print("Mean all SimToPrev:")
    print(work_df[work_df['SimToPrev'].notnull()]['SimToClosePrev'].mean())
    print("Med all SimToPrev:")
    print(work_df[work_df['SimToPrev'].notnull()]['SimToPrev'].median())
    print("Mean not ClueWeb09 SimToClueWeb:")
    print(no_clueweb_df['SimToClueWeb'].mean())
    print("Med not ClueWeb09 SimToClueWeb:")
    print(no_clueweb_df['SimToClueWeb'].median())

    all_docno_and_queries_df = work_df[['Docno', 'QueryNum']].drop_duplicates()
    only_clueweb_num = 0

    summary_df = pd.DataFrame(columns = ['Interval', 'QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb', 'QueryNum'])
    next_index = 0
    exact_mathchs = 0
    # exact_mathch_df = pd.DataFrame(columns = ['Docno'])
    # exact_mathch_index = 0
    for index, row_ in all_docno_and_queries_df.iterrows():
        doc_df = work_df[work_df['Docno'] == row_['Docno']].copy()
        doc_df = doc_df[doc_df['QueryNum'] == row_['QueryNum']].copy()
        if len(doc_df) == 1:
            only_clueweb_num += 1
            continue
        bench_df = doc_df[doc_df['Interval'] == 'ClueWeb09']
        bench_query_term_ratio = list(bench_df['QueryTermsRatio'])[0]
        bench_stopword_ratio   = list(bench_df['StopwordsRatio'])[0]
        bench_entropy_ratio    = list(bench_df['Entropy'])[0]
        if bench_stopword_ratio  == 0:
            bench_stopword_ratio = 1.0
        if bench_query_term_ratio == 0:
            bench_query_term_ratio = 1.0
        for index, row in doc_df.iterrows():
            if row['Interval'] != 'ClueWeb09':
                interval         = row['Interval']
                query_term_ratio = (row['QueryTermsRatio'] - bench_query_term_ratio)/float(bench_query_term_ratio)
                stopword_ratio   = (row['StopwordsRatio'] - bench_stopword_ratio) / float(bench_stopword_ratio)
                entropy_ratio    = (row['Entropy'] - bench_entropy_ratio) / float(bench_entropy_ratio)
                sim_cluweb       = row['SimToClueWeb']
                summary_df.loc[next_index] = [interval, query_term_ratio,stopword_ratio, entropy_ratio, sim_cluweb, int(row_['QueryNum'])]
                next_index += 1
            else:
                if row['SimToPrev'] == 1.0:
                    exact_mathchs += 1
                    # exact_mathch_df.loc[exact_mathch_index] = [row_['Docno']]
                    # exact_mathch_index += 1
    exact_mathch_df = work_df[work_df['Interval'] != 'ClueWeb09'].copy()
    exact_mathch_df = exact_mathch_df[exact_mathch_df['SimToClueWeb'] == 1.0][['Docno']].drop_duplicates()
    exact_mathch_df.to_csv(filename.replace('.tsv', '_Exact_matched_docnos.tsv') , sep = '\t', index= False)
    print("Num of one snapshot:")
    print(only_clueweb_num)
    print("Num of Exact Match:")
    print(len(exact_mathch_df))
    summary_df['ExactMatchs'] = summary_df['SimClueWeb'].apply(lambda x: 1 if x == 1.0 else 0)
    plot_df = summary_df[['Interval', 'ExactMatchs']].groupby(['Interval']).sum()
    plt.cla()
    plt.clf()
    plot_df.plot(kind='bar', color='c', legend= False)
    plt.xlabel('Interval')
    plt.tick_params(axis='x', labelsize=5)
    plt.xticks(rotation=45)
    plt.ylabel("#Exact Matches")
    plt.title("#Exact Matches Per Interval")
    plt.savefig( filename_for_save + "_Exact_Matches_per_interval.png", dpi=300)
    summary_df['IsCompQuery'] = summary_df['QueryNum'].apply(lambda x: 1 if x in comp_queries_list else 0)
    interval_quantity_df = summary_df[['Interval', 'QueryNum']].groupby(['Interval']).count()
    plt.cla()
    plt.clf()
    interval_quantity_df.plot(kind='bar', color='r', legend= False)
    plt.xlabel('Interval')
    plt.tick_params(axis='x', labelsize=5)
    plt.xticks(rotation=45)
    plt.ylabel("#Snapshots")
    plt.title("#Snapshots Per Interval")
    plt.savefig(filename_for_save + "_Snapshots_per_interval.png", dpi=300)
    for measure in ['QueryTermsRatio', 'StopwordsRatio', 'Entropy', 'SimClueWeb']:
        plt.cla()
        plt.clf()
        plot_df = summary_df[['Interval', measure]].groupby(['Interval']).mean()
        plot_df.rename(columns = {measure : 'ALL'}, inplace =True)
        plot_df_ = summary_df[summary_df['IsCompQuery'] == 1][['Interval', measure]].groupby(['Interval']).mean()
        plot_df_.rename(columns={measure: 'Comp Queries'}, inplace=True)
        plot_df = pd.merge(
            plot_df,
            plot_df_,
            left_index=True,
            right_index=True)
        plot_df_ = summary_df[summary_df['IsCompQuery'] == 0][['Interval', measure]].groupby(['Interval']).mean()
        plot_df_.rename(columns={measure: 'Non Comp'}, inplace=True)
        plot_df = pd.merge(
            plot_df,
            plot_df_,
            left_index=True,
            right_index=True)

        plot_df.plot(kind='bar')
        plt.xlabel('Interval')
        plt.tick_params(axis='x', labelsize=6)
        plt.xticks(rotation=45)
        plt.legend(loc="center left", bbox_to_anchor=(1,0.5))
        plt.subplots_adjust(right=0.75, bottom=0.15)
        if measure != 'SimClueWeb':
            plt.ylabel("%" + measure)
            plt.title("Mean " + measure + " Precentage Difference from ClueWeb09")
        else:
            plt.ylabel(measure)
            plt.title("Mean " + measure)

        plt.savefig(measure + filename + "_Precentage_Difference_per_interval.png", dpi=300)
        plot_df.to_csv(os.path.join( 'plot_df', filename_for_save + '_' + measure + '.tsv'), sep='\t')
        # tikzplotlib.save(measure + "Precentage_Difference_per_interval.tex")
        # plt.cla()
        # plt.clf()
        # plot_df = summary_df[['Interval', measure]].groupby(['Interval']).median()
        # plot_df.plot(legend=False, kind='bar', color='r')
        # plt.xlabel('Interval')
        # plt.tick_params(axis='x', labelsize=5)
        # plt.xticks(rotation=45)
        # if measure != 'SimClueWeb':
        #     plt.ylabel("%" + measure)
        #     plt.title("Median " + measure + " Precentage Difference from ClueWeb09")
        # else:
        #     plt.ylabel(measure)
        #     plt.title("Median " + measure)
        #
        # plt.savefig(measure + "Median_Precentage_Difference_per_interval.png", dpi=300)

def plot_retrival_stats(
        lookup_method,
        interval_freq,
        comp_queries_list):
    work_df = pd.read_csv(interval_freq + '_' + lookup_method + '_Per_query_stats.tsv' , sep = '\t', index_col=False)
    work_df['IsCompQuery'] = work_df['Query_Num'].apply(lambda x: 1 if x in comp_queries_list else 0)

    save_df = pd.DataFrame({})
    for measure in ['Map', 'P@5', 'P@10', 'Overlap@5_ClueWeb','Overlap@5_Last','RBO_0.95_Ext_ClueWeb', 'RBO_0.975_Ext_ClueWeb','RBO_0.99_Ext_ClueWeb','RBO_0.95_Ext_Last', 'RBO_0.975_Ext_Last','RBO_0.99_Ext_Last']:
        plt.cla()
        plt.clf()
        plot_df = work_df[['Interval', measure]].groupby(['Interval']).mean()
        plot_df.rename(columns={measure: 'ALL'}, inplace=True)
        plot_df_ = work_df[work_df['IsCompQuery'] == 1][['Interval', measure]].groupby(['Interval']).mean()
        plot_df_.rename(columns={measure: 'Comp Queries'}, inplace=True)
        plot_df = pd.merge(
            plot_df,
            plot_df_,
            left_index=True,
            right_index=True)
        plot_df_ = work_df[work_df['IsCompQuery'] == 0][['Interval', measure]].groupby(['Interval']).mean()
        plot_df_.rename(columns={measure: 'Non Comp'}, inplace=True)
        plot_df = pd.merge(
            plot_df,
            plot_df_,
            left_index=True,
            right_index=True)
        measure = measure.replace('Ext_', '')

        plot_df.plot(kind='bar')
        plt.xlabel('Interval')
        plt.tick_params(axis='x', labelsize=6)
        plt.xticks(rotation=45)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.75, bottom=0.15)

        plt.ylabel(measure)
        plt.title(lookup_method + " Mean " + measure)

        plt.savefig(interval_freq + '_' + lookup_method + "_" + measure + "_per_interval.png", dpi=300)
        for col in list(plot_df.columns):
            plot_df.rename(columns = {col : measure + ' ' + col }, inplace=True)
        if save_df.empty == True:
            save_df = plot_df
        else:
            save_df = pd.merge(
                save_df,
                plot_df,
                left_index=True,
                right_index=True)
    save_df.to_csv(os.path.join('plot_df', interval_freq + '_' +  lookup_method + '_Retrival_Stats.tsv'), sep='\t')


def create_per_query_stats(filename = 'Summay_snapshot_stats.tsv'):
    work_df = pd.read_csv(filename, sep = '\t', index_col = False)
    work_df = work_df[work_df['Interval'] != 'ClueWeb09'].copy()

    plot_df = work_df[['QueryNum', 'SimToClueWeb']].groupby(['QueryNum']).mean()
    plot_df = plot_df.sort_values('SimToClueWeb', ascending=False)
    plot_df.plot(legend=False, kind='bar', color='c')
    plt.xlabel('Query')
    plt.xticks(visible=False)
    plt.ylabel('Mean SimToClueWeb')
    plt.savefig("Mean_SimToClueWeb_per_query.png", dpi=300)

def plot_stats_vs_winner(
        interval_freq = '1M',
        lookup_method = 'Backward'):

    work_df = pd.read_csv('Summay_vs_winner_' + interval_freq + '_'+ lookup_method+'.csv', index_col=False)
    all_queries   = list(work_df['QueryNum'].drop_duplicates())
    all_intervals = sorted(list(work_df['Interval'].drop_duplicates()))

    summary_df = pd.DataFrame(columns = ['QueryNum','Interval','Docno','Round','Winner','Sim_PW','QueryTermsRatio','StopwordsRatio','Entropy','Trend'])
    next_index = 0
    spearman_corr_df = pd.DataFrame({})
    for query_num in all_queries:
        print (query_num )
        query_df = work_df[work_df['QueryNum'] == query_num].copy()
        interval_winners_df_dict = {}
        for interval_idx in range(len(all_intervals)):
            interval = all_intervals[interval_idx]
            query_interval_df = query_df[query_df['Interval'] == interval]
            query_interval_df['Sim_PD_Rank'] = query_interval_df['Sim_PW'].rank(ascending= False)
            # interval_winners_df_dict[interval] = query_interval_df[query_interval_df['Rank'] == 1]
            if interval != '2008-01-01':
                spearman_corr_df = spearman_corr_df.append(query_interval_df[['Sim_PD_Rank', 'Rank']])
                addition = 0
                while (interval_idx - addition) >= 0:
                    temp_interval    = all_intervals[interval_idx - addition]
                    temp_interval_df = query_df[query_df['Interval'] == temp_interval].copy()
                    temp_interval_df = pd.merge(
                        temp_interval_df,
                        query_interval_df[['Rank', 'Docno']].rename(columns = {'Rank' : 'CurrRank'}),
                        on=['Docno'],
                        how='inner')
                    for index, row  in temp_interval_df.iterrows():
                        trend = '='
                        if row['Rank'] > row['CurrRank']:
                            trend = 'W'
                        elif row['Rank'] < row['CurrRank']:
                            trend = 'L'
                        insert_row = [query_num, interval, row['Docno'],(-1)*addition, row['Rank'] == 1, row['Sim_PW'],
                                      row['QueryTermsRatio'],row['StopwordsRatio'],row['Entropy'], trend]

                        summary_df.loc[next_index] = insert_row
                        next_index += 1
                    addition += 1

    print('Spearman corr: ' +str(spearman_corr_df.corr().loc['Sim_PD_Rank']['Rank']))
    summary_df.to_csv('Summay_vs_winner_processed_' + interval_freq + '_'+ lookup_method +'_' +
                      str(round(spearman_corr_df.corr().loc['Sim_PD_Rank']['Rank'], 6))  + '.tsv', sep ='\t', index=False)



def plot_stats_vs_winner_plots(
        comp_queries_list,
        file_name,
        interval_freq = '1M',
        lookup_method = 'Backward'):

    work_df = pd.read_csv(file_name, sep = '\t', index_col=False)
    winner_df = work_df[work_df['Winner'] == True]
    non_winner_df = work_df[work_df['Winner'] == False]

    work_df['IsCompQuery'] = work_df['QueryNum'].apply(lambda x: 1 if x in comp_queries_list else 0)
    comp_df = work_df[work_df['IsCompQuery'] == 1]
    comp_winner_df = comp_df[comp_df['Winner'] == True]
    comp_non_winner_df = comp_df[comp_df['Winner'] == False]

    n_plots = 2
    f, axarr = plt.subplots(n_plots,n_plots)
    idx_r = 0
    idx_c = 0
    for measure in ['SimClueWeb','QueryTermsRatio','StopwordsRatio','Entropy']:
        plot_df = winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color = 'b', label = 'ALL_Winner')
        plot_df = non_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='b',linestyle='--', label='ALL_Loser')
        plot_df = comp_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='r', label='Comp_Winner')
        plot_df = comp_non_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='r', linestyle='--', label='Comp_Loser')
        axarr[idx_r, idx_c].set_title(measure)

        if idx_r == (n_plots - 1):
            idx_r = 0
            idx_c += 1
            continue

        idx_r += 1
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75)
    plt.savefig(file_name.replace('.tsv', '.png'), dpi =300)

    plt.cla()
    plt.clf()


    winner_df_ = work_df[work_df['Trend'] == 'W']
    non_winner_df = work_df[work_df['Trend'] == 'L']

    comp_winner_df_ = comp_df[comp_df['Trend'] == 'W']
    comp_non_winner_df = comp_df[comp_df['Trend'] == 'L']

    n_plots = 2
    f, axarr = plt.subplots(n_plots, n_plots)
    idx_r = 0
    idx_c = 0
    for measure in ['SimClueWeb', 'QueryTermsRatio', 'StopwordsRatio', 'Entropy']:
        plot_df = winner_df_[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='b', label='ALL_Up')
        plot_df = non_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='b', linestyle='--', label='ALL_Down')
        plot_df = winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='b', linestyle=':', label='ALL_Winner', alpha =0.7)

        plot_df = comp_winner_df_[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='r', label='Comp_Up')
        plot_df = comp_non_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='r', linestyle='--', label='Comp_Down')
        plot_df = comp_winner_df[['Round', measure]].groupby(['Round']).mean()
        plot_df = plot_df.reset_index()
        axarr[idx_r, idx_c].plot(plot_df['Round'], plot_df[measure], color='r', linestyle=':', label='Comp_Winner',
                                 alpha=0.7)
        axarr[idx_r, idx_c].set_title(measure)


        if idx_r == (n_plots - 1):
            idx_r = 0
            idx_c += 1
            continue

        idx_r += 1
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75)
    plt.savefig(file_name.replace('.tsv', '_WL.png'), dpi=300)

comp_queries_list = [195,193,188,180,177,161,144,59,51,45,36,34,32,29,11,10,
                     9,2,78,4,167,69,166,33,164,18,182,17,98,124,48]

# df_1 = pd.read_csv('Summay_snapshot_stats.tsv', sep = '\t', index_col = False)
# df_2 = pd.read_csv('Summay_snapshot_stats_old.tsv', sep = '\t', index_col = False)
#
# merged = pd.merge(
#     df_1,
#     df_2,
#     on = ['Docno','Interval'],
#     how = 'inner')
# pass
# create_snapshot_changes_stats()
# create_lost_snapshot_stats()
#
# plot_retrival_stats()
# create_interval_coverage_plot()

# plot_retrival_stats(lookup_method = 'Backward',interval_freq='1M', comp_queries_list=comp_queries_list)
# create_per_query_stats()
plot_stats_vs_winner()
# plot_stats_vs_winner_plots(comp_queries_list, 'Summay_vs_winner_processed_2M_NoLookup_0.562329.tsv')