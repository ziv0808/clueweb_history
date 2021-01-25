from pymongo import MongoClient
import os
import re
import ast
import sys
import datetime
import unicodedata
import krovetzstemmer
import xml.etree.ElementTree as ET
import csv
import subprocess
from bs4 import BeautifulSoup
import pandas as pd
from utils import calc_cosine, calc_tfidf_dict, calc_releational_measure


def changeStatus():
    client = MongoClient('asr2.iem.technion.ac.il',27017)
    db = client.asr16
    status = db.status.find_one({})
    indeterminate = status['indeterminate']
    status['indeterminate'] = not(indeterminate)
    db.status.save(status)


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')



def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)

    out, err = p.communicate()
    return out


def retrieve_users():
    return ["K5W94I","FQ5ADE","ZZ9G1J","55BITZ","0H1M36","AS64YR","MPNM2K","B8F6E9","632S70","YZ797R","R63H9T","1HAIVS","IL2Z4J","OAR3JM","SQK6UG","6WBO8B"
        , "4IB6XC","P3VUZZ","U3BWFS","ECXXPC","YF1WGX","T04YHN","GVWD05","KN0P86","IR1LI3","TXOH1U","USDW0H","YR85MD","W0N7XO","E4XDKY","WZZIQQ","PW656R","UAYNLZ","319WBN","S8C22R"
        , "37ZSCA","PBXOHX","S33OJM","TX5GTH","8G7EXD","5PFRI1","H8MTZJ","P4XXKM","B4UN6O","GF7KCF","R3CHA3","ELOCB2","5RJQ5W","8QHNXA","0H4T7V","93TF3E","FBZ7PM","J2O460"
        ,"XK1O1C","SYK2VV","EHOVYP","WPGM77","6OJZI1","5440N4","5KBCIG","G3UP0M","XW13YM","678QUM","JUKQRU","BCI0MP","J5YEUV","ITIHJZ","C07X4Q","MR6B6Q","Y150M4","7DSLJS","KKS1QA"
        ,"MRCU75","IM4W6P","VS0C4F","6QRS72","JJN2DN","E9NUZ6","AQ5V72","F30TB1","BNFNVS","8ZJ42Z","4EE20G","4BJPNQ"]




def createTrecTextForCurrentDocuments(baseDir):
    """
    Create a trec text file of current documents
    """
    pathToFolder = baseDir + 'Collections/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)
    currentTime = str(datetime.datetime.now()).replace(":", "-").replace(" ", "-").replace(".", "-")
    pathToTrecText = pathToFolder+"TrecText/"
    if not os.path.exists(pathToTrecText):
        os.makedirs(pathToTrecText)
    filename = pathToTrecText + currentTime
    client = MongoClient('asr2.iem.technion.ac.il',27017)
    db = client.asr16
    documents = db.documents.find({}).sort('query_id',1)
    queryToDocnos= {}
    current_users = retrieve_users()
    f = open(filename, 'w')
    for document in documents:
        if document['username'] in current_users:
            print(document['query_id'], document['username'])
            f.write('<DOC>\n')
            docno = str(document['query_id']).zfill(3) + '-' + str(document['username'])
            f.write('<DOCNO>' + docno + '</DOCNO>\n')
            docnos = queryToDocnos.get(str(document['query_id']).zfill(3), [])
            docnos.append(docno)
            queryToDocnos[str(document['query_id']).zfill(3)] = docnos
            f.write('<TEXT>\n')
            f.write(unicodedata.normalize('NFKD', document['current_document']).encode('cp1252', "ignore").decode('utf-8', 'replace').replace(u'\uFFFD', ' ').rstrip())
            f.write('\n</TEXT>\n')
            f.write('</DOC>\n')
    f.close()
    pathToWorkingSet = pathToFolder+ 'WorkingSets/'
    if not os.path.exists(pathToWorkingSet):
        os.makedirs(pathToWorkingSet)
    workingSetFilename = pathToWorkingSet + currentTime
    f = open(workingSetFilename, 'w')
    for query, docnos in queryToDocnos.items():
        i = 1
        for docid in docnos:
            f.write(query.zfill(3) + ' Q0 ' + docid + ' ' + str(i) + ' -' + str(i) + ' indri\n')
            i +=1
    f.close()
    return filename, workingSetFilename, currentTime

def buildIndex(filename, currentTime, baseDir):
    """
    Parse the trectext file given, and create an index.
    """
    pathToFolder = baseDir + 'Collections/IndriIndices/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)
    INDRI_BUILD_INDEX = '/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/indri/bin/IndriBuildIndex'
    CORPUS_PATH = filename
    CORPUS_CLASS = 'trectext'
    MEMORY = '1G'
    INDEX = pathToFolder + currentTime
    STEMMER =  'krovetz'
    run_bash_command(INDRI_BUILD_INDEX + ' -corpus.path='+CORPUS_PATH + ' -corpus.class='+CORPUS_CLASS + ' -index='+INDEX + ' -memory='+MEMORY + ' -stemmer.name=' + STEMMER)
    return INDEX


def mergeIndices(asrIndex, baseDir):
    """
    Merge indices of ASR and ClueWeb09. If MergedIndx is exist, it will be deleted.
    """

    INDRI_DUMP_INDEX = '/mnt/bi-strg3/v/zivvasilisky/ziv/env/indri/indri/bin/dumpindex'
    CLUEWEB = '/mnt/bi-strg3/v/zivvasilisky/index/CW09ForMerge'
    pathToFolder = baseDir+'Collections/'
    MERGED_INDEX = pathToFolder + '/mergedindex'
    run_bash_command('rm -r ' + MERGED_INDEX)
    run_bash_command(INDRI_DUMP_INDEX + ' ' + MERGED_INDEX + ' merge ' + CLUEWEB + ' ' + asrIndex)
    return MERGED_INDEX



def run_bert_model(currentTime):
    command = 'python3 /lv_local/home/zivvasilisky/ziv/clueweb_history/run_bert_on_curr_comp_docs.py /lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/TrecText/' + currentTime
    out = run_bash_command(command)
    print(out)



def rewrite_features(scores, features_fname):
    new_file = features_fname + "_with_waterloo"
    new = open(new_file,'w')
    with open(features_fname) as old:
        for line in old:
            name = line.split(" # ")[1].rstrip()
            data= line.split(" # ")[0]
            qid = name.split("-")[0]
            user = name.split("-")[1]
            waterloo_score = scores[qid][user]
            new_data = data+" 26:"+str(waterloo_score)+" # "+name+"\n"
            new.write(new_data)
    new.close()
    return new_file



def get_waterloo_scores():
    stats = {}
    client = MongoClient('asr2.iem.technion.ac.il', 27017)
    db = client.asr16
    documents = db.documents.find({})
    for doc in documents:
        qid = doc["query_id"]
        username = doc["username"]
        waterloo = doc["waterloo"]
        if qid not in stats:
            stats[qid]={}
        stats[qid][username]=waterloo
    return stats


def create_normalized_scores(waterloo_scores):
    normalized_scores = {}
    for query in waterloo_scores:
        normalized_scores[query]={}
        all_scores = [waterloo_scores[query][u] for u in waterloo_scores[query]]
        max_score = max(all_scores)
        min_score = min(all_scores)
        for user in waterloo_scores[query]:
            if max_score!=min_score:
                normalized_scores[query][user] = (waterloo_scores[query][user]-min_score)/(max_score-min_score)
            else:
                normalized_scores[query][user] = 0
    return normalized_scores



def runRankingModels_old(mergedIndex, workingSet, currentTime, baseDir):
    scriptDir = '/lv_local/home/zivvasilisky/ziv/content_modification_code/scripts/'
    pathToFolder = baseDir + 'Results/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)
    INDEX = mergedIndex
    WORKING_SET_FILE = workingSet
    MODEL_DIR = baseDir+"content_modification_code/rank_models/"
    MODEL_FILE = MODEL_DIR+"model_lambdatamart"
    QUERIES_FILE = baseDir+'Data/QueriesFile.xml'
    FEATURES_DIR = pathToFolder + 'Features/' +  currentTime
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
    ORIGINAL_FEATURES_FILE = 'features'
    command = scriptDir+'LTRFeatures ' + QUERIES_FILE + ' -stream=doc -index=' + INDEX + ' -repository='+ INDEX +' -useWorkingSet=true -workingSetFile='+ WORKING_SET_FILE + ' -workingSetFormat=trec'
    print(command)
    out = run_bash_command(command)
    print(out)
    run_command('mv doc*_* ' + FEATURES_DIR)
    command='perl '+scriptDir+'generate.pl ' + FEATURES_DIR + ' ' + WORKING_SET_FILE
    print(command)
    out=run_bash_command(command)
    print(out)
    # waterloo=get_waterloo_scores()
    # normalized_scores = create_normalized_scores(waterloo)
    # FEATURES_FILE=rewrite_features(normalized_scores,ORIGINAL_FEATURES_FILE)
    FEATURES_FILE = ORIGINAL_FEATURES_FILE
    command = 'java -jar '+scriptDir+'RankLib.jar -load ' + MODEL_FILE + ' -rank '+FEATURES_FILE+' -score predictions.tmp'
    print(command)
    out=run_bash_command(command)
    print(out)
    command = 'cut -f3 predictions.tmp > predictions'
    print(command)
    out=run_bash_command(command)
    print(out)
    run_bash_command('rm predictions.tmp')
    RANKED_LIST_DIR = pathToFolder+'RankedLists/'
    if not os.path.exists(RANKED_LIST_DIR):
        os.makedirs(RANKED_LIST_DIR)
    PREDICTIONS_FILE = 'predictions'
    command='perl '+scriptDir+'order.pl ' + RANKED_LIST_DIR+ 'LambdaMART' + currentTime + ' ' +FEATURES_FILE + ' ' + PREDICTIONS_FILE
    print(command)
    out=run_bash_command(command)
    print(out)
    return RANKED_LIST_DIR+ 'LambdaMART' + currentTime


def turn_df_to_feature_str_for_model(
        df,
        feature_list):

    feature_str = ""
    for index, row in df.iterrows():
        feature_str += "0 qid:" + str(row['QueryNum'])
        feat_num = 1
        for feature in feature_list:
            feature_str += " " + str(feat_num) + ":" + str(row[feature])
            feat_num += 1

        feature_str += " # " +row['Docno'] + '\n'

    return feature_str

def read_current_doc_file(doc_filepath):
    stats = {}
    with open(doc_filepath, 'r') as f:
        soup = BeautifulSoup(f.read())
    all_docs = soup.find_all('doc')

    for doc_ in list(all_docs):
        docno = doc_.find('docno').text
        fulltext = doc_.find('text').text
        query = docno.split('-')[0]
        user =docno.split('-')[1]
        if query not in stats:
            stats[query] = {}

        stats[query][user] = {}
        stats[query][user]['FullText'] = fulltext

    return stats

def create_tdfidf_dicts_per_doc_for_file(doc_filepath):
    fulltext_dict = read_current_doc_file(doc_filepath)
    df_dict = {}
    stemmer = krovetzstemmer.Stemmer()
    for query in fulltext_dict:
        for user in fulltext_dict[query]:
            fulltext = re.sub('[^a-zA-Z0-9 ]', ' ', fulltext_dict[query][user]['FullText'])
            fulltext_dict[query][user]['TfDict'] = {}
            fulltext_dict[query][user]['StemList'] = []
            curr_fulltext_list = fulltext.split(" ")
            for stem in curr_fulltext_list:
                stem = stemmer.stem(stem)
                if stem == '' or stem == '\n':
                    continue
                if stem not in fulltext_dict[query][user]['TfDict']:
                    fulltext_dict[query][user]['StemList'].append(stem)
                    fulltext_dict[query][user]['TfDict'][stem] = 1
                else:
                    fulltext_dict[query][user]['TfDict'][stem] += 1
            for stem in fulltext_dict[query][user]['StemList']:
                if stem in df_dict:
                    df_dict[stem] += 1
                else:
                    df_dict[stem] = 1

    for query in fulltext_dict:
        for user in fulltext_dict[query]:
            fulltext_dict[query][user]['DFList'] = []
            fulltext_dict[query][user]['TFList'] = []
            for stem in fulltext_dict[query][user]['StemList']:
                fulltext_dict[query][user]['DFList'].append(df_dict[stem])
                fulltext_dict[query][user]['TFList'].append(fulltext_dict[query][user]['TfDict'][stem])
            fulltext_dict[query][user]['TfIdf'] = calc_tfidf_dict(
                stem_list=fulltext_dict[query][user]['StemList'],
                tf_list=fulltext_dict[query][user]['TFList'],
                df_list=fulltext_dict[query][user]['DFList'])

    return fulltext_dict




def create_model_ready_feature_df(
        currentTime,
        feature_list = ['Ent', 'FracStops', 'Len', 'LMIR.DIR', 'StopCover', 'TFSum', 'TFNormSum', 'SimClueWeb', 'BM25Score',
                    'IDF', 'BERTScore'],
        static = True,
        prev_rounds_list = [],
        baseDir =None):
    with open('/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/FeatureIdx/' + currentTime +  '_BERT.json', 'r') as f:
        bert_dict = ast.literal_eval(f.read())

    all_feature_dict = create_ltr_feature_dict('/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/Features/' + currentTime +'/')

    if static == True:
        feature_df = pd.DataFrame(columns = ['QueryNum', 'Docno'] + feature_list)
        next_idx = 0
        for query in all_feature_dict:
            if query.split('_')[1] == '0':
                for docno in all_feature_dict[query]:
                    insert_row = [query, docno]
                    for feature in feature_list:
                        if feature == 'SimClueWeb':
                            insert_row.append(1.0)
                        elif feature == 'BERTScore':
                            insert_row.append(bert_dict[query][docno.split('-')[1]]['BERTScore'])
                        else:
                            insert_row.append(all_feature_dict[query][docno][feature])
                    feature_df.loc[next_idx] = insert_row
                    next_idx += 1

        all_queries = list(feature_df['QueryNum'].drop_duplicates())
        fin_df = pd.DataFrame({})
        for q in sorted(all_queries):
            tmp_q_df = feature_df[feature_df['QueryNum'] == q].copy()
            for feature in feature_list:
                min_feat = tmp_q_df[feature].min()
                max_feat = tmp_q_df[feature].max()
                tmp_q_df[feature] = tmp_q_df[feature].apply(
                    lambda x: (x - min_feat) / float(max_feat - min_feat) if (max_feat - min_feat) != 0 else 0.0)

            fin_df = fin_df.append(tmp_q_df, ignore_index=True)

        fin_df.fillna(0.0, inplace=True)
        feature_str = turn_df_to_feature_str_for_model(df=fin_df, feature_list=feature_list)
        fin_df.to_csv('df_features_0.tsv', sep='\t', index=False)
        with open('features_0', 'w') as f:
            f.write(feature_str)
    else:
        base_feature_list = feature_list[:]
        broken_feature_groupname = "Static_MXXSnap_STDXXSnap_MinXXSnap_MaxXXSnap_MGXXSnap".split('_')
        if 'MGXXSnap' in broken_feature_groupname:
            for base_feat in base_feature_list:
                feature_list.append(base_feat + '_MGXXSnaps')
        if 'MXXSnap' in broken_feature_groupname:
            for base_feat in base_feature_list:
                feature_list.append(base_feat + '_MXXSnaps')
        if 'STDXXSnap' in broken_feature_groupname:
            for base_feat in base_feature_list:
                feature_list.append(base_feat + '_STDXXSnaps')
        if 'MinXXSnap' in broken_feature_groupname:
            for base_feat in base_feature_list:
                feature_list.append(base_feat + '_MinXXSnaps')
        if 'MaxXXSnap' in broken_feature_groupname:
            for base_feat in base_feature_list:
                feature_list.append(base_feat + '_MaxXXSnaps')

        historical_feature_dict = {}
        parsed_curr_file_dict = create_tdfidf_dicts_per_doc_for_file(baseDir + 'Collections/TrecText/' + currentTime)
        for historical_snap in prev_rounds_list:
            with open('/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/FeatureIdx/' + historical_snap + '_BERT.json','r') as f:
                curr_snap_bert_dict = ast.literal_eval(f.read())
            curr_snap_all_feature_dict = create_ltr_feature_dict('/lv_local/home/zivvasilisky/ASR20/epoch_run/Results/Features/' + historical_snap + '/')
            parsed_curr_snap_file_dict = create_tdfidf_dicts_per_doc_for_file(baseDir + 'Collections/TrecText/' + historical_snap)

            for query in curr_snap_all_feature_dict:
                if query.split('_')[1] != '0':
                    if query not in historical_feature_dict:
                        historical_feature_dict[query] = {}
                    for docno in curr_snap_all_feature_dict[query]:
                        if docno not in historical_feature_dict[query]:
                            historical_feature_dict[query][docno] = {}
                        for feature in base_feature_list:
                            if feature not in historical_feature_dict[query][docno]:
                                historical_feature_dict[query][docno][feature] = []
                            if feature == 'SimClueWeb':
                                curr_sim = calc_cosine(parsed_curr_file_dict[query][docno.split('-')[1]]['TfIdf'], parsed_curr_snap_file_dict[query][docno.split('-')[1]]['TfIdf'])
                                historical_feature_dict[query][docno][feature].append(curr_sim)
                            elif feature == 'BERTScore':
                                historical_feature_dict[query][docno][feature].append(curr_snap_bert_dict[query][docno.split('-')[1]]['BERTScore'])
                            else:
                                historical_feature_dict[query][docno][feature].append(curr_snap_all_feature_dict[query][docno][feature])
        print(feature_list)
        feature_df = pd.DataFrame(columns=['QueryNum', 'Docno'] + feature_list, sep=r'\s*,\s*')
        next_idx = 0
        for query in all_feature_dict:
            if query.split('_')[1] != '0':
                for docno in all_feature_dict[query]:
                    insert_row = [query, docno]
                    for feature in feature_list:
                        if 'XXSnap' in feature:
                            if '_MXXSnaps' in feature:
                                insert_row.append(pd.np.mean(historical_feature_dict[query][docno][feature.replace('_MXXSnaps','')]))
                            elif '_STDXXSnaps' in feature:
                                insert_row.append(pd.np.std(historical_feature_dict[query][docno][feature.replace('_STDXXSnaps', '')]))
                            elif '_MinXXSnaps' in feature:
                                insert_row.append(pd.np.min(historical_feature_dict[query][docno][feature.replace('_MinXXSnaps', '')]))
                            elif '_MaxXXSnaps' in feature:
                                insert_row.append(pd.np.max(historical_feature_dict[query][docno][feature.replace('_MaxXXSnaps', '')]))
                            elif '_MGXXSnaps' in feature:
                                raw_feat = feature.replace('_MGXXSnaps', '')
                                if raw_feat == 'SimClueWeb':
                                    curr_snap_score = 1.0
                                elif raw_feat == 'BERTScore':
                                    curr_snap_score = bert_dict[query][docno.split('-')[1]]['BERTScore']
                                else:
                                    curr_snap_score = all_feature_dict[query][docno][raw_feat]
                                curr_feat_list = historical_feature_dict[query][docno][raw_feat]
                                curr_feat_list.append(curr_snap_score)
                                avg_score = 0.0
                                for i in range(1,len(curr_feat_list)):
                                    avg_score += calc_releational_measure(measure_obs=curr_feat_list[i], reletional_measure_obs=curr_feat_list[i-1])
                                avg_score = avg_score / float(len(curr_feat_list) - 1)
                                insert_row.append(avg_score)

                        elif feature == 'SimClueWeb':
                            insert_row.append(1.0)
                        elif feature == 'BERTScore':
                            insert_row.append(bert_dict[query][docno.split('-')[1]]['BERTScore'])
                        else:
                            insert_row.append(all_feature_dict[query][docno][feature])
                    feature_df.loc[next_idx] = insert_row
                    next_idx += 1
        all_queries = list(feature_df['QueryNum'].drop_duplicates())
        fin_df = pd.DataFrame({})
        for q in sorted(all_queries):
            tmp_q_df = feature_df[feature_df['QueryNum'] == q].copy()
            for feature in feature_list:
                min_feat = tmp_q_df[feature].min()
                max_feat = tmp_q_df[feature].max()
                tmp_q_df[feature] = tmp_q_df[feature].apply(
                    lambda x: (x - min_feat) / float(max_feat - min_feat) if (max_feat - min_feat) != 0 else 0.0)

            fin_df = fin_df.append(tmp_q_df, ignore_index=True)

        fin_df.fillna(0.0, inplace=True)
        feature_str = turn_df_to_feature_str_for_model(df=fin_df, feature_list=feature_list)
        fin_df.to_csv('df_features_1.tsv', sep ='\t', index =False)
        with open('features_1', 'w') as f:
            f.write(feature_str)

def create_ltr_feature_dict(
        feature_folder):
    res_dict = {}
    for filename in os.listdir(feature_folder):
        if filename.startswith('doc') and '_' in filename:
            print(filename)
            sys.stdout.flush()
            feature = filename.split('_')[0].replace('doc', '')
            if feature == 'BM25':
                feature = 'BM25Score'
            query = filename.split('_')[1] + '_' + filename.split('_')[2]
            if query not in res_dict:
                res_dict[query] = {}
            with open(os.path.join(feature_folder, filename), 'r') as f:
                file_str = f.read()
            file_lines = file_str.split('\n')
            first_ = True
            for line in file_lines:
                if line == '':
                    continue
                docno = line.split(' ')[0]
                if docno not in res_dict[query]:
                    res_dict[query][docno] = {}
                if len(line.split(' ')) > 2:
                    if first_ == True:
                        print('multiple Features!')
                        first_ = False
                    feature_additions = ['Sum', 'Min', 'Max', 'Mean', 'Std']
                    for i in range(len(feature_additions)):
                        res_dict[query][docno][feature + feature_additions[i]] = float(line.split(' ')[i + 1])
                else:
                    val = float(line.split(' ')[1])
                    res_dict[query][docno][feature] = val
    return res_dict


def runRankingModels(mergedIndex, workingSet, currentTime, baseDir, curr_static_model, curr_s_msmm_mg_model, prev_rounds_list):
    scriptDir = '/lv_local/home/zivvasilisky/ziv/content_modification_code/scripts/'
    pathToFolder = baseDir + 'Results/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)
    INDEX = mergedIndex
    WORKING_SET_FILE = workingSet
    MODEL_DIR = baseDir+"content_modification_code/rank_models/"
    STATIC_MODEL_FILE = MODEL_DIR+curr_static_model
    FULL_MODEL_FILE = MODEL_DIR+curr_s_msmm_mg_model
    QUERIES_FILE = baseDir+'Data/QueriesFile.xml'
    FEATURES_DIR = pathToFolder + 'Features/' +  currentTime
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
    # CONTROL_FEATURES_FILE = 'features_0'
    # command = scriptDir+'LTRFeatures ' + QUERIES_FILE + ' -stream=doc -index=' + INDEX + ' -repository='+ INDEX +' -useWorkingSet=true -workingSetFile='+ WORKING_SET_FILE + ' -workingSetFormat=trec'
    # print(command)
    # out = run_bash_command(command)
    # print(out)
    # run_command('mv doc*_* ' + FEATURES_DIR)
    # run_bert_model(currentTime)
    # create_model_ready_feature_df(currentTime, static=True)
    # FEATURES_FILE = CONTROL_FEATURES_FILE
    # command = 'java -jar '+scriptDir+'RankLib.jar -load ' + STATIC_MODEL_FILE + ' -rank '+FEATURES_FILE+' -score predictions.tmp'
    # print(command)
    # out=run_bash_command(command)
    # print(out)
    # command = 'cut -f3 predictions.tmp > predictions'
    # print(command)
    # out=run_bash_command(command)
    # print(out)
    # run_bash_command('rm predictions.tmp')
    RANKED_LIST_DIR = pathToFolder+'RankedLists/'
    if not os.path.exists(RANKED_LIST_DIR):
        os.makedirs(RANKED_LIST_DIR)
    # PREDICTIONS_FILE = 'predictions'
    # command='perl '+scriptDir+'order.pl ' + RANKED_LIST_DIR+ 'LambdaMART_0_' + currentTime + ' ' +FEATURES_FILE + ' ' + PREDICTIONS_FILE
    # print(command)
    # out=run_bash_command(command)
    # print(out)

    create_model_ready_feature_df(currentTime, static=False,prev_rounds_list=prev_rounds_list,baseDir=baseDir)
    TEST_FEATURES_FILE = 'features_1'
    FEATURES_FILE = TEST_FEATURES_FILE

    command = 'java -jar ' + scriptDir + 'RankLib.jar -load ' + FULL_MODEL_FILE + ' -rank ' + FEATURES_FILE + ' -score predictions.tmp'
    print(command)
    out = run_bash_command(command)
    print(out)
    command = 'cut -f3 predictions.tmp > predictions'
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command('rm predictions.tmp')
    PREDICTIONS_FILE = 'predictions'
    command = 'perl ' + scriptDir + 'order.pl ' + RANKED_LIST_DIR + 'LambdaMART_1_' + currentTime + ' ' + FEATURES_FILE + ' ' + PREDICTIONS_FILE
    print(command)
    out = run_bash_command(command)
    print(out)

    static_res_file = RANKED_LIST_DIR + 'LambdaMART_0_' + currentTime
    full_model_res_file = RANKED_LIST_DIR + 'LambdaMART_1_' + currentTime
    run_bash_command('cat '  + static_res_file + ' ' + full_model_res_file + ' > ' +RANKED_LIST_DIR +'LambdaMART' + currentTime)

    return RANKED_LIST_DIR+ 'LambdaMART' + currentTime




def updateScores(rankedLists):
    """
    Update scores and positions to main documents.
    """
    docToRank = {}
    for rankedList in rankedLists:

        f = open(rankedList, 'r')
        for line in f:
            documentID = line.split()[2]
            docno = documentID
            score = float(line.split()[4])
            position = int(line.split()[3])
            docToRank[docno] = (position,score)
        f.close()
    client = MongoClient('asr2.iem.technion.ac.il',27017)
    db = client.asr16
    documents = db.documents.find({})
    for document in documents:
        key = document["query_id"]+"-"+document["username"]
        document['position'] = docToRank[key][0]
        document['score'] = docToRank[key][1]
        document['posted_document'] = document['current_document']
        db.documents.save(document)


featureNames = [
    'docCoverQueryNum',
    'docCoverQueryRatio',
    'docLen',
    'docIDF',
    'docBM25',
    'docLMIR.DIR',
    'docLMIR.JM',
    'docEnt',
    'docStopCover',
    'docFracStops',
    'docTF',
    'docTFNorm',
    'docTFIDF',
    'docCent',
    'docDocToDocCosine',
    'docQueryCosine',
    'docDocToDocCent']

def parseFeatures(featuresDirectory):
    """
    Parse LTR features for each document.
    """
    client = MongoClient('asr2.iem.technion.ac.il',27017)
    db = client.asr16
    queries = db.documents.distinct("query_id")
    docToFeatureVector = {}
    for featureName in featureNames:
        for query in queries:
            if not os.path.exists(featuresDirectory + '/' +featureName + '_' + str(query)):
                continue
            f = open(featuresDirectory + '/' +featureName + '_' + str(query), 'r')

            for line in f:
                documentID = line.split()[0]
                docno = documentID
                features = line.split()[1:]
                floatFeatures = [float(val) for val in features]
                documentFeaturesDict = docToFeatureVector.get(docno, {})
                documentFeaturesDict[featureName.replace(".","_")] = floatFeatures
                docToFeatureVector[docno] = documentFeaturesDict
    return docToFeatureVector


def backupDocuments(currentTime,baseDir):
    """
    Backup documents in archive and their LTR features.
    """
    client = MongoClient('asr2.iem.technion.ac.il',27017)
    db = client.asr16
    pathToFolder = baseDir +'Results/'
    FEATURES_DIR = pathToFolder + '/Features/' +  currentTime
    docToFeatureVector = parseFeatures(FEATURES_DIR)
    documents = db.documents.find({})
    for document in documents:
        document['text']= document.pop('current_document')
        document['id']= document.pop('_id')
        document['features'] = docToFeatureVector[document["query_id"]+"-"+document["username"]]
        del document['posted_document']
        document['iteration'] = currentTime
        db.archive.save(document)


if __name__=="__main__":
    baseDir = '/lv_local/home/zivvasilisky/ASR20/epoch_run/'
    # if not os.path.exists(baseDir):
    #     os.makedirs(baseDir)
    # changeStatus()
    # print('Status Changed!')
    # sys.stdout.flush()
    # trecFileName, workingSetFilename, currentTime = createTrecTextForCurrentDocuments(baseDir)
    # print('Files created!')
    # sys.stdout.flush()
    # asrIndex = buildIndex(trecFileName, currentTime, baseDir)
    # print('Index Built!')
    # sys.stdout.flush()
    # mergedIndex = mergeIndices(asrIndex, baseDir)
    # print('Index Merged!')
    # sys.stdout.flush()
    curr_static_model = 'static_model_lambdamart_round_4'
    curr_s_msmm_mg_model = 'static_msmm_mg_model_lambdamart_round_4'
    prev_rounds_list = ['2021-01-10-22-42-25-566245','2021-01-14-22-37-00-218428','2021-01-19-01-59-48-181622']
    mergedIndex = '/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/mergedindex'
    workingSetFilename = '/lv_local/home/zivvasilisky/ASR20/epoch_run/Collections/WorkingSets/2021-01-24-22-40-57-628006'
    currentTime = '2021-01-24-22-40-57-628006'
    rankedLists = runRankingModels(mergedIndex,workingSetFilename,currentTime,baseDir, curr_static_model, curr_s_msmm_mg_model, prev_rounds_list)
    print('Ranked docs!')
    rankedLists = baseDir + 'Results/RankedLists/LambdaMART' + currentTime
    sys.stdout.flush()
    updateScores((rankedLists,))
    print('Updated docs!')
    sys.stdout.flush()
    backupDocuments(currentTime,baseDir)
    changeStatus()
