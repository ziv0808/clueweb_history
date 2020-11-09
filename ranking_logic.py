from pymongo import MongoClient
import os
import sys
import datetime
import xml.etree.ElementTree as ET
import csv
import subprocess




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
    pathToTrecText = pathToFolder+"/TrecText/"
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
            f.write(document['current_document'].encode('cp1252', "ignore").decode('utf-8', 'ignore').rstrip())
            f.write('\n</TEXT>\n')
            f.write('</DOC>\n')
    f.close()
    pathToWorkingSet = pathToFolder+ '/WorkingSets/'
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



def runRankingModels(mergedIndex, workingSet, currentTime, baseDir):
    scriptDir = '/lv_local/home/zivvasilisky/ziv/content_modification_code/scripts/'
    pathToFolder = baseDir + 'Results/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)
    INDEX = mergedIndex
    WORKING_SET_FILE = workingSet
    MODEL_DIR = baseDir+"content_modification_code/rank_models/"
    MODEL_FILE = MODEL_DIR+"model_lambdatamart"
    QUERIES_FILE = baseDir+'Data/QueriesFile.xml'
    FEATURES_DIR = pathToFolder + '/Features/' +  currentTime
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
    command = '~/jdk1.8.0_181/bin/java -jar '+scriptDir+'RankLib.jar -load ' + MODEL_FILE + ' -rank '+FEATURES_FILE+' -score predictions.tmp'
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
    command='perl '+scriptDir+'order.pl ' + RANKED_LIST_DIR+ '/LambdaMART' + currentTime + ' ' +FEATURES_FILE + ' ' + PREDICTIONS_FILE
    print(command)
    out=run_bash_command(command)
    print(out)
    return RANKED_LIST_DIR+ '/LambdaMART' + currentTime









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
    if not os.path.exists(baseDir):
        os.makedirs(baseDir)
    # changeStatus()
    # print('Status Changed!')
    # sys.stdout.flush()
    trecFileName, workingSetFilename, currentTime = createTrecTextForCurrentDocuments(baseDir)
    print('Files created!')
    sys.stdout.flush()
    asrIndex = buildIndex(trecFileName, currentTime, baseDir)
    print('Index Built!')
    sys.stdout.flush()
    mergedIndex = mergeIndices(asrIndex, baseDir)
    print('Index Merged!')
    sys.stdout.flush()
    rankedLists = runRankingModels(mergedIndex,workingSetFilename,currentTime,baseDir)
    print('Ranked docs!')
    sys.stdout.flush()
    updateScores((rankedLists,))
    print('Updated docs!')
    sys.stdout.flush()
    backupDocuments(currentTime,baseDir)
    changeStatus()
