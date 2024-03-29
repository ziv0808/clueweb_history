import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from random import shuffle
import pandas as pd


train_data_path = '/lv_local/home/zivvasilisky/dataset/processed_queries/tsv_files/'
test_data_path = '/lv_local/home/zivvasilisky/dataset/processed_queries/test_tsv_files_fixed/'
res_path = '/lv_local/home/zivvasilisky/dataset/nn_res/'

# hidden_size = 64
learning_rate = 0.000001
num_epochs = 10
features_num = 769

train_q_file_list = []
for filename in os.listdir(train_data_path):
    train_q_file_list.append(filename)

test_file_list =[]
for filename in os.listdir(test_data_path):
    test_file_list.append(filename)


class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, num_classes)
        # self.logsoftmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        out = self.dropout(x)
        out = self.fc1(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.logsoftmax(out)
        return out

net = NeuralNet(features_num, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

feature_cols =  ['BM25'] + ['CLS_'+ str(i) for i in range(1,769)]
for i in range(num_epochs):
    print("Epoch: " + str(i + 1))
    sys.stdout.flush()
    train_files = 0
    shuffle(train_q_file_list)
    for train_filename in train_q_file_list:
        df = pd.read_csv(train_data_path + train_filename, sep = '\t', index_col = False)
        # df[feature_cols] = df[feature_cols].applymap(lambda x: float(x))
        X = Variable(torch.from_numpy(df[feature_cols].values).float())
        Y = Variable(torch.from_numpy(df['Relevance'].values))

        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        train_files += 1
        if train_files % 10000 == 0:
            print("[" + str(train_files) + " : " + str(len(train_q_file_list)) + "] train file processed Loss " + str(loss.data))
            sys.stdout.flush()

    correct = 0
    total = 0
    total_rel = 0
    total_non_rel = 0
    total_rel_corr = 0
    total_non_rel_corr = 0
    for test_file in test_file_list:
        try:
            df = pd.read_csv(test_data_path + test_file, sep='\t', index_col=False)
        except Exception as e:
            print(test_file)
            print(str(e))
            sys.stdout.flush()
            continue
        # df[feature_cols] = df[feature_cols].applymap(lambda x: float(x))
        X = Variable(torch.from_numpy(df[feature_cols].values).float())
        labels = torch.from_numpy(df['Relevance'].values)
        outputs = net(X)
        proba = pd.np.array(torch.softmax(outputs.data, dim=1).tolist())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        num_rel = (labels == 1).sum()
        total_rel += num_rel
        total_non_rel += ((labels.size(0)) - num_rel)
        correct += (predicted == labels).sum()
        for j in range(labels.size(0)):
            if labels[j] == 1 and predicted[j] == labels[j]:
                total_rel_corr += 1
            if labels[j] == 0 and predicted[j] == labels[j]:
                total_non_rel_corr += 1
        df = df[['Docno', 'Relevance']]
        df['NonRelProba'] = list(proba[:,0])
        df['RelProba'] = list(proba[:,1])
        df.to_csv(res_path + 'Epoch_' + str(i) + '_' + test_file, sep ='\t', index = False)

    print('Accuracy of the model on the test queries: %f %%' % (100 * float(correct) / total))
    print('Accuracy of the model on relevant docs: %f %%' % (100 * float(total_rel_corr) / total_rel))
    print('Accuracy of the model on non-relevant docs: %f %%' % (100 * float(total_non_rel_corr) / total_non_rel))
    sys.stdout.flush()