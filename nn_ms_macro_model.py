import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd


train_data_path = '/lv_local/home/zivvasilisky/dataset/processed_queries/tsv_files/'
test_data_path = '/lv_local/home/zivvasilisky/dataset/processed_queries/test_tsv_files_fixed/'
res_path = '/lv_local/home/zivvasilisky/dataset/nn_res/'

hidden_size = 64
learning_rate = 0.001
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
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.logsoftmax = nn.LogSoftmax()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out

net = NeuralNet(features_num, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=5e-4)

feature_cols =  ['BM25'] + ['CLS_'+ str(i) for i in range(1,769)]
for i in range(num_epochs):
    print("Epoch: " + str(i + 1))
    sys.stdout.flush()
    train_files = 0
    for train_filename in train_q_file_list[:12]:
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
        if train_files % 10 == 0:
            print("[" + str(train_files) + " : " + str(len(train_q_file_list)) + "] train file processed ")
            sys.stdout.flush()

    correct = 0
    total = 0
    for test_file in test_file_list[:10]:
        df = pd.read_csv(test_data_path + test_file, sep='\t', index_col=False)
        # df[feature_cols] = df[feature_cols].applymap(lambda x: float(x))
        X = Variable(torch.from_numpy(df[feature_cols].values).float())
        labels = torch.from_numpy(df['Relevance'].values)
        outputs = net(X)
        proba = pd.np.array(torch.softmax(outputs.data, dim=1).tolist())
        print(test_file)
        print(proba)
        # print(proba[:,0])
        print(proba[:,1])

        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        sys.stdout.flush()
        total += labels.size(0)
        correct += (predicted == labels).sum()
        df = df[['Docno', 'Relevance']]
        df['NonRelProba'] = list(proba[:,0])
        df['RelProba'] = list(proba[:,1])
        df.to_csv(res_path + 'Epoch_' + str(i) + test_file, sep ='\t', index = False)

    print('Accuracy of the model on the test queries: %f %%' % (100 * float(correct) / total))
    sys.stdout.flush()