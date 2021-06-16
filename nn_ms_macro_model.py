import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd


train_data_path = '/lv_local/home/zivvasilisky/dataset/processed_queries/tsv_files/'
test_data_path = '/lv_local/home/zivvasilisky/dataset/processed_queries/test_tsv_files_fixed/'


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
    for train_filename in train_q_file_list[:10]:
        print(train_filename)
        sys.stdout.flush()
        df = pd.read_csv(train_data_path + train_filename, sep = '\t', index_col = False)
        df[feature_cols] = df[feature_cols].apply(lambda x: float(x))
        X = Variable(torch.from_numpy(df[feature_cols].values))
        Y = Variable(torch.from_numpy(df['Relevance'].values))

        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    for test_file in test_file_list[:10]:
        df = pd.read_csv(test_data_path + test_file, sep='\t', index_col=False)
        df[feature_cols] = df[feature_cols].apply(lambda x: float(x))
        X = Variable(torch.from_numpy(df[feature_cols].values))
        labels = torch.from_numpy(df['Relevance'].values)
        outputs = net(X)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the model on the test queries: %f %%' % (100 * float(correct) / total))
    sys.stdout.flush()