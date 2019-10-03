import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description='Q4_regression')
parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=130, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--fc', action='store_true', default=True,
                    help='if True, use fully-connected neural network for training')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')


class FCNet(nn.Module):   
    ''' fully-connected neural network '''
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(2, 50, bias=True)
        self.fc2 = nn.Linear(50, 20, bias = True)
        self.fc3 = nn.Linear(20, 1, bias = True)
        
    def forward(self, x):
        x = x.view(-1, 2)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.squeeze(self.fc3(x))
        return x



class OrganizeData(Dataset):

    def __init__(self, dataset, transform=None):
        self.data = dataset['data']
        self.label = dataset['target']
        self.transform = transform

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)


def adam_optimizer(args, model):
    ''' define optimizer '''
    optimizer = optim.Adam(model.parameters())

    print('\n---Training Details---')
    print('batch size:', args.batch_size)
    print('seed number', args.seed)

    print('\n---Optimization Information---')
    print('optimizer: Adam')
    
    return optimizer

def get_model(args):
    ''' define model '''
    if args.fc == True:
        model = FCNet()
    
    return model




def train(model, optimizer, train_data, test_data, train_loss, test_loss, epoch):
    print("\n--- start training ---\n")
    ''' define training function '''
    training_loss = train_loss
    testing_loss = test_loss
    CLIP = 50.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = Variable(data.float()), Variable(target.float())
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, reduction='mean')
        loss.backward()
        #  scaling the gradients down by the same amount in order to reduce the norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        training_loss[batch_idx+(epoch-1)*len(train_data)] = loss.item()

        with torch.no_grad():
            for batch_idx_test, (data_test, target_test) in enumerate(test_data):
                data_test, target_test = Variable(data_test.float()), Variable(target_test.float())
                output = model(data_test)
                testing_loss[batch_idx+(epoch-1)*len(train_data)] += F.mse_loss(output, target_test, reduction='mean').item()
    
        if batch_idx % args.log_interval == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)] \ttrain_Loss: {:.6f} \ttest_loss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_data.sampler),
                    100.* (batch_idx+1) / len(train_data), training_loss[batch_idx+(epoch-1)*len(train_data)],
                 testing_loss[batch_idx+(epoch-1)*len(train_data)]))

    return training_loss, testing_loss

def test(model, test_data, train_data):
    
    model.eval()
    test_loss = 0
    train_loss = 0
    
    print("\n--- start testing ---\n")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = Variable(data.float()), Variable(target.float())
            output = model(data)
            train_loss += F.mse_loss(output, target, reduction='mean').item()
        for batch_idx, (data, target) in enumerate(test_data):
            data, target = Variable(data.float()), Variable(target.float())
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='mean').item()
            
    test_loss /= len(test_data)
    train_loss /= len(train_data)


    print('Test set: Average loss: {:.4f}\nTrain set: Average loss: {:.4f}'.format(
           test_loss,  train_loss))
    
            

if __name__ == "__main__":

    args = parser.parse_args()

    dataset = {}

    N = 5000
    x1 = np.random.uniform(-10, 10, N)
    x2 = np.random.uniform(-10, 10, N)
    x = np.array([x1.T, x2.T])
    y = x1 ** 2 + x1 * x2 + x2 ** 2

    dataset['data'] = x.T
    dataset['target'] = y
    data = OrganizeData(dataset)

    index = np.arange(len(y))
    train_index, test_index = train_test_split(index, test_size=0.1)

    train_index = SubsetRandomSampler(train_index)
    test_index = SubsetRandomSampler(test_index)

    train_data = DataLoader(data, batch_size=args.batch_size, drop_last=False, sampler=train_index)
    test_data = DataLoader(data, batch_size=len(test_index), drop_last=False, sampler=test_index)

    model = get_model(args)
    optimizer = adam_optimizer(args, model)

    train_loss = np.zeros(args.epochs*len(train_data)) 
    test_loss = np.zeros(args.epochs*len(train_data))

    for epoch in range(1, args.epochs + 1):
        train_loss, test_loss = train(model, optimizer, train_data, test_data, train_loss, test_loss, epoch)
        test(model, test_data, train_data)

    plot1 = plt.subplot(111, projection='3d')
    plot1.scatter(x[0], x[1], y, c='b')
    plt.show()

    loader_whole = DataLoader(data, batch_size=N, shuffle=False, drop_last=False)

    with torch.no_grad():
        for whole, (data, target) in enumerate(loader_whole):
            model_output = model(Variable(data).float())

    plot2 = plt.subplot(111, projection='3d')
    plot2.scatter(x[0], x[1], model_output, c='r')
    plt.show()
    
    train_time = np.arange(len(train_loss))
    plt.plot(train_time, train_loss, 'g', train_time, test_loss, 'y')
    plt.title("Train loss in green Vs. Test loss in yellow")
    plt.show()
