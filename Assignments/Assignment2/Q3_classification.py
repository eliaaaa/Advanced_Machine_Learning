import torch
import torchfile
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable



parser = argparse.ArgumentParser(description='Q3')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
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
        self.fc1 = nn.Linear(2, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 4)
        

    def forward(self, x):
        x = x.view(-1, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))    
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class organizeData(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = dataset.data
        self.label = dataset.label
        self.transform = transform

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

def get_data_loader():
    ''' define training and testing data loader'''
    # load trainig data loader
    
    dataset = torchfile.load("dataset.t7b")
    dataset.label -= 1    # transform label to index
    data = organizeData(dataset)
    permute_index = torch.randperm(len(data.label))  #shuffle
    train_index, test_index = train_test_split(np.array(permute_index), test_size = 0.1)
    train_index = torch.utils.data.sampler.SubsetRandomSampler(train_index)
    test_index = torch.utils.data.sampler.SubsetRandomSampler(test_index) # split training and testing set
                                               
    train_loader = DataLoader(data, batch_size = args.batch_size, drop_last = False, sampler = train_index)
    test_loader = DataLoader(data, batch_size = len(test_index), drop_last = False, sampler = test_index)
    return train_loader, test_loader

def get_model(args):
    ''' define model '''
    if args.fc == True:
        model = FCNet()
    
    return model


def get_optimizer(args, model):
    ''' define optimizer '''
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print('\n---Training Details---')
    print('batch size:',args.batch_size)
    print('seed number', args.seed)

    print('\n---Optimization Information---')
    print('optimizer: SGD')
    print('lr:', args.lr)
    print('momentum:', args.momentum)
    
    return optimizer


def train(model, optimizer, training_data, testing_data, train_loss, test_loss, epoch):
    print("\n--- training ---\n")
    training_loss = train_loss
    testing_loss = test_loss
    model.train()
    for batch_idx, (data, label) in enumerate(training_data):
        data, target = Variable(data), Variable(label.long())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        training_loss[batch_idx+(epoch-1)*len(training_data)] = loss.item()
        
        with torch.no_grad():
            for batch_idx_test, (data_test, label_test) in enumerate(testing_data):
                data_test, label_test = Variable(data_test), Variable(label_test.long())
                output = model(data_test)
                testing_loss[batch_idx+(epoch-1)*len(training_data)] += F.nll_loss(output, label_test, size_average= False).item()
        
        testing_loss[batch_idx+(epoch-1)*len(training_data)] /= len(testing_data.sampler)

        if batch_idx % args.log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\ttrain_Loss: {:.6f} \t test_loss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(training_data.sampler),
                100.*(batch_idx) / len(train_data), loss.item(),
                 testing_loss[batch_idx+(epoch-1)*len(training_data)]))
    
    return training_loss, testing_loss


def test(model, test_data, train_data):
    print("\n--- testing ---\n")
    model.eval()
    test_loss = 0
    test_correct = 0
    train_loss = 0
    train_correct = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_data):
            data, target = Variable(data), Variable(label.long())
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average= False).item()
            pred = output.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        for batch_idx, (data, label) in enumerate(train_data):
            data_train, target_train = Variable(data), Variable(label.long())
            output_train = model(data_train)
            train_loss += F.nll_loss(output_train, target_train, size_average=False).item()
            pred_train = output_train.data.max(1, keepdim=True)[1]
            train_correct += pred_train.eq(target_train.data.view_as(pred_train)).cpu().sum()

    train_loss /= len(train_data.sampler)
    test_loss /= len(test_data.sampler)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           test_loss, test_correct, len(test_data.sampler),
           100. * test_correct / len(test_data.sampler), train_loss, train_correct, len(train_data.sampler),
           100. * train_correct / len(train_data.sampler)))
    

if __name__ == "__main__":
    args = parser.parse_args()  
    model = get_model(args)
    optimizer = get_optimizer(args, model)
    train_data, test_data = get_data_loader()
    train_loss = np.zeros(args.epochs*len(train_data)) 
    test_loss = np.zeros(args.epochs*len(train_data))

    for epoch in range(0, args.epochs):
        train_loss, test_loss = train(model, optimizer, train_data, test_data, train_loss, test_loss, epoch+1)
        test(model, test_data, train_data)  
    
    train_time = np.arange(len(train_loss))

    plt.plot(train_time, train_loss, 'g', train_time, test_loss, 'y')
    plt.title("Train loss in green Vs. Test loss in yellow")
    plt.show()

