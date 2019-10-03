import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, BatchSampler
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from torch.optim import lr_scheduler


parser = argparse.ArgumentParser(description='Q5')
parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                    help='SGD momentum (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--cn', action='store_true', default=True,
                    help='if True, use convolutional neural network for training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


class CNN_Net(nn.Module):

    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 16, 5, 1)
        self.fc1 = nn.Linear(16 * 4 * 4, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x



def get_model(args):
    ''' define model '''
    if args.cn == True:
        model = CNN_Net()

    return model


def adam_optimizer(args, model):
    ''' use adam as optimizer '''
    optimizer = optim.Adam(model.parameters())

    print('\n---Training Details---')
    print('batch size:', args.batch_size)
    print('seed number', args.seed)
    print('\n---Optimization Information---')
    print('optimizer: Adam')

    return optimizer



class Two_MNIST(Dataset):
    """
        Train: For each sample creates randomly a positive or a negative pair
        Test: Creates fixed pairs for testing
        """

    def __init__(self, mnist_data):
        self.mnist_data = mnist_data
        self.train = mnist_data.train
        self.transform = self.mnist_data.transform

        if self.train:
            self.train_labels = self.mnist_data.train_labels
            self.train_data = self.mnist_data.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:

            self.test_labels = self.mnist_data.test_labels
            self.test_data = self.mnist_data.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(12)

            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.test_labels[i].item()]), 1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i, random_state.choice(
                self.label_to_indices[np.random.choice(list(self.labels_set - set([self.test_labels[i].item()])))]), 0]
                              for i in range(1, len(self.test_data), 2)]

            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_data)

def get_data_loader():
    train_data = datasets.MNIST('../data/MNIST', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    test_data = datasets.MNIST('../data/MNIST', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

    index_train = np.arange(len(train_data))
    index_test = np.arange(len(test_data))

    # Split 10% of the MNIST dataset
    t_train, train_index = train_test_split(index_train, test_size=0.1)
    t_test, test_index = train_test_split(index_test, test_size=0.1)

    train_index = SubsetRandomSampler(train_index)
    test_index = SubsetRandomSampler(test_index)

    train_sampler = Two_MNIST(train_data)
    test_sampler = Two_MNIST(test_data)

    train_loader = DataLoader(train_sampler, batch_size=args.batch_size, drop_last=True, sampler=train_index)
    test_loader = DataLoader(test_sampler, batch_size=len(test_index), drop_last=True, sampler=test_index)

    return train_loader, test_loader


def train(args, model, optimizer, train_loader, epoch):
    ''' define training function '''
    model.train()

    print("\n--- training ---\n")

    for batch_idx, ((img1, img2), target) in enumerate(train_loader):
        # Concatenate two images to train
        data, label = torch.tensor(np.concatenate((img1, img2), axis=1)), target
        data, label = Variable(data), Variable(label).float()
        optimizer.zero_grad()
        output = model(data)
        output = torch.squeeze(output)
        loss = F.binary_cross_entropy(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain_Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, test_loader, trainging_loss, testing_loss, epoch):
    ''' define testing function '''
    model.eval()
    test_loss = 0
    train_loss = 0
    test_correct = 0
    train_correct = 0

    with torch.no_grad():
        for batch_idx, ((img1, img2), target) in enumerate(test_loader):
            data, label = torch.tensor(np.concatenate((img1, img2), axis=1)), target
            data, label = Variable(data), Variable(label).float()
            output = torch.squeeze(model(data))
            test_loss += F.binary_cross_entropy(output, label, reduction='sum').item()
            pred = torch.round(output)
            test_correct += pred.eq(label.view_as(pred)).sum().item()

        for b_idx, ((i1, i2), t) in enumerate(train_loader):
            data, label = torch.tensor(np.concatenate((i1, i2), axis=1)), t
            data, label = Variable(data), Variable(label).float()
            output = torch.squeeze(model(data))
            train_loss += F.binary_cross_entropy(output, label, reduction='sum').item()
            predi = torch.round(output)
            train_correct += predi.eq(label.view_as(predi)).sum().item()

    train_loss /= len(train_loader.sampler)
    test_loss /= len(test_loader.sampler)

    trainging_loss[epoch - 1] = train_loss
    testing_loss[epoch - 1] = test_loss

    print("\n--- testing ---\n")
    print('\nTest set: Average loss: {:.4f},\t Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(test_loader.sampler),
        100. * test_correct / len(test_loader.sampler)))

    return train_loss, test_loss


if __name__ == "__main__":
    args = parser.parse_args()  # load options

    train_loader, test_loader = get_data_loader()

    model = get_model(args)
    # use adam as optimizer
    optimizer = adam_optimizer(args, model)

    trainging_loss = np.zeros(args.epochs)
    testing_loss = np.zeros(args.epochs)

    for i in range(1, args.epochs + 1):
        train(args, model, optimizer, train_loader, i)
        test(args, model, test_loader, trainging_loss, testing_loss, i)

    train_time = np.arange(len(trainging_loss))

    plt.plot(train_time, trainging_loss, 'g', train_time, testing_loss, 'y')
    plt.title("Train loss in green Vs. Test loss in yellow")
    plt.show()



