import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('image', cmap='gray')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.dropout2d1 = nn.Dropout2d(0.6)
        self.dropout2d2 = nn.Dropout2d(0.6)
        self.fc1 = nn.Linear(1176, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout1 = nn.Dropout(0.9)
        self.dropout2 = nn.Dropout(0.8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d1(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch, verbose=True):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if (batch_idx+1) % (len(train_loader)//args.log_numbers) == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, name='Test', verbose=True):
    model.eval()    # Set the model to inference mode
    test_loss = 0.
    correct = 0.
    preds = np.zeros((0,), dtype=np.int)
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            preds = np.hstack((preds, pred.cpu().numpy().flatten()))
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.sampler)
    
    if verbose:
        print('{:s} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            name, test_loss, correct, len(test_loader.sampler),
            100. * correct / len(test_loader.sampler)))
    
    return test_loss, correct, preds

def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-percentage', type=float, default=15., metavar='P',
                        help='percentage of training data used for validation (default: 15)')
    parser.add_argument('--training-division', type=float, default=1., metavar='D',
                        help='divide the remaining training data by this factor')
    parser.add_argument('--epochs', type=int, default=12, metavar='N',
                        help='number of epochs to train (default: 12)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=1, metavar='M',
                        help='Learning rate step gamma (default: 1)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='disables data augmentation')
    parser.add_argument('--seed', type=int, default=2020, metavar='S',
                        help='random seed (default: 2020)')
    parser.add_argument('--log-numbers', type=int, default=1, metavar='N',
                        help='how many entries of logging training status to show per epoch')
    parser.add_argument('--name', type=str, default='default', metavar='name',
                        help='name of the model')
    parser.add_argument('--root', type=str, default='../data/hw03_outputs/', metavar='path',
                        help='path to save all model and plots')
    parser.add_argument('--plot', action='store_true',
                        help='plot the training curve')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate your model on the official test set')
    parser.add_argument('--save-model', action='store_true',
                        help='save the current model');    
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        path_model = args.root+args.name+'.pt'
        
        assert os.path.exists(path_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(path_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        test_loss, correct, preds = test(model, device, test_loader)
        
        np.save(args.root+args.name+'_test_loss.npy', test_loss)
        np.save(args.root+args.name+'_test_accuracy.npy', correct/len(test_loader.sampler)*100)
        np.save(args.root+args.name+'_preds.npy', preds)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
        transform=transforms.Compose([       # Data preprocessing
            transforms.ToTensor(),           # Add data augmentation here
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    train_dataset_augmented = datasets.MNIST('../data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomAffine(4, translate=(.1, .1), scale=(.9, 1.1), shear=(2, 2, 2, 2)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    
    train_labels = np.array([data[1] for data in train_dataset])
    labels = np.unique(train_labels)

    rng = np.random.default_rng(args.seed)
    train_label_idc = [rng.permutation(np.argwhere(train_labels==l)) for l in labels]

    subset_indices_train = [idx[0] for idc in train_label_idc for idx in idc[:np.round(len(idc)*(1-args.validation_percentage/100)/args.training_division).astype(int)]]
    subset_indices_valid = [idx[0] for idc in train_label_idc for idx in idc[np.round(len(idc)*(1-args.validation_percentage/100)).astype(int):]]
    
    if args.no_augmentation:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(subset_indices_train)
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset_augmented, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(subset_indices_train)
        )
    
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    train_loss = np.zeros((args.epochs,))
    val_loss = np.zeros((args.epochs,))
    train_correct = np.zeros((args.epochs,))
    val_correct = np.zeros((args.epochs,))
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        train_loss[epoch], train_correct[epoch] = test(model, device, train_loader, name='Training')
        val_loss[epoch], val_correct[epoch] = test(model, device, val_loader, name='Validation')
        print()
        scheduler.step()    # learning rate scheduler

    np.save(args.root+args.name+'_train_loss.npy', train_loss)
    np.save(args.root+args.name+'_val_loss.npy', val_loss)
    np.save(args.root+args.name+'_train_accuracy.npy', train_correct/len(train_loader.sampler)*100)
    np.save(args.root+args.name+'_train_accuracy.npy', val_correct/len(val_loader.sampler)*100)

    if args.save_model:
        torch.save(model.state_dict(), args.root+args.name+'.pt')
    
    if args.plot:
        fig = plt.figure(figsize=(8, 6), tight_layout=True)
        ax1 = plt.axes()
        ax1.plot(np.arange(args.epochs), train_loss, 'b-', label='Training Loss')
        ax1.plot(np.arange(args.epochs), val_loss, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epochs', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Negative Log Likelihood Loss', fontsize=14, fontweight='bold')
        ax2 = ax1.twinx()
        ax2.plot(np.arange(args.epochs), train_correct/len(train_loader.sampler)*100, 'b:', label='Training Accuracy')
        ax2.plot(np.arange(args.epochs), val_correct/len(val_loader.sampler)*100, 'r:', label='Validation Accuracy')
        # ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy %', fontsize=14, fontweight='bold')
        lines1, line_labels1 = ax1.get_legend_handles_labels()
        lines2, line_labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, line_labels1 + line_labels2, loc='right', fontsize=12)
        plt.savefig(args.root+args.name+'.pdf', pad_inches=0, bbox_inches='tight')
        plt.show()
        
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
