import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
from torch.utils.data import DataLoader
import os
import random as rn
from tqdm import tqdm
import pandas as pd
from torcheval.metrics.functional import multiclass_f1_score

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class config:
    weights = torch.tensor([1, 1, 1, 1], dtype=torch.double)


# ------------------------------------------------ Define Dataset ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def game_image(frame):
    state = frame[-4:]
    state[1], state[2] = state[1]/720, state[2]/24
    state = torch.FloatTensor(state)

    f = np.zeros((9, 50, 94))

    bx, by = min(int(frame[0]), 93), min(int(frame[1]), 49)
    f[0, by, bx], f[1, by, bx], f[2, by, bx] = 1, frame[4], frame[5]

    for j in range(10):
        px, py = min(int(frame[6 + 4*j]), 93), min(int(frame[7 + 4*j]), 49)
        t = int(j >= 5) * 3
        f[3+t, py, px], f[4+t, py, px], f[5+t, py, px] = 1, frame[8 + 4*j], frame[9 + 4*j]

    channels = torch.from_numpy(f)

    return channels, state


class BBDataset(torch.utils.data.Dataset):
    def __init__(self, frames, transforms, class_labels, point_labels):
        self.frames = frames
        self.transforms = transforms
        self.class_labels = class_labels
        self.point_labels = point_labels

    def __getitem__(self, idx):
        # load frame
        frame = self.frames[idx]

        channels, state = self.transforms(frame)

        return channels, state, self.class_labels[idx], self.point_labels[idx]

    def __len__(self):
        return len(self.frames)


def get_data(frames, game_image, yc, yn):
    train_size = int(0.8 * len(frames))
    test_size = len(frames) - train_size

    data = BBDataset(frames, game_image, yc, yn)

    # train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

    # train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
    dataloader = DataLoader(data, batch_size=128, shuffle=True)

    # return train_dataloader, test_dataloader
    return dataloader

# ----------------------------------------------- Define C Blocks ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class convBlock(nn.Module):
    def __init__(self):
        super(convBlock, self).__init__()

        # Define the layers of the network
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lrelu = torch.nn.LeakyReLU()
        self.batchNorm = nn.BatchNorm2d(64)

    def forward(self, x):
        # Define the forward pass of the network
        x1 = self.lrelu(self.conv1(x))
        x1 = self.batchNorm(x1)
        x1 = self.lrelu(self.conv2(x1))
        x1 = x1 + x
        x1 = self.batchNorm(x1)

        return x1


# ------------------------------------------------ Define Network ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

class BBNet(nn.Module):
    def __init__(self):
        super(BBNet, self).__init__()

        # Define the layers of the network
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=7, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * 4 * 10, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(68, 5)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x, s):
        # Define the forward pass of the network

        x = self.lrelu(self.conv1(x))
        x = self.pool(x)
        x = self.lrelu(self.conv2(x))
        x = self.pool(x)
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.fc1(x))
        x = self.dropout1(x)
        x = self.lrelu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.cat((x, s), dim=1)
        x = self.fc3(x)

        # x[:, :4] = self.sigmoid(x[:, :4])
        x[:, 4] = torch.tanh(x[:, 4])
        x[:, 4] = torch.mul(x[:, 4], 3)
        return x


# ------------------------------------------------- Read in Data -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def read_file(fpath):
    with open(fpath) as f:
        freader = csv.reader(f, delimiter = ',')
        frames = []
        yc = []
        yn = []
        for row in freader:
            if row[-4] == '':
                row[-4] = row[-5]

            r = [float(e) for e in row[:-2]]
            frames.append(r[:6] + r[7:11] + r[12:16] + r[17:21] + r[22:26] + r[27:31] + r[32:36] + r[37:41] + r[42:46] + r[47:51] + r[52:])
            yc.append(int(row[-2]))
            yn.append(int(row[-1]))

    return frames, yc, yn

# Can use torch.div() for normalising in future
# Need Dataset object + Dataloader + sampler most likely

# ---------------------------------------------- Define Loss + Model ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# print(frames[-1][-2], len(frames[-1]))

# Define the training data and targets
# x_train = torch.FloatTensor(frames)


# Define the neural network
net = BBNet().to(device)


# Define the loss function as a weighted sum of the two objective functions
def loss_fn(y_pred, yc_true, yn_true):
    # Define the loss function and optimizer
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss(weight=config.weights)
    loss1 = criterion1(y_pred[:, 4], yn_true)
    loss2 = criterion2(y_pred[:, :4].view(-1, 4), yc_true.type(torch.LongTensor))
    # alpha = 0.75
    # loss = alpha * loss1 + (1 - alpha) * loss2
    loss = loss1 + loss2
    return loss


# criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.00075)

# ------------------------------------------------- Train Model --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def train(dataloader, optimizer, loss_fn, net, device):
    net.double()
    net.train()
    step = 0
    running_loss = 0
    r_acc = 0
    rf1 = 0
    stuffer = 0
    f1 = 0

    # Loop over the training data and perform back-propagation
    for channels, state, class_labels, point_labels in tqdm(dataloader):
        # Forward pass
        channels = channels.to(device)
        state = state.to(device)

        y_pred = net(channels.double(), state.double())

        # Compute the loss
        # loss = criterion(y_pred, y_train)
        loss = loss_fn(y_pred, class_labels.double(), point_labels.double())

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        # optimizer.step()
        # Update the parameters
        optimizer.step()

        _, p = torch.max(y_pred[:, :4].view(-1, 4), 1)
        acc = (p == class_labels).sum().item()/len(p)
        r_acc += acc

        try:
            f1 = multiclass_f1_score(y_pred[:, :4].view(-1, 4), class_labels.long(), num_classes=4, average="weighted")
            rf1 += f1
        except IndexError:
            stuffer += 1

        running_loss += loss.item()

        # Print the loss every 100 epochs
        # if step % 100 == 0:
        #     print('Step {}, Loss: {:.4f}'.format(step, loss.item()))

        if step == len(dataloader)-1:
            print('Train Step {}, Loss: {:.4f}, Acc: {:.4f}, F1:{:.4f}'.format(step, loss.item(), acc, f1))
            print(sum(y_pred[:, 4])/len(y_pred[:, 4]))
            print(max(y_pred[:, 0]), max(y_pred[:, 1]), max(y_pred[:, 2]), max(y_pred[:, 3]))
            print(sum(point_labels[:]) / len(point_labels))
            print(y_pred[1, :])

        step += 1

    return running_loss/len(dataloader), r_acc/len(dataloader), rf1/(len(dataloader)-stuffer)


def valid(dataloader, loss_fn, net, epoch, device):
    net.double()
    net.eval()
    running_loss = 0
    step = 0
    r_acc = 0
    rf1 = 0
    stuffer = 0
    f1 = 0

    # Loop over the training data and perform back-propagation
    for channels, state, class_labels, point_labels in tqdm(dataloader):
        # Forward pass
        channels = channels.to(device)
        state = state.to(device)

        y_pred = net(channels.double(), state.double())

        # Compute the loss
        # loss = criterion(y_pred, y_train)
        loss = loss_fn(y_pred, class_labels.double(), point_labels.double())

        running_loss += loss.item()

        _, p = torch.max(y_pred[:, :4], 1)
        acc = (p == class_labels).sum().item() / len(p)
        r_acc += acc

        try:
            f1 = multiclass_f1_score(y_pred[:, :4].view(-1, 4), class_labels.long(), num_classes=4, average="weighted")
            rf1 += f1
        except IndexError:
            stuffer += 1

        if step == len(dataloader)-1:
            print('Valid epoch {}, Loss: {:.4f}, Acc: {:.4f}, F1:{:.4f}'.format(step, loss.item(), acc, f1))
            print(sum(y_pred[:, 4])/len(y_pred[:, 4]))
            print(max(y_pred[:, 0]), max(y_pred[:, 1]), max(y_pred[:, 2]), max(y_pred[:, 3]))
            print(sum(point_labels[:]) / len(point_labels))
            print(y_pred[1, :])

        step += 1

    return running_loss/len(dataloader), r_acc/len(dataloader), rf1/(len(dataloader)-stuffer)


# ------------------------------------------------ Build Dataset -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def build_dataset(directory, files, device):
    dSet = []
    yc = []
    yn = []
    # breaker = 0
    classes = [0, 0, 0, 0]
    for file in files:
        f = os.path.join(directory, file)
        # checking if it is a file
        if os.path.isfile(f):
            frames, cl, nl = read_file(f)
            n = int(len(frames)/125)
            rframes = rn.choices(range(len(frames)), k=n)

            for i in rframes:
                fs = frames[i]
                dSet.append(fs)
                yc.append(cl[i])
                yn.append(nl[i])
                classes[cl[i]] += 1

        # if breaker == 2:
        #     break
        # breaker += 1

    # Converting class (yc) and points (yn) labels to tensors
    yn = torch.FloatTensor(yn)
    yc = torch.FloatTensor(yc)

    # print(classes)
    bincount = np.array(classes)
    n_samples = sum(bincount)
    # print(n_samples)
    weights = n_samples / (4 * bincount)
    config.weights = torch.tensor(weights, dtype=torch.double)
    config.weights = config.weights.to(device)
    # print(config.weights)

    return dSet, yc, yn


# --------------------------------------------------- Testing ----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

no_files = 629
directory = '/home/jpg23/Documents/BB NN project/BB Data Cleaned'

trainset = []
validset = []

randfiles = rn.choices(range(no_files), k=int(no_files*0.2))

for i, filename in enumerate(os.listdir(directory)):
    if i in randfiles:
        validset.append(filename)
    else:
        trainset.append(filename)

print(len(trainset), len(validset))

vdf = pd.DataFrame(validset)
vdf.to_csv(f'valid_files.csv')

maxepochs = 31
t_loss = []
v_loss = []
t_acc = []
v_acc = []
tf1 = []
vf1 = []

for e in range(maxepochs):
    vframes, vyc, vyn = build_dataset(directory, validset, device)
    tframes, tyc, tyn = build_dataset(directory, trainset, device)

    train_dataloader = get_data(tframes, game_image, tyc, tyn)
    test_dataloader = get_data(vframes, game_image, vyc, vyn)

    t, ta, tf = train(train_dataloader, optimizer, loss_fn, net, device)
    v, va, vf = valid(test_dataloader, loss_fn, net, e, device)

    t_loss.append(t)
    v_loss.append(v)

    t_acc.append(ta)
    v_acc.append(va)

    tf1.append(tf.item())
    vf1.append(vf.item())

    if e % 10 == 0:
        checkpoint = {'model': net, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                      'loss': v}
        torch.save(checkpoint, f'./model_{e}.bin')
        print(' Average Training Loss: {:.4f}, Average Valid Loss: {:.4f}'.format(t, v))

print(t_loss)
print(v_loss)

print(t_acc)
print(v_acc)

print(tf1)
print(vf1)

def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.train()
    return model





# f = frames[65000]
# ch, st = game_image(f)

# img = ch[3, :, :]

# img = img.reshape(50, 94, 1)
# plt.imshow(img.numpy())
# plt.show()



