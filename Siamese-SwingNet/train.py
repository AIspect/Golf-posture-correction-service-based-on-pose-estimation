from dataloader import GolfDB, Normalize, ToTensor
from model import EventDetector, SiameseNetwork
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import random

class Dataset(object):
    
    def __init__(self, x0, x1, label):
        self.size = label.shape[0]
        self.x0 = torch.from_numpy(x0)
        self.x1 = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return (self.x0[index],
                self.x1[index],
                self.label[index])

    def __len__(self):
        return self.size


def create_pairs(data, digit_indices):
    x0_data = []
    x1_data = []
    label = []

    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        # make n pairs with each number
        for i in range(n):
            # make pairs of the same class
            # label is 1
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            # scale data to 0-1
            # XXX this does ToTensor also
            x0_data.append(data[z1] / 255.0)
            x1_data.append(data[z2] / 255.0)
            label.append(1)

            # make pairs of different classes
            # since the minimum value is 1, it is not the same class
            # label is 0
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # scale data to 0-1
            # XXX this does ToTensor also
            x0_data.append(data[z1] / 255.0)
            x1_data.append(data[z2] / 255.0)
            label.append(0)

    x0_data = np.array(x0_data, dtype=np.float32)
    x0_data = x0_data.reshape([-1, 1, 28, 28])
    x1_data = np.array(x1_data, dtype=np.float32)
    x1_data = x1_data.reshape([-1, 1, 28, 28])
    label = np.array(label, dtype=np.int32)
    return x0_data, x1_data, label


def create_iterator(data, label, batchsize, shuffle=False):
    digit_indices = [np.where(label == i)[0] for i in range(10)]
    x0, x1, label = create_pairs(data, digit_indices)
    ret = Dataset(x0, x1, label)
    return ret

if __name__ == '__main__':

    # training configuration
    split = 1
    iterations = 2000
    it_save = 100  # save model every 100 iterations
    n_cpu = 6
    seq_length = 64
    bs = 22  # batch size
    k = 10  # frozen layers

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    model.train()
    model.cuda()

    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=True)

    train_iter = create_iterator(
        dataset.train_data.numpy(),
        dataset.train_labels.numpy(),
        bs)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    while i < iterations:
        for sample in data_loader:
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            logits = model(images)
            labels = labels.view(bs*seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/swingnet_{}.pth.tar'.format(i))
            if i == iterations:
                break



