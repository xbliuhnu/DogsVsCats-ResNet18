from dataset import CatsVSDogsDataset as CVDD
from torch.utils.data import DataLoader as DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models

dataset_dir = './data/'
model_cp = './checkpoint/'
workers = 10
batch_size = 16
lr = 0.0001
nepoch = 10


def train():
    datafile = CVDD('train', dataset_dir)
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers)

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    model = models.resnet18(num_classes=2)
    model = model.cuda()
    model = nn.DataParallel(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for epoch in range(nepoch):
        for img, label in dataloader:
            img, label = Variable(img).cuda(), Variable(label).cuda()
            out = model(img)
            loss = criterion(out, label.squeeze())      # the parameter 'target' must be a 1D Tensor
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1

            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt * batch_size, loss / batch_size))

    torch.save(model.state_dict(), '{0}/model.pth'.format(model_cp))


if __name__ == '__main__':
    train()










