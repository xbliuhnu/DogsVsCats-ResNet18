import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import dataset

np.set_printoptions(suppress=True)

test_data_dir = './data/test/'
model_file = './checkpoint/model.pth'

N = 10


def test():

    # setting model
    model = models.resnet18(num_classes=2)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # get data
    files = random.sample(os.listdir(test_data_dir), N)
    imgs = []           # img
    imgs_data = []      # img data
    for file in files:
        img = Image.open(test_data_dir + file)
        img_data = dataset.dataTransform(img)

        imgs.append(img)
        imgs_data.append(img_data)
    imgs_data = torch.stack(imgs_data)
    imgs_data = imgs_data.cuda()

    # calculation
    with torch.no_grad():
        out = model(imgs_data)
    out = F.softmax(out, dim=1)
    out = out.data.cpu().numpy()

    # pring results
    for idx in range(N):
        plt.figure()
        if out[idx, 0] > out[idx, 1]:
            plt.suptitle('cat:{:.1%},dog:{:.1%}'.format(out[idx, 0], out[idx, 1]))
        else:
            plt.suptitle('dog:{:.1%},cat:{:.1%}'.format(out[idx, 1], out[idx, 0]))
        plt.imshow(imgs[idx])
    plt.show()


if __name__ == '__main__':
    test()


