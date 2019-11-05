import os
import torch.utils.data as data
from PIL import Image
import torch
import torchvision.transforms as transforms


IMAGE_SIZE = 200

dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CatsVSDogsDataset(data.Dataset):
    def __init__(self, mode, dir):
        self.mode = mode
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.transform = dataTransform

        if self.mode == 'train':
            dir = dir + '/train/'
            for file in os.listdir(dir):
                self.list_img.append(dir + file)        # add cat or dog image
                self.data_size += 1
                name = file.split(sep='.')
                if name[0] == 'cat':
                    self.list_label.append(0)         # the label of cat is 0
                else:
                    self.list_label.append(1)         # the label of dog is 1
        elif self.mode == 'test':
            dir = dir + '/test/'
            for file in os.listdir(dir):
                self.list_img.append(dir + file)
                self.data_size += 1
                self.list_label.append(2)
        else:
            print('Undefined Dataset!')

    def __getitem__(self, item):
        if self.mode == 'train':
            img = Image.open(self.list_img[item])
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])
        elif self.mode == 'test':
            img = Image.open(self.list_img[item])
            return self.transform(img)
        else:
            print('None')

    def __len__(self):
        return self.data_size

    def get_item(self, index):
        if self.mode == 'train':
            return self.list_img[index], self.list_label[index]
        elif self.mode == 'test':
            return self.list_img[index]
        else:
            print('None')







