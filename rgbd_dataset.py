import os
import glob

from itertools import groupby 
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.io import decode_image

class RgbdDataset(Dataset):
    """RGB-D dataset."""
    def __init__(self, split="train", transform=None):
        """
        Arguments:
            split (string, optional): Define the split to apply to the dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = 'dataset/rgbd-dataset'
        self.class_set =  ['camera', 'flashlight', 'lightbulb', 'pitcher', 'stapler']
        self.split = split
        self.img_labels = self.get_labels()
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        left_img_path = self.img_labels[idx][0]
        left_image = decode_image(left_img_path, mode='RGB')
        right_img_path = self.img_labels[idx][1]
        right_image = decode_image(right_img_path, mode='RGB')
        images = left_image, right_image
        label = self.img_labels[idx][2]
        if self.transform:
            images = self.transform(images[0]), self.transform(images[1])
        return images, label

    def get_data_from_file(self):
        data_dir = os.path.join(self.dataset_path, '{}', '*')
        target_dataset = defaultdict(list)

        for label, target in enumerate(self.class_set):
            target_videos = glob.glob(data_dir.format(target))
            target_videos = sorted(target_videos, key=lambda x: int(os.path.basename(x).split("_")[-1]))
            for target_video in target_videos:
                dataset = glob.glob(target_video + "/*_crop.png")
                dataset = sorted(dataset, key=lambda x: (int(os.path.basename(x).split("_")[2]),
                                                        int(os.path.basename(x).split("_")[3])))
                prefix = os.path.basename(target_video) + "_"
                for key, value in groupby(dataset, lambda x: os.path.basename(x).split(prefix)[1].split("_")[0]):
                    value = list(value)
                    # Roll value list backwards 
                    value_roll = list(value)
                    value_roll.append(value_roll.pop(0))
                    target_dataset[target].append(list(zip(value, value_roll, [label]*len(value))))
        return target_dataset
    
    def get_labels(self):
        img_labels = []
        target_dataset = self.get_data_from_file()
        for category, data in target_dataset.items():
            for instance in data:
                if (self.split == 'train'):
                    # get 2/3 of 1/3 of the dataset
                    img_labels.extend(instance[:(len(instance) // 9) * 2])
                elif (self.split == 'val'):
                    # get 1/3 of 1/3 of the dataset
                    img_labels.extend(instance[(len(instance) // 9) * 2 : (len(instance) // 9) * 3])
                elif (self.split == 'test'):
                    # get 2/3 of the dataset
                    img_labels.extend(instance[(len(instance) // 3):])
                else:
                    raise RuntimeError("Dataset split name unknown.")
        return img_labels