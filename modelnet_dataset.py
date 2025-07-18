import os
from torch.utils.data import Dataset
from torchvision.io import decode_image

class ModelNetDataset(Dataset):
    """ModelNet2D dataset."""
    def __init__(self, transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = 'dataset/modelnet2d'
        self.class_set =  ['airplane', 'car', 'chair', 'lamp', 'person']
        self.azimuth_angles = 72
        self.elevations = 9
        self.model_instances = 15
        self.img_labels = self.get_labels()
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        left_img_path = os.path.join(self.dataset_path, self.img_labels[idx][0])
        left_image = decode_image(left_img_path, mode='GRAY')
        right_img_path = os.path.join(self.dataset_path, self.img_labels[idx][1])
        right_image = decode_image(right_img_path, mode='GRAY')
        images = left_image, right_image
        label = self.img_labels[idx][3]
        azimuth = self.img_labels[idx][2]
        if self.transform:
            images = self.transform(images[0]), self.transform(images[1])
        return images, label, azimuth

    def get_labels(self):
        total_imgs = self.azimuth_angles * self.elevations
        img_labels = []
        for c in self.class_set: # 5
            label = self.class_set.index(c)
            for m in range(1, self.model_instances + 1): # 15
                for n in range(1, total_imgs, 2): # 72/2
                    left_img = '{}/{}/{}.png'.format(c, m, n)
                    right_img = '{}/{}/{}.png'.format(c, m, n+1)
                    azimuth = (n % self.azimuth_angles - 1) * 5
                    img_labels.append([left_img, right_img, azimuth, label])
        return img_labels