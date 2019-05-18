import torch.utils.data as data

from PIL import Image

import os

import sys
import numpy as np

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(list_file, labels,absolute_prefix,extensions,label_filter):
    label_filter = label_filter or (lambda x : True)
    images=[]
    try:
        file = open(list_file,'r')
    except:
        raise RuntimeError("List file doesn't exist")

    image_list = file.readlines()

    if labels:
      raise NotImplementedError("Labels not implemented")
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        for file_and_label in image_list:
            if(len(file_and_label.split())==2 and has_file_allowed_extension(file_and_label.split()[0], extensions)):
                data_array = file_and_label.split()

                path = os.path.join(absolute_prefix,data_array[0])
                label = int(data_array[1])
                if label_filter(label):
                    item = (path, label)
                    images.append(item)
    file.close()
    return images

class ListFile(data.Dataset):

    def __init__(self, list_file ,loader ,extensions,labels=None ,absolute_prefix='',transform=None, target_transform=None, label_filter=None):
        samples = make_dataset(list_file, labels,absolute_prefix,extensions,label_filter)
        if len(samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root + "\n")

        self.samples = samples
        self.loader = loader
        self.extensions = extensions

        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

class ImageList(ListFile):
    def __init__(self, list_file , absolute_prefix='',labels=None , transform=None, target_transform=None,
                loader=default_loader, label_filter=None):
        super(ImageList, self).__init__(list_file ,loader ,IMG_EXTENSIONS,labels,absolute_prefix=absolute_prefix,
                                         transform=transform, target_transform=target_transform, label_filter=label_filter)
        self.imgs = self.samples



