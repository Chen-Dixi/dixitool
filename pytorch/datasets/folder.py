import os

from PIL import Image

from torchvision.datasets import DatasetFolder

class DIXIDatasetFolder(DatasetFolder):
    """Custom Folder Dataset to meet some extra requirements
    Args:
        same as torchvision.dataset.DatasetFolder

        folder_filter: exlude some folder
    """
    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, folder_filter=None):
        self.folder_filter = folder_filter or (lambda x : True)

        super(DIXIDatasetFolder, self).__init__(root, loader, extensions, transform,
                 target_transform, is_valid_file)
    
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Customized by overriding the _find_classes() method for torchvision 0.9.0.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and self.folder_filter(d.name)]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Customized by overriding the _find_classes() method for torchvision 0.10.0.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and self.folder_filter(d.name)]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


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

class DIXIImageFolder(DIXIDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    
    Example:
    >>> source_dataset = DIXIImageFolder(source_path, transform = data_transforms['source'], folder_filter = (lambda x: x not in ['__MACOSX']))
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, folder_filter=None):
        super(DIXIImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          folder_filter=folder_filter)
        self.imgs = self.samples