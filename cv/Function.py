import cv2
import skimage
import numpy as np
import glob
import os
import sys
from .. import utils as util
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

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# images_paths = glob.glob(images_dir+'*.jpg')
# images_paths += glob.glob(images_dir+'*.jpeg')
# images_paths += glob.glob(images_dir+'*.png')
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def adjust_gamma_of_folder(dir, gamma):
    if not os.path.exists(dir):
        raise RuntimeError('Directory not found!')
    d = os.path.expanduser(dir)
    for root,_,fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                path = os.path.join(root, fname)
                original = skimage.io.imread(path)
                adjusted = adjust_gamma(original, gamma=gamma)
                util.save_image_from_numpy(root,"gamma_"+fname,adjusted.transpose(2,0,1)/255.0)

