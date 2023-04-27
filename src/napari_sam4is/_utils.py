import os
import urllib

from segment_anything import sam_model_registry
from skimage.color import gray2rgb, rgba2rgb
from skimage.measure import find_contours


def load_model(model_name):
    """Load model

    Args:
        model_name (str): model name

    :return: model
    """
    model_urls = dict(default="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                      vit_h="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                      vit_l="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                      vit_b="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    model_url = model_urls[model_name]
    model_path = os.path.join(os.path.expanduser("~"), '.cache', 'napari-SAM4IS', os.path.basename(model_url))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        autodownload(model_url)
    sam = sam_model_registry[model_name](checkpoint=model_path)
    return sam


def autodownload(model_url):
    """Download model

    Args:
        model_url (str): model url

    """

    urllib.request.urlretrieve(model_url, os.path.join(os.path.expanduser("~"), '.cache', 'napari-SAM4IS', os.path.basename(model_url)))


def preprocess(image):
    if len(image.shape) == 2:
        image = gray2rgb(image)
    elif image.shape[-1] == 4:
        image = rgba2rgb(image)
    else:
        pass
    return image


def label2polygon(label):
    """Convert label to polygon

    Args:
        label (np.ndarray): label image

    :return: polygons
    """
    polygons = [find_contours(label)[0].astype(int)]
    print(polygons)
    return polygons