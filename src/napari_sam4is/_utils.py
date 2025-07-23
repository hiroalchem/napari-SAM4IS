import os
import urllib

import numpy as np
from segment_anything import sam_model_registry
from skimage.color import gray2rgb, rgba2rgb
from skimage.draw import polygon2mask
from skimage.measure import find_contours


def load_model(model_name: str = 'default') -> sam_model_registry:
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


def autodownload(model_url: str):
    """Download model

    Args:
        model_url (str): model url

    """

    urllib.request.urlretrieve(model_url, os.path.join(os.path.expanduser("~"), '.cache', 'napari-SAM4IS', os.path.basename(model_url)))


def preprocess(image, layer_type, current_step=None):
    if layer_type == "Gray":
        image = gray2rgb(np.array(image))
    elif layer_type == "RGBA":
        image = rgba2rgb(np.array(image))
    elif layer_type == "Gray with channel":
        image = gray2rgb(np.array(image[:, :, 0]))
    elif layer_type == "stacked gray images":
        if current_step is not None:
            image = gray2rgb(np.array(image[current_step, :, :]))
    elif layer_type == "stacked gray images with channel":
        if current_step is not None:
            image = gray2rgb(np.array(image[current_step, :, :, 0]))
    elif layer_type == "stacked RGB images":
        if current_step is not None:
            image = np.array(image[current_step, :, :, :])
    elif layer_type == "RGB":
        pass
    elif layer_type == "Not supported":
        raise ValueError("image shape is not supported")
    else:
        pass
    return np.array(image)


def check_image_type(viewer, layer_name):
    image = viewer.layers[layer_name].data
    print(f'current image shape = {image.shape}')
    if len(image.shape) == 2: # Gray
        return "Gray"
    elif len(image.shape) > 4:
        return "Not supported"
    elif (len(image.shape) == 3)&(image.shape[-1] == 4):
        return "RGBA"
    elif (len(image.shape) == 3)&(image.shape[-1] == 1): # Gray
        return "Gray with channel"
    elif (len(image.shape) == 3)&(image.shape[-1] == 2):
        return "Not supported"
    elif (len(image.shape) == 3)&(image.shape[-1] > 4): # maybe stacked gray images
        return "stacked gray images"
    elif (len(image.shape) == 4)&(image.shape[-1] == 1): # maybe stacked gray images
        return "stacked gray images with channel"
    elif (len(image.shape) == 4)&(image.shape[-1] == 3): # maybe stacked RGB images
        return "stacked RGB images"
    elif (len(image.shape) == 4)&(image.shape[-1] == 2) or (len(image.shape) == 4)&(image.shape[-1] > 4):
        return "Not supported"
    elif (len(image.shape) == 3)&(image.shape[-1] == 3):
        return "RGB"
    else:
        return "Not supported"


def label2polygon(label):
    """Convert label to polygon

    Args:
        label (np.ndarray): label image

    :return: polygons
    """
    polygons = [find_contours(label)[0].astype(int)]
    return polygons


def create_json(image, name, data):
    images = [
        {
            "file_name": name,
            "height": image.shape[0],
            "width": image.shape[1],
            "id": 0
        }
    ]
    annotations = []
    for i, polygon in enumerate(data):
        annotation = {
            "id": i,
            "image_id": 0,
            "category_id": 0,
            "segmentation": [polygon.flatten().tolist()[::-1]],
            "area": np.count_nonzero(polygon2mask(image.shape, polygon)),
            "bbox": [min(polygon[:, 1]), min(polygon[:, 0]), max(polygon[:, 1]) - min(polygon[:, 1]), max(polygon[:, 0]) - min(polygon[:, 0])],  # [x, y, width, height]
            "iscrowd": 0
        }
        annotations.append(annotation)
    categories = [
        {
            "id": 0,
            "name": "object",
            "supercategory": "object"
        }
    ]
    json = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    return json


def find_first_missing(arr):
    arr = np.unique(arr)  # remove duplicates
    arr = arr[arr >= 0]  # keep only positive values and zero
    arr.sort()  # sort the array

    # check for missing integers
    for index, value in np.ndenumerate(arr):
        if index[0] != value:
            return index[0]
    return len(arr)

