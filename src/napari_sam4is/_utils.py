import os
import urllib

import numpy as np
from skimage.color import gray2rgb, rgba2rgb
from skimage.draw import polygon2mask
from skimage.measure import find_contours

MODEL_URLS = {
    "default": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def get_available_model_names():
    """Return model names without importing heavy dependencies."""
    return list(MODEL_URLS.keys())


def load_model(model_name: str = "default"):
    """Load model

    Args:
        model_name (str): model name

    :return: model
    """
    try:
        from segment_anything import sam_model_registry
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "segment_anything is required to load local SAM models."
        ) from exc

    if model_name not in MODEL_URLS:
        raise ValueError(f"Unsupported model name: {model_name}")

    model_url = MODEL_URLS[model_name]
    model_path = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "napari-SAM4IS",
        os.path.basename(model_url),
    )
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

    urllib.request.urlretrieve(
        model_url,
        os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "napari-SAM4IS",
            os.path.basename(model_url),
        ),
    )


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
    print(f"current image shape = {image.shape}")
    if len(image.shape) == 2:  # Gray
        return "Gray"
    elif len(image.shape) > 4:
        return "Not supported"
    elif (len(image.shape) == 3) & (image.shape[-1] == 4):
        return "RGBA"
    elif (len(image.shape) == 3) & (image.shape[-1] == 1):  # Gray
        return "Gray with channel"
    elif (len(image.shape) == 3) & (image.shape[-1] == 2):
        return "Not supported"
    elif (len(image.shape) == 3) & (
        image.shape[-1] > 4
    ):  # maybe stacked gray images
        return "stacked gray images"
    elif (len(image.shape) == 4) & (
        image.shape[-1] == 1
    ):  # maybe stacked gray images
        return "stacked gray images with channel"
    elif (len(image.shape) == 4) & (
        image.shape[-1] == 3
    ):  # maybe stacked RGB images
        return "stacked RGB images"
    elif (len(image.shape) == 4) & (image.shape[-1] == 2) or (
        len(image.shape) == 4
    ) & (image.shape[-1] > 4):
        return "Not supported"
    elif (len(image.shape) == 3) & (image.shape[-1] == 3):
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


def create_json(image, name, data, categories=None, category_ids=None):
    if categories is None:
        categories = [{"id": 0, "name": "object", "supercategory": "object"}]
    if category_ids is None:
        category_ids = [0] * len(data)

    images = [
        {
            "file_name": name,
            "height": image.shape[0],
            "width": image.shape[1],
            "id": 0,
        }
    ]
    annotations = []
    for i, polygon in enumerate(data):
        cat_id = category_ids[i] if i < len(category_ids) else 0
        annotation = {
            "id": i,
            "image_id": 0,
            "category_id": cat_id,
            "segmentation": [polygon.flatten().tolist()[::-1]],
            "area": int(np.count_nonzero(polygon2mask(image.shape, polygon))),
            "bbox": [
                float(min(polygon[:, 1])),
                float(min(polygon[:, 0])),
                float(max(polygon[:, 1]) - min(polygon[:, 1])),
                float(max(polygon[:, 0]) - min(polygon[:, 0])),
            ],
            "iscrowd": 0,
        }
        annotations.append(annotation)
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def find_first_missing(arr):
    arr = np.unique(arr)  # remove duplicates
    arr = arr[arr >= 0]  # keep only positive values and zero
    arr.sort()  # sort the array

    # check for missing integers
    for index, value in np.ndenumerate(arr):
        if index[0] != value:
            return index[0]
    return len(arr)


def find_missing_class_number(numbers):
    """Find the smallest missing non-negative integer in a list.

    Args:
        numbers: list of int (class IDs currently in use)

    Returns:
        int: smallest missing non-negative integer
    """
    if not numbers:
        return 0
    numbers = sorted(set(numbers))
    for i, val in enumerate(numbers):
        if i != val:
            return i
    return len(numbers)
