import json
import logging
import os
import urllib

import numpy as np
from skimage.color import gray2rgb, rgba2rgb
from skimage.draw import polygon2mask
from skimage.measure import find_contours

logger = logging.getLogger(__name__)

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


def create_json(
    image,
    name,
    data,
    categories=None,
    category_ids=None,
    attributes_list=None,
):
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
        if attributes_list is not None and i < len(attributes_list):
            annotation["attributes"] = attributes_list[i]
        annotations.append(annotation)
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def load_json(json_path):
    """Load COCO JSON and return parsed annotations.

    Returns:
        dict with keys:
        - "categories": list of COCO category dicts
        - "annotations": list of dicts, each with:
            "polygon": np.ndarray(N,2), "category_id": int,
            "attributes": dict or None
        - "image_info": dict with "file_name", "height", "width"
    """
    with open(json_path) as f:
        coco = json.load(f)

    image_info = coco["images"][0] if coco.get("images") else {}
    categories = coco.get("categories", [])

    parsed = []
    for i, ann in enumerate(coco.get("annotations", [])):
        ann_id = ann.get("id")
        seg = ann.get("segmentation")

        if seg is None:
            continue

        # RLE format (dict) → skip
        if isinstance(seg, dict):
            logger.warning(
                "Skipping annotation (id=%s, index=%d): RLE not supported",
                ann_id,
                i,
            )
            continue

        # seg should be list-of-list (polygon format)
        if not isinstance(seg, list):
            continue

        cat_id = ann.get("category_id", 0)
        attrs = ann.get("attributes")

        # Convert reviewed_at None → ""
        if isinstance(attrs, dict) and attrs.get("reviewed_at") is None:
            attrs["reviewed_at"] = ""

        sub_polygons = []
        for sub in seg:
            if not isinstance(sub, list) or len(sub) < 6:
                continue
            coords = np.array(sub[::-1], dtype=float).reshape(-1, 2)
            sub_polygons.append(coords)

        if len(sub_polygons) > 1:
            logger.info(
                "Annotation (id=%s, index=%d): split into %d shapes",
                ann_id,
                i,
                len(sub_polygons),
            )

        for poly in sub_polygons:
            entry = {
                "polygon": poly,
                "category_id": cat_id,
                "attributes": dict(attrs) if attrs else None,
            }
            parsed.append(entry)

    return {
        "categories": categories,
        "annotations": parsed,
        "image_info": image_info,
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
