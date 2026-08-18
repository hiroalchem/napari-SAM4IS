import json
import logging
import os
import urllib

import numpy as np
from skimage.color import gray2rgb, rgba2rgb
from skimage.draw import polygon2mask
from skimage.measure import find_contours, points_in_poly

logger = logging.getLogger(__name__)

MODEL_URLS = {
    "default": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def get_available_model_names():
    """Return model names without importing heavy dependencies."""
    return list(MODEL_URLS.keys()) + ["sam3"]


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


def to_uint8(image):
    """Min-max normalize to uint8 (pass through if already uint8)."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    lo, hi = float(image.min()), float(image.max())
    if hi - lo > 0:
        return ((image - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.zeros_like(image, dtype=np.uint8)


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


def _open_ring(ring):
    """Drop an explicit closing vertex if the ring has one."""
    ring = np.asarray(ring, dtype=float)
    if len(ring) > 1 and np.array_equal(ring[0], ring[-1]):
        return ring[:-1]
    return ring


def _dedupe_consecutive(ring):
    """Drop consecutive duplicate vertices.

    Repeated vertices are not robust for triangulation algorithms, and
    rounding contour coordinates to integers readily produces them.
    """
    ring = np.asarray(ring, dtype=float)
    if len(ring) < 2:
        return ring
    keep = np.ones(len(ring), dtype=bool)
    keep[1:] = np.any(ring[1:] != ring[:-1], axis=1)
    return ring[keep]


def _signed_area(ring):
    """Shoelace signed area of an open ring in (row, col) coordinates."""
    ring = _open_ring(ring)
    if len(ring) < 3:
        return 0.0
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def concatenate_rings(rings):
    """Chain closed rings into one napari polygon.

    napari (>= 0.6.0) opens up a hole by discarding every edge that the
    outline walks twice, so each ring has to be entered and left along
    the same bridge. Chaining the rings head to tail would instead form
    a loop of bridges that each appear once and survive, which leaves
    the holes filled in, so every ring after the first is walked as a
    there-and-back detour from a single shared anchor vertex. The
    implicit closing edge supplies the last hop back to the anchor.

    Args:
        rings (list): open rings, the first one acting as the anchor

    :return: (M, 2) vertex array in napari's concatenated-ring form
    """
    anchor = rings[0][:1]
    parts = [np.vstack([rings[0], anchor])]
    for ring in rings[1:]:
        parts.append(np.vstack([ring, ring[:1], anchor]))
    polygon = np.concatenate(parts)
    return polygon[:-1] if len(parts) > 1 else polygon


def normalize_ring_group(outer, holes=()):
    """Return ``[outer, *holes]`` with holes wound against the outer ring.

    Raises ValueError if the outer ring is degenerate.
    """
    outer = _dedupe_consecutive(_open_ring(outer))
    if len(outer) < 3:
        raise ValueError("outer ring needs at least 3 distinct vertices")

    outer_sign = np.sign(_signed_area(outer)) or 1.0
    rings = [outer]
    for hole in holes:
        ring = _dedupe_consecutive(_open_ring(hole))
        if len(ring) < 3:
            continue
        # napari expects holes to wind opposite to the outer ring
        if np.sign(_signed_area(ring)) == outer_sign:
            ring = ring[::-1]
        rings.append(ring)
    return rings


def rings_to_polygon(outer, holes=()):
    """Combine an outer ring and its holes into one napari polygon.

    Args:
        outer (np.ndarray): (N, 2) outer ring, open or explicitly closed
        holes (iterable): rings to subtract from ``outer``

    :return: (M, 2) vertex array in napari's concatenated-ring form
    """
    return concatenate_rings(normalize_ring_group(outer, holes))


def polygon_to_rings(polygon):
    """Split a concatenated-ring polygon back into its rings.

    Inverse of :func:`concatenate_rings`. The first ring is an outer
    boundary and the rest are holes or further components. A polygon
    that is not in the canonical form is returned unchanged as a single
    ring, since splitting it would be ambiguous.

    Args:
        polygon (np.ndarray): (N, 2) vertex array

    :return: list of (M, 2) open rings
    """
    poly = _dedupe_consecutive(polygon)
    rings = []
    i, n = 0, len(poly)
    while i < n:
        match = np.flatnonzero(np.all(poly[i + 1 :] == poly[i], axis=1))
        if len(match) == 0:
            # trailing vertices with no closure: not canonical
            rings.append(poly[i:])
            break
        ring = poly[i : i + 1 + int(match[0])]
        # a vertex revisited inside a ring makes the split ambiguous
        if len(ring) < 3 or len(np.unique(ring, axis=0)) != len(ring):
            return [poly]
        rings.append(ring)
        i += 2 + int(match[0])
        # step over the hop back to the anchor before the next ring
        if i < n and np.array_equal(poly[i], poly[0]):
            i += 1
    return rings or [poly]


def group_rings_by_nesting(rings):
    """Group rings into outer/hole pairs by containment.

    Nesting is resolved by containment rather than by input order: rings
    at even depth are outer boundaries, rings at odd depth are the holes
    of whichever ring encloses them. A ring nested inside a hole is an
    island and becomes an outer boundary again.

    Args:
        rings (iterable): (N, 2) vertex arrays, open or explicitly closed

    :return: list of ``(outer_ring, [hole_rings])`` tuples
    """
    rings = [_dedupe_consecutive(_open_ring(r)) for r in rings]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return []

    depth = np.zeros(len(rings), dtype=int)
    for i, inner in enumerate(rings):
        for j, outer in enumerate(rings):
            if i != j and points_in_poly(inner[:1], outer)[0]:
                depth[i] += 1

    groups = []
    for i, ring in enumerate(rings):
        if depth[i] % 2:
            continue
        holes = [
            rings[j]
            for j in range(len(rings))
            if depth[j] == depth[i] + 1
            and points_in_poly(rings[j][:1], ring)[0]
        ]
        groups.append((ring, holes))
    return groups


def mask_to_rings(mask):
    """Extract outer/hole ring groups from a binary mask.

    The mask is padded with a background border first. Without it a
    region running into the edge of the image yields an open contour,
    and closing that with a straight chord draws a segment right
    through the other vertices sitting on the same edge, which is
    degenerate and makes triangulation fail.

    Args:
        mask (np.ndarray): 2D mask; non-zero pixels are foreground

    :return: list of ``(outer_ring, [hole_rings])`` tuples
    """
    mask = np.asarray(mask) > 0
    limit = np.array(mask.shape) - 1
    contours = [
        np.clip(contour - 1, 0, limit)
        for contour in find_contours(np.pad(mask, 1), 0.5)
    ]
    return group_rings_by_nesting(contours)


def merge_rings_to_polygon(rings):
    """Combine independent rings into one concatenated-ring polygon.

    Used to turn separately drawn shapes into a single annotation with
    holes, which is awkward to draw directly in the napari GUI. All
    groups share one chain so that every bridge cancels.

    Args:
        rings (iterable): (N, 2) vertex arrays

    :return: (M, 2) vertex array, or None if no usable ring was given
    """
    chain = []
    for outer, holes in group_rings_by_nesting(rings):
        try:
            chain.extend(normalize_ring_group(outer, holes))
        except ValueError:
            continue
    return concatenate_rings(chain) if chain else None


def label2polygon(label):
    """Convert label to polygon

    The mask becomes a single polygon in napari's concatenated-ring
    form, so holes and disconnected components are preserved without
    changing the one-mask-one-annotation model.

    Args:
        label (np.ndarray): label image

    :return: polygons
    """
    chain = []
    for outer, holes in mask_to_rings(label):
        try:
            chain.extend(
                normalize_ring_group(
                    np.round(outer), [np.round(hole) for hole in holes]
                )
            )
        except ValueError:
            continue
    return [concatenate_rings(chain)] if chain else []


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
            # shape[:2]: an RGB image would otherwise count every channel
            "area": int(
                np.count_nonzero(polygon2mask(image.shape[:2], polygon))
            ),
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
    """Load single-image COCO JSON and return parsed annotations.

    Only single-image COCO files are supported (as produced by
    this plugin's Save function). Multi-image files are rejected.

    Returns:
        dict with keys:
        - "categories": list of COCO category dicts
        - "annotations": list of dicts, each with:
            "polygon": np.ndarray(N,2), "category_id": int,
            "attributes": dict or None
        - "image_info": dict with "file_name", "height", "width"
    """
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    if len(images) > 1:
        raise ValueError(
            f"Multi-image COCO files are not supported "
            f"({len(images)} images found)"
        )
    image_info = images[0] if images else {}
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
            if len(sub) % 2 != 0:
                logger.warning(
                    "Skipping sub-polygon in annotation "
                    "(id=%s, index=%d): odd coordinate count %d",
                    ann_id,
                    i,
                    len(sub),
                )
                continue
            try:
                coords = np.array(sub[::-1], dtype=float).reshape(-1, 2)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping sub-polygon in annotation "
                    "(id=%s, index=%d): %s",
                    ann_id,
                    i,
                    exc,
                )
                continue
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


def load_sam3_model(checkpoint_path: str | None = None):
    """Load SAM3 model (HuggingFace auto or local checkpoint).

    Args:
        checkpoint_path: Path to locally downloaded sam3.pt.
            If None, downloads from HuggingFace (gated access required).

    Returns:
        tuple: (model, processor, cleanup) where cleanup is a callable
            that reverts the pin_memory monkey-patch (or None on CUDA).

    Raises:
        ImportError: sam3 not installed
        Exception: any error during model load
    """
    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "sam3 is not installed. Install with: uv sync --extra sam3"
        ) from exc

    import torch

    # MPS not yet supported by sam3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Temporarily set default device to "cpu" so that sam3 submodules
    # (e.g. build_tracker) that don't accept a device argument cannot
    # accidentally inherit MPS from a previously loaded SAM1 model.
    _prev_default = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        if checkpoint_path:
            model = build_sam3_image_model(
                enable_inst_interactivity=True,
                load_from_HF=False,
                checkpoint_path=checkpoint_path,
                device=device,
            )
        else:
            model = build_sam3_image_model(
                enable_inst_interactivity=True,
                load_from_HF=True,
                device=device,
            )
    finally:
        torch.set_default_device(_prev_default)

    # _setup_device_and_mode only handles "cuda"; move to CPU explicitly.
    # Force all parameters to cpu to avoid MPS leakage on macOS Apple Silicon.
    if device != "cuda":
        model.to("cpu")

    # Confirm the actual device from model parameters (model.device is a
    # dynamic property: next(self.parameters()).device).
    actual_device = str(next(model.parameters()).device)

    # sam3 geometry_encoders.py:648 calls pin_memory().to(device=..., non_blocking=True)
    # which raises RuntimeError on MPS ("Attempted to set storage on device cpu to mps:0")
    # because pin_memory() is CUDA-only.  Monkey-patch Tensor.pin_memory to be a no-op
    # on non-CUDA platforms so sam3 works on macOS Apple Silicon.
    # The patch must stay active during inference, so we return a cleanup
    # callable for the caller to invoke when the model is unloaded.
    cleanup = None
    if not torch.cuda.is_available():
        # Capture the *real* original once; reloads must not snapshot the
        # already-patched version.
        if not hasattr(load_sam3_model, "_orig_pin_memory"):
            load_sam3_model._orig_pin_memory = torch.Tensor.pin_memory
        _orig = load_sam3_model._orig_pin_memory

        def _safe_pin_memory(self, device=None):
            return self

        torch.Tensor.pin_memory = _safe_pin_memory

        def cleanup():  # noqa: E306
            torch.Tensor.pin_memory = _orig

        try:
            processor = Sam3Processor(model, device=actual_device)
        except Exception:
            cleanup()
            raise
    else:
        processor = Sam3Processor(model, device=actual_device)
    return model, processor, cleanup


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
