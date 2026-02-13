import json
import os
import tempfile

import numpy as np

from napari_sam4is import SAMWidget
from napari_sam4is._utils import (
    create_json,
    find_missing_class_number,
    load_json,
)


# make_napari_viewer is a pytest fixture that returns a napari viewer object
def test_sam_widget_creation(make_napari_viewer):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    widget = SAMWidget(viewer)

    # check that the widget was created successfully
    assert widget is not None
    assert widget._viewer == viewer


def test_sam_widget_basic_functionality(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create the SAM widget
    widget = SAMWidget(viewer)

    # check basic attributes exist
    assert hasattr(widget, "_viewer")
    assert hasattr(widget, "_image_type")
    assert hasattr(widget, "_current_slice")


def test_class_management(make_napari_viewer):
    """Test adding and deleting classes."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    # Add classes
    widget._class_name_input.setText("cat")
    widget._add_class()
    assert widget._class_list_widget.count() == 1
    assert widget._class_list_widget.item(0).text() == "0: cat"

    widget._class_name_input.setText("dog")
    widget._add_class()
    assert widget._class_list_widget.count() == 2
    assert widget._class_list_widget.item(1).text() == "1: dog"

    # Duplicate should not be added
    widget._class_name_input.setText("cat")
    widget._add_class()
    assert widget._class_list_widget.count() == 2

    # Delete class (not in use, should succeed)
    widget._class_list_widget.setCurrentRow(0)
    widget._del_class()
    assert widget._class_list_widget.count() == 1
    assert widget._class_list_widget.item(0).text() == "1: dog"


def test_build_categories(make_napari_viewer):
    """Test COCO categories generation."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    widget._class_name_input.setText("cat")
    widget._add_class()
    widget._class_name_input.setText("dog")
    widget._add_class()

    categories = widget._build_categories_list()
    assert len(categories) == 2
    assert categories[0]["id"] == 0
    assert categories[0]["name"] == "cat"
    assert categories[1]["id"] == 1
    assert categories[1]["name"] == "dog"


def test_get_selected_class_default(make_napari_viewer):
    """Test default class auto-creation when list is empty."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    # List is empty, should auto-create "0: object"
    result = widget._get_selected_class()
    assert result == "0: object"
    assert widget._class_list_widget.count() == 1


def test_find_missing_class_number():
    """Test find_missing_class_number utility."""
    assert find_missing_class_number([]) == 0
    assert find_missing_class_number([0, 1, 2]) == 3
    assert find_missing_class_number([0, 2, 3]) == 1
    assert find_missing_class_number([1, 2, 3]) == 0


def test_create_json_with_categories():
    """Test create_json with multi-category support."""
    image = np.zeros((100, 100), dtype=np.uint8)
    polygon = np.array([[10, 10], [10, 20], [20, 20], [20, 10]])
    categories = [
        {"id": 0, "name": "cat", "supercategory": "cat"},
        {"id": 1, "name": "dog", "supercategory": "dog"},
    ]

    result = create_json(
        image,
        "test.png",
        [polygon],
        categories=categories,
        category_ids=[1],
    )
    assert len(result["categories"]) == 2
    assert result["annotations"][0]["category_id"] == 1
    assert result["categories"][0]["name"] == "cat"
    assert result["categories"][1]["name"] == "dog"


def test_create_json_backward_compatible():
    """Test create_json without categories (backward compat)."""
    image = np.zeros((100, 100), dtype=np.uint8)
    polygon = np.array([[10, 10], [10, 20], [20, 20], [20, 10]])

    result = create_json(image, "test.png", [polygon])
    assert len(result["categories"]) == 1
    assert result["categories"][0]["name"] == "object"
    assert result["annotations"][0]["category_id"] == 0


def test_manual_mode_toggle(make_napari_viewer):
    """Test manual mode toggle enables/disables correct controls."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    # Enable manual mode
    widget._manual_mode_checkbox.setChecked(True)
    assert not widget._model_selection.isEnabled()
    assert not widget._model_load_btn.isEnabled()
    assert not widget._sam_box_layer.visible
    assert not widget._sam_positive_point_layer.visible
    assert not widget._sam_negative_point_layer.visible

    # Disable manual mode
    widget._manual_mode_checkbox.setChecked(False)
    assert widget._model_selection.isEnabled()
    assert widget._model_load_btn.isEnabled()
    assert widget._sam_box_layer.visible
    assert widget._sam_positive_point_layer.visible
    assert widget._sam_negative_point_layer.visible


# --- Annotation Attributes Tests ---


def _make_polygon():
    return np.array([[10, 10], [10, 20], [20, 20], [20, 10]])


def test_create_json_with_attributes():
    """Test attributes are included in COCO JSON."""
    image = np.zeros((100, 100), dtype=np.uint8)
    polygon = _make_polygon()
    attrs = [
        {
            "unclear": True,
            "uncertain": False,
            "review_status": "approved",
            "reviewed_at": None,
        }
    ]
    result = create_json(
        image,
        "test.png",
        [polygon],
        attributes_list=attrs,
    )
    ann = result["annotations"][0]
    assert "attributes" in ann
    assert ann["attributes"]["unclear"] is True
    assert ann["attributes"]["reviewed_at"] is None


def test_create_json_without_attributes():
    """Test backward compat: no attributes key."""
    image = np.zeros((100, 100), dtype=np.uint8)
    polygon = _make_polygon()
    result = create_json(image, "test.png", [polygon])
    ann = result["annotations"][0]
    assert "attributes" not in ann


def test_load_json_roundtrip():
    """Test create_json → load_json coordinate roundtrip."""
    image = np.zeros((100, 100), dtype=np.uint8)
    polygon = _make_polygon()
    attrs = [
        {
            "unclear": True,
            "uncertain": False,
            "review_status": "approved",
            "reviewed_at": "2026-02-13T10:00:00+09:00",
        }
    ]
    coco = create_json(
        image,
        "test.png",
        [polygon],
        attributes_list=attrs,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(coco, f)
        tmp_path = f.name

    try:
        result = load_json(tmp_path)
        loaded = result["annotations"]
        assert len(loaded) == 1
        loaded_poly = loaded[0]["polygon"]
        np.testing.assert_array_equal(loaded_poly, polygon)
        assert loaded[0]["attributes"]["unclear"] is True
        assert (
            loaded[0]["attributes"]["reviewed_at"]
            == "2026-02-13T10:00:00+09:00"
        )
    finally:
        os.unlink(tmp_path)


def test_load_json_multi_polygon():
    """Test multi-polygon split into separate shapes."""
    polygon1 = _make_polygon()
    polygon2 = np.array([[30, 30], [30, 40], [40, 40], [40, 30]])
    coco = {
        "images": [
            {
                "file_name": "test.png",
                "height": 100,
                "width": 100,
                "id": 0,
            }
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "segmentation": [
                    polygon1.flatten().tolist()[::-1],
                    polygon2.flatten().tolist()[::-1],
                ],
                "area": 100,
                "bbox": [10, 10, 10, 10],
                "iscrowd": 0,
                "attributes": {
                    "unclear": True,
                    "uncertain": False,
                    "review_status": "unreviewed",
                    "reviewed_at": None,
                },
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "object",
                "supercategory": "object",
            }
        ],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(coco, f)
        tmp_path = f.name

    try:
        result = load_json(tmp_path)
        loaded = result["annotations"]
        assert len(loaded) == 2
        # Both should have same attributes (copied)
        assert loaded[0]["attributes"]["unclear"] is True
        assert loaded[1]["attributes"]["unclear"] is True
        np.testing.assert_array_equal(loaded[0]["polygon"], polygon1)
        np.testing.assert_array_equal(loaded[1]["polygon"], polygon2)
    finally:
        os.unlink(tmp_path)


def test_load_json_rle_skipped():
    """Test RLE segmentation is skipped."""
    coco = {
        "images": [
            {
                "file_name": "test.png",
                "height": 100,
                "width": 100,
                "id": 0,
            }
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "segmentation": {
                    "counts": [10, 20],
                    "size": [100, 100],
                },
                "area": 100,
                "bbox": [10, 10, 10, 10],
                "iscrowd": 1,
            }
        ],
        "categories": [],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(coco, f)
        tmp_path = f.name

    try:
        result = load_json(tmp_path)
        assert len(result["annotations"]) == 0
    finally:
        os.unlink(tmp_path)


def test_load_json_reviewed_at_null():
    """Test reviewed_at null → empty string."""
    polygon = _make_polygon()
    coco = {
        "images": [
            {
                "file_name": "test.png",
                "height": 100,
                "width": 100,
                "id": 0,
            }
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "segmentation": [polygon.flatten().tolist()[::-1]],
                "area": 100,
                "bbox": [10, 10, 10, 10],
                "iscrowd": 0,
                "attributes": {
                    "unclear": False,
                    "uncertain": False,
                    "review_status": "unreviewed",
                    "reviewed_at": None,
                },
            }
        ],
        "categories": [],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(coco, f)
        tmp_path = f.name

    try:
        result = load_json(tmp_path)
        attrs = result["annotations"][0]["attributes"]
        assert attrs["reviewed_at"] == ""
    finally:
        os.unlink(tmp_path)


def test_reviewed_at_save_load_consistency():
    """Test "" → null → "" roundtrip."""
    image = np.zeros((100, 100), dtype=np.uint8)
    polygon = _make_polygon()
    attrs = [
        {
            "unclear": False,
            "uncertain": False,
            "review_status": "unreviewed",
            "reviewed_at": None,  # "" converted to None
        }
    ]
    coco = create_json(
        image,
        "test.png",
        [polygon],
        attributes_list=attrs,
    )
    # JSON null
    assert coco["annotations"][0]["attributes"]["reviewed_at"] is None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(coco, f)
        tmp_path = f.name

    try:
        result = load_json(tmp_path)
        loaded_at = result["annotations"][0]["attributes"]["reviewed_at"]
        assert loaded_at == ""
    finally:
        os.unlink(tmp_path)


def test_annotation_attributes_defaults(make_napari_viewer):
    """Test features defaults after accept."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    # Set up instance mode with Accepted layer
    widget._radio_btn_shape.setChecked(True)
    output_name = "Accepted"
    widget._shapes_layer_selection.setCurrentText(output_name)

    output_layer = viewer.layers[output_name]

    # Simulate adding a polygon via _ensure_features_columns
    from napari_sam4is._widget import _ATTR_DEFAULTS

    widget._ensure_features_columns(output_layer)

    # Check columns exist
    for col in (
        "class",
        "unclear",
        "uncertain",
        "review_status",
        "reviewed_at",
    ):
        assert col in output_layer.features.columns

    # Set defaults and add a shape manually
    output_layer.feature_defaults["class"] = "0: object"
    for key, val in _ATTR_DEFAULTS.items():
        if key != "class":
            output_layer.feature_defaults[key] = val

    output_layer.add_polygons([_make_polygon()], edge_width=2)

    row = output_layer.features.iloc[0]
    assert row["unclear"] is False or row["unclear"] == 0
    assert row["review_status"] == "unreviewed"
    assert row["reviewed_at"] == ""


def test_features_none_mixed_in():
    """Test save/load with None mixed in features."""
    image = np.zeros((100, 100), dtype=np.uint8)
    polygon = _make_polygon()

    # Simulate _save() conversion: "" → None
    attrs = [
        {
            "unclear": False,
            "uncertain": False,
            "review_status": "unreviewed",
            "reviewed_at": None,
        },
        {
            "unclear": True,
            "uncertain": True,
            "review_status": "approved",
            "reviewed_at": "2026-02-13T10:00:00+09:00",
        },
    ]
    polygon2 = np.array([[30, 30], [30, 40], [40, 40], [40, 30]])
    coco = create_json(
        image,
        "test.png",
        [polygon, polygon2],
        attributes_list=attrs,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(coco, f)
        tmp_path = f.name

    try:
        result = load_json(tmp_path)
        loaded = result["annotations"]
        assert len(loaded) == 2
        # First: None → ""
        assert loaded[0]["attributes"]["reviewed_at"] == ""
        # Second: timestamp preserved
        assert (
            loaded[1]["attributes"]["reviewed_at"]
            == "2026-02-13T10:00:00+09:00"
        )
    finally:
        os.unlink(tmp_path)
