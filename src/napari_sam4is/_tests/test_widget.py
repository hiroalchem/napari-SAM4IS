import json
import os
import sys
import tempfile

import numpy as np
import pytest
import yaml

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
    assert widget._class_list_widget.topLevelItemCount() == 1
    assert widget._class_list_widget.topLevelItem(0).text(0) == "0: cat"

    widget._class_name_input.setText("dog")
    widget._add_class()
    assert widget._class_list_widget.topLevelItemCount() == 2
    assert widget._class_list_widget.topLevelItem(1).text(0) == "1: dog"

    # Duplicate should not be added
    widget._class_name_input.setText("cat")
    widget._add_class()
    assert widget._class_list_widget.topLevelItemCount() == 2

    # Delete class (not in use, should succeed)
    widget._class_list_widget.setCurrentItem(
        widget._class_list_widget.topLevelItem(0)
    )
    widget._del_class()
    assert widget._class_list_widget.topLevelItemCount() == 1
    assert widget._class_list_widget.topLevelItem(0).text(0) == "1: dog"


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
    """Test default class auto-creation when tree is empty."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    # Tree is empty, should auto-create "0: object"
    result = widget._get_selected_class()
    assert result == "0: object"
    assert widget._class_list_widget.topLevelItemCount() == 1


def test_subclass_management(make_napari_viewer):
    """Test adding subclasses and hierarchical class strings."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    # Add top-level class
    widget._class_name_input.setText("Animal")
    widget._add_class()
    assert widget._class_list_widget.topLevelItemCount() == 1

    # Select the top-level class and add subclass
    parent_item = widget._class_list_widget.topLevelItem(0)
    widget._class_list_widget.setCurrentItem(parent_item)
    widget._class_name_input.setText("Cat")
    widget._add_subclass()
    assert parent_item.childCount() == 1
    assert parent_item.child(0).text(0) == "1: Cat"

    # Add sub-subclass
    cat_item = parent_item.child(0)
    widget._class_list_widget.setCurrentItem(cat_item)
    widget._class_name_input.setText("Persian")
    widget._add_subclass()
    assert cat_item.childCount() == 1

    # Verify full path
    persian_item = cat_item.child(0)
    class_str = widget._get_item_class_string(persian_item)
    assert class_str == "2: Animal-Cat-Persian"

    # Verify selected class returns full path
    widget._class_list_widget.setCurrentItem(persian_item)
    assert widget._get_selected_class() == "2: Animal-Cat-Persian"


def test_subclass_categories_hierarchy(make_napari_viewer):
    """Test COCO categories include hierarchy via supercategory."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    widget._class_name_input.setText("Animal")
    widget._add_class()
    parent_item = widget._class_list_widget.topLevelItem(0)
    widget._class_list_widget.setCurrentItem(parent_item)
    widget._class_name_input.setText("Cat")
    widget._add_subclass()

    categories = widget._build_categories_list()
    assert len(categories) == 2
    # Top-level has empty supercategory
    assert categories[0]["name"] == "Animal"
    assert categories[0]["supercategory"] == ""
    # Subclass has parent as supercategory
    assert categories[1]["name"] == "Animal-Cat"
    assert categories[1]["supercategory"] == "Animal"


def test_separator_in_class_name_rejected(make_napari_viewer):
    """Test that class names containing the separator are rejected."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    widget._class_name_input.setText("my-class")
    widget._add_class()
    assert widget._class_list_widget.topLevelItemCount() == 0


def test_del_class_with_children_blocked(make_napari_viewer):
    """Test that deleting a class with subclasses is blocked."""
    from unittest.mock import patch

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    widget._class_name_input.setText("Animal")
    widget._add_class()
    parent_item = widget._class_list_widget.topLevelItem(0)
    widget._class_list_widget.setCurrentItem(parent_item)
    widget._class_name_input.setText("Cat")
    widget._add_subclass()

    # Try to delete parent — should be blocked and show a warning dialog
    widget._class_list_widget.setCurrentItem(parent_item)
    with patch("napari_sam4is._widget.QMessageBox.warning") as mock_warning:
        widget._del_class()
        mock_warning.assert_called_once()

    # Parent and child must still be present
    assert widget._class_list_widget.topLevelItemCount() == 1
    assert widget._class_list_widget.topLevelItem(0).childCount() == 1


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


def test_load_json_multi_image_rejected():
    """Test multi-image COCO files are rejected."""
    coco = {
        "images": [
            {"file_name": "a.png", "height": 100, "width": 100, "id": 0},
            {"file_name": "b.png", "height": 100, "width": 100, "id": 1},
        ],
        "annotations": [],
        "categories": [],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(coco, f)
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="Multi-image"):
            load_json(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_load_json_odd_coords_skipped():
    """Test polygon with odd coordinate count is skipped."""
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
                "segmentation": [[10, 20, 30, 40, 50, 60, 70]],
                "area": 100,
                "bbox": [10, 10, 10, 10],
                "iscrowd": 0,
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


def test_yaml_unicode_roundtrip():
    """Test that Japanese class names survive YAML serialization."""
    class_data = {"names": {0: "猫", 1: "犬", 2: "自転車"}}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(
            class_data,
            f,
            default_flow_style=False,
            allow_unicode=True,
        )
        tmp_path = f.name

    try:
        with open(tmp_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["names"][0] == "猫"
        assert loaded["names"][1] == "犬"
        assert loaded["names"][2] == "自転車"

        # Verify raw file content contains actual characters, not escapes
        with open(tmp_path, encoding="utf-8") as f:
            raw = f.read()
        assert "猫" in raw
        assert "\\u" not in raw
    finally:
        os.unlink(tmp_path)


# --- Display Settings Tests ---


def test_load_settings_defaults(tmp_path, monkeypatch):
    """_load_settings returns defaults when no settings.json exists."""
    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )
    from napari_sam4is._widget import _SETTINGS_DEFAULTS, _load_settings

    settings = _load_settings()
    assert settings == _SETTINGS_DEFAULTS


def test_load_settings_from_file(tmp_path, monkeypatch):
    """_load_settings reads and sanitizes an existing settings.json."""
    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "accepted_edge_color": "#ff0000",
                "text_color": "#0000ff",
                "text_size": 20,
            }
        ),
        encoding="utf-8",
    )

    from napari_sam4is._widget import _load_settings

    settings = _load_settings()
    assert settings["accepted_edge_color"] == "#ff0000"
    assert settings["text_color"] == "#0000ff"
    assert settings["text_size"] == 20


def test_load_settings_invalid_color_falls_back(tmp_path, monkeypatch):
    """Invalid color strings fall back to defaults."""
    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )
    (tmp_path / "settings.json").write_text(
        json.dumps({"accepted_edge_color": "not-a-color", "text_size": 20}),
        encoding="utf-8",
    )

    from napari_sam4is._widget import _SETTINGS_DEFAULTS, _load_settings

    settings = _load_settings()
    assert (
        settings["accepted_edge_color"]
        == _SETTINGS_DEFAULTS["accepted_edge_color"]
    )
    assert settings["text_size"] == 20


def test_load_settings_invalid_size_falls_back(tmp_path, monkeypatch):
    """Non-integer text_size falls back to default."""
    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )
    (tmp_path / "settings.json").write_text(
        json.dumps({"text_size": "large"}), encoding="utf-8"
    )

    from napari_sam4is._widget import _SETTINGS_DEFAULTS, _load_settings

    settings = _load_settings()
    assert settings["text_size"] == _SETTINGS_DEFAULTS["text_size"]


def test_load_settings_clamps_size(tmp_path, monkeypatch):
    """text_size is clamped to [_TEXT_SIZE_MIN, _TEXT_SIZE_MAX]."""
    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )
    from napari_sam4is._widget import (
        _TEXT_SIZE_MAX,
        _TEXT_SIZE_MIN,
        _load_settings,
    )

    path = tmp_path / "settings.json"

    path.write_text(json.dumps({"text_size": 9999}), encoding="utf-8")
    assert _load_settings()["text_size"] == _TEXT_SIZE_MAX

    path.write_text(json.dumps({"text_size": -5}), encoding="utf-8")
    assert _load_settings()["text_size"] == _TEXT_SIZE_MIN


def test_load_settings_malformed_json_falls_back(tmp_path, monkeypatch):
    """Malformed JSON falls back to defaults without raising."""
    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )
    (tmp_path / "settings.json").write_text("not json {{{", encoding="utf-8")

    from napari_sam4is._widget import _SETTINGS_DEFAULTS, _load_settings

    assert _load_settings() == _SETTINGS_DEFAULTS


def test_load_settings_non_dict_json_falls_back(tmp_path, monkeypatch):
    """Non-dict JSON (e.g. list) falls back to defaults."""
    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )
    (tmp_path / "settings.json").write_text(
        json.dumps([1, 2, 3]), encoding="utf-8"
    )

    from napari_sam4is._widget import _SETTINGS_DEFAULTS, _load_settings

    assert _load_settings() == _SETTINGS_DEFAULTS


def test_save_settings_writes_file(tmp_path, monkeypatch):
    """_save_settings writes valid JSON to the config path."""
    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )

    from napari_sam4is._widget import _save_settings

    _save_settings(
        {
            "accepted_edge_color": "#123456",
            "text_color": "#abcdef",
            "text_size": 15,
        }
    )

    written = json.loads(
        (tmp_path / "settings.json").read_text(encoding="utf-8")
    )
    assert written["accepted_edge_color"] == "#123456"
    assert written["text_size"] == 15


def test_save_settings_oserror_warns_once(tmp_path, monkeypatch):
    """_save_settings fails gracefully when write raises OSError."""
    from unittest.mock import patch

    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )

    import napari_sam4is._widget as wmod

    wmod._SETTINGS_SAVE_WARNED = False

    with patch(
        "napari_sam4is._widget.Path.write_text",
        side_effect=OSError("disk full"),
    ):
        # Should not raise
        wmod._save_settings(
            {
                "accepted_edge_color": "#ff0000",
                "text_color": "#ff0000",
                "text_size": 12,
            }
        )
    assert wmod._SETTINGS_SAVE_WARNED is True

    # Second call should not warn again (warned flag already set)
    warned_before = wmod._SETTINGS_SAVE_WARNED
    with patch(
        "napari_sam4is._widget.Path.write_text",
        side_effect=OSError("disk full"),
    ):
        wmod._save_settings(
            {
                "accepted_edge_color": "#ff0000",
                "text_color": "#ff0000",
                "text_size": 12,
            }
        )
    assert warned_before == wmod._SETTINGS_SAVE_WARNED


def test_widget_accepted_layer_edge_color_from_settings(
    make_napari_viewer, tmp_path, monkeypatch
):
    """Accepted layer current_edge_color is initialized from loaded settings."""
    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "accepted_edge_color": "#ff0000",
                "text_color": "#ffffff",
                "text_size": 10,
            }
        ),
        encoding="utf-8",
    )

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    assert widget._settings["accepted_edge_color"] == "#ff0000"
    # Verify edge_color_btn stylesheet reflects the setting
    style = widget._edge_color_btn.styleSheet()
    assert "#ff0000" in style


def test_widget_text_size_spinbox_from_settings(
    make_napari_viewer, tmp_path, monkeypatch
):
    """Text size spinbox reflects the loaded setting on startup."""
    monkeypatch.setattr(
        "napari_sam4is._widget.platformdirs.user_config_dir",
        lambda _: str(tmp_path),
    )
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "accepted_edge_color": "#ffff00",
                "text_color": "#ffff00",
                "text_size": 24,
            }
        ),
        encoding="utf-8",
    )

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)

    assert widget._text_size_spin.value() == 24


def test_json_unicode_roundtrip():
    """Test that non-ASCII category names persist in COCO JSON."""
    image = np.zeros((100, 100), dtype=np.uint8)
    polygon = _make_polygon()
    categories = [
        {"id": 0, "name": "猫", "supercategory": "動物"},
        {"id": 1, "name": "犬", "supercategory": "動物"},
    ]

    coco = create_json(
        image,
        "test.png",
        [polygon],
        categories=categories,
        category_ids=[0],
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(coco, f, ensure_ascii=False)
        tmp_path = f.name

    try:
        with open(tmp_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["categories"][0]["name"] == "猫"
        assert loaded["categories"][1]["name"] == "犬"
        assert loaded["categories"][0]["supercategory"] == "動物"

        # Verify raw file contains actual Unicode, not \uXXXX escapes
        with open(tmp_path, encoding="utf-8") as f:
            raw = f.read()
        assert "猫" in raw
        assert "\\u" not in raw
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# SAM3 tests
# ---------------------------------------------------------------------------


def test_model_selection_includes_sam3(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    items = [
        widget._model_selection.itemText(i)
        for i in range(widget._model_selection.count())
    ]
    assert "sam3" in items


def test_load_sam3_model_import_error(make_napari_viewer):
    from unittest.mock import patch

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    with (
        patch(
            "napari_sam4is._utils.load_sam3_model",
            side_effect=ImportError("sam3 not installed"),
        ),
        patch("napari_sam4is._widget.QMessageBox") as mock_mb,
    ):
        widget._model_selection.setCurrentText("sam3")
        widget._load_model()
        assert widget._sam3_processor is None
        mock_mb.critical.assert_called_once()


def test_pixel_box_to_sam3_norm_cxcywh(make_napari_viewer):
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock, patch

    mock_box_ops = MagicMock()
    mock_viz_utils = MagicMock()

    def _xyxy_to_cxcywh(t):
        x1, y1, x2, y2 = t[0].tolist()
        return torch.tensor([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]])

    def _norm(t, W, H):
        return t / torch.tensor([W, H, W, H], dtype=torch.float32)

    mock_box_ops.box_xyxy_to_cxcywh = _xyxy_to_cxcywh
    mock_viz_utils.normalize_bbox = _norm
    mock_model = MagicMock()
    mock_model.box_ops = mock_box_ops
    with patch.dict(
        sys.modules,
        {
            "sam3": MagicMock(),
            "sam3.model": mock_model,
            "sam3.model.box_ops": mock_box_ops,
            "sam3.visualization_utils": mock_viz_utils,
        },
    ):
        viewer = make_napari_viewer()
        viewer.add_image(np.random.random((100, 200)))  # H=100, W=200
        widget = SAMWidget(viewer)
        widget._sam3_processor = MagicMock()
        # [x1,y1,x2,y2]=[10,20,110,70]
        # → cx=60,cy=45,w=100,h=50 → norm by W=200,H=100: [0.3, 0.45, 0.5, 0.5]
        result = widget._pixel_box_to_sam3_norm_cxcywh(
            np.array([10.0, 20.0, 110.0, 70.0])
        )
        assert len(result) == 4
        assert abs(result[0] - 0.3) < 1e-4
        assert abs(result[1] - 0.45) < 1e-4


def test_accept_multi_masks(make_napari_viewer):
    torch = pytest.importorskip("torch")
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    widget._radio_btn_shape.setChecked(True)
    masks = torch.zeros(2, 1, 100, 100)
    masks[0, 0, 10:30, 10:30] = 1
    masks[1, 0, 50:70, 50:70] = 1
    n = widget._accept_multi_masks(masks, "0: dog")
    assert n == 2
    assert len(widget._accepted_layer.data) == 2


def test_detect_all_text_mode(make_napari_viewer):
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    mock_proc = MagicMock()
    fake_masks = torch.zeros(1, 1, 100, 100)
    fake_masks[0, 0, 10:30, 10:30] = 1
    mock_proc.set_text_prompt.return_value = {
        "masks": fake_masks,
        "boxes": None,
        "scores": None,
    }
    widget._sam3_processor = mock_proc
    widget._sam3_inference_state = {}
    widget._sam3_model = MagicMock()
    widget._radio_btn_shape.setChecked(True)
    widget._sam3_prompt_text_radio.setChecked(True)
    widget._on_sam3_detect_all()
    mock_proc.set_text_prompt.assert_called_once()
    mock_proc.add_geometric_prompt.assert_not_called()
    assert len(widget._accepted_layer.data) == 1


def test_detect_all_box_mode(make_napari_viewer):
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock, patch

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    fake_masks = torch.zeros(1, 1, 100, 100)
    fake_masks[0, 0, 10:30, 10:30] = 1
    geo_result = {"pred_masks": fake_masks}
    mock_proc = MagicMock()
    mock_proc.add_geometric_prompt.return_value = geo_result
    widget._sam3_processor = mock_proc
    widget._sam3_inference_state = {}
    widget._sam3_model = MagicMock()
    widget._input_box = np.array([10.0, 10.0, 50.0, 50.0])
    widget._radio_btn_shape.setChecked(True)
    widget._sam3_prompt_box_radio.setChecked(True)
    with patch.dict(
        sys.modules,
        {
            "sam3": MagicMock(),
            "sam3.model.box_ops": MagicMock(),
            "sam3.visualization_utils": MagicMock(),
        },
    ):
        widget._on_sam3_detect_all()
    mock_proc.add_geometric_prompt.assert_called_once()
    mock_proc.set_text_prompt.assert_not_called()
    assert len(widget._accepted_layer.data) == 1


def test_detect_all_text_box_mode(make_napari_viewer):
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock, patch

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    mock_proc = MagicMock()
    fake_masks = torch.zeros(1, 1, 100, 100)
    fake_masks[0, 0, 10:30, 10:30] = 1
    mock_proc.set_text_prompt.return_value = {
        "masks": fake_masks,
        "boxes": None,
        "scores": None,
    }
    mock_proc.add_geometric_prompt.return_value = {}
    widget._sam3_processor = mock_proc
    widget._sam3_inference_state = {}
    widget._sam3_model = MagicMock()
    widget._input_box = np.array([10.0, 10.0, 50.0, 50.0])
    widget._radio_btn_shape.setChecked(True)
    widget._sam3_prompt_both_radio.setChecked(True)
    with patch.dict(
        sys.modules,
        {
            "sam3": MagicMock(),
            "sam3.model.box_ops": MagicMock(),
            "sam3.visualization_utils": MagicMock(),
        },
    ):
        widget._on_sam3_detect_all()
    mock_proc.add_geometric_prompt.assert_called_once()
    mock_proc.set_text_prompt.assert_called_once()
    assert len(widget._accepted_layer.data) == 1


def test_send_selected_to_predict(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    polygon = np.array([[10, 10], [10, 40], [40, 40], [40, 10]])
    widget._accepted_layer.add_polygons([polygon])
    widget._accepted_layer.selected_data = {0}
    widget._shapes_layer_selection.setCurrentText("Accepted")
    widget._send_selected_to_predict()
    assert len(widget._accepted_layer.data) == 0
    assert widget._labels_layer.data.any()


def test_detect_all_box_mode_key_error(make_napari_viewer):
    from unittest.mock import MagicMock, patch

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    mock_proc = MagicMock()
    mock_proc.add_geometric_prompt.return_value = {}
    widget._sam3_processor = mock_proc
    widget._sam3_inference_state = {}
    widget._sam3_model = MagicMock()
    widget._input_box = np.array([10.0, 10.0, 50.0, 50.0])
    widget._sam3_prompt_box_radio.setChecked(True)
    with (
        patch("napari_sam4is._widget.QMessageBox") as mock_mb,
        patch.dict(
            sys.modules,
            {
                "sam3": MagicMock(),
                "sam3.model": MagicMock(),
                "sam3.model.box_ops": MagicMock(),
                "sam3.utils": MagicMock(),
                "sam3.utils.misc": MagicMock(),
            },
        ),
    ):
        widget._on_sam3_detect_all()
    mock_mb.critical.assert_called_once()
    assert len(widget._accepted_layer.data) == 0


def test_set_attr_ui_disabled_disables_buttons(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    widget._send_to_predict_btn.setEnabled(True)
    widget._set_attr_ui_disabled()
    assert not widget._send_to_predict_btn.isEnabled()


def test_detect_all_box_mode_with_exemplar(make_napari_viewer):
    """Exemplar boxes from selected output shapes are used."""
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock, patch

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    fake_masks = torch.zeros(1, 1, 100, 100)
    fake_masks[0, 0, 10:30, 10:30] = 1
    geo_result = {"pred_masks": fake_masks}
    mock_proc = MagicMock()
    mock_proc.add_geometric_prompt.return_value = geo_result
    widget._sam3_processor = mock_proc
    widget._sam3_inference_state = {}
    widget._sam3_model = MagicMock()
    widget._input_box = None
    widget._radio_btn_shape.setChecked(True)
    widget._sam3_prompt_box_radio.setChecked(True)

    # Add a polygon to accepted layer and select it
    polygon = np.array([[10, 10], [10, 50], [30, 50], [30, 10]])
    widget._accepted_layer.add_polygons([polygon])
    widget._ensure_features_columns(widget._accepted_layer)
    widget._accepted_layer.features.at[0, "class"] = "1: cell"
    widget._accepted_layer.selected_data = {0}
    widget._shapes_layer_selection.setCurrentText("Accepted")
    # Make accepted layer active
    viewer.layers.selection.active = widget._accepted_layer

    with patch.dict(
        sys.modules,
        {
            "sam3": MagicMock(),
            "sam3.model.box_ops": MagicMock(),
            "sam3.visualization_utils": MagicMock(),
        },
    ):
        widget._on_sam3_detect_all()
    mock_proc.add_geometric_prompt.assert_called_once()
    mock_proc.set_text_prompt.assert_not_called()


def test_detect_all_box_mode_combined(make_napari_viewer):
    """Both _input_box and exemplar shapes are used as box prompts."""
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock, patch

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    fake_masks = torch.zeros(1, 1, 100, 100)
    fake_masks[0, 0, 10:30, 10:30] = 1
    geo_result = {"pred_masks": fake_masks}
    mock_proc = MagicMock()
    mock_proc.add_geometric_prompt.return_value = geo_result
    widget._sam3_processor = mock_proc
    widget._sam3_inference_state = {}
    widget._sam3_model = MagicMock()
    widget._input_box = np.array([5.0, 5.0, 40.0, 40.0])
    widget._radio_btn_shape.setChecked(True)
    widget._sam3_prompt_box_radio.setChecked(True)

    polygon = np.array([[10, 10], [10, 50], [30, 50], [30, 10]])
    widget._accepted_layer.add_polygons([polygon])
    widget._ensure_features_columns(widget._accepted_layer)
    widget._accepted_layer.features.at[0, "class"] = "1: cell"
    widget._accepted_layer.selected_data = {0}
    widget._shapes_layer_selection.setCurrentText("Accepted")
    viewer.layers.selection.active = widget._accepted_layer

    with patch.dict(
        sys.modules,
        {
            "sam3": MagicMock(),
            "sam3.model.box_ops": MagicMock(),
            "sam3.visualization_utils": MagicMock(),
        },
    ):
        widget._on_sam3_detect_all()
    # 2 calls: one for _input_box, one for exemplar
    assert mock_proc.add_geometric_prompt.call_count == 2


def test_detect_all_ignores_exemplar_when_not_active(
    make_napari_viewer,
):
    """Exemplar shapes are ignored when output layer is not active."""
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock, patch

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    fake_masks = torch.zeros(1, 1, 100, 100)
    fake_masks[0, 0, 10:30, 10:30] = 1
    geo_result = {"pred_masks": fake_masks}
    mock_proc = MagicMock()
    mock_proc.add_geometric_prompt.return_value = geo_result
    widget._sam3_processor = mock_proc
    widget._sam3_inference_state = {}
    widget._sam3_model = MagicMock()
    widget._input_box = np.array([5.0, 5.0, 40.0, 40.0])
    widget._radio_btn_shape.setChecked(True)
    widget._sam3_prompt_box_radio.setChecked(True)

    polygon = np.array([[10, 10], [10, 50], [30, 50], [30, 10]])
    widget._accepted_layer.add_polygons([polygon])
    widget._accepted_layer.selected_data = {0}
    widget._shapes_layer_selection.setCurrentText("Accepted")
    # Keep a different layer active (not Accepted)
    viewer.layers.selection.active = widget._sam_box_layer

    with patch.dict(
        sys.modules,
        {
            "sam3": MagicMock(),
            "sam3.model.box_ops": MagicMock(),
            "sam3.visualization_utils": MagicMock(),
        },
    ):
        widget._on_sam3_detect_all()
    # Only _input_box used, not exemplar
    assert mock_proc.add_geometric_prompt.call_count == 1


def test_detect_all_no_box_no_exemplar(make_napari_viewer, capsys):
    """Error message when no box source is available."""
    pytest.importorskip("torch")
    from unittest.mock import MagicMock

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    mock_proc = MagicMock()
    widget._sam3_processor = mock_proc
    widget._sam3_inference_state = {}
    widget._sam3_model = MagicMock()
    widget._input_box = None
    widget._sam3_prompt_box_radio.setChecked(True)
    widget._on_sam3_detect_all()
    captured = capsys.readouterr()
    assert "SAM-Box" in captured.out


def test_detect_all_exemplar_mixed_classes(make_napari_viewer, capsys):
    """Mixed-class exemplars produce an error message."""
    pytest.importorskip("torch")
    from unittest.mock import MagicMock

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    mock_proc = MagicMock()
    widget._sam3_processor = mock_proc
    widget._sam3_inference_state = {}
    widget._sam3_model = MagicMock()
    widget._input_box = None
    widget._radio_btn_shape.setChecked(True)
    widget._sam3_prompt_box_radio.setChecked(True)

    p1 = np.array([[10, 10], [10, 30], [30, 30], [30, 10]])
    p2 = np.array([[40, 40], [40, 60], [60, 60], [60, 40]])
    widget._accepted_layer.add_polygons([p1, p2])
    widget._ensure_features_columns(widget._accepted_layer)
    widget._accepted_layer.features.at[0, "class"] = "1: cell"
    widget._accepted_layer.features.at[1, "class"] = "2: bg"
    widget._accepted_layer.selected_data = {0, 1}
    widget._shapes_layer_selection.setCurrentText("Accepted")
    viewer.layers.selection.active = widget._accepted_layer

    widget._on_sam3_detect_all()
    captured = capsys.readouterr()
    assert "混在" in captured.out
    # No prompts should have been added after reset
    mock_proc.add_geometric_prompt.assert_not_called()


def test_accept_multi_masks_iou_filtering(make_napari_viewer):
    """Masks overlapping existing shapes above IoU threshold are skipped."""
    torch = pytest.importorskip("torch")
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    widget._radio_btn_shape.setChecked(True)
    widget._iou_threshold_spin.setValue(0.5)
    widget._iou_same_class_checkbox.setChecked(False)

    # Add existing polygon covering [10:30, 10:30]
    polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]])
    widget._accepted_layer.add_polygons([polygon])
    widget._ensure_features_columns(widget._accepted_layer)
    widget._accepted_layer.features.at[0, "class"] = "1: cell"

    # Mask that overlaps significantly with existing
    m1 = torch.zeros(1, 1, 100, 100)
    m1[0, 0, 10:30, 10:30] = 1  # same region
    # Mask that doesn't overlap
    m2 = torch.zeros(1, 1, 100, 100)
    m2[0, 0, 60:80, 60:80] = 1
    masks = torch.cat([m1, m2], dim=0)

    n = widget._accept_multi_masks(masks, "1: cell")
    assert n == 1  # only m2 accepted


def test_accept_multi_masks_iou_same_class_only(make_napari_viewer):
    """Same-class-only IoU skips only same-class overlaps."""
    torch = pytest.importorskip("torch")
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    widget._radio_btn_shape.setChecked(True)
    widget._iou_threshold_spin.setValue(0.5)
    widget._iou_same_class_checkbox.setChecked(True)

    # Existing polygon with different class
    polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]])
    widget._accepted_layer.add_polygons([polygon])
    widget._ensure_features_columns(widget._accepted_layer)
    widget._accepted_layer.features.at[0, "class"] = "2: bg"

    # Mask that overlaps existing but is a different class
    m1 = torch.zeros(1, 1, 100, 100)
    m1[0, 0, 10:30, 10:30] = 1

    n = widget._accept_multi_masks(m1, "1: cell")
    # Should be accepted because existing is class "2: bg"
    assert n == 1


def test_accept_multi_masks_iou_all_classes(make_napari_viewer):
    """Cross-class IoU skips overlaps regardless of class."""
    torch = pytest.importorskip("torch")
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    widget._radio_btn_shape.setChecked(True)
    widget._iou_threshold_spin.setValue(0.5)
    widget._iou_same_class_checkbox.setChecked(False)

    polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]])
    widget._accepted_layer.add_polygons([polygon])
    widget._ensure_features_columns(widget._accepted_layer)
    widget._accepted_layer.features.at[0, "class"] = "2: bg"

    m1 = torch.zeros(1, 1, 100, 100)
    m1[0, 0, 10:30, 10:30] = 1

    n = widget._accept_multi_masks(m1, "1: cell")
    # Should be skipped because IoU is high regardless of class
    assert n == 0


def test_accept_multi_masks_iou_zero(make_napari_viewer):
    """IoU threshold of 0 disables filtering."""
    torch = pytest.importorskip("torch")
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    widget = SAMWidget(viewer)
    widget._radio_btn_shape.setChecked(True)
    widget._iou_threshold_spin.setValue(0.0)

    polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]])
    widget._accepted_layer.add_polygons([polygon])
    widget._ensure_features_columns(widget._accepted_layer)
    widget._accepted_layer.features.at[0, "class"] = "1: cell"

    m1 = torch.zeros(1, 1, 100, 100)
    m1[0, 0, 10:30, 10:30] = 1

    n = widget._accept_multi_masks(m1, "1: cell")
    # No filtering, so should be accepted
    assert n == 1
