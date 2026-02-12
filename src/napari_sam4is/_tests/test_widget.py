import numpy as np

from napari_sam4is import SAMWidget
from napari_sam4is._utils import (
    create_json,
    find_missing_class_number,
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
    layer = viewer.add_image(np.random.random((100, 100)))

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
