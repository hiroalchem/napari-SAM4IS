import numpy as np
import pytest
from napari.utils.key_bindings import KeyBinding

from napari_sam4is import SAMWidget
from napari_sam4is._visibility import hidden_indices


def _square(y, x, size=10):
    return np.array(
        [[y, x], [y, x + size], [y + size, x + size], [y + size, x]],
        dtype=float,
    )


@pytest.fixture
def vis_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((200, 200)))
    widget = SAMWidget(viewer)
    layer = widget._accepted_layer
    layer.add_polygons([_square(10, 10), _square(10, 100), _square(100, 10)])
    widget._shapes_layer_selection.setCurrentText(layer.name)
    widget._connect_output_layer_events(layer)
    return widget, layer


def test_keys_bound_on_output_layer(vis_widget):
    widget, layer = vis_widget
    for key in ("Alt-H", "Alt-Shift-H"):
        assert KeyBinding.from_str(key) in layer.keymap
    # the existing H (merge as hole) binding is kept
    assert KeyBinding.from_str("H") in layer.keymap
    widget._disconnect_output_layer_events()
    assert KeyBinding.from_str("Alt-H") not in layer.keymap


def test_hide_and_show_selected(vis_widget):
    widget, layer = vis_widget
    vis = widget._outline_visibility
    rgb_before = np.array(layer.edge_color)[:, :3].copy()

    layer.selected_data = {0, 2}
    vis.toggle_selected()
    assert hidden_indices(layer) == [0, 2]
    # only the alpha changes, so the color comes back unchanged
    np.testing.assert_allclose(np.array(layer.edge_color)[:, :3], rgb_before)

    vis.toggle_selected()  # all selected are hidden -> show them
    assert hidden_indices(layer) == []
    np.testing.assert_allclose(np.array(layer.edge_color)[:, 3], 1.0)


def test_toggle_mixed_selection_hides_all(vis_widget):
    widget, layer = vis_widget
    vis = widget._outline_visibility

    layer.selected_data = {0}
    vis.toggle_selected()
    layer.selected_data = {0, 1}
    vis.toggle_selected()
    assert hidden_indices(layer) == [0, 1]


def test_show_all(vis_widget):
    widget, layer = vis_widget
    vis = widget._outline_visibility

    layer.selected_data = {0, 1}
    vis.toggle_selected()
    layer.selected_data = set()
    vis.show_all()
    assert hidden_indices(layer) == []


def test_hidden_annotation_keeps_data_and_attributes(vis_widget):
    widget, layer = vis_widget
    vis = widget._outline_visibility
    widget._ensure_features_columns(layer)
    data_before = [d.copy() for d in layer.data]
    features_before = layer.features.copy()

    layer.selected_data = {1}
    vis.toggle_selected()

    assert len(layer.data) == 3
    for before, after in zip(data_before, layer.data, strict=True):
        np.testing.assert_allclose(before, after)
    assert layer.features.equals(features_before)


def test_new_annotation_is_visible_after_selecting_hidden_one(vis_widget):
    """napari copies the selection's edge color into new shapes."""
    widget, layer = vis_widget
    vis = widget._outline_visibility

    layer.selected_data = {0}
    vis.toggle_selected()
    layer.selected_data = set()
    layer.selected_data = {0}  # current_edge_color becomes transparent
    layer.selected_data = set()

    layer.add_polygons([_square(150, 150)])
    assert hidden_indices(layer) == [0]


def test_hidden_index_follows_deletion(vis_widget):
    widget, layer = vis_widget
    vis = widget._outline_visibility

    layer.selected_data = {2}
    vis.toggle_selected()
    layer.selected_data = {0}
    layer.remove_selected()
    assert hidden_indices(layer) == [1]  # the old annotation 2


def test_show_all_button_shows_count(vis_widget):
    widget, layer = vis_widget
    vis = widget._outline_visibility
    layer.selected_data = {0, 1}
    vis.toggle_selected()

    buttons = [
        w
        for w in widget.findChildren(type(widget._split_rings_btn))
        if w.text().startswith("Show All")
    ]
    assert len(buttons) == 1
    assert buttons[0].isEnabled()
    assert "(2)" in buttons[0].text()

    vis.show_all()
    assert not buttons[0].isEnabled()
