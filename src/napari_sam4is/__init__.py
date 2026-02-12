# Monkey patch for napari 0.6.6 Selection.replace_selection bug
# napari 0.6.6's Shapes.selected_data setter calls replace_selection,
# but Selection class is missing this method. We patch both:
# 1. Add replace_selection to Selection class
# 2. Replace Shapes.selected_data setter to handle missing method
#
# Note: This patch is applied conditionally based on the presence of the
# replace_selection method, making it compatible with both affected versions
# (e.g., 0.6.6) and future versions where the bug may be fixed.
from collections.abc import Collection

from napari.layers.shapes.shapes import Shapes
from napari.utils.events.containers import Selection

# Add replace_selection to Selection class if missing (version guard)
if not hasattr(Selection, "replace_selection"):

    def _replace_selection(self, items):
        """Replace current selection with items."""
        self.clear()
        self.update(items)

    Selection.replace_selection = _replace_selection


def _patched_selected_data_setter(
    self, selected_data: Collection[int]
) -> None:
    """Patched version that handles missing replace_selection gracefully.

    This reimplements napari's Shapes.selected_data setter logic with a
    fallback for missing Selection.replace_selection method. We copy the
    entire setter implementation rather than wrapping the original because
    the original would fail before we could intercept the error.
    """
    # Use replace_selection if available, otherwise fall back
    if hasattr(self._selected_data, "replace_selection"):
        self._selected_data.replace_selection(selected_data)
    else:
        self._selected_data.clear()
        self._selected_data.update(selected_data)

    # Call the rest of the original setter logic
    # Import here to avoid circular dependencies
    import numpy as np
    from napari.layers.shapes.shapes import _unique_element

    self._selected_box = self.interaction_box(self._selected_data)

    if len(selected_data) > 0:
        selected_data_indices = list(selected_data)
        selected_face_colors = self._data_view._face_color[
            selected_data_indices
        ]
        if (
            unique_face_color := _unique_element(selected_face_colors)
        ) is not None:
            with self.block_update_properties():
                self.current_face_color = unique_face_color

        selected_edge_colors = self._data_view._edge_color[
            selected_data_indices
        ]
        if (
            unique_edge_color := _unique_element(selected_edge_colors)
        ) is not None:
            with self.block_update_properties():
                self.current_edge_color = unique_edge_color

        unique_edge_width = _unique_element(
            np.array(
                [self._data_view.shapes[i].edge_width for i in selected_data]
            )
        )
        if unique_edge_width is not None:
            with self.block_update_properties():
                self.current_edge_width = unique_edge_width

        unique_properties = {}
        for k, v in self.properties.items():
            unique_properties[k] = _unique_element(v[selected_data_indices])

        if all(p is not None for p in unique_properties.values()):
            with self.block_update_properties():
                self.current_properties = unique_properties

    self._set_highlight()


# Replace the property
Shapes.selected_data = property(
    Shapes.selected_data.fget, _patched_selected_data_setter
)

__all__ = ("SAMWidget",)


def __getattr__(name):
    if name == "SAMWidget":
        from ._widget import SAMWidget

        globals()["SAMWidget"] = SAMWidget
        return SAMWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
