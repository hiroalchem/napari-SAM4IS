"""Hide the outlines of individual annotations in the output Shapes layer.

napari's Shapes layer can only show or hide a whole layer, so a hidden
annotation is drawn with a fully transparent edge. Only the alpha
channel changes, so showing it again restores its original color. The
annotation itself, its class and its attributes are not changed, and it
is still saved.

- ``Alt-H``: hide the selected annotations' outlines, or show them again
  if they are all hidden already
- ``Alt-Shift-H``: show all hidden outlines in the layer
"""

import contextlib

import numpy as np
from qtpy.QtWidgets import QHBoxLayout, QPushButton

_KEYS = ("Alt-H", "Alt-Shift-H")


def hidden_indices(layer):
    """Indices of annotations whose outline is hidden."""
    if len(layer.data) == 0:
        return []
    alpha = np.asarray(layer.edge_color)[:, 3]
    return [int(i) for i in np.flatnonzero(alpha == 0)]


def set_outline_visible(layer, indices, visible):
    """Show or hide the outlines of ``indices`` (alpha 1 or 0)."""
    indices = [i for i in indices if i < len(layer.data)]
    if not indices:
        return
    colors = np.array(layer.edge_color, dtype=float)
    colors[indices, 3] = 1.0 if visible else 0.0
    # napari copies the edge color of the selection into
    # current_edge_color; keep it so new annotations stay visible
    current = layer.current_edge_color
    layer.edge_color = colors
    if not layer.selected_data:
        layer.current_edge_color = current


class OutlineVisibility:
    """Per-annotation outline hiding on one Shapes layer."""

    def __init__(self, viewer):
        self._viewer = viewer
        self._layer = None
        self._size_before_add = None
        self.on_change = None

    @property
    def layer(self):
        return self._layer

    def connect(self, layer):
        """Bind the keys and watch ``layer`` for added annotations."""
        self.disconnect()
        layer.bind_key(
            "Alt-H", lambda _: self.toggle_selected(), overwrite=True
        )
        layer.bind_key(
            "Alt-Shift-H", lambda _: self.show_all(), overwrite=True
        )
        layer.events.data.connect(self._on_data)
        self._layer = layer
        self._changed()

    def disconnect(self):
        if self._layer is None:
            return
        for key in _KEYS:
            with contextlib.suppress(TypeError, RuntimeError, KeyError):
                self._layer.bind_key(key, None)
        with contextlib.suppress(TypeError, RuntimeError):
            self._layer.events.data.disconnect(self._on_data)
        self._layer = None
        self._changed()

    # --- actions ---

    def toggle_selected(self):
        """Hide the selection, or show it if it is all hidden already."""
        layer = self._layer
        if layer is None:
            return
        selected = sorted(layer.selected_data)
        if not selected:
            self._set_status("Select an annotation first")
            return
        hidden = set(hidden_indices(layer))
        visible = all(i in hidden for i in selected)
        set_outline_visible(layer, selected, visible)
        self._changed()

    def show_all(self):
        """Show every hidden outline in the layer."""
        layer = self._layer
        if layer is None:
            return
        hidden = hidden_indices(layer)
        if not hidden:
            self._set_status("No hidden outlines")
            return
        set_outline_visible(layer, hidden, True)
        self._changed()

    # --- helpers ---

    def _on_data(self, event):
        """Make annotations added while a hidden one was selected visible.

        A new shape takes current_edge_color, which napari copies from
        the selection, so it can come out transparent.
        """
        action = getattr(event, "action", None)
        if action == "adding":
            self._size_before_add = len(self._layer.data)
        elif action == "added" and self._size_before_add is not None:
            new = range(self._size_before_add, len(self._layer.data))
            self._size_before_add = None
            if any(i in set(hidden_indices(self._layer)) for i in new):
                set_outline_visible(self._layer, list(new), True)
            self._changed()
        elif action == "removed":
            self._changed()

    def _changed(self):
        if self._layer is not None:
            count = len(hidden_indices(self._layer))
            self._set_status(f"Hidden outlines: {count}")
        if self.on_change is not None:
            self.on_change()

    def _set_status(self, text):
        self._viewer.status = text


def make_visibility_buttons(visibility):
    """Row with the hide / show buttons for ``visibility``."""
    row = QHBoxLayout()
    toggle_btn = QPushButton("Hide/Show Selected (Alt+H)")
    toggle_btn.setToolTip(
        "選択した shape の輪郭線を非表示にする\n"
        "（すべて非表示の shape を選ぶと再表示）\n"
        "表示だけの変更で、保存されるアノテーションは変わらない"
    )
    toggle_btn.clicked.connect(visibility.toggle_selected)
    row.addWidget(toggle_btn)

    show_btn = QPushButton("Show All (Alt+Shift+H)")
    show_btn.setToolTip("非表示にした輪郭線をすべて再表示する")
    show_btn.clicked.connect(visibility.show_all)
    row.addWidget(show_btn)

    def _update():
        layer = visibility.layer
        has_layer = layer is not None
        toggle_btn.setEnabled(has_layer)
        count = len(hidden_indices(layer)) if has_layer else 0
        show_btn.setEnabled(count > 0)
        show_btn.setText(
            f"Show All ({count}) (Alt+Shift+H)"
            if count
            else "Show All (Alt+Shift+H)"
        )

    visibility.on_change = _update
    _update()
    return row
