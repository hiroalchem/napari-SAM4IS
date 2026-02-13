import base64
import io
import json
import os
from datetime import datetime

import napari
import numpy as np
import pandas as pd
import requests
import torch
import yaml
from PIL import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ._utils import (
    check_image_type,
    create_json,
    find_first_missing,
    find_missing_class_number,
    get_available_model_names,
    label2polygon,
    load_json,
    load_model,
    preprocess,
)

_ATTR_COLUMNS = {
    "class": str,
    "unclear": bool,
    "uncertain": bool,
    "review_status": str,
    "reviewed_at": str,
}
_ATTR_DEFAULTS = {
    "class": "",
    "unclear": False,
    "uncertain": False,
    "review_status": "unreviewed",
    "reviewed_at": "",
}


class SAMWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer

        self._shapes_layer_selection = None
        self._labels_layer_selection = None
        self._image_type = None
        self._current_slice = None
        self._input_box = None
        self._input_point = None
        self._point_label = None
        self._label_id_to_class = {}
        self._current_target_image_name = None
        self._current_target_image_layer = None

        # self._corner = None

        self.vbox = QVBoxLayout()

        # API settings group
        self._api_group = QGroupBox("API Settings")
        self._api_layout = QVBoxLayout()

        self._use_api_checkbox = QCheckBox("Use API")
        self._use_api_checkbox.toggled.connect(self._on_api_checkbox_toggled)
        self._api_layout.addWidget(self._use_api_checkbox)

        # API URL input
        self._api_url_layout = QHBoxLayout()
        self._api_url_layout.addWidget(QLabel("API URL:"))
        self._api_url_input = QLineEdit()
        self._api_url_input.setPlaceholderText("https://your-api-endpoint.com")
        self._api_url_layout.addWidget(self._api_url_input)
        self._api_layout.addLayout(self._api_url_layout)

        # API Key input
        self._api_key_layout = QHBoxLayout()
        self._api_key_layout.addWidget(QLabel("API Key:"))
        self._api_key_input = QLineEdit()
        self._api_key_input.setPlaceholderText("Enter your API key")
        self._api_key_input.setEchoMode(QLineEdit.Password)
        self._api_key_layout.addWidget(self._api_key_input)
        self._api_layout.addLayout(self._api_key_layout)

        self._api_group.setLayout(self._api_layout)
        self.vbox.addWidget(self._api_group)

        # Initially hidden
        self._api_url_input.setVisible(False)
        self._api_key_input.setVisible(False)
        self._api_url_layout.itemAt(0).widget().setVisible(False)
        self._api_key_layout.itemAt(0).widget().setVisible(False)

        # Manual Mode
        self._manual_mode_checkbox = QCheckBox("Manual Mode")
        self._manual_mode_checkbox.toggled.connect(
            self._on_manual_mode_toggled
        )
        self.vbox.addWidget(self._manual_mode_checkbox)

        # Model selection
        self._model_selection = QComboBox()
        self._model_selection.addItems(get_available_model_names())
        self.vbox.addWidget(self._model_selection)
        self._model_load_btn = QPushButton("load model")
        self._model_load_btn.clicked.connect(self._load_model)
        self.vbox.addWidget(self._model_load_btn)
        self.vbox.addWidget(QLabel("input image layer"))
        self._image_layer_selection = QComboBox()
        self._image_layer_selection.addItems(
            [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.image.image.Image)
            ]
        )
        self._image_layer_selection.currentTextChanged.connect(
            self._on_image_layer_changed
        )
        self.vbox.addWidget(self._image_layer_selection)

        self.vbox.addWidget(QLabel("Output type (3D: labels only)"))
        self._radio_btn_group = QButtonGroup()
        self._radio_btn_shape = QRadioButton("instance (Shapes layer)")
        self._radio_btn_shape.toggled.connect(self._on_radio_btn_toggled)
        self._radio_btn_label = QRadioButton("labels (Labels layer)")
        self._radio_btn_label.toggled.connect(self._on_radio_btn_toggled)
        self._radio_btn_group.addButton(self._radio_btn_shape, 0)
        self._radio_btn_group.addButton(self._radio_btn_label, 1)
        self.vbox.addWidget(self._radio_btn_shape)
        self.vbox.addWidget(self._radio_btn_label)

        self.check_box = QCheckBox("instance labels")
        self.vbox.addWidget(self.check_box)

        self.vbox.addWidget(QLabel("output shapes layer"))
        self._shapes_layer_selection = QComboBox()
        self._shapes_layer_selection.addItems(
            [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.shapes.shapes.Shapes)
            ]
        )
        self._shapes_layer_selection.currentTextChanged.connect(
            self._on_shapes_layer_combo_changed
        )
        self.vbox.addWidget(self._shapes_layer_selection)

        self.vbox.addWidget(QLabel("output labels layer"))
        self._labels_layer_selection = QComboBox()
        self._labels_layer_selection.addItems(
            [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.labels.labels.Labels)
            ]
        )
        self.vbox.addWidget(self._labels_layer_selection)

        # Class Management group
        self._class_group = QGroupBox("Class Management")
        self._class_layout = QVBoxLayout()

        self._class_list_widget = QListWidget()
        self._class_list_widget.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self._class_list_widget.itemClicked.connect(self._on_class_clicked)
        self._class_layout.addWidget(self._class_list_widget)

        self._class_input_layout = QHBoxLayout()
        self._class_name_input = QLineEdit()
        self._class_name_input.setPlaceholderText("Enter class name")
        self._class_name_input.returnPressed.connect(self._add_class)
        self._class_input_layout.addWidget(self._class_name_input)
        self._add_class_btn = QPushButton("Add")
        self._add_class_btn.clicked.connect(self._add_class)
        self._class_input_layout.addWidget(self._add_class_btn)
        self._class_layout.addLayout(self._class_input_layout)

        self._class_btn_layout = QHBoxLayout()
        self._del_class_btn = QPushButton("Delete")
        self._del_class_btn.clicked.connect(self._del_class)
        self._class_btn_layout.addWidget(self._del_class_btn)
        self._load_class_btn = QPushButton("Load")
        self._load_class_btn.clicked.connect(self._load_classes)
        self._class_btn_layout.addWidget(self._load_class_btn)
        self._class_layout.addLayout(self._class_btn_layout)

        self._class_group.setLayout(self._class_layout)
        self.vbox.addWidget(self._class_group)

        # --- Annotation Attributes group ---
        self._attr_group = QGroupBox("Annotation Attributes")
        self._attr_layout = QVBoxLayout()

        self._unclear_checkbox = QCheckBox("Unclear boundary")
        self._unclear_checkbox.setEnabled(False)
        self._unclear_checkbox.stateChanged.connect(self._on_unclear_toggled)
        self._attr_layout.addWidget(self._unclear_checkbox)

        self._uncertain_checkbox = QCheckBox("Uncertain class")
        self._uncertain_checkbox.setEnabled(False)
        self._uncertain_checkbox.stateChanged.connect(
            self._on_uncertain_toggled
        )
        self._attr_layout.addWidget(self._uncertain_checkbox)

        self._review_btn_layout = QHBoxLayout()
        self._accept_selected_btn = QPushButton("Accept Selected")
        self._accept_selected_btn.clicked.connect(self._accept_selected)
        self._accept_selected_btn.setEnabled(False)
        self._review_btn_layout.addWidget(self._accept_selected_btn)
        self._accept_all_btn = QPushButton("Accept All")
        self._accept_all_btn.clicked.connect(self._accept_all)
        self._review_btn_layout.addWidget(self._accept_all_btn)
        self._attr_layout.addLayout(self._review_btn_layout)

        self._attr_status_label = QLabel("No annotation selected")
        self._attr_layout.addWidget(self._attr_status_label)

        self._attr_group.setLayout(self._attr_layout)
        self.vbox.addWidget(self._attr_group)

        self._updating_attr_ui = False
        self._connected_output_layer = None

        # --- Save / Load buttons ---
        self.vbox.addWidget(
            QLabel(
                "Save as coco format \nin the same directory"
                " \nwith the input image"
            )
        )
        self._save_load_layout = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._save)
        self._save_load_layout.addWidget(self._save_btn)
        self._load_annotations_btn = QPushButton("Load Annotations")
        self._load_annotations_btn.clicked.connect(
            self._on_load_annotations_clicked
        )
        self._save_load_layout.addWidget(self._load_annotations_btn)
        self.vbox.addLayout(self._save_load_layout)

        self._sam_box_layer = self._viewer.add_shapes(
            name="SAM-Box",
            edge_color="red",
            edge_width=2,
            face_color="transparent",
        )
        self._sam_box_layer.mouse_drag_callbacks.append(
            self._on_sam_box_created
        )
        self._sam_box_layer.bind_key("R", self._reject_all_boxes)
        self.lock_controls(self._sam_box_layer)
        self._sam_positive_point_layer = self._viewer.add_points(
            name="SAM-Positive", face_color="green", size=10
        )
        self._sam_negative_point_layer = self._viewer.add_points(
            name="SAM-Negative", face_color="red", size=10
        )
        # self._sam_positive_point_layer.mouse_drag_callbacks.append(self._on_sam_point_created)
        # self._sam_negative_point_layer.mouse_drag_callbacks.append(self._on_sam_point_created)
        self._sam_positive_point_layer.events.data.connect(
            self._on_sam_point_changed
        )
        self._sam_negative_point_layer.events.data.connect(
            self._on_sam_point_changed
        )

        if (self._image_layer_selection.currentText() != "") & (
            self._image_layer_selection.currentText() in self._viewer.layers
        ):
            image_layer = self._get_layer_by_name_safe(
                self._image_layer_selection.currentText()
            )
            if image_layer is not None:
                self._image_type = check_image_type(
                    self._viewer, self._image_layer_selection.currentText()
                )
                if "stack" in self._image_type:
                    shape = image_layer.data.shape[1:3]
                else:
                    shape = image_layer.data.shape[:2]
            else:
                shape = (100, 100)
        else:
            shape = (100, 100)

        self._labels_layer = self._viewer.add_labels(
            np.zeros(shape, dtype="uint8"),
            name="SAM-Predict",
            blending="additive",
            opacity=0.5,
        )

        self._accepted_layer = self._viewer.add_shapes(
            name="Accepted",
            edge_color="green",
            edge_width=6,
            face_color="transparent",
        )

        # Wrap content in a scroll area
        scroll_content = QWidget()
        scroll_content.setLayout(self.vbox)
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll_area)
        self.setLayout(outer_layout)
        self.show()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self._sam_model = None
        self.sam_predictor = None

        self._viewer.layers.events.inserted.connect(
            self._on_layer_list_changed
        )
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

        # Track connected layers to avoid duplicate connections
        self._connected_layers = set()

        # Connect to existing layers' name events
        self._connect_to_existing_layers()

        self._labels_layer.bind_key("A", self._accept_mask)
        self._labels_layer.bind_key("R", self._reject_mask)

        self._on_layer_list_changed(None)
        self._radio_btn_shape.setChecked(True)

    def _on_api_checkbox_toggled(self):
        """Handle API checkbox state changes"""
        is_checked = self._use_api_checkbox.isChecked()
        self._api_url_input.setVisible(is_checked)
        self._api_key_input.setVisible(is_checked)
        self._api_url_layout.itemAt(0).widget().setVisible(is_checked)
        self._api_key_layout.itemAt(0).widget().setVisible(is_checked)

        # Toggle local model enable/disable
        self._model_selection.setEnabled(not is_checked)
        self._model_load_btn.setEnabled(not is_checked)

    def _get_image_shape(self):
        """Return 2D shape of the currently selected image layer."""
        layer_name = self._image_layer_selection.currentText()
        if layer_name and layer_name in self._viewer.layers:
            image_layer = self._get_layer_by_name_safe(layer_name)
            if image_layer is not None:
                img_type = check_image_type(self._viewer, layer_name)
                if "stack" in img_type:
                    return image_layer.data.shape[1:3]
                else:
                    return image_layer.data.shape[:2]
        return None

    def _resize_labels_layer(self):
        """Resize SAM-Predict layer to match selected image."""
        shape = self._get_image_shape()
        if shape is not None and shape != self._labels_layer.data.shape:
            self._labels_layer.data = np.zeros(shape, dtype="uint8")
        else:
            self._labels_layer.data = np.zeros_like(self._labels_layer.data)

    def _on_manual_mode_toggled(self, is_checked):
        """Handle Manual Mode checkbox state changes."""
        if is_checked:
            # Disable SAM-related controls
            self._api_group.setEnabled(False)
            self._model_selection.setEnabled(False)
            self._model_load_btn.setEnabled(False)

            # Hide SAM layers
            self._sam_box_layer.visible = False
            self._sam_positive_point_layer.visible = False
            self._sam_negative_point_layer.visible = False

            # Resize and clear SAM-Predict, then set to paint mode
            self._resize_labels_layer()
            self._labels_layer.selected_label = 1
            self._labels_layer.brush_size = 10
            self._labels_layer.mode = "paint"
            self._viewer.layers.selection.active = self._labels_layer
        else:
            # Re-enable SAM-related controls
            self._api_group.setEnabled(True)
            is_api = self._use_api_checkbox.isChecked()
            self._model_selection.setEnabled(not is_api)
            self._model_load_btn.setEnabled(not is_api)

            # Show SAM layers
            self._sam_box_layer.visible = True
            self._sam_positive_point_layer.visible = True
            self._sam_negative_point_layer.visible = True

            # Reset SAM-Predict
            self._resize_labels_layer()
            self._labels_layer.mode = "pan_zoom"
            self._viewer.layers.selection.active = self._sam_box_layer

    def _on_layer_list_changed(self, event):
        if event is not None:
            print(event.value)

            # Connect to name change event for newly added layers
            if hasattr(event, "value") and event.value is not None:
                self._connect_to_layer_name_event(event.value)

            self._refresh_layer_selections()
            # Move image layers to front
            for i, layer in enumerate(self._viewer.layers):
                if isinstance(layer, napari.layers.image.image.Image):
                    self._viewer.layers.move(i, 0)
            if isinstance(event.value, napari.layers.image.image.Image):
                self._on_image_layer_changed(None)
        else:
            self._refresh_layer_selections()

    def _connect_to_existing_layers(self):
        """Connect to name change events for all existing layers"""
        for layer in self._viewer.layers:
            self._connect_to_layer_name_event(layer)

    def _connect_to_layer_name_event(self, layer):
        """Connect to a layer's name change event if not already connected"""
        if id(layer) not in self._connected_layers:
            layer.events.name.connect(self._on_layer_name_changed)
            self._connected_layers.add(id(layer))

    def _on_layer_name_changed(self, event):
        """Handle layer name changes"""
        # Get the layer from the event source
        layer = event.source if hasattr(event, "source") else None
        layer_name = layer.name if layer else "unknown"
        print(f"Layer name changed to '{layer_name}'")
        self._refresh_layer_selections()

    def _on_layer_removed(self, event):
        """Handle layer removal and cleanup connections"""
        if (
            event is not None
            and hasattr(event, "value")
            and event.value is not None
        ):
            # Remove from connected layers set
            layer_id = id(event.value)
            if layer_id in self._connected_layers:
                self._connected_layers.remove(layer_id)

        # Refresh layer selections
        self._refresh_layer_selections()

    def _refresh_layer_selections(self):
        """Refresh all layer selection ComboBoxes with current layer names"""
        # Store current selections
        current_image = self._image_layer_selection.currentText()

        # Update image layer selection
        self._image_layer_selection.clear()
        image_layers = [
            layer.name
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.image.image.Image)
        ]
        self._image_layer_selection.addItems(image_layers)

        # Restore selection if layer still exists
        if current_image in image_layers:
            self._image_layer_selection.setCurrentText(current_image)

        # Update shapes and labels selections via radio button toggle
        self._on_radio_btn_toggled()

    def _on_radio_btn_toggled(self):
        button_id = self._radio_btn_group.checkedId()
        if (self._shapes_layer_selection is not None) & (
            self._labels_layer_selection is not None
        ):
            # Store current selections
            current_shapes = self._shapes_layer_selection.currentText()
            current_labels = self._labels_layer_selection.currentText()

            if button_id == 0:
                self._shapes_layer_selection.clear()
                shape_layers = [
                    layer.name
                    for layer in self._viewer.layers
                    if (isinstance(layer, napari.layers.shapes.shapes.Shapes))
                    and (layer.name != self._sam_box_layer.name)
                ]
                self._shapes_layer_selection.addItems(shape_layers)

                # Restore selection if layer still exists
                if current_shapes in shape_layers:
                    self._shapes_layer_selection.setCurrentText(current_shapes)

                self._labels_layer_selection.clear()
                self._save_btn.setEnabled(True)
                self.check_box.setEnabled(False)
                self.check_box.setStyleSheet("text-decoration: line-through")

            else:
                self._labels_layer_selection.clear()
                label_layers = [
                    layer.name
                    for layer in self._viewer.layers
                    if (isinstance(layer, napari.layers.labels.labels.Labels))
                    and (layer.name != self._labels_layer.name)
                ]
                self._labels_layer_selection.addItems(label_layers)

                # Restore selection if layer still exists
                if current_labels in label_layers:
                    self._labels_layer_selection.setCurrentText(current_labels)

                self._shapes_layer_selection.clear()
                self._save_btn.setEnabled(False)
                self.check_box.setEnabled(True)
                self.check_box.setStyleSheet("text-decoration: none")

    # --- Features DataFrame Helpers ---

    def _has_text_set(self, layer):
        """Check if layer already has a non-empty text template."""
        if not hasattr(layer, "text") or layer.text is None:
            return False
        try:
            s = layer.text.string
            const = getattr(s, "constant", None)
            if const is not None:
                return bool(str(const))
            return bool(s)
        except (AttributeError, TypeError):
            return False

    def _ensure_features_columns(self, layer):
        """Ensure output shapes layer has all required columns."""
        features = layer.features
        changed = False
        for col, dtype in _ATTR_COLUMNS.items():
            if col not in features.columns:
                features[col] = pd.Series(
                    [_ATTR_DEFAULTS[col]] * len(features),
                    dtype=dtype,
                )
                changed = True
        if changed:
            layer.features = features

    def _reset_output_layer(self, layer):
        """Reset data / features / feature_defaults."""
        layer.data = []
        layer.features = pd.DataFrame(
            {
                col: pd.Series(dtype=dtype)
                for col, dtype in _ATTR_COLUMNS.items()
            }
        )
        for key, val in _ATTR_DEFAULTS.items():
            layer.feature_defaults[key] = val

    # --- Output Layer Event Connection ---

    def _on_shapes_layer_combo_changed(self, text):
        """Handle output shapes layer ComboBox change."""
        self._disconnect_output_layer_events()
        if text:
            layer = self._get_layer_by_name_safe(text)
            if layer is not None and isinstance(
                layer, napari.layers.shapes.shapes.Shapes
            ):
                self._connect_output_layer_events(layer)
        self._on_output_selection_changed()

    def _connect_output_layer_events(self, layer):
        """Connect to output shapes layer highlight events."""
        self._disconnect_output_layer_events()
        layer.events.highlight.connect(self._on_output_selection_changed)
        self._connected_output_layer = layer

    def _disconnect_output_layer_events(self):
        """Disconnect from previous output layer events."""
        if self._connected_output_layer is not None:
            import contextlib

            with contextlib.suppress(TypeError, RuntimeError):
                self._connected_output_layer.events.highlight.disconnect(
                    self._on_output_selection_changed
                )
            self._connected_output_layer = None

    # --- Annotation Attributes Handlers ---

    def _on_output_selection_changed(self, event=None):
        """Update attribute UI when output layer selection changes."""
        if self._updating_attr_ui:
            return
        self._updating_attr_ui = True
        try:
            self._update_attr_ui()
        finally:
            self._updating_attr_ui = False

    def _update_attr_ui(self):
        """Sync attribute checkboxes and status label."""
        output_name = self._shapes_layer_selection.currentText()
        if not output_name:
            self._set_attr_ui_disabled()
            return

        layer = self._get_layer_by_name_safe(output_name)
        if layer is None or not isinstance(
            layer, napari.layers.shapes.shapes.Shapes
        ):
            self._set_attr_ui_disabled()
            return

        selected = list(layer.selected_data)
        if not selected:
            self._set_attr_ui_disabled()
            return

        self._ensure_features_columns(layer)
        features = layer.features

        # Enable controls
        self._unclear_checkbox.setEnabled(True)
        self._uncertain_checkbox.setEnabled(True)
        self._accept_selected_btn.setEnabled(True)

        # Unclear
        unclear_vals = features.loc[selected, "unclear"]
        self._set_tristate_checkbox(self._unclear_checkbox, unclear_vals)

        # Uncertain
        uncertain_vals = features.loc[selected, "uncertain"]
        self._set_tristate_checkbox(self._uncertain_checkbox, uncertain_vals)

        # Status label
        if len(selected) == 1:
            idx = selected[0]
            status = features.at[idx, "review_status"]
            reviewed_at = features.at[idx, "reviewed_at"]
            reviewed_str = self._format_reviewed_at(reviewed_at)
            self._attr_status_label.setText(
                f"Status: {status} | Reviewed: {reviewed_str}"
            )
        else:
            statuses = features.loc[selected, "review_status"].unique()
            status_text = statuses[0] if len(statuses) == 1 else "mixed"
            self._attr_status_label.setText(
                f"Status: {status_text} ({len(selected)} selected)"
            )

        # Sync class list selection
        self._sync_class_list_to_selection(features, selected)

    def _sync_class_list_to_selection(self, features, selected):
        """Sync class list widget to match selected polygon(s)."""
        if len(selected) == 1:
            target = features.at[selected[0], "class"]
        else:
            class_vals = features.loc[selected, "class"].unique()
            target = class_vals[0] if len(class_vals) == 1 else None

        if target is not None:
            matched = False
            for i in range(self._class_list_widget.count()):
                if self._class_list_widget.item(i).text() == target:
                    self._class_list_widget.setCurrentRow(i)
                    matched = True
                    break
            if not matched:
                self._class_list_widget.clearSelection()
        else:
            self._class_list_widget.clearSelection()

    def _set_attr_ui_disabled(self):
        """Disable attribute UI when no selection."""
        self._unclear_checkbox.blockSignals(True)
        self._unclear_checkbox.setCheckState(Qt.Unchecked)
        self._unclear_checkbox.setEnabled(False)
        self._unclear_checkbox.blockSignals(False)

        self._uncertain_checkbox.blockSignals(True)
        self._uncertain_checkbox.setCheckState(Qt.Unchecked)
        self._uncertain_checkbox.setEnabled(False)
        self._uncertain_checkbox.blockSignals(False)

        self._accept_selected_btn.setEnabled(False)
        self._attr_status_label.setText("No annotation selected")

    def _set_tristate_checkbox(self, checkbox, values):
        """Set checkbox state based on selection values."""
        checkbox.blockSignals(True)
        all_true = values.all()
        all_false = not values.any()
        if all_true:
            checkbox.setCheckState(Qt.Checked)
        elif all_false:
            checkbox.setCheckState(Qt.Unchecked)
        else:
            checkbox.setCheckState(Qt.PartiallyChecked)
        checkbox.blockSignals(False)

    def _format_reviewed_at(self, reviewed_at):
        """Format reviewed_at for display."""
        if not reviewed_at:
            return "\u2014"
        try:
            dt = datetime.fromisoformat(reviewed_at)
            return dt.astimezone().strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            return str(reviewed_at)

    def _on_unclear_toggled(self, state):
        """Handle unclear checkbox toggle."""
        if self._updating_attr_ui:
            return
        int_state = int(state)
        if int_state == 1:  # PartiallyChecked → normalize to Checked
            self._unclear_checkbox.setCheckState(Qt.Checked)
            return  # setCheckState re-fires stateChanged with Checked
        value = int_state != 0
        self._set_selected_attr("unclear", value)

    def _on_uncertain_toggled(self, state):
        """Handle uncertain checkbox toggle."""
        if self._updating_attr_ui:
            return
        int_state = int(state)
        if int_state == 1:  # PartiallyChecked → normalize to Checked
            self._uncertain_checkbox.setCheckState(Qt.Checked)
            return
        value = int_state != 0
        self._set_selected_attr("uncertain", value)

    def _set_selected_attr(self, attr_name, value):
        """Set attribute for all selected shapes."""
        output_name = self._shapes_layer_selection.currentText()
        if not output_name:
            return
        layer = self._get_layer_by_name_safe(output_name)
        if layer is None:
            return

        self._ensure_features_columns(layer)
        selected = list(layer.selected_data)
        if selected:
            features = layer.features.copy()
            features.loc[selected, attr_name] = value
            self._updating_attr_ui = True
            try:
                layer.features = features
            finally:
                self._updating_attr_ui = False

    def _accept_selected(self):
        """Mark selected annotations as accepted."""
        output_name = self._shapes_layer_selection.currentText()
        if not output_name:
            return
        layer = self._get_layer_by_name_safe(output_name)
        if layer is None:
            return

        self._ensure_features_columns(layer)
        selected = list(layer.selected_data)
        if not selected:
            return

        now = datetime.now().astimezone().isoformat()
        features = layer.features.copy()
        features.loc[selected, "review_status"] = "approved"
        features.loc[selected, "reviewed_at"] = now
        self._updating_attr_ui = True
        try:
            layer.features = features
        finally:
            self._updating_attr_ui = False
        self._on_output_selection_changed()

    def _accept_all(self):
        """Mark all annotations as accepted."""
        output_name = self._shapes_layer_selection.currentText()
        if not output_name:
            return
        layer = self._get_layer_by_name_safe(output_name)
        if layer is None:
            return

        self._ensure_features_columns(layer)
        if len(layer.features) == 0:
            return

        now = datetime.now().astimezone().isoformat()
        features = layer.features.copy()
        features["review_status"] = "approved"
        features["reviewed_at"] = now
        self._updating_attr_ui = True
        try:
            layer.features = features
        finally:
            self._updating_attr_ui = False
        self._on_output_selection_changed()
        print(f"Accepted all {len(layer.features)} annotations")

    # --- Class Management Methods ---

    def _on_class_clicked(self):
        """Handle class selection in the class list widget."""
        selected_items = self._class_list_widget.selectedItems()
        if not selected_items:
            return
        class_name = selected_items[0].text()

        button_id = self._radio_btn_group.checkedId()
        if button_id == 0:
            output_name = self._shapes_layer_selection.currentText()
            if output_name:
                output_layer = self._get_layer_by_name_safe(output_name)
                if output_layer is not None and isinstance(
                    output_layer,
                    napari.layers.shapes.shapes.Shapes,
                ):
                    output_layer.feature_defaults["class"] = class_name
                    idxs = list(output_layer.selected_data)
                    if idxs:
                        features = output_layer.features.copy()
                        features.loc[idxs, "class"] = class_name
                        self._updating_attr_ui = True
                        try:
                            output_layer.features = features
                        finally:
                            self._updating_attr_ui = False
                        output_layer.refresh_text()

    def _add_class(self):
        """Add a new class to the class list."""
        class_name = self._class_name_input.text().strip()
        if not class_name:
            return

        existing = [
            self._class_list_widget.item(i).text()
            for i in range(self._class_list_widget.count())
        ]
        existing_names = [
            name.split(": ", 1)[1] for name in existing if ": " in name
        ]
        if class_name in existing_names:
            print("Class already exists")
            return

        if existing:
            numbers = [int(n.split(":")[0]) for n in existing]
            next_id = find_missing_class_number(numbers)
        else:
            next_id = 0

        formatted = f"{next_id}: {class_name}"
        self._class_list_widget.addItem(formatted)
        self._sort_class_list()
        self._class_name_input.clear()

    def _del_class(self):
        """Delete the selected class (blocked if in use)."""
        selected_items = self._class_list_widget.selectedItems()
        if not selected_items:
            return

        class_text = selected_items[0].text()

        # Check if class is in use in the output shapes layer
        button_id = self._radio_btn_group.checkedId()
        if button_id == 0:
            output_name = self._shapes_layer_selection.currentText()
            if output_name:
                output_layer = self._get_layer_by_name_safe(output_name)
                if (
                    output_layer is not None
                    and isinstance(
                        output_layer,
                        napari.layers.shapes.shapes.Shapes,
                    )
                    and "class" in output_layer.features.columns
                ):
                    in_use = (
                        output_layer.features["class"] == class_text
                    ).any()
                    if in_use:
                        QMessageBox.warning(
                            self,
                            "Cannot delete",
                            f"Class '{class_text}' is "
                            f"assigned to existing "
                            f"annotations.",
                        )
                        return

        # Check if class is in use in labels mode mapping
        if class_text in self._label_id_to_class.values():
            QMessageBox.warning(
                self,
                "Cannot delete",
                f"Class '{class_text}' is assigned to "
                f"existing label annotations.",
            )
            return

        self._class_list_widget.takeItem(
            self._class_list_widget.row(selected_items[0])
        )

    def _sort_class_list(self):
        """Sort class list items by numeric ID prefix."""
        items = [
            self._class_list_widget.item(i).text()
            for i in range(self._class_list_widget.count())
        ]
        sorted_items = sorted(
            items, key=lambda x: int(x.split(":")[0].strip())
        )
        self._class_list_widget.clear()
        for item_text in sorted_items:
            self._class_list_widget.addItem(item_text)

    def _load_classes(self):
        """Load class definitions from a YAML file."""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Load class file",
            "",
            "YAML files (*.yaml *.yml)",
        )
        if not fname:
            return
        with open(fname) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "names" not in data:
            print("Invalid class file format")
            return
        self._class_list_widget.clear()
        for key, value in data["names"].items():
            class_id = int(key)
            class_name = str(value).strip()
            self._class_list_widget.addItem(f"{class_id}: {class_name}")
        self._sort_class_list()

    def _get_selected_class(self):
        """Return currently selected class string.

        Falls back to first item, or auto-creates
        '0: object' if list is empty.
        """
        selected = self._class_list_widget.selectedItems()
        if selected:
            return selected[0].text()
        if self._class_list_widget.count() > 0:
            return self._class_list_widget.item(0).text()
        # Auto-create default class
        self._class_list_widget.addItem("0: object")
        return "0: object"

    def _get_selected_class_id(self):
        """Return the numeric ID of the selected class."""
        class_str = self._get_selected_class()
        if class_str and ": " in class_str:
            return int(class_str.split(":")[0].strip())
        return 0

    def _build_categories_list(self):
        """Build COCO categories list from class list."""
        categories = []
        for i in range(self._class_list_widget.count()):
            text = self._class_list_widget.item(i).text()
            parts = text.split(": ", 1)
            cat_id = int(parts[0].strip())
            cat_name = parts[1].strip() if len(parts) > 1 else "object"
            categories.append(
                {
                    "id": cat_id,
                    "name": cat_name,
                    "supercategory": cat_name,
                }
            )
        if not categories:
            categories = [
                {
                    "id": 0,
                    "name": "object",
                    "supercategory": "object",
                }
            ]
        return categories

    def _save_class_yaml(self, directory):
        """Save class definitions to class.yaml."""
        names = {}
        for i in range(self._class_list_widget.count()):
            text = self._class_list_widget.item(i).text()
            parts = text.split(": ", 1)
            cat_id = int(parts[0].strip())
            cat_name = parts[1].strip() if len(parts) > 1 else ""
            names[cat_id] = cat_name
        if names:
            class_data = {"names": names}
            path = os.path.join(directory, "class.yaml")
            with open(path, "w") as f:
                yaml.dump(class_data, f, default_flow_style=False)

    def _load_model(self):
        if self._use_api_checkbox.isChecked():
            print("Local model loading is not required in API mode")
            return
        try:
            from segment_anything import SamPredictor
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"Failed to import segment_anything: {exc}")
            return

        model_name = self._model_selection.currentText()
        self._sam_model = load_model(model_name)
        self._sam_model.to(device=self.device)
        self.sam_predictor = SamPredictor(self._sam_model)
        print("model loaded")
        if self._image_layer_selection.currentText() != "":
            self._on_image_layer_changed(True)

    def _prompt_save_before_switch(self, prev_image_name):
        """Prompt user to save annotations before switching images.

        Returns True if the switch should proceed, False to cancel.
        """
        output_name = self._shapes_layer_selection.currentText()
        if not output_name:
            return True

        output_layer = self._get_layer_by_name_safe(output_name)
        if output_layer is None or not isinstance(
            output_layer, napari.layers.shapes.shapes.Shapes
        ):
            return True

        if len(output_layer.data) == 0:
            return True

        reply = QMessageBox.question(
            self,
            "Save annotations?",
            "Current image has unsaved annotations.\n"
            "Save before switching?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save and not self._save_impl(
            image_layer_name=prev_image_name
        ):
            print("Save failed, cancelling image switch")
            return False
        return True

    def _notify_discard_on_deleted_image(self):
        """Notify that annotations will be discarded (previous image gone).

        When the previous image layer has been deleted, there is no valid
        save target and no image to revert to, so discard is the only option.
        """
        output_name = self._shapes_layer_selection.currentText()
        if not output_name:
            return

        output_layer = self._get_layer_by_name_safe(output_name)
        if output_layer is None or not isinstance(
            output_layer, napari.layers.shapes.shapes.Shapes
        ):
            return

        if len(output_layer.data) == 0:
            return

        QMessageBox.information(
            self,
            "Annotations discarded",
            "Previous image was removed.\n"
            "Unsaved annotations have been discarded.",
        )

    def _on_image_layer_changed(self, set_image=False):
        print("image_layer_changed")

        layer_name = self._image_layer_selection.currentText()

        # Resolve current layer object
        new_layer = None
        if layer_name and layer_name in self._viewer.layers:
            new_layer = self._get_layer_by_name_safe(layer_name)

        # Detect real image switch (not just a rename)
        prev_layer = getattr(self, "_current_target_image_layer", None)
        prev_name = getattr(self, "_current_target_image_name", None)
        is_real_switch = (
            new_layer is not None
            and prev_layer is not None
            and new_layer is not prev_layer
        )

        if is_real_switch:
            prev_still_exists = prev_layer in self._viewer.layers
            if prev_still_exists:
                if not self._prompt_save_before_switch(prev_name):
                    # Cancel → revert ComboBox to previous image
                    self._image_layer_selection.blockSignals(True)
                    self._image_layer_selection.setCurrentText(prev_layer.name)
                    self._image_layer_selection.blockSignals(False)
                    return
            else:
                # Previous image deleted — no save target, must discard
                self._notify_discard_on_deleted_image()

            # Clear output layer for new image
            output_name = self._shapes_layer_selection.currentText()
            if output_name:
                output_layer = self._get_layer_by_name_safe(output_name)
                if output_layer is not None and isinstance(
                    output_layer,
                    napari.layers.shapes.shapes.Shapes,
                ):
                    self._reset_output_layer(output_layer)

        # Update image metadata (needed for all modes)
        image_layer = new_layer
        if image_layer is not None:
            self._current_target_image_name = layer_name
            self._current_target_image_layer = image_layer
            self._image_type = check_image_type(self._viewer, layer_name)
            if "stack" in self._image_type:
                self._current_slice, _, _ = self._viewer.dims.current_step
            else:
                self._current_slice = None

        # Update SAM predictor (for local SAM mode, even
        # if in manual mode). Ensures predictor has correct
        # image when switching back to SAM mode.
        if (
            self.sam_predictor is not None
            and image_layer is not None
            and not self._use_api_checkbox.isChecked()
        ):
            self.sam_predictor.set_image(
                preprocess(
                    image_layer.data,
                    self._image_type,
                    self._current_slice,
                )
            )
            print("Set image for SAM predictor")

        # In manual mode, resize SAM-Predict and return
        if self._manual_mode_checkbox.isChecked():
            self._resize_labels_layer()
            self._labels_layer.mode = "paint"
            self._viewer.layers.selection.active = self._labels_layer
            self._try_auto_load_annotations()
            return

        # API mode: no predictor setup needed
        if self._use_api_checkbox.isChecked():
            print("Image selected for API mode")
            self._try_auto_load_annotations()
            return

        self._try_auto_load_annotations()

    def _on_sam_box_created(self, layer, event):
        if self._manual_mode_checkbox.isChecked():
            return
        # mouse click
        yield
        # mouse move
        while event.type == "mouse_move":
            yield
        # mouse release
        if len(self._sam_box_layer.data) == 1:
            if "stack" in self._image_type:
                if self._current_slice == self._viewer.dims.current_step[0]:
                    pass
                else:
                    self._on_image_layer_changed(None)
            else:
                pass
            coords = self._sam_box_layer.data[0]
            y1 = int(coords[0][0])
            x1 = int(coords[0][1])
            y2 = int(coords[2][0])
            x2 = int(coords[2][1])
            print(f"x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}")
            self._input_box = np.array([x1, y1, x2, y2])
            self._create_input_point()
            self._predict()
            self._sam_box_layer.data = []

    def _reject_all_boxes(self, layer):
        self._sam_box_layer.data = []

    def _on_sam_point_changed(self):
        if (len(self._sam_positive_point_layer.data) != 0) or (
            self._input_box is not None
        ):
            if "stack" in self._image_type:
                if self._current_slice == self._viewer.dims.current_step[0]:
                    pass
                else:
                    self._on_image_layer_changed(None)
            else:
                pass
            self._create_input_point()
            self._predict()

    def _predict(self):
        if self._use_api_checkbox.isChecked():
            self._predict_api()
        else:
            self._predict_local()

    def _predict_local(self):
        if self.sam_predictor is not None:
            masks, _, _ = self.sam_predictor.predict(
                point_coords=self._input_point,
                point_labels=self._point_label,
                box=(
                    self._input_box[None, :]
                    if self._input_box is not None
                    else None
                ),
                multimask_output=False,
            )
            self._labels_layer.data = masks[0] * 1
        self._viewer.layers.selection.active = self._labels_layer

    def _predict_api(self):
        """Execute prediction using API"""
        if self._input_box is None:
            print("Error: Bounding box is not set")
            return

        api_url = self._api_url_input.text().strip()
        api_key = self._api_key_input.text().strip()

        if not api_url or not api_key:
            print("Error: Please enter API URL and API Key")
            return

        try:
            # Get current image
            image_layer = self._get_layer_by_name_safe(
                self._image_layer_selection.currentText()
            )
            if image_layer is None:
                print("Error: Image layer not found or was renamed")
                return
            if "stack" in self._image_type:
                current_slice = self._viewer.dims.current_step[0]
                image_data = image_layer.data[current_slice]
            else:
                image_data = image_layer.data

            # Convert to PIL Image and encode as JPEG
            if image_data.ndim == 2:  # Grayscale
                pil_image = Image.fromarray(image_data).convert("RGB")
            else:  # RGB
                pil_image = Image.fromarray(image_data.astype(np.uint8))

            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=100)
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Wrap bounding box coordinates in list
            coords = [self._input_box.tolist()]

            # Create API request data
            request_data = {
                "input": {
                    "image_data": image_b64,
                    "coords": coords,
                    "output_format": "geojson",
                    "image_format": "jpeg",
                }
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            # Create session with retry functionality
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504, 520],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)

            print("Sending request to API...")
            response = session.post(
                api_url, headers=headers, json=request_data, timeout=300
            )
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                print(f"API error: {result['error']}")
                return

            # Generate mask from GeoJSON
            geojson_data = result["output"]["geojson"]
            mask = self._geojson_to_mask(geojson_data, image_data.shape)

            self._labels_layer.data = mask.astype(np.uint8)
            print("API prediction completed")

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
        ) as e:
            print(f"API error: {str(e)}")

        self._viewer.layers.selection.active = self._labels_layer

    def _geojson_to_mask(self, geojson_data, image_shape):
        """Convert GeoJSON data to mask"""
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for feature in geojson_data["features"]:
            if feature["geometry"]["type"] == "Polygon":
                coordinates = feature["geometry"]["coordinates"][0]
                coords_array = np.array(coordinates)

                from matplotlib.path import Path

                path = Path(coords_array)

                y_coords, x_coords = np.meshgrid(
                    np.arange(height), np.arange(width), indexing="ij"
                )
                points = np.column_stack((x_coords.ravel(), y_coords.ravel()))

                inside = path.contains_points(points)
                inside_2d = inside.reshape(height, width)

                mask = np.where(inside_2d, 1, mask)

        return mask

    def _create_input_point(self):
        positive_points = self._sam_positive_point_layer.data
        negative_points = self._sam_negative_point_layer.data
        if len(positive_points) + len(negative_points) == 0:
            self._input_point = None
            self._point_label = None
            return
        self._point_label = np.array(
            len(positive_points) * [1] + len(negative_points) * [0]
        )
        coords = np.concatenate((positive_points, negative_points), axis=0)
        self._input_point = coords[:, ::-1].astype(np.int32)

    def _get_layer_by_name_safe(self, layer_name):
        """Safely get layer by name, handling case where layer was renamed"""
        try:
            return self._viewer.layers[layer_name]
        except KeyError:
            print(
                f"Warning: Layer '{layer_name}' not found."
                " It may have been renamed."
            )
            # Refresh layer selections and return None
            self._refresh_layer_selections()
            return None

    def _accept_mask(self, layer):
        # Guard: skip if mask is empty (no painted pixels)
        if not np.any(self._labels_layer.data):
            print("Nothing to accept (empty mask)")
            return
        selected_class = self._get_selected_class()
        button_id = self._radio_btn_group.checkedId()
        if button_id == 0:
            if self._shapes_layer_selection.currentText() != "":
                output_layer = self._get_layer_by_name_safe(
                    self._shapes_layer_selection.currentText()
                )
                if output_layer and isinstance(
                    output_layer,
                    napari.layers.shapes.shapes.Shapes,
                ):
                    # Initialize features/text if needed
                    self._ensure_features_columns(output_layer)
                    if not self._has_text_set(output_layer):
                        output_layer.text = {
                            "string": "{class}",
                            "anchor": "upper_left",
                            "size": 10,
                            "color": "green",
                        }
                    output_layer.feature_defaults["class"] = selected_class
                    for key, val in _ATTR_DEFAULTS.items():
                        if key != "class":
                            output_layer.feature_defaults[key] = val
                    output_layer.add_polygons(
                        label2polygon(self._labels_layer.data),
                        edge_width=2,
                    )
                    output_layer.refresh_text()
                    self._viewer.layers.selection.active = self._sam_box_layer
                elif output_layer is None:
                    print("Output shapes layer not found or was renamed")
                    return
            else:
                pass
        else:
            if self._labels_layer_selection.currentText() != "":
                output_layer = self._get_layer_by_name_safe(
                    self._labels_layer_selection.currentText()
                )
                if output_layer and isinstance(
                    output_layer,
                    napari.layers.labels.labels.Labels,
                ):
                    if self._current_slice is not None:
                        if self.check_box.isChecked():
                            num = find_first_missing(
                                output_layer.data[self._current_slice]
                            )
                        else:
                            num = 1
                        current_data = output_layer.data[self._current_slice]
                        new_mask = self._labels_layer.data * num
                        mask_to_apply = (current_data == 0) & (new_mask > 0)
                        output_layer.data[self._current_slice] = (
                            current_data + new_mask * mask_to_apply
                        )
                        output_layer.refresh()
                    else:
                        if self.check_box.isChecked():
                            num = find_first_missing(output_layer.data)
                        else:
                            num = 1
                        current_data = output_layer.data
                        new_mask = self._labels_layer.data * num
                        mask_to_apply = (current_data == 0) & (new_mask > 0)
                        output_layer.data = (
                            current_data + new_mask * mask_to_apply
                        )
                    self._label_id_to_class[num] = selected_class
                    self._viewer.layers.selection.active = self._sam_box_layer
                elif output_layer is None:
                    print("Output labels layer not found or was renamed")
                    return
                else:
                    pass
        self._labels_layer.data = np.zeros_like(self._labels_layer.data)
        self._input_box = None
        self._sam_positive_point_layer.data = []
        self._sam_negative_point_layer.data = []
        self._input_point = None
        self._point_label = None

        # In manual mode, return to paint mode on SAM-Predict
        if self._manual_mode_checkbox.isChecked():
            self._viewer.layers.selection.active = self._labels_layer
            self._labels_layer.mode = "paint"

    def _reject_mask(self, layer):
        self._labels_layer.data = np.zeros_like(self._labels_layer.data)
        self._input_box = None
        self._sam_positive_point_layer.data = []
        self._sam_negative_point_layer.data = []

        if self._manual_mode_checkbox.isChecked():
            self._viewer.layers.selection.active = self._labels_layer
            self._labels_layer.mode = "paint"
        else:
            self._viewer.layers.selection.active = self._sam_box_layer

    def _save(self):
        """Save button handler (absorbs bool from clicked signal)."""
        self._save_impl()

    def _save_impl(self, image_layer_name=None):
        """Save annotations to COCO JSON.

        Args:
            image_layer_name: Override image layer name.
                If None, uses the current ComboBox selection.

        Returns:
            True if saved successfully, False otherwise.
        """
        if self._shapes_layer_selection.currentText() == "":
            return False

        if image_layer_name is None:
            image_layer_name = self._image_layer_selection.currentText()
        image_layer = self._get_layer_by_name_safe(image_layer_name)
        if image_layer is None:
            print("Image layer not found or was renamed")
            return False

        image_path = image_layer.source.path
        image_name = os.path.basename(image_path)
        output_layer = self._get_layer_by_name_safe(
            self._shapes_layer_selection.currentText()
        )
        if output_layer is None:
            print("Output layer not found or was renamed")
            return False
        if not isinstance(
            output_layer,
            napari.layers.shapes.shapes.Shapes,
        ):
            return False

        categories = self._build_categories_list()

        category_ids = []
        if (
            "class" in output_layer.features.columns
            and len(output_layer.features) > 0
        ):
            for cls_str in output_layer.features["class"]:
                try:
                    cat_id = int(str(cls_str).split(":")[0].strip())
                except (ValueError, IndexError):
                    cat_id = 0
                category_ids.append(cat_id)
        else:
            category_ids = [0] * len(output_layer.data)

        # Build attributes_list from features
        attributes_list = None
        if (
            "unclear" in output_layer.features.columns
            and len(output_layer.features) > 0
        ):
            attributes_list = []
            for _, row in output_layer.features.iterrows():
                reviewed_at_val = row.get("reviewed_at", "")
                attrs = {
                    "unclear": bool(row.get("unclear", False)),
                    "uncertain": bool(row.get("uncertain", False)),
                    "review_status": str(
                        row.get(
                            "review_status",
                            "unreviewed",
                        )
                    ),
                    "reviewed_at": (
                        reviewed_at_val if reviewed_at_val else None
                    ),
                }
                attributes_list.append(attrs)

        output_path = os.path.join(
            os.path.dirname(image_path),
            os.path.splitext(image_name)[0] + ".json",
        )
        data = create_json(
            image_layer.data,
            image_name,
            output_layer.data,
            categories=categories,
            category_ids=category_ids,
            attributes_list=attributes_list,
        )
        with open(output_path, "w") as f:
            json.dump(data, f)

        self._save_class_yaml(os.path.dirname(image_path))
        print("saved")
        return True

    # --- Annotation Loading ---

    def _on_load_annotations_clicked(self):
        """Handle Load Annotations button click."""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Load annotations JSON",
            "",
            "JSON files (*.json)",
        )
        if not fname:
            return
        self._load_annotations_with_confirm(fname)

    def _try_auto_load_annotations(self):
        """Auto-load annotations for current image if output is empty."""
        # Guard: instance mode only
        if self._radio_btn_group.checkedId() != 0:
            return

        image_layer = self._get_layer_by_name_safe(
            self._image_layer_selection.currentText()
        )
        if image_layer is None:
            return
        image_path = getattr(
            getattr(image_layer, "source", None), "path", None
        )
        if not image_path:
            return

        json_path = os.path.join(
            os.path.dirname(image_path),
            os.path.splitext(os.path.basename(image_path))[0] + ".json",
        )
        if not os.path.exists(json_path):
            return

        # Guard: output shapes layer must exist and be empty
        output_name = self._shapes_layer_selection.currentText()
        if not output_name:
            return
        output_layer = self._get_layer_by_name_safe(output_name)
        if output_layer is None or not isinstance(
            output_layer, napari.layers.shapes.shapes.Shapes
        ):
            return
        if len(output_layer.data) > 0:
            return

        self._load_annotations(json_path, output_layer, needs_replace=False)

    def _load_annotations_with_confirm(self, json_path):
        """Load annotations, asking to replace if shapes exist.

        Returns True if annotations were loaded successfully.
        """
        button_id = self._radio_btn_group.checkedId()
        if button_id != 0:
            print("Annotation loading is only supported in instance mode")
            return False

        output_name = self._shapes_layer_selection.currentText()
        if not output_name:
            print("No output shapes layer selected")
            return False

        output_layer = self._get_layer_by_name_safe(output_name)
        if output_layer is None:
            return False

        needs_replace = len(output_layer.data) > 0
        if needs_replace:
            reply = QMessageBox.question(
                self,
                "Replace annotations?",
                "Output layer already has annotations. Replace them?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return False

        return self._load_annotations(json_path, output_layer, needs_replace)

    def _load_annotations(self, json_path, output_layer, needs_replace=False):
        """Load annotations from COCO JSON into output layer.

        Returns True if annotations were loaded successfully.
        """
        try:
            result = load_json(json_path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"Failed to load JSON: {e}")
            return False

        annotations = result["annotations"]
        categories = result["categories"]
        image_info = result["image_info"]

        # Check image size match
        image_layer = self._get_layer_by_name_safe(
            self._image_layer_selection.currentText()
        )
        if image_layer is not None and image_info:
            img_h, img_w = image_layer.data.shape[:2]
            json_h = image_info.get("height", 0)
            json_w = image_info.get("width", 0)
            if (json_h, json_w) != (img_h, img_w):
                reply = QMessageBox.warning(
                    self,
                    "Image size mismatch",
                    f"JSON image size ({json_w}x{json_h})"
                    f" differs from current image "
                    f"({img_w}x{img_h}). Continue?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if reply != QMessageBox.Yes:
                    return False

        # All validation passed — safe to reset now
        if needs_replace:
            self._reset_output_layer(output_layer)

        # Restore categories to class list
        if categories:
            self._class_list_widget.clear()
            for cat in categories:
                cat_id = cat.get("id", 0)
                cat_name = cat.get("name", "object")
                self._class_list_widget.addItem(f"{cat_id}: {cat_name}")
            self._sort_class_list()

        # Build category_id → class string mapping
        cat_id_to_str = {}
        for i in range(self._class_list_widget.count()):
            text = self._class_list_widget.item(i).text()
            parts = text.split(": ", 1)
            cid = int(parts[0].strip())
            cat_id_to_str[cid] = text

        self._ensure_features_columns(output_layer)

        if not self._has_text_set(output_layer):
            output_layer.text = {
                "string": "{class}",
                "anchor": "upper_left",
                "size": 10,
                "color": "green",
            }

        loaded_count = 0
        for ann in annotations:
            polygon = ann["polygon"]
            cat_id = ann["category_id"]
            attrs = ann["attributes"]

            class_str = cat_id_to_str.get(cat_id, f"{cat_id}: object")

            # Set feature_defaults before adding
            output_layer.feature_defaults["class"] = class_str
            if attrs:
                for key in (
                    "unclear",
                    "uncertain",
                    "review_status",
                    "reviewed_at",
                ):
                    val = attrs.get(key, _ATTR_DEFAULTS[key])
                    if val is None:
                        val = _ATTR_DEFAULTS[key]
                    output_layer.feature_defaults[key] = val
            else:
                for key, val in _ATTR_DEFAULTS.items():
                    if key != "class":
                        output_layer.feature_defaults[key] = val

            output_layer.add_polygons([polygon], edge_width=2)
            loaded_count += 1

        output_layer.refresh_text()
        print(
            f"Loaded {loaded_count} annotations from "
            f"{os.path.basename(json_path)}"
        )
        return True

    def lock_controls(self, layer, locked=True):
        import warnings

        widget_list = [
            "ellipse_button",
            "line_button",
            "path_button",
            "vertex_remove_button",
            "vertex_insert_button",
            "move_back_button",
            "move_front_button",
            "polygon_button",
            "select_button",
            "direct_button",
            "delete_button",
        ]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Public access to Window.qt_viewer",
                    category=FutureWarning,
                )
                qctrl = self._viewer.window.qt_viewer.controls.widgets[layer]
        except (AttributeError, KeyError):
            # Controls container not available for this layer
            return

        # Lock/unlock each control individually
        for wdg in widget_list:
            ctrl = getattr(qctrl, wdg, None)
            if ctrl is not None and hasattr(ctrl, "setEnabled"):
                ctrl.setEnabled(not locked)

    def print_corner_value(self):
        print(self._viewer.dims.current_step)
        image_layer = self._get_layer_by_name_safe(
            self._image_layer_selection.currentText()
        )
        if image_layer is not None:
            print(image_layer.corner_pixels)
        else:
            print("Image layer not found")
