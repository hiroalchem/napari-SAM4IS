import base64
import io
import json
import math
import os
from datetime import datetime
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import platformdirs
import requests
import torch
import yaml
from PIL import Image
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
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

_SETTINGS_DEFAULTS = {
    "accepted_edge_color": "#ffff00",
    "text_color": "#ffff00",
    "text_size": 12,
}
_TEXT_SIZE_MIN = 6
_TEXT_SIZE_MAX = 72
_SETTINGS_SAVE_WARNED = False
_TEXT_SIZE_SAVE_DEBOUNCE_MS = 300


def _sanitize_settings(data: dict) -> dict:
    settings = dict(_SETTINGS_DEFAULTS)
    if not isinstance(data, dict):
        return settings

    for key in ("accepted_edge_color", "text_color"):
        value = data.get(key)
        if isinstance(value, str) and QColor(value).isValid():
            settings[key] = QColor(value).name()

    raw_size = data.get("text_size")
    try:
        size = int(raw_size)
    except (TypeError, ValueError):
        size = _SETTINGS_DEFAULTS["text_size"]
    settings["text_size"] = max(_TEXT_SIZE_MIN, min(_TEXT_SIZE_MAX, size))
    return settings


def _load_settings() -> dict:
    path = (
        Path(platformdirs.user_config_dir("napari-SAM4IS")) / "settings.json"
    )
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _sanitize_settings(data)
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            pass
    return dict(_SETTINGS_DEFAULTS)


def _save_settings(settings: dict) -> None:
    global _SETTINGS_SAVE_WARNED
    path = (
        Path(platformdirs.user_config_dir("napari-SAM4IS")) / "settings.json"
    )
    safe_settings = _sanitize_settings(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(safe_settings, indent=2), encoding="utf-8")
        _SETTINGS_SAVE_WARNED = False
    except OSError as exc:
        if not _SETTINGS_SAVE_WARNED:
            print(f"Warning: Failed to save settings to {path}: {exc}")
            _SETTINGS_SAVE_WARNED = True


class SAMWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._settings = _load_settings()
        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.timeout.connect(self._flush_settings_save)

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

        # SAM3 checkpoint input (shown only when "sam3" is selected)
        self._sam3_ckpt_lineedit = QLineEdit()
        self._sam3_ckpt_lineedit.setPlaceholderText(
            "(blank = auto download from HuggingFace)"
        )
        self._sam3_ckpt_browse_btn = QPushButton("Browse...")
        self._sam3_ckpt_browse_btn.clicked.connect(
            self._browse_sam3_checkpoint
        )
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self._sam3_ckpt_lineedit)
        ckpt_row.addWidget(self._sam3_ckpt_browse_btn)
        self._sam3_ckpt_widget = QWidget()
        ckpt_vbox = QVBoxLayout()
        ckpt_vbox.setContentsMargins(0, 0, 0, 0)
        ckpt_vbox.addWidget(QLabel("SAM3 checkpoint:"))
        ckpt_vbox.addLayout(ckpt_row)
        self._sam3_ckpt_widget.setLayout(ckpt_vbox)
        self._sam3_ckpt_widget.setVisible(False)
        self.vbox.addWidget(self._sam3_ckpt_widget)
        self._model_selection.currentTextChanged.connect(
            lambda text: self._sam3_ckpt_widget.setVisible(text == "sam3")
        )

        self._model_load_btn = QPushButton("load model")
        self._model_load_btn.clicked.connect(self._load_model)
        self.vbox.addWidget(self._model_load_btn)

        # SAM3 Detect All prompt group (shown after SAM3 is loaded)
        self._sam3_prompt_group = QGroupBox("Detect All Prompt")
        sam3_prompt_layout = QVBoxLayout()
        self._sam3_prompt_text_radio = QRadioButton("Text (class name)")
        self._sam3_prompt_box_radio = QRadioButton("Box (exemplar)")
        self._sam3_prompt_both_radio = QRadioButton("Text + Box")
        self._sam3_prompt_both_radio.setChecked(True)
        for rb in (
            self._sam3_prompt_text_radio,
            self._sam3_prompt_box_radio,
            self._sam3_prompt_both_radio,
        ):
            sam3_prompt_layout.addWidget(rb)
        self._sam3_detect_all_btn = QPushButton("Detect All (SAM3)")
        self._sam3_detect_all_btn.clicked.connect(self._on_sam3_detect_all)
        sam3_prompt_layout.addWidget(self._sam3_detect_all_btn)

        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU threshold:"))
        self._iou_threshold_spin = QDoubleSpinBox()
        self._iou_threshold_spin.setRange(0.0, 1.0)
        self._iou_threshold_spin.setSingleStep(0.05)
        self._iou_threshold_spin.setValue(0.5)
        self._iou_threshold_spin.setToolTip(
            "IoU がこの閾値以上の既存 shape がある場合、"
            "新しいマスクを追加しません"
        )
        iou_layout.addWidget(self._iou_threshold_spin)
        sam3_prompt_layout.addLayout(iou_layout)

        self._iou_same_class_checkbox = QCheckBox("Same class only")
        self._iou_same_class_checkbox.setChecked(True)
        self._iou_same_class_checkbox.setToolTip(
            "チェック時: 同一クラスの既存 shape のみ IoU 比較\n"
            "未チェック時: 全クラス横断で IoU 比較"
        )
        sam3_prompt_layout.addWidget(self._iou_same_class_checkbox)

        self._sam3_prompt_group.setLayout(sam3_prompt_layout)
        self._sam3_prompt_group.setVisible(False)
        self.vbox.addWidget(self._sam3_prompt_group)

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

        self._class_list_widget = QTreeWidget()
        self._class_list_widget.setHeaderLabels(["Class"])
        self._class_list_widget.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self._class_list_widget.setMinimumHeight(200)
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
        self._add_subclass_btn = QPushButton("Add Sub")
        self._add_subclass_btn.clicked.connect(self._add_subclass)
        self._class_input_layout.addWidget(self._add_subclass_btn)
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

        # --- Edit in SAM-Predict button (always available) ---
        self._send_to_predict_btn = QPushButton("Edit in SAM-Predict (E)")
        self._send_to_predict_btn.setToolTip(
            "選択した shape を SAM-Predict レイヤーに戻す\n"
            "A キーで再 Accept 可能"
        )
        self._send_to_predict_btn.clicked.connect(
            self._send_selected_to_predict
        )
        self._send_to_predict_btn.setEnabled(False)
        self.vbox.addWidget(self._send_to_predict_btn)

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

        # --- Display Settings group ---
        self._display_group = QGroupBox("Display Settings")
        self._display_layout = QVBoxLayout()

        # Accepted edge color row
        _edge_row = QHBoxLayout()
        _edge_row.addWidget(QLabel("Accepted edge color:"))
        self._edge_color_btn = QPushButton()
        self._edge_color_btn.setFixedSize(32, 20)
        self._edge_color_btn.setStyleSheet(
            f"background-color: {self._settings['accepted_edge_color']};"
        )
        self._edge_color_btn.clicked.connect(self._on_edge_color_btn_clicked)
        _edge_row.addWidget(self._edge_color_btn)
        _edge_row.addStretch()
        self._display_layout.addLayout(_edge_row)

        # Text color row
        _text_color_row = QHBoxLayout()
        _text_color_row.addWidget(QLabel("Annotation text color:"))
        self._text_color_btn = QPushButton()
        self._text_color_btn.setFixedSize(32, 20)
        self._text_color_btn.setStyleSheet(
            f"background-color: {self._settings['text_color']};"
        )
        self._text_color_btn.clicked.connect(self._on_text_color_btn_clicked)
        _text_color_row.addWidget(self._text_color_btn)
        _text_color_row.addStretch()
        self._display_layout.addLayout(_text_color_row)

        # Text size row
        _text_size_row = QHBoxLayout()
        _text_size_row.addWidget(QLabel("Annotation text size:"))
        self._text_size_spin = QSpinBox()
        self._text_size_spin.setRange(_TEXT_SIZE_MIN, _TEXT_SIZE_MAX)
        self._text_size_spin.setValue(self._settings["text_size"])
        self._text_size_spin.valueChanged.connect(self._on_text_size_changed)
        _text_size_row.addWidget(self._text_size_spin)
        _text_size_row.addWidget(QLabel("px"))
        _text_size_row.addStretch()
        self._display_layout.addLayout(_text_size_row)

        self._display_group.setLayout(self._display_layout)
        self.vbox.addWidget(self._display_group)

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
            edge_color=self._settings["accepted_edge_color"],
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

        self._sam3_model = None
        self._sam3_processor = None
        self._sam3_inference_state = None
        self._sam3_cleanup = None

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

        # SAM3 is local-only: hide its UI elements in API mode
        self._sam3_ckpt_widget.setVisible(
            not is_checked and self._model_selection.currentText() == "sam3"
        )
        self._sam3_prompt_group.setVisible(
            not is_checked and self._sam3_processor is not None
        )

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
            self._sam3_prompt_group.setEnabled(False)

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
            # Only enable SAM3 group if not in API mode and SAM3 is loaded
            self._sam3_prompt_group.setEnabled(
                not is_api and self._sam3_processor is not None
            )

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
        removed = None
        if (
            event is not None
            and hasattr(event, "value")
            and event.value is not None
        ):
            removed = event.value
            layer_id = id(removed)
            if layer_id in self._connected_layers:
                self._connected_layers.remove(layer_id)

        # Restore any critical layer that was deleted
        if removed is not None:
            self._restore_critical_layer_if_needed(removed)

        # Refresh layer selections
        self._refresh_layer_selections()

    def _restore_critical_layer_if_needed(self, removed_layer):
        """Re-create a critical layer if it was deleted by the user."""
        if removed_layer is self._sam_box_layer:
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

        elif removed_layer is self._sam_positive_point_layer:
            self._sam_positive_point_layer = self._viewer.add_points(
                name="SAM-Positive", face_color="green", size=10
            )
            self._sam_positive_point_layer.events.data.connect(
                self._on_sam_point_changed
            )

        elif removed_layer is self._sam_negative_point_layer:
            self._sam_negative_point_layer = self._viewer.add_points(
                name="SAM-Negative", face_color="red", size=10
            )
            self._sam_negative_point_layer.events.data.connect(
                self._on_sam_point_changed
            )

        elif removed_layer is self._labels_layer:
            shape = removed_layer.data.shape
            self._labels_layer = self._viewer.add_labels(
                np.zeros(shape, dtype="uint8"),
                name="SAM-Predict",
                blending="additive",
                opacity=0.5,
            )
            self._labels_layer.bind_key("A", self._accept_mask)
            self._labels_layer.bind_key("R", self._reject_mask)

        elif removed_layer is self._accepted_layer:
            self._accepted_layer = self._viewer.add_shapes(
                name="Accepted",
                edge_color=self._settings["accepted_edge_color"],
                edge_width=6,
                face_color="transparent",
            )
            # Re-connect output layer events (including E key binding)
            self._connect_output_layer_events(self._accepted_layer)

    def _refresh_layer_selections(self):
        """Refresh all layer selection ComboBoxes with current layer names"""
        # Store current selections
        current_image = self._image_layer_selection.currentText()

        # Block signals to prevent cascading updates during refresh
        self._image_layer_selection.blockSignals(True)
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
        self._image_layer_selection.blockSignals(False)

        # If effective selection changed while signals were blocked,
        # manually invoke handler to update widget state.
        new_image = self._image_layer_selection.currentText()
        if new_image != current_image and new_image:
            self._on_image_layer_changed()

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
                self._shapes_layer_selection.blockSignals(True)
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
                self._shapes_layer_selection.blockSignals(False)
                # Fire once after rebuild
                self._on_shapes_layer_combo_changed(
                    self._shapes_layer_selection.currentText()
                )

                self._labels_layer_selection.blockSignals(True)
                self._labels_layer_selection.clear()
                self._labels_layer_selection.blockSignals(False)
                self._save_btn.setEnabled(True)
                self.check_box.setEnabled(False)
                self.check_box.setStyleSheet("text-decoration: line-through")

            else:
                self._labels_layer_selection.blockSignals(True)
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
                self._labels_layer_selection.blockSignals(False)

                self._shapes_layer_selection.blockSignals(True)
                self._shapes_layer_selection.clear()
                self._shapes_layer_selection.blockSignals(False)
                self._on_shapes_layer_combo_changed("")
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
        layer.bind_key("E", lambda _: self._send_selected_to_predict())
        self._connected_output_layer = layer

    def _disconnect_output_layer_events(self):
        """Disconnect from previous output layer events."""
        if self._connected_output_layer is not None:
            import contextlib

            with contextlib.suppress(TypeError, RuntimeError):
                self._connected_output_layer.events.highlight.disconnect(
                    self._on_output_selection_changed
                )
            with contextlib.suppress(TypeError, RuntimeError):
                self._connected_output_layer.bind_key("E", None)
            self._connected_output_layer = None

    # --- Display Settings Handlers ---

    def _on_edge_color_btn_clicked(self):
        current = QColor(self._settings["accepted_edge_color"])
        color = QColorDialog.getColor(current, self, "Accepted Edge Color")
        if color.isValid():
            hex_color = color.name()
            self._settings["accepted_edge_color"] = hex_color
            _save_settings(self._settings)
            self._edge_color_btn.setStyleSheet(
                f"background-color: {hex_color};"
            )
            n = len(self._accepted_layer.data)
            if n > 0:
                self._accepted_layer.edge_color = [hex_color] * n
            self._accepted_layer.current_edge_color = hex_color

    def _on_text_color_btn_clicked(self):
        current = QColor(self._settings["text_color"])
        color = QColorDialog.getColor(current, self, "Annotation Text Color")
        if color.isValid():
            hex_color = color.name()
            self._settings["text_color"] = hex_color
            _save_settings(self._settings)
            self._text_color_btn.setStyleSheet(
                f"background-color: {hex_color};"
            )
            self._apply_text_settings_to_all_layers()

    def _on_text_size_changed(self, value: int):
        self._settings["text_size"] = value
        self._settings_save_timer.start(_TEXT_SIZE_SAVE_DEBOUNCE_MS)
        self._apply_text_settings_to_all_layers()

    def _flush_settings_save(self):
        _save_settings(self._settings)

    def _apply_text_settings_to_all_layers(self):
        """Apply text settings to plugin-managed output Shapes layers only.

        Plugin output layers are identified by having a "class" feature
        column (added by _ensure_features_columns).
        """
        for layer in self._viewer.layers:
            if (
                isinstance(layer, napari.layers.Shapes)
                and self._has_text_set(layer)
                and "class" in layer.features.columns
            ):
                layer.text.color = self._settings["text_color"]
                layer.text.size = self._settings["text_size"]
                layer.refresh_text()

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
        self._send_to_predict_btn.setEnabled(len(selected) == 1)

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
        """Sync class tree widget to match selected polygon(s)."""
        if len(selected) == 1:
            target = features.at[selected[0], "class"]
        else:
            class_vals = features.loc[selected, "class"].unique()
            target = class_vals[0] if len(class_vals) == 1 else None

        if target is not None:
            matched = False
            for item in self._iter_all_tree_items():
                class_str = self._get_item_class_string(item)
                if class_str == target:
                    self._class_list_widget.setCurrentItem(item)
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
        self._send_to_predict_btn.setEnabled(False)
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

    # --- Class Hierarchy Helper Methods ---

    _CLASS_SEPARATOR = "-"

    def _iter_all_tree_items(self):
        """Iterate all items in the class tree (depth-first)."""
        stack = []
        for i in range(
            self._class_list_widget.topLevelItemCount() - 1, -1, -1
        ):
            stack.append(self._class_list_widget.topLevelItem(i))
        while stack:
            item = stack.pop()
            yield item
            for i in range(item.childCount() - 1, -1, -1):
                stack.append(item.child(i))

    def _get_all_class_ids(self):
        """Return list of all class IDs in the tree."""
        import contextlib

        ids = []
        for item in self._iter_all_tree_items():
            text = item.text(0)
            with contextlib.suppress(ValueError, IndexError):
                ids.append(int(text.split(":")[0].strip()))
        return ids

    def _get_item_local_name(self, item):
        """Get the local name (without ID prefix) from a tree item."""
        text = item.text(0)
        if ": " in text:
            return text.split(": ", 1)[1]
        return text

    def _get_item_path(self, item):
        """Compute full hierarchical path for a tree item.

        Example: for item "Persian" under "Cat" under "Animal",
        returns "Animal-Cat-Persian".
        """
        parts = []
        current = item
        while current is not None:
            parts.append(self._get_item_local_name(current))
            current = current.parent()
        parts.reverse()
        return self._CLASS_SEPARATOR.join(parts)

    def _get_item_class_string(self, item):
        """Get the full class string 'id: path' for a tree item."""
        text = item.text(0)
        try:
            class_id = int(text.split(":")[0].strip())
        except (ValueError, IndexError):
            class_id = 0
        path = self._get_item_path(item)
        return f"{class_id}: {path}"

    def _get_item_id(self, item):
        """Get the numeric class ID from a tree item."""
        text = item.text(0)
        try:
            return int(text.split(":")[0].strip())
        except (ValueError, IndexError):
            return 0

    def _tree_item_count(self):
        """Return total number of items in the tree (all levels)."""
        return sum(1 for _ in self._iter_all_tree_items())

    def _find_tree_item_by_class_string(self, class_str):
        """Find a tree item matching the given class string."""
        for item in self._iter_all_tree_items():
            if self._get_item_class_string(item) == class_str:
                return item
        return None

    def _get_first_tree_item(self):
        """Return the first item in the tree, or None."""
        if self._class_list_widget.topLevelItemCount() > 0:
            return self._class_list_widget.topLevelItem(0)
        return None

    # --- Class Management Methods ---

    def _on_class_clicked(self):
        """Handle class selection in the class tree widget."""
        selected_items = self._class_list_widget.selectedItems()
        if not selected_items:
            return
        class_name = self._get_item_class_string(selected_items[0])

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
        """Add a new top-level class to the class tree."""
        class_name = self._class_name_input.text().strip()
        if not class_name:
            return

        if self._CLASS_SEPARATOR in class_name:
            print(
                f"Class name cannot contain '{self._CLASS_SEPARATOR}' "
                f"(reserved as hierarchy separator)"
            )
            return

        # Check for duplicate top-level names
        for i in range(self._class_list_widget.topLevelItemCount()):
            existing_name = self._get_item_local_name(
                self._class_list_widget.topLevelItem(i)
            )
            if existing_name == class_name:
                print("Class already exists")
                return

        ids = self._get_all_class_ids()
        next_id = find_missing_class_number(ids)

        item = QTreeWidgetItem([f"{next_id}: {class_name}"])
        self._class_list_widget.addTopLevelItem(item)
        self._sort_class_tree()
        self._class_name_input.clear()

    def _add_subclass(self):
        """Add a subclass under the currently selected class."""
        class_name = self._class_name_input.text().strip()
        if not class_name:
            return

        if self._CLASS_SEPARATOR in class_name:
            print(
                f"Class name cannot contain '{self._CLASS_SEPARATOR}' "
                f"(reserved as hierarchy separator)"
            )
            return

        selected_items = self._class_list_widget.selectedItems()
        if not selected_items:
            print("Select a parent class first")
            return

        parent_item = selected_items[0]

        # Check for duplicate sibling names
        for i in range(parent_item.childCount()):
            existing_name = self._get_item_local_name(parent_item.child(i))
            if existing_name == class_name:
                print("Subclass already exists under this parent")
                return

        ids = self._get_all_class_ids()
        next_id = find_missing_class_number(ids)

        child_item = QTreeWidgetItem([f"{next_id}: {class_name}"])
        parent_item.addChild(child_item)
        parent_item.setExpanded(True)
        self._class_name_input.clear()

    def _del_class(self):
        """Delete the selected class (blocked if in use or has children)."""
        selected_items = self._class_list_widget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]

        # Block deletion if item has children
        if item.childCount() > 0:
            QMessageBox.warning(
                self,
                "Cannot delete",
                "Cannot delete a class that has subclasses. "
                "Delete subclasses first.",
            )
            return

        class_text = self._get_item_class_string(item)

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

        parent = item.parent()
        if parent is not None:
            parent.removeChild(item)
        else:
            idx = self._class_list_widget.indexOfTopLevelItem(item)
            self._class_list_widget.takeTopLevelItem(idx)

    def _sort_class_tree(self):
        """Sort top-level tree items by numeric ID (not lexicographic)."""
        root = self._class_list_widget.invisibleRootItem()
        count = root.childCount()
        items = [root.takeChild(0) for _ in range(count)]
        items.sort(key=lambda it: self._get_item_id(it))
        for it in items:
            root.addChild(it)

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
        with open(fname, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            print("Invalid class file format")
            return

        self._class_list_widget.clear()

        if "classes" in data:
            self._load_classes_from_list(data["classes"], None)
        elif "hierarchy" in data:
            # Backward compatibility: old dict-keyed format
            self._load_hierarchy_from_dict(data["hierarchy"], None)
        elif "names" in data:
            # Backward compatibility: flat format
            for key, value in data["names"].items():
                class_id = int(key)
                class_name = str(value).strip()
                item = QTreeWidgetItem([f"{class_id}: {class_name}"])
                self._class_list_widget.addTopLevelItem(item)
        else:
            print("Invalid class file format")
            return
        self._sort_class_tree()

    def _load_classes_from_list(self, classes_list, parent_item):
        """Recursively load class hierarchy from list-of-dicts into tree."""
        for entry in classes_list:
            class_id = int(entry.get("id", 0))
            class_name = str(entry.get("name", "")).strip()
            children = entry.get("children", [])

            item = QTreeWidgetItem([f"{class_id}: {class_name}"])
            if parent_item is None:
                self._class_list_widget.addTopLevelItem(item)
            else:
                parent_item.addChild(item)

            if children:
                self._load_classes_from_list(children, item)
                item.setExpanded(True)

    def _load_hierarchy_from_dict(self, hierarchy_dict, parent_item):
        """Recursively load class hierarchy from dict into tree (legacy format)."""
        for key, value in hierarchy_dict.items():
            class_id = int(key)
            if isinstance(value, dict):
                class_name = str(value.get("name", "")).strip()
                children = value.get("children", {})
            else:
                class_name = str(value).strip()
                children = {}

            item = QTreeWidgetItem([f"{class_id}: {class_name}"])
            if parent_item is None:
                self._class_list_widget.addTopLevelItem(item)
            else:
                parent_item.addChild(item)

            if children:
                self._load_hierarchy_from_dict(children, item)
                item.setExpanded(True)

    def _get_selected_class(self):
        """Return currently selected class string (id: full_path).

        Falls back to first item, or auto-creates
        '0: object' if tree is empty.
        """
        selected = self._class_list_widget.selectedItems()
        if selected:
            return self._get_item_class_string(selected[0])
        first = self._get_first_tree_item()
        if first is not None:
            return self._get_item_class_string(first)
        # Auto-create default class
        item = QTreeWidgetItem(["0: object"])
        self._class_list_widget.addTopLevelItem(item)
        return "0: object"

    def _get_selected_class_id(self):
        """Return the numeric ID of the selected class."""
        class_str = self._get_selected_class()
        if class_str and ": " in class_str:
            return int(class_str.split(":")[0].strip())
        return 0

    def _build_categories_list(self):
        """Build COCO categories list from class tree.

        Each category includes a supercategory field that encodes
        the parent path for hierarchy reconstruction.
        """
        categories = []
        for item in self._iter_all_tree_items():
            cat_id = self._get_item_id(item)
            cat_name = self._get_item_path(item)
            parent = item.parent()
            if parent is not None:
                supercategory = self._get_item_path(parent)
            else:
                supercategory = ""
            categories.append(
                {
                    "id": cat_id,
                    "name": cat_name,
                    "supercategory": supercategory,
                }
            )
        if not categories:
            categories = [
                {
                    "id": 0,
                    "name": "object",
                    "supercategory": "",
                }
            ]
        return categories

    def _save_class_yaml(self, directory):
        """Save class definitions to class.yaml with hierarchy."""

        def _build_list(parent_item):
            count = (
                parent_item.childCount()
                if parent_item is not None
                else self._class_list_widget.topLevelItemCount()
            )
            result = []
            for i in range(count):
                child = (
                    parent_item.child(i)
                    if parent_item is not None
                    else self._class_list_widget.topLevelItem(i)
                )
                entry = {
                    "id": self._get_item_id(child),
                    "name": self._get_item_local_name(child),
                }
                if child.childCount() > 0:
                    entry["children"] = _build_list(child)
                result.append(entry)
            return result

        classes = _build_list(None)
        if classes:
            class_data = {"classes": classes}
            path = os.path.join(directory, "class.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(
                    class_data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                )

    def _restore_categories_to_tree(self, categories):
        """Rebuild class tree from COCO categories list.

        Uses the supercategory field to reconstruct the hierarchy.
        Categories with empty supercategory are top-level.
        Categories whose name contains the separator are treated
        as hierarchical paths.
        """
        sep = self._CLASS_SEPARATOR

        # Build path → category mapping
        # Sort by path depth so parents are created before children
        sorted_cats = sorted(
            categories, key=lambda c: c.get("name", "").count(sep)
        )

        path_to_item = {}
        for cat in sorted_cats:
            cat_id = cat.get("id", 0)
            cat_name = cat.get("name", "object")
            supercategory = cat.get("supercategory", "")

            # Use supercategory to determine hierarchy.
            # This plugin writes cat_name as the full path (e.g. "A-B-C")
            # and supercategory as the parent path (e.g. "A-B"), so
            # cat_name always starts with supercategory + sep.
            # Legacy/external COCO files use supercategory == cat_name or a
            # name that does NOT start with supercategory + sep, so we
            # preserve the full name as-is in those cases.
            plugin_hierarchy = supercategory and cat_name.startswith(
                supercategory + sep
            )
            if plugin_hierarchy:
                local_name = cat_name[len(supercategory) + len(sep) :]
                parent_path = supercategory
            else:
                parent_path = ""
                local_name = cat_name

            item = QTreeWidgetItem([f"{cat_id}: {local_name}"])

            if parent_path and parent_path in path_to_item:
                parent_item = path_to_item[parent_path]
                parent_item.addChild(item)
                parent_item.setExpanded(True)
            else:
                self._class_list_widget.addTopLevelItem(item)

            path_to_item[cat_name] = item

    def _browse_sam3_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM3 Checkpoint",
            "",
            "PyTorch checkpoint (*.pt *.pth)",
        )
        if path:
            self._sam3_ckpt_lineedit.setText(path)

    def _load_sam3_model_internal(self):
        """Load SAM3 model (HuggingFace auto or local checkpoint)."""
        # Move SAM1 model off MPS/CUDA before loading SAM3 on CPU,
        # so no MPS tensors remain that could contaminate SAM3 inference.
        if self._sam_model is not None:
            try:
                self._sam_model.to("cpu")
            except Exception:
                pass
        self._sam_model = None
        self.sam_predictor = None

        # Revert previous pin_memory patch before applying a new one,
        # so reload doesn't capture the patched version as the original.
        if self._sam3_cleanup is not None:
            self._sam3_cleanup()
            self._sam3_cleanup = None

        ckpt = self._sam3_ckpt_lineedit.text().strip() or None
        try:
            from ._utils import load_sam3_model

            self._sam3_model, self._sam3_processor, self._sam3_cleanup = (
                load_sam3_model(ckpt)
            )
        except ImportError as exc:
            self._sam3_model = None
            self._sam3_processor = None
            self._sam3_inference_state = None
            QMessageBox.critical(
                self,
                "SAM3 Not Installed",
                f"sam3 package is not installed.\n\n"
                f"Install with: uv sync --extra sam3\n\nDetail: {exc}",
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._sam3_model = None
            self._sam3_processor = None
            self._sam3_inference_state = None
            hint = (
                f"Checkpoint: {ckpt}"
                if ckpt
                else (
                    "Make sure you have requested access at:\n"
                    "https://huggingface.co/facebook/sam3\n"
                    "or specify a locally downloaded checkpoint."
                )
            )
            QMessageBox.critical(
                self,
                "SAM3 Load Failed",
                f"Failed to load SAM3 model.\n\n{hint}\n\nDetail: {exc}",
            )
            return
        # Reset SAM1 state
        self._sam_model = None
        self.sam_predictor = None
        is_manual = self._manual_mode_checkbox.isChecked()
        self._sam3_prompt_group.setVisible(True)
        self._sam3_prompt_group.setEnabled(not is_manual)
        print("SAM3 loaded")
        if self._image_layer_selection.currentText():
            self._on_image_layer_changed(True)

    def _load_model(self):
        if self._use_api_checkbox.isChecked():
            print("Local model loading is not required in API mode")
            return

        model_name = self._model_selection.currentText()
        if model_name == "sam3":
            self._load_sam3_model_internal()
            return

        try:
            from segment_anything import SamPredictor
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"Failed to import segment_anything: {exc}")
            return

        self._sam_model = load_model(model_name)
        self._sam_model.to(device=self.device)
        self.sam_predictor = SamPredictor(self._sam_model)
        # Reset SAM3 state
        if self._sam3_cleanup is not None:
            self._sam3_cleanup()
            self._sam3_cleanup = None
        self._sam3_model = None
        self._sam3_processor = None
        self._sam3_inference_state = None
        self._sam3_prompt_group.setVisible(False)
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

        # Update predictor image (for local SAM/SAM3 mode, even
        # if in manual mode). Ensures predictor has correct
        # image when switching back to SAM mode.
        if image_layer is not None and not self._use_api_checkbox.isChecked():
            preprocessed = preprocess(
                image_layer.data,
                self._image_type,
                self._current_slice,
            )
            if self._sam3_processor is not None:
                from PIL import Image as _PILImage

                if preprocessed.dtype != np.uint8:
                    lo = float(preprocessed.min())
                    hi = float(preprocessed.max())
                    if hi - lo > 0:
                        preprocessed = (
                            (preprocessed - lo) / (hi - lo) * 255
                        ).astype(np.uint8)
                    else:
                        preprocessed = np.zeros_like(
                            preprocessed, dtype=np.uint8
                        )
                self._sam3_inference_state = self._sam3_processor.set_image(
                    _PILImage.fromarray(preprocessed)
                )
                print("SAM3: image set")
            elif self.sam_predictor is not None:
                self.sam_predictor.set_image(preprocessed)
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

    def _predict_local_sam3(self):
        """SAM3 SAM1-style single prediction (box prompt)."""
        if self._sam3_inference_state is None or self._input_box is None:
            return
        masks, _, _ = self._sam3_model.predict_inst(
            self._sam3_inference_state,
            point_coords=None,
            point_labels=None,
            box=self._input_box[None, :],
            multimask_output=False,
        )
        self._labels_layer.data = masks[0].astype("uint8")

    def _pixel_box_to_sam3_norm_cxcywh(self, box_xyxy) -> list:
        """Convert pixel [x1,y1,x2,y2] to SAM3 normalized CXCYWH list."""
        from sam3.model.box_ops import box_xyxy_to_cxcywh

        shape = self._get_image_shape()
        if shape is None:
            raise ValueError("No image layer selected")
        H, W = shape
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        xyxy = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        cxcywh = box_xyxy_to_cxcywh(xyxy)
        norm = cxcywh / torch.tensor([W, H, W, H], dtype=torch.float32)
        return norm.flatten().tolist()

    def _predict_local(self):
        if self._sam3_processor is not None:
            self._predict_local_sam3()
        elif self.sam_predictor is not None:
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

    def _accept_multi_masks(self, masks, class_str: str) -> int:
        """Add N binary masks as polygons to the Accepted Shapes layer.

        Masks whose bbox-IoU with an existing shape exceeds the
        threshold set in ``_iou_threshold_spin`` are skipped.
        """
        output_name = self._shapes_layer_selection.currentText()
        if not output_name:
            return 0
        output_layer = self._get_layer_by_name_safe(output_name)
        if not isinstance(output_layer, napari.layers.shapes.shapes.Shapes):
            return 0

        self._ensure_features_columns(output_layer)
        if not self._has_text_set(output_layer):
            output_layer.text = {
                "string": "{class}",
                "anchor": "upper_left",
                "size": self._settings["text_size"],
                "color": self._settings["text_color"],
            }

        from skimage.measure import find_contours

        iou_threshold = self._iou_threshold_spin.value()
        same_class_only = self._iou_same_class_checkbox.isChecked()

        # Pre-compute existing bboxes (inclusive int coords)
        existing_bboxes = []
        if iou_threshold > 0:
            features = output_layer.features.reset_index(drop=True)
            for i, poly in enumerate(output_layer.data):
                if same_class_only:
                    ex_class = str(features.iloc[i]["class"])
                    if ex_class != class_str:
                        continue
                r_min, c_min = poly.min(axis=0)
                r_max, c_max = poly.max(axis=0)
                existing_bboxes.append(
                    (
                        math.floor(r_min),
                        math.floor(c_min),
                        math.ceil(r_max),
                        math.ceil(c_max),
                    )
                )

        count = 0
        skipped = 0
        for i in range(len(masks)):
            m = masks[i]
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            m = m.squeeze().astype(bool)
            if not m.any():
                continue

            # bbox-IoU check
            if existing_bboxes and iou_threshold > 0:
                rows, cols = np.where(m)
                nr_min = int(rows.min())
                nr_max = int(rows.max())
                nc_min = int(cols.min())
                nc_max = int(cols.max())
                dup = False
                for er, ec, er2, ec2 in existing_bboxes:
                    ir_min = max(nr_min, er)
                    ic_min = max(nc_min, ec)
                    ir_max = min(nr_max, er2)
                    ic_max = min(nc_max, ec2)
                    if ir_min <= ir_max and ic_min <= ic_max:
                        inter = (ir_max - ir_min + 1) * (ic_max - ic_min + 1)
                    else:
                        inter = 0
                    a_new = (nr_max - nr_min + 1) * (nc_max - nc_min + 1)
                    a_ex = (er2 - er + 1) * (ec2 - ec + 1)
                    union = a_new + a_ex - inter
                    if union > 0 and inter / union >= iou_threshold:
                        dup = True
                        break
                if dup:
                    skipped += 1
                    continue

            contours = find_contours(m)
            if not contours:
                continue
            polygon = contours[0].astype(int)
            output_layer.feature_defaults["class"] = class_str
            for k, v in _ATTR_DEFAULTS.items():
                if k != "class":
                    output_layer.feature_defaults[k] = v
            output_layer.add_polygons([polygon], edge_width=2)

            # Add new bbox for subsequent duplicate checks
            if iou_threshold > 0:
                r_min, c_min = polygon.min(axis=0)
                r_max, c_max = polygon.max(axis=0)
                existing_bboxes.append(
                    (
                        int(r_min),
                        int(c_min),
                        int(r_max),
                        int(c_max),
                    )
                )
            count += 1

        if skipped > 0:
            print(f"SAM3: {skipped} masks skipped (IoU >= {iou_threshold})")
        if count > 0:
            output_layer.refresh_text()
        return count

    def _get_selected_exemplar_boxes(self) -> list:
        """Return bbox [x1,y1,x2,y2] list from selected output shapes."""
        output_name = self._shapes_layer_selection.currentText()
        output_layer = self._get_layer_by_name_safe(output_name)
        if not isinstance(output_layer, napari.layers.shapes.shapes.Shapes):
            return []
        selected = sorted(output_layer.selected_data)
        if not selected:
            return []
        boxes = []
        for idx in selected:
            polygon = output_layer.data[idx]
            r_min, c_min = polygon.min(axis=0).astype(float)
            r_max, c_max = polygon.max(axis=0).astype(float)
            boxes.append(np.array([c_min, r_min, c_max, r_max]))
        return boxes

    def _get_sam3_box_only_masks(self, inference_state):
        """Return masks from inference_state after add_geometric_prompt.

        Raises TypeError if inference_state is not a dict, or KeyError if
        none of the candidate keys are found (to surface API mismatches
        clearly rather than silently returning None).
        """
        _CANDIDATE_KEYS = ("pred_masks", "masks")

        if not isinstance(inference_state, dict):
            raise TypeError(
                f"SAM3 Box-only: inference_state is "
                f"{type(inference_state).__name__}, expected dict. "
                "Check add_geometric_prompt() return type."
            )
        for key in _CANDIDATE_KEYS:
            if key in inference_state and inference_state[key] is not None:
                return inference_state[key]
        raise KeyError(
            f"SAM3 Box-only: none of {_CANDIDATE_KEYS} found in "
            f"inference_state (keys={list(inference_state.keys())}). "
            "Update _CANDIDATE_KEYS after checking "
            "add_geometric_prompt() output."
        )

    def _on_sam3_detect_all(self):
        """Detect All: accept all instances using selected prompt mode."""
        if self._sam3_processor is None or self._sam3_inference_state is None:
            print("SAM3 not loaded")
            return

        use_text = not self._sam3_prompt_box_radio.isChecked()
        use_box = not self._sam3_prompt_text_radio.isChecked()

        # ① Reset prompts upfront so early returns leave clean state
        self._sam3_processor.reset_all_prompts(self._sam3_inference_state)

        # ② Collect exemplar boxes (only when output layer is active)
        output_name = self._shapes_layer_selection.currentText()
        output_layer = self._get_layer_by_name_safe(output_name)
        active = self._viewer.layers.selection.active
        has_exemplar = (
            use_box
            and active is output_layer
            and isinstance(output_layer, napari.layers.shapes.shapes.Shapes)
        )
        exemplar_boxes = (
            self._get_selected_exemplar_boxes() if has_exemplar else []
        )

        # Collect all box sources
        all_boxes = []
        if use_box:
            if self._input_box is not None:
                all_boxes.append(self._input_box)
            all_boxes.extend(exemplar_boxes)
            if not all_boxes:
                print(
                    "Box mode: SAM-Box に矩形を描くか、"
                    "出力 Shapes layer で shape を選択してください"
                )
                return

        # ③ Determine class_str
        if exemplar_boxes:
            self._ensure_features_columns(output_layer)
            selected = sorted(output_layer.selected_data)
            classes = {
                str(output_layer.features.at[idx, "class"]) for idx in selected
            }
            if len(classes) > 1:
                print(
                    "Detect All: 選択 shape のクラスが混在しています。"
                    "同じクラスの shape のみ選択してください"
                )
                return
            class_str = classes.pop()
        else:
            class_str = self._get_selected_class()
        text = (
            str(class_str).split(": ", 1)[-1].split("-")[-1].strip()
            if class_str
            else ""
        )

        # ④ Add box prompts
        if use_box:
            for box_xyxy in all_boxes:
                norm_box = self._pixel_box_to_sam3_norm_cxcywh(box_xyxy)
                self._sam3_inference_state = (
                    self._sam3_processor.add_geometric_prompt(
                        state=self._sam3_inference_state,
                        box=norm_box,
                        label=True,
                    )
                )

        # ⑤ Run inference
        if use_text:
            output = self._sam3_processor.set_text_prompt(
                state=self._sam3_inference_state, prompt=text
            )
            masks = output.get("masks")
        else:
            try:
                masks = self._get_sam3_box_only_masks(
                    self._sam3_inference_state
                )
            except (TypeError, KeyError) as exc:
                QMessageBox.critical(
                    self,
                    "SAM3 Box-only Error",
                    f"Box-only マスクの取得に失敗しました。\n\n"
                    f"Detail: {exc}\n\n"
                    "sam3 の API が変わった可能性があります。"
                    "_get_sam3_box_only_masks() の "
                    "_CANDIDATE_KEYS を確認してください。",
                )
                return

        if masks is None or len(masks) == 0:
            print("SAM3: no objects detected")
            return

        n = self._accept_multi_masks(masks, class_str)
        self._sam_box_layer.data = []
        self._input_box = None
        self._labels_layer.data = np.zeros_like(self._labels_layer.data)
        self._sam3_processor.reset_all_prompts(self._sam3_inference_state)
        print(f"SAM3 Detect All: {n} objects accepted as '{class_str}'")

    def _send_selected_to_predict(self):
        """Move selected Accepted shape back to SAM-Predict for editing."""
        output_name = self._shapes_layer_selection.currentText()
        output_layer = self._get_layer_by_name_safe(output_name)
        if not isinstance(output_layer, napari.layers.shapes.shapes.Shapes):
            return

        selected = list(output_layer.selected_data)
        if len(selected) != 1:
            print("1つの shape を選択してください")
            return

        idx = selected[0]
        polygon = output_layer.data[idx]
        H, W = self._labels_layer.data.shape

        from skimage.draw import polygon2mask as _polygon2mask

        mask = _polygon2mask((H, W), polygon).astype("uint8")
        self._labels_layer.data = mask
        output_layer.selected_data = {idx}
        output_layer.remove_selected()
        self._viewer.layers.selection.active = self._labels_layer
        print(
            "Shape を SAM-Predict に送りました。A で再 Accept してください。"
        )

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
                            "size": self._settings["text_size"],
                            "color": self._settings["text_color"],
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

        image_path = getattr(
            getattr(image_layer, "source", None), "path", None
        )
        if not image_path:
            print("Image layer has no file path")
            return False
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
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        self._save_class_yaml(os.path.dirname(image_path))
        print("saved")
        return True

    # --- Annotation Loading ---

    def _on_load_annotations_clicked(self):
        """Handle Load Annotations button click."""
        if self._radio_btn_group.checkedId() != 0:
            QMessageBox.warning(
                self,
                "Cannot load annotations",
                "Annotation loading is only supported in instance mode.",
            )
            return
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

        if not self._load_annotations(
            json_path, output_layer, needs_replace=False
        ):
            print(
                f"Warning: auto-load failed for "
                f"{os.path.basename(json_path)}"
            )

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
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            OSError,
        ) as e:
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

        # Restore categories to class tree (with hierarchy)
        if categories:
            self._class_list_widget.clear()
            self._restore_categories_to_tree(categories)
            self._sort_class_tree()

        # Build category_id → class string mapping
        cat_id_to_str = {}
        for item in self._iter_all_tree_items():
            cid = self._get_item_id(item)
            cat_id_to_str[cid] = self._get_item_class_string(item)

        self._ensure_features_columns(output_layer)

        if not self._has_text_set(output_layer):
            output_layer.text = {
                "string": "{class}",
                "anchor": "upper_left",
                "size": self._settings["text_size"],
                "color": self._settings["text_color"],
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
