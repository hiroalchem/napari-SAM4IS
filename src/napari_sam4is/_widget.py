import base64
import io
import json
import os

import napari
import numpy as np
import requests
import torch
from PIL import Image
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from requests.adapters import HTTPAdapter
from segment_anything import SamPredictor, sam_model_registry
from urllib3.util.retry import Retry

from ._utils import (
    check_image_type,
    create_json,
    find_first_missing,
    label2polygon,
    load_model,
    preprocess,
)


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
        self._current_target_image_name = None

        #self._corner = None

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

        # Model selection
        self._model_selection = QComboBox()
        self._model_selection.addItems(list(sam_model_registry.keys()))
        self.vbox.addWidget(self._model_selection)
        self._model_load_btn = QPushButton("load model")
        self._model_load_btn.clicked.connect(self._load_model)
        self.vbox.addWidget(self._model_load_btn)
        self.vbox.addWidget(QLabel("input image layer"))
        self._image_layer_selection = QComboBox()
        self._image_layer_selection.addItems([layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.image.image.Image)])
        # self._image_layer_selection.currentTextChanged.connect(self._on_image_layer_changed)
        self.vbox.addWidget(self._image_layer_selection)

        self.vbox.addWidget(QLabel("select output layer type \nif you want to use the output\n"
                                   "as a mask, select 'labels'.\n"
                                   "3D image is currently only\nsupported for 'labels'"))
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
        self._shapes_layer_selection.addItems([layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.shapes.shapes.Shapes)])
        self.vbox.addWidget(self._shapes_layer_selection)

        self.vbox.addWidget(QLabel("output labels layer"))
        self._labels_layer_selection = QComboBox()
        self._labels_layer_selection.addItems([layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.labels.labels.Labels)])
        self.vbox.addWidget(self._labels_layer_selection)

        self.vbox.addWidget(QLabel("Save as coco format \nin the same directory \nwith the input image"))
        self._save_btn = QPushButton("save")
        self._save_btn.clicked.connect(self._save)
        self.vbox.addWidget(self._save_btn)
        """
        self._test_btn = QPushButton("corner")
        self._test_btn.clicked.connect(self.print_corner_value)
        self.vbox.addWidget(self._test_btn)
        """

        self._sam_box_layer = self._viewer.add_shapes(name="SAM-Box", edge_color="red", edge_width=2, face_color="transparent")
        self._sam_box_layer.mouse_drag_callbacks.append(self._on_sam_box_created)
        self._sam_box_layer.bind_key("R", self._reject_all_boxes)
        self.lock_controls(self._sam_box_layer)
        self._sam_positive_point_layer = self._viewer.add_points(name="SAM-Positive", face_color="green", size=10)
        self._sam_negative_point_layer = self._viewer.add_points(name="SAM-Negative", face_color="red", size=10)
        #self._sam_positive_point_layer.mouse_drag_callbacks.append(self._on_sam_point_created)
        #self._sam_negative_point_layer.mouse_drag_callbacks.append(self._on_sam_point_created)
        self._sam_positive_point_layer.events.data.connect(self._on_sam_point_changed)
        self._sam_negative_point_layer.events.data.connect(self._on_sam_point_changed)

        if (self._image_layer_selection.currentText() != "")&(self._image_layer_selection.currentText() in self._viewer.layers):
            image_layer = self._get_layer_by_name_safe(self._image_layer_selection.currentText())
            if image_layer is not None:
                self._image_type = check_image_type(self._viewer, self._image_layer_selection.currentText())
                if "stack" in self._image_type:
                    shape = image_layer.data.shape[1:3]
                else:
                    shape = image_layer.data.shape[:2]
            else:
                shape = (100, 100)
        else:
            shape = (100, 100)

        self._labels_layer = self._viewer.add_labels(np.zeros(shape, dtype='uint8'), name="SAM-Predict", blending="additive", opacity=0.5)

        self._accepted_layer = self._viewer.add_shapes(name="Accepted", edge_color="green", edge_width=6,
                                                      face_color="transparent")

        self.setLayout(self.vbox)
        self.show()

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self._sam_model = None
        self.sam_predictor = None

        self._viewer.layers.events.inserted.connect(self._on_layer_list_changed)
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

    def _on_layer_list_changed(self, event):
        if event is not None:
            print(event.value)
            
            # Connect to name change event for newly added layers
            if hasattr(event, 'value') and event.value is not None:
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
        layer = event.source if hasattr(event, 'source') else None
        layer_name = layer.name if layer else 'unknown'
        print(f"Layer name changed to '{layer_name}'")
        self._refresh_layer_selections()
    
    def _on_layer_removed(self, event):
        """Handle layer removal and cleanup connections"""
        if event is not None and hasattr(event, 'value') and event.value is not None:
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
        current_shapes = self._shapes_layer_selection.currentText() if self._shapes_layer_selection else ""
        current_labels = self._labels_layer_selection.currentText() if self._labels_layer_selection else ""

        # Update image layer selection
        self._image_layer_selection.clear()
        image_layers = [layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.image.image.Image)]
        self._image_layer_selection.addItems(image_layers)

        # Restore selection if layer still exists
        if current_image in image_layers:
            self._image_layer_selection.setCurrentText(current_image)

        # Update shapes and labels selections via radio button toggle
        self._on_radio_btn_toggled()

    def _on_radio_btn_toggled(self):
        button_id = self._radio_btn_group.checkedId()
        if (self._shapes_layer_selection is not None) & (self._labels_layer_selection is not None):
            # Store current selections
            current_shapes = self._shapes_layer_selection.currentText()
            current_labels = self._labels_layer_selection.currentText()

            if button_id == 0:
                self._shapes_layer_selection.clear()
                shape_layers = [layer.name for layer in self._viewer.layers
                              if (isinstance(layer, napari.layers.shapes.shapes.Shapes)) and
                                 (layer.name != self._sam_box_layer.name)]
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
                label_layers = [layer.name for layer in self._viewer.layers
                              if (isinstance(layer, napari.layers.labels.labels.Labels)) and
                                 (layer.name != self._labels_layer.name)]
                self._labels_layer_selection.addItems(label_layers)

                # Restore selection if layer still exists
                if current_labels in label_layers:
                    self._labels_layer_selection.setCurrentText(current_labels)

                self._shapes_layer_selection.clear()
                self._save_btn.setEnabled(False)
                self.check_box.setEnabled(True)
                self.check_box.setStyleSheet("text-decoration: none")

    def _load_model(self):
        if self._use_api_checkbox.isChecked():
            print("Local model loading is not required in API mode")
            return

        model_name = self._model_selection.currentText()
        self._sam_model = load_model(model_name)
        self._sam_model.to(device=self.device)
        self.sam_predictor = SamPredictor(self._sam_model)
        print("model loaded")
        if self._image_layer_selection.currentText() != "":
            self._on_image_layer_changed(True)

    def _on_image_layer_changed(self, set_image=False):
        print("image_layer_changed")

        # Skip local setup in API mode
        if self._use_api_checkbox.isChecked():
            if (self._image_layer_selection.currentText() != "")&(self._image_layer_selection.currentText() in self._viewer.layers):
                image_layer = self._get_layer_by_name_safe(self._image_layer_selection.currentText())
                if image_layer is not None:
                    self._current_target_image_name = self._image_layer_selection.currentText()
                    self._image_type = check_image_type(self._viewer, self._image_layer_selection.currentText())
                    if "stack" in self._image_type:
                        self._current_slice, _, _ = self._viewer.dims.current_step
                    else:
                        self._current_slice = None
                    print('Image selected for API mode')
            return

        if self.sam_predictor is not None:
            if (self._image_layer_selection.currentText() != "")&(self._image_layer_selection.currentText() in self._viewer.layers):
                image_layer = self._get_layer_by_name_safe(self._image_layer_selection.currentText())
                if image_layer is not None and ((self._current_target_image_name != self._image_layer_selection.currentText()) or set_image):
                    self._current_target_image_name = self._image_layer_selection.currentText()
                    self._image_type = check_image_type(self._viewer, self._image_layer_selection.currentText())
                    if "stack" in self._image_type:
                        self._current_slice, _, _ = self._viewer.dims.current_step
                    else:
                        self._current_slice = None
                    self.sam_predictor.set_image(preprocess(image_layer.data, self._image_type, self._current_slice))
                    print('Set image')
                    # self._corner = self._viewer.layers[self._image_layer_selection.currentText()].corner_pixels

    def _on_sam_box_created(self, layer, event):
        # mouse click
        yield
        # mouse move
        while event.type == 'mouse_move':
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
        if (len(self._sam_positive_point_layer.data) != 0) or (self._input_box is not None):
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
                box=self._input_box[None, :] if self._input_box is not None else None,
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
            image_layer = self._get_layer_by_name_safe(self._image_layer_selection.currentText())
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
                pil_image = Image.fromarray(image_data).convert('RGB')
            else:  # RGB
                pil_image = Image.fromarray(image_data.astype(np.uint8))

            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=100)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Wrap bounding box coordinates in list
            coords = [self._input_box.tolist()]

            # Create API request data
            request_data = {
                "input": {
                    "image_data": image_b64,
                    "coords": coords,
                    "output_format": "geojson",
                    "image_format": "jpeg"
                }
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Create session with retry functionality
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504, 520]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)

            print("Sending request to API...")
            response = session.post(api_url, headers=headers, json=request_data, timeout=300)
            response.raise_for_status()
            result = response.json()

            if 'error' in result:
                print(f"API error: {result['error']}")
                return

            # Generate mask from GeoJSON
            geojson_data = result['output']['geojson']
            mask = self._geojson_to_mask(geojson_data, image_data.shape)

            self._labels_layer.data = mask.astype(np.uint8)
            print("API prediction completed")

        except Exception as e:
            print(f"API error: {str(e)}")

        self._viewer.layers.selection.active = self._labels_layer

    def _geojson_to_mask(self, geojson_data, image_shape):
        """Convert GeoJSON data to mask"""
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                coordinates = feature['geometry']['coordinates'][0]
                coords_array = np.array(coordinates)

                from matplotlib.path import Path
                path = Path(coords_array)

                y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
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
        self._point_label = np.array(len(positive_points) * [1] + len(negative_points) * [0])
        coords = np.concatenate((positive_points, negative_points), axis=0)
        self._input_point = coords[:, ::-1].astype(np.int32)


    def _get_layer_by_name_safe(self, layer_name):
        """Safely get layer by name, handling case where layer was renamed"""
        try:
            return self._viewer.layers[layer_name]
        except KeyError:
            print(f"Warning: Layer '{layer_name}' not found. It may have been renamed.")
            # Refresh layer selections and return None
            self._refresh_layer_selections()
            return None

    def _accept_mask(self, layer):
        button_id = self._radio_btn_group.checkedId()
        if button_id == 0:
            if self._shapes_layer_selection.currentText() != "":
                output_layer = self._get_layer_by_name_safe(self._shapes_layer_selection.currentText())
                if output_layer and isinstance(output_layer, napari.layers.shapes.shapes.Shapes):
                    output_layer.add_polygons(label2polygon(self._labels_layer.data), edge_width=2)
                    self._viewer.layers.selection.active = self._sam_box_layer
                elif output_layer is None:
                    print("Output shapes layer not found or was renamed")
                    return
            else:
                pass
        else:
            if self._labels_layer_selection.currentText() != "":
                output_layer = self._get_layer_by_name_safe(self._labels_layer_selection.currentText())
                if output_layer and isinstance(output_layer, napari.layers.labels.labels.Labels):
                    if self._current_slice is not None:
                        if self.check_box.isChecked():
                            num = find_first_missing(output_layer.data[self._current_slice])
                        else:
                            num = 1
                        # 既存のラベル（1以上）がある場所は上書きしない
                        current_data = output_layer.data[self._current_slice]
                        new_mask = self._labels_layer.data * num
                        # 既存のラベルが0の場所のみ新しいマスクを適用
                        mask_to_apply = (current_data == 0) & (new_mask > 0)
                        output_layer.data[self._current_slice] = current_data + new_mask * mask_to_apply
                        output_layer.refresh()
                    else:
                        if self.check_box.isChecked():
                            num = find_first_missing(output_layer.data)
                        else:
                            num = 1
                        # 既存のラベル（1以上）がある場所は上書きしない
                        current_data = output_layer.data
                        new_mask = self._labels_layer.data * num
                        # 既存のラベルが0の場所のみ新しいマスクを適用
                        mask_to_apply = (current_data == 0) & (new_mask > 0)
                        output_layer.data = current_data + new_mask * mask_to_apply
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

    def _reject_mask(self, layer):
        self._labels_layer.data = np.zeros_like(self._labels_layer.data)
        self._viewer.layers.selection.active = self._sam_box_layer
        self._input_box = None
        self._sam_positive_point_layer.data = []
        self._sam_negative_point_layer.data = []

    def _save(self):
        if self._shapes_layer_selection.currentText() != "":
            image_layer = self._get_layer_by_name_safe(self._image_layer_selection.currentText())
            if image_layer is None:
                print("Image layer not found or was renamed")
                return

            image_path = image_layer.source.path
            image_name = os.path.basename(image_path)
            output_layer = self._get_layer_by_name_safe(self._shapes_layer_selection.currentText())
            if output_layer is None:
                print("Output layer not found or was renamed")
                return
            if isinstance(output_layer, napari.layers.shapes.shapes.Shapes):
                output_path = os.path.join(os.path.dirname(image_path), os.path.splitext(image_name)[0] + ".json")
                data = create_json(image_layer.data, image_name, output_layer.data)
                with open(output_path, 'w') as f:
                    json.dump(data, f)
                print("saved")
        else:
            pass


    def lock_controls(self, layer, locked=True):
        widget_list = [
            'ellipse_button',
            'line_button',
            'path_button',
            'vertex_remove_button',
            'vertex_insert_button',
            'move_back_button',
            'move_front_button',
            'polygon_button',
            'select_button',
            'direct_button',
            'delete_button',
        ]
        qctrl = self._viewer.window.qt_viewer.controls.widgets[layer]
        for wdg in widget_list:
            getattr(qctrl, wdg).setEnabled(not locked)


    def print_corner_value(self):
        print(self._viewer.dims.current_step)
        image_layer = self._get_layer_by_name_safe(self._image_layer_selection.currentText())
        if image_layer is not None:
            print(image_layer.corner_pixels)
        else:
            print("Image layer not found")









