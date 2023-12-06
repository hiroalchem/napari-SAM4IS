import json
import os

import napari
import numpy as np
import torch
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QComboBox, QLabel, QButtonGroup, QRadioButton, QCheckBox
from segment_anything import sam_model_registry, SamPredictor

from ._utils import load_model, preprocess, label2polygon, create_json, check_image_type, find_first_missing


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
        self.lock_controls(self._sam_box_layer)
        self._sam_positive_point_layer = self._viewer.add_points(name="SAM-Positive", face_color="green", size=10)
        self._sam_negative_point_layer = self._viewer.add_points(name="SAM-Negative", face_color="red", size=10)
        #self._sam_positive_point_layer.mouse_drag_callbacks.append(self._on_sam_point_created)
        #self._sam_negative_point_layer.mouse_drag_callbacks.append(self._on_sam_point_created)
        self._sam_positive_point_layer.events.data.connect(self._on_sam_point_changed)
        self._sam_negative_point_layer.events.data.connect(self._on_sam_point_changed)

        if (self._image_layer_selection.currentText() != "")&(self._image_layer_selection.currentText() in self._viewer.layers):
            self._image_type = check_image_type(self._viewer, self._image_layer_selection.currentText())
            if "stack" in self._image_type:
                shape = self._viewer.layers[self._image_layer_selection.currentText()].data.shape[1:3]
            else:
                shape = self._viewer.layers[self._image_layer_selection.currentText()].data.shape[:2]
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
        self._viewer.layers.events.removed.connect(self._on_layer_list_changed)

        self._labels_layer.bind_key("A", self._accept_mask)
        self._labels_layer.bind_key("R", self._reject_mask)

        self._on_layer_list_changed(None)
        self._radio_btn_shape.setChecked(True)

    def _on_layer_list_changed(self, event):
        if event is not None:
            print(event.value)
            self._image_layer_selection.clear()
            self._image_layer_selection.addItems([layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.image.image.Image)])
            [self._viewer.layers.move(i, 0) for i, layer in enumerate(self._viewer.layers) if isinstance(layer, napari.layers.image.image.Image)]
            if isinstance(event.value, napari.layers.image.image.Image):
               self._on_image_layer_changed(None)
        self._on_radio_btn_toggled()

    def _on_radio_btn_toggled(self):
        button_id = self._radio_btn_group.checkedId()
        if (self._shapes_layer_selection is not None) & (self._labels_layer_selection is not None):
            if button_id == 0:
                self._shapes_layer_selection.clear()
                self._shapes_layer_selection.addItems([layer.name for layer in self._viewer.layers if (isinstance(layer, napari.layers.shapes.shapes.Shapes))&(layer.name != self._sam_box_layer.name)])
                self._labels_layer_selection.clear()
                self._save_btn.setEnabled(True)
                self.check_box.setEnabled(False)
                self.check_box.setStyleSheet("text-decoration: line-through")

            else:
                self._labels_layer_selection.clear()
                self._labels_layer_selection.addItems([layer.name for layer in self._viewer.layers if (isinstance(layer, napari.layers.labels.labels.Labels))&(layer.name != self._labels_layer.name)])
                self._shapes_layer_selection.clear()
                self._save_btn.setEnabled(False)
                self.check_box.setEnabled(True)
                self.check_box.setStyleSheet("text-decoration: none")

    def _load_model(self):
        model_name = self._model_selection.currentText()
        self._sam_model = load_model(model_name)
        self._sam_model.to(device=self.device)
        self.sam_predictor = SamPredictor(self._sam_model)
        print("model loaded")
        if self._image_layer_selection.currentText() != "":
            self._on_image_layer_changed(None)

    def _on_image_layer_changed(self, index):
        print("image_layer_changed")
        if self.sam_predictor is not None:
            if (self._image_layer_selection.currentText() != "")&(self._image_layer_selection.currentText() in self._viewer.layers):
                if self._current_target_image_name != self._image_layer_selection.currentText():
                    self._current_target_image_name = self._image_layer_selection.currentText()
                    self._image_type = check_image_type(self._viewer, self._image_layer_selection.currentText())
                    if "stack" in self._image_type:
                        self._current_slice, _, _ = self._viewer.dims.current_step
                    else:
                        self._current_slice = None
                    self.sam_predictor.set_image(preprocess(self._viewer.layers[self._image_layer_selection.currentText()].data, self._image_type, self._current_slice))
                    print('set image')
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
        if self.sam_predictor is not None:
            masks, _, _ = self.sam_predictor.predict(
                point_coords=self._input_point,
                point_labels=self._point_label,
                box=self._input_box[None, :] if self._input_box is not None else None,
                multimask_output=False,
            )
            self._labels_layer.data = masks[0] * 1
        self._viewer.layers.selection.active = self._labels_layer


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


    def _accept_mask(self, layer):
        button_id = self._radio_btn_group.checkedId()
        if button_id == 0:
            if self._shapes_layer_selection.currentText() != "":
                output_layer = self._viewer.layers[self._shapes_layer_selection.currentText()]
                if isinstance(output_layer, napari.layers.shapes.shapes.Shapes):
                    output_layer.add_polygons(label2polygon(self._labels_layer.data), edge_width=6)
                    self._viewer.layers.selection.active = self._sam_box_layer
                else:
                    pass
            else:
                pass
        else:
            if self._labels_layer_selection.currentText() != "":
                output_layer = self._viewer.layers[self._labels_layer_selection.currentText()]
                if isinstance(output_layer, napari.layers.labels.labels.Labels):
                    if self._current_slice is not None:
                        if self.check_box.isChecked():
                            num = find_first_missing(output_layer.data[self._current_slice])
                        else:
                            num = 1
                        output_layer.data[self._current_slice] = output_layer.data[self._current_slice] | self._labels_layer.data * num
                        output_layer.refresh()
                    else:
                        if self.check_box.isChecked():
                            num = find_first_missing(output_layer.data)
                        else:
                            num = 1
                        output_layer.data = output_layer.data | self._labels_layer.data * num
                    self._viewer.layers.selection.active = self._sam_box_layer
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
            image_layer = self._viewer.layers[self._image_layer_selection.currentText()]
            image_path = image_layer.source.path
            image_name = os.path.basename(image_path)
            output_layer = self._viewer.layers[self._shapes_layer_selection.currentText()]
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
        print(self._viewer.layers[self._image_layer_selection.currentText()].corner_pixels)









