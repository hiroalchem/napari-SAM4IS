import napari
import numpy as np
import torch
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QComboBox, QLabel
from segment_anything import sam_model_registry, SamPredictor
from skimage.color import gray2rgb, rgba2rgb

from ._utils import load_model, preprocess, label2polygon


class SAMWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer

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
        self._image_layer_selection.currentTextChanged.connect(self._on_image_layer_changed)
        self.vbox.addWidget(self._image_layer_selection)
        self.vbox.addWidget(QLabel("output shapes layer"))
        self._shapes_layer_selection = QComboBox()
        self._shapes_layer_selection.addItems([layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.shapes.shapes.Shapes)])
        self.vbox.addWidget(self._shapes_layer_selection)

        self._sam_box_layer = self._viewer.add_shapes(name="SAM-Box", edge_color="red", edge_width=2, face_color="transparent")
        self._sam_box_layer.mouse_drag_callbacks.append(self._on_sam_box_created)

        if self._image_layer_selection.currentText() != "":
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

    def _on_layer_list_changed(self, event):
        self._image_layer_selection.clear()
        self._image_layer_selection.addItems([layer.name for layer in self._viewer.layers if isinstance(layer, napari.layers.image.image.Image)])
        [self._viewer.layers.move(i, 0) for i, layer in enumerate(self._viewer.layers) if isinstance(layer, napari.layers.image.image.Image)]
        self._on_image_layer_changed(None)
        self._shapes_layer_selection.clear()
        self._shapes_layer_selection.addItems([layer.name for layer in self._viewer.layers if (isinstance(layer, napari.layers.shapes.shapes.Shapes))&(layer.name != self._sam_box_layer.name)])

    def _load_model(self):
        model_name = self._model_selection.currentText()
        self._sam_model = load_model(model_name)
        self._sam_model.to(device=self.device)
        self.sam_predictor = SamPredictor(self._sam_model)
        if self._image_layer_selection.currentText() != "":
            self.sam_predictor.set_image(preprocess(self._viewer.layers[self._image_layer_selection.currentText()].data))
            print('set image')

    def _on_image_layer_changed(self, index):
        print("image_layer_changed")
        if self.sam_predictor is not None:
            self.sam_predictor.set_image(preprocess(self._viewer.layers[self._image_layer_selection.currentText()].data))
            print('set image')

    def _on_sam_box_created(self, layer, event):
        # mouse click
        yield
        # mouse move
        while event.type == 'mouse_move':
            yield
        # mouse release
        if len(self._sam_box_layer.data) == 1:
            coords = self._sam_box_layer.data[0]
            y1 = int(coords[0][0])
            x1 = int(coords[0][1])
            y2 = int(coords[2][0])
            x2 = int(coords[2][1])
            print(x1, y1, x2, y2)
            input_box = np.array([x1, y1, x2, y2])
            if self.sam_predictor is not None:
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                self._labels_layer.data = masks[0] * 1
            self._sam_box_layer.data = []
            self._viewer.layers.selection.active = self._labels_layer

    def _accept_mask(self, layer):
        if self._shapes_layer_selection.currentText() != "":
            output_layer = self._viewer.layers[self._shapes_layer_selection.currentText()]
            if isinstance(output_layer, napari.layers.shapes.shapes.Shapes):
                output_layer.add_polygons(label2polygon(self._labels_layer.data), edge_width=6)
                self._viewer.layers.selection.active = self._sam_box_layer
        else:
            pass
        self._labels_layer.data = np.zeros_like(self._labels_layer.data)

    def _reject_mask(self, layer):
        self._labels_layer.data = np.zeros_like(self._labels_layer.data)
        self._viewer.layers.selection.active = self._sam_box_layer








