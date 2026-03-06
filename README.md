# napari-SAM4IS

[English](README.md) | [日本語](README.ja.md)

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/napari-SAM4IS.svg?color=green)](https://github.com/hiroalchem/napari-SAM4IS/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-SAM4IS.svg?color=green)](https://pypi.org/project/napari-SAM4IS)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-SAM4IS.svg?color=green)](https://python.org)
[![tests](https://github.com/hiroalchem/napari-SAM4IS/workflows/tests/badge.svg)](https://github.com/hiroalchem/napari-SAM4IS/actions)
[![codecov](https://codecov.io/gh/hiroalchem/napari-SAM4IS/branch/main/graph/badge.svg)](https://codecov.io/gh/hiroalchem/napari-SAM4IS)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-SAM4IS)](https://napari-hub.org/plugins/napari-SAM4IS)


### napari plugin for instance and semantic segmentation annotation using Segment Anything Model (SAM)

This is a plugin for [napari](https://napari.org/), a multi-dimensional image viewer for Python, that allows for instance and semantic segmentation annotation. This plugin provides an easy-to-use interface for annotating images with the option to output annotations as COCO format.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

**Requirements**: Python 3.10-3.13

### Step 1: Install napari-SAM4IS

You can install `napari-SAM4IS` via [pip]:

```bash
pip install napari-SAM4IS
```

Or via conda:

```bash
conda install -c conda-forge napari-SAM4IS
```

To install the latest development version:

```bash
pip install git+https://github.com/hiroalchem/napari-SAM4IS.git
```

### Step 2: Install Segment Anything Model (Optional - for local model usage)

**Note**: Installing a SAM model is only required if you plan to use local models. If you're using the API mode, you can skip this step.

#### SAM 1 (vit_h / vit_l / vit_b)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

For more detailed instructions, please refer to the [SAM installation guide](https://github.com/facebookresearch/segment-anything#installation).

#### SAM 3

SAM 3 requires a patched fork for macOS compatibility. Install with:

```bash
pip install git+https://github.com/hiroalchem/sam3.git@patched-macos
```

If you use [uv](https://docs.astral.sh/uv/), sam3 and its dependencies are resolved automatically:

```bash
uv sync --group sam3
```

SAM 3 also requires the following packages (installed automatically with uv, manually with pip):

```bash
pip install "numpy>=1.26" "timm>=1.0.17" "einops>=0.8,<0.9" "huggingface-hub>=0.30,<1" "av>=12" "pycocotools>=2"
```

> **Important**: Before using SAM 3, you must request access to the checkpoints on the [SAM 3 Hugging Face repo](https://huggingface.co/facebook/sam3).
>
> - **Automatic download**: Run `huggingface-cli login` to authenticate, then leave the checkpoint field empty and click the Load button. The weights will be downloaded automatically.
> - **Manual download**: Download `sam3.pt` from the HuggingFace repo and specify the local path in the plugin's checkpoint field.

## Usage
### Preparation
1. Open an image in napari and launch the plugin. (Opening an image after launching the plugin is also possible.)
2. Upon launching the plugin, several layers will be automatically created: SAM-Box, SAM-Positive, SAM-Negative, SAM-Predict, and Accepted. The usage of these layers will be explained later.
3. Choose between local model or API mode:
   - **Local Model Mode (SAM 1)**: Select the model you want to use and click the load button. (The default option is recommended.)
   - **Local Model Mode (SAM 3)**: Select "SAM3" from the model dropdown. The model weights will be downloaded automatically from HuggingFace (gated access required), or you can specify a local checkpoint path.
   - **API Mode**: Check the "Use API" checkbox, then enter your API URL and API Key. No model loading is required. This mode is designed to work with the SAM API provided by [LPIXEL Inc.](https://lpixel.net/) via [IMACEL](https://imacel.net/). For API access, please contact [IMACEL](https://imacel.net/contact) directly.
4. Next, select the image layer you want to annotate.
5. Then, select whether you want to do instance segmentation or semantic segmentation. (Note that for 3D images, semantic segmentation should be chosen in the current version.)
6. Finally, select the output layer as "shapes" for instance segmentation or "labels" for semantic segmentation. (For instance segmentation, the "Accept" layer can also be used.)

### Class Management
You can define annotation classes to assign to each segmented object. Classes support a hierarchical (parent–child) structure.

1. In the **Class Management** section, type a class name and click **Add** (or press Enter) to add a new top-level class.
2. To add a subclass, first select a parent class in the tree, type the subclass name, and click **Add Sub**.
3. Click a class in the tree to select it. The selected class will be assigned to subsequent annotations.
4. To reassign a class, select an existing annotation in the output Shapes layer, then click the desired class.
5. Classes with subclasses cannot be deleted. Remove all subclasses first. Classes in use cannot be deleted either.
6. You can load class definitions from a YAML file (click **Load**). The supported formats are:
   ```yaml
   # New hierarchical format
   classes:
     - id: 0
       name: animal
       children:
         - id: 1
           name: cat
         - id: 2
           name: dog
     - id: 3
       name: vehicle
   ```
   ```yaml
   # Legacy flat format (still supported)
   names:
     0: cat
     1: dog
     2: bird
   ```
7. Class definitions are automatically saved as `class.yaml` alongside the COCO JSON output.

### Annotation with SAM
1. Select the SAM-Box layer and use the rectangle tool to enclose the object you want to segment.
2. An automatic segmentation mask will be created and output to the SAM-Predict layer.
3. You can refine the prediction by adding point prompts: click on the SAM-Positive layer to add points that should be included, or on the SAM-Negative layer to add points that should be excluded.
4. If you want to make further adjustments, do so in the SAM-Predict layer.
5. To accept or reject the annotation, press **A** or **R** on the keyboard, respectively.
6. If you accept the annotation, it will be output as label 1 for semantic segmentation or converted to a polygon and output to the designated layer for instance segmentation. The currently selected class will be assigned to the annotation.
7. If you reject the annotation, the segmentation mask in the SAM-Predict layer will be discarded.
8. After accepting or rejecting the annotation, the SAM-Predict layer will automatically reset to blank and return to the SAM-Box layer.

### SAM 3: Detect All

When using SAM 3, additional features are available for batch detection:

1. **Prompt Modes**: Choose between Text, Box, or Text+Box prompts using the radio buttons in the SAM3 Prompt section.
2. **Box prompt**: Draw a bounding box on the SAM-Box layer, then click **Detect All** to detect all instances within the box.
3. **Text prompt**: Enter a class name or description as the text prompt. The text is derived from the currently selected class name.
4. **Exemplar-based detection**: Select one or more shapes in the output Shapes layer (e.g., Accepted), then click **Detect All**. The selected shapes are used as exemplar box prompts. This is useful for finding similar objects across the image.
   - The output Shapes layer must be the active layer for exemplar selection to take effect.
   - All selected exemplar shapes must belong to the same class.
   - Exemplar boxes and SAM-Box input can be combined.
5. **IoU duplicate filtering**: To prevent overlapping annotations, the plugin automatically filters out masks that overlap with existing shapes above a configurable IoU threshold (default: 0.5).
   - Adjust the threshold using the **IoU Threshold** spinner.
   - Check **Same class only** to only filter duplicates within the same class, or uncheck it to filter across all classes.

### Manual Annotation (without SAM)
You can also annotate without using SAM by enabling **Manual Mode**.

1. Check the **Manual Mode** checkbox. SAM-related controls and layers will be hidden.
2. The SAM-Predict layer switches to paint mode. Use napari's standard Labels tools (paint brush, eraser, fill) from the layer controls panel to draw your annotation.
3. Adjust brush size using napari's standard Labels controls.
4. Press **A** to accept or **R** to reject, just like SAM mode.
5. After accepting, the painted mask is converted to a polygon (instance mode) or merged into the output Labels layer (semantic mode), with the selected class assigned.

### Annotation Attributes
Each annotation can have additional attributes to support quality control workflows.

1. Select one or more annotations in the output Shapes layer.
2. In the **Annotation Attributes** panel, you can set:
   - **Unclear boundary**: Mark annotations where the object boundary is ambiguous.
   - **Uncertain class**: Mark annotations where the object class is uncertain.
3. Click **Accept Selected** to mark the selected annotations as reviewed (sets status to "approved" with a timestamp), or **Accept All** to review all annotations at once.
4. Attributes are saved as part of the COCO JSON output under each annotation's `"attributes"` field.
5. When multiple annotations are selected with mixed attribute values, checkboxes show a mixed state indicator.

### Display Settings
The **Display Settings** panel lets you customize the appearance of annotation layers and persist those preferences across sessions.

- **Accepted edge color**: Click the color swatch to open a color picker and change the outline color of accepted annotations. The new color is applied immediately to all existing shapes on the Accepted layer and saved for future sessions.
- **Annotation text color**: Click the color swatch to change the color of class label text shown on annotation shapes.
- **Annotation text size**: Adjust the integer spinner to change the font size of class label text.

All settings are saved to `settings.json` in the OS user-config directory (e.g. `~/Library/Preferences/napari-SAM4IS/` on macOS) and restored automatically on the next launch.

### Saving and Loading Annotations
1. If you have output to the labels layer, use napari's standard functionality to save the mask.
2. If you have output to the shapes layer, you can save the shapes layer using napari's standard functionality, or you can click the **Save** button to output a JSON file in COCO format for each image in the folder. (The JSON file will have the same name as the image.) Class definitions will also be saved as `class.yaml` in the same directory.
3. To load previously saved annotations, click the **Load** button and select a COCO JSON file. Annotations, class definitions, and attributes will be restored.
4. When switching images via the Image ComboBox, the plugin will:
   - Prompt to save unsaved annotations (Save / Discard / Cancel)
   - Automatically clear the output layer
   - Auto-load annotations from a matching JSON file if one exists



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"napari-SAM4IS" is free and open source software

## Citation

If you use napari-SAM4IS in your research, please cite it:

```bibtex
@software{kawai_napari_sam4is,
  author  = {Kawai, Hiroki},
  title   = {napari-SAM4IS},
  url     = {https://github.com/hiroalchem/napari-SAM4IS},
  license = {Apache-2.0},
}
```

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/hiroalchem/napari-SAM4IS/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
