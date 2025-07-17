# napari-SAM4IS

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

### Step 1: Install napari-SAM4IS

You can install `napari-SAM4IS` via [pip]:

```bash
pip install napari-SAM4IS
```

Or via conda

```bash
conda install -c conda-forge napari-SAM4IS
```


### Step 2: Install Segment Anything Model

**IMPORTANT**: You must install the Segment Anything Model separately to use this plugin:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Development Installation

To install the latest development version:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Or you can install from source by cloning the repository:

```
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

For more detailed instructions, please refer to the [SAM installation guide](https://github.com/facebookresearch/segment-anything#installation).

### napari-SAM4IS Installation

You can install `napari-SAM4IS` via [pip]:

    pip install napari-SAM4IS


Or via conda

    conda install -c conda-forge napari-SAM4IS



To install latest development version :

    pip install git+https://github.com/hiroalchem/napari-SAM4IS.git

## Usage
### Preparation
1. Open an image in napari and launch the plugin. (Opening an image after launching the plugin is also possible.)
2. Upon launching the plugin, three layers will be automatically created: SAM-Box, SAM-Predict, and Accepted. The usage of these layers will be explained later.
3. In the widget that appears, select the model you want to use and click the load button. (The default option is recommended.)
4. Next, select the image layer you want to annotate.
5. Then, select whether you want to do instance segmentation or semantic segmentation. (Note that for 3D images, semantic segmentation should be chosen in the current version.)
6. Finally, select the output layer as "shapes" for instance segmentation or "labels" for semantic segmentation. (For instance segmentation, the "Accept" layer can also be used.)

### Annotation
1. Select the SAM-Box layer and use the rectangle tool to enclose the object you want to segment.
2. An automatic segmentation mask will be created and output to the SAM-Predict layer.
3. If you want to make adjustments, do so in the SAM-Predict layer.
4. To accept or reject the annotation, press "a" or "r" on the keyboard, respectively.
5. If you accept the annotation, it will be output as label 1 for semantic segmentation or converted to a polygon and output to the designated layer for instance segmentation.
6. If you reject the annotation, the segmentation mask in the SAM-Predict layer will be discarded.
7. After accepting or rejecting the annotation, the SAM-Predict layer will automatically reset to blank and return to the SAM-Box layer.

### Saving
1. If you have output to the labels layer, use napari's standard functionality to save the mask.
2. If you have output to the shapes layer, you can save the shapes layer using napari's standard functionality, or you can click the "save" button to output a JSON file in COCO format for each image in the folder. (The JSON file will have the same name as the image.)



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"napari-SAM4IS" is free and open source software

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
