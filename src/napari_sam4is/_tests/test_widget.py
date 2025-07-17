import numpy as np

from napari_sam4is import SAMWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
def test_sam_widget_creation(make_napari_viewer):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    widget = SAMWidget(viewer)

    # check that the widget was created successfully
    assert widget is not None
    assert widget._viewer == viewer


def test_sam_widget_basic_functionality(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # create the SAM widget
    widget = SAMWidget(viewer)

    # check basic attributes exist
    assert hasattr(widget, '_viewer')
    assert hasattr(widget, '_image_type')
    assert hasattr(widget, '_current_slice')
