"""Tests for polygon-with-hole (concatenated ring) support."""

import json

import numpy as np
import pytest
from skimage.draw import disk

from napari_sam4is._utils import (
    _signed_area,
    create_json,
    label2polygon,
    load_json,
    mask_to_rings,
    polygon_to_rings,
    rings_to_polygon,
)

# napari's own canonical example, from
# napari/layers/shapes/_tests/conftest.py::poly_hole
POLY_HOLE = np.array(
    [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
        [0, 0],
        [2, 5],
        [5, 8],
        [8, 5],
        [5, 2],
        [2, 5],
    ],
    dtype=float,
)


def donut_mask(shape=(100, 100), outer=30, inner=12, center=(50, 50)):
    mask = np.zeros(shape, np.uint8)
    mask[disk(center, outer, shape=shape)] = 1
    mask[disk(center, inner, shape=shape)] = 0
    return mask


def rasterize(polygon):
    """Paint the polygon's actual triangulation onto a canvas.

    Counting triangle centroids instead would miss slivers, and would
    also flag long triangles that merely pass over a region.
    """
    from napari.layers.shapes._shapes_models import Polygon
    from skimage.draw import polygon as draw_polygon

    verts = np.asarray(polygon, dtype=np.float32)
    shape = Polygon(verts)
    size = int(np.ceil(verts.max())) + 2
    canvas = np.zeros((size, size), np.uint8)
    for triangle in shape._face_vertices[shape._face_triangles]:
        rows, cols = draw_polygon(triangle[:, 0], triangle[:, 1], canvas.shape)
        canvas[rows, cols] = 1
    return canvas


def triangles_in_hole(polygon, center, radius):
    """Painted pixels well inside a hole; 0 means the hole is open."""
    canvas = rasterize(polygon)
    rows, cols = disk(center, max(radius - 2, 1), shape=canvas.shape)
    return int(canvas[rows, cols].sum())


def bridges_removed(polygon):
    """How many bridge edges napari cancelled out.

    Each ring past the first is reached by a there-and-back detour, so
    a correctly built polygon cancels exactly two edges per extra ring.
    """
    from napari.layers.shapes._accelerated_triangulate_dispatch import (
        normalize_vertices_and_edges,
    )

    _, edges = normalize_vertices_and_edges(
        np.asarray(polygon, dtype=np.float32), close=True
    )
    return len(polygon) - len(edges)


class TestRingConversion:
    def test_split_canonical_polygon(self):
        rings = polygon_to_rings(POLY_HOLE)
        assert len(rings) == 2
        assert [len(r) for r in rings] == [4, 4]

    def test_round_trip_is_lossless(self):
        rings = polygon_to_rings(POLY_HOLE)
        assert np.array_equal(rings_to_polygon(rings[0], rings[1:]), POLY_HOLE)

    def test_holes_wind_opposite_to_outer(self):
        outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
        hole = np.array([[2, 2], [8, 2], [8, 8], [2, 8]], float)
        # same winding on input; must be flipped on output
        rings = polygon_to_rings(rings_to_polygon(outer, [hole]))
        assert np.sign(_signed_area(rings[0])) != np.sign(
            _signed_area(rings[1])
        )

    def test_outer_ring_needs_three_vertices(self):
        with pytest.raises(ValueError):
            rings_to_polygon(np.array([[0, 0], [1, 1]], float))

    def test_degenerate_holes_are_dropped(self):
        outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
        poly = rings_to_polygon(outer, [np.array([[5, 5], [5, 5]], float)])
        assert len(polygon_to_rings(poly)) == 1

    def test_legacy_simple_polygon_stays_one_ring(self):
        legacy = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float)
        assert len(polygon_to_rings(legacy)) == 1

    def test_ambiguous_polygon_is_not_split(self):
        """A self-touching ring cannot be split unambiguously."""
        ambiguous = np.array(
            [[0, 0], [10, 0], [0, 0], [10, 10], [0, 10]], float
        )
        assert len(polygon_to_rings(ambiguous)) == 1


class TestMaskToRings:
    def test_donut_yields_one_group_with_one_hole(self):
        groups = mask_to_rings(donut_mask())
        assert len(groups) == 1
        assert len(groups[0][1]) == 1

    def test_border_touching_region_is_closed(self):
        mask = np.zeros((60, 60), np.uint8)
        mask[:30, 10:50] = 1
        mask[disk((15, 30), 6)] = 0
        groups = mask_to_rings(mask)
        assert len(groups) == 1 and len(groups[0][1]) == 1

    def test_empty_mask(self):
        assert mask_to_rings(np.zeros((10, 10), np.uint8)) == []


class TestLabel2Polygon:
    def test_donut_renders_with_an_empty_hole(self):
        polygons = label2polygon(donut_mask())
        assert len(polygons) == 1
        assert triangles_in_hole(polygons[0], (50, 50), 12) == 0

    def test_border_touching_donut_renders_with_hole(self):
        mask = np.zeros((60, 60), np.uint8)
        mask[:30, 10:50] = 1
        mask[disk((15, 30), 6)] = 0
        polygons = label2polygon(mask)
        assert len(polygons) == 1
        assert triangles_in_hole(polygons[0], (15, 30), 6) == 0

    def test_disconnected_components_stay_one_annotation(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[disk((30, 30), 12)] = 1
        mask[disk((70, 70), 12)] = 1
        polygons = label2polygon(mask)
        assert len(polygons) == 1
        assert len(polygon_to_rings(polygons[0])) == 2

    def test_island_inside_hole_is_kept(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[disk((50, 50), 40)] = 1
        mask[disk((50, 50), 25)] = 0
        mask[disk((50, 50), 10)] = 1
        polygons = label2polygon(mask)
        assert len(polygons) == 1
        # outer boundary, its hole, and the island inside that hole
        assert len(polygon_to_rings(polygons[0])) == 3

    def test_empty_mask_yields_no_polygon(self):
        assert label2polygon(np.zeros((10, 10), np.uint8)) == []

    def test_solid_mask_is_a_single_ring(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[disk((50, 50), 20)] = 1
        polygons = label2polygon(mask)
        assert len(polygons) == 1
        assert len(polygon_to_rings(polygons[0])) == 1


class TestCocoRoundTrip:
    def test_donut_survives_save_and_load(self, tmp_path):
        image = np.zeros((100, 100, 3), np.uint8)
        polygon = label2polygon(donut_mask())[0]

        data = create_json(image, "t.png", [polygon])
        assert len(data["annotations"][0]["segmentation"]) == 1

        path = tmp_path / "t.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_json(str(path))

        assert len(loaded["annotations"]) == 1
        assert np.array_equal(loaded["annotations"][0]["polygon"], polygon)

    def test_area_subtracts_the_hole(self):
        image = np.zeros((100, 100), np.uint8)
        donut = label2polygon(donut_mask())[0]
        solid = np.zeros((100, 100), np.uint8)
        solid[disk((50, 50), 30)] = 1

        donut_area = create_json(image, "t", [donut])["annotations"][0]["area"]
        solid_area = create_json(image, "t", label2polygon(solid))[
            "annotations"
        ][0]["area"]
        assert donut_area < solid_area

    def test_area_ignores_image_channels(self):
        gray = np.zeros((100, 100), np.uint8)
        rgb = np.zeros((100, 100, 3), np.uint8)
        polygon = label2polygon(donut_mask())[0]
        assert (
            create_json(gray, "t", [polygon])["annotations"][0]["area"]
            == create_json(rgb, "t", [polygon])["annotations"][0]["area"]
        )


def geojson_polygon(rings):
    return {
        "features": [{"geometry": {"type": "Polygon", "coordinates": rings}}]
    }


class TestGeojsonToMask:
    """The API backend can return polygons with interior rings."""

    OUTER = [[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]]
    HOLE = [[40, 40], [60, 40], [60, 60], [40, 60], [40, 40]]

    def widget(self, make_napari_viewer):
        from napari_sam4is import SAMWidget

        viewer = make_napari_viewer()
        viewer.add_image(np.zeros((100, 100), np.uint8))
        return SAMWidget(viewer)

    def test_interior_ring_becomes_a_hole(self, make_napari_viewer):
        widget = self.widget(make_napari_viewer)
        mask = widget._geojson_to_mask(
            geojson_polygon([self.OUTER, self.HOLE]), (100, 100)
        )
        assert mask[50, 50] == 0  # inside the hole
        assert mask[20, 20] == 1  # inside the ring

    def test_multipolygon_is_supported(self, make_napari_viewer):
        widget = self.widget(make_napari_viewer)
        data = {
            "features": [
                {
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [[self.OUTER, self.HOLE]],
                    }
                }
            ]
        }
        mask = widget._geojson_to_mask(data, (100, 100))
        assert mask[50, 50] == 0
        assert mask[20, 20] == 1

    def test_hole_does_not_erase_an_overlapping_feature(
        self, make_napari_viewer
    ):
        widget = self.widget(make_napari_viewer)
        data = geojson_polygon([self.OUTER, self.HOLE])
        data["features"].append(
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[45, 45], [55, 45], [55, 55], [45, 55], [45, 45]]
                    ],
                }
            }
        )
        mask = widget._geojson_to_mask(data, (100, 100))
        assert mask[50, 50] == 1


class TestMergeAndSplit:
    """Rings drawn as separate shapes are merged into one annotation."""

    OUTER = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], float)
    INNER = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], float)
    APART = np.array([[95, 95], [99, 95], [99, 99], [95, 99]], float)

    def setup_widget(self, make_napari_viewer, shapes):
        from napari_sam4is import SAMWidget

        viewer = make_napari_viewer()
        viewer.add_image(np.zeros((100, 100), np.uint8))
        widget = SAMWidget(viewer)
        layer = widget._accepted_layer
        widget._ensure_features_columns(layer)
        layer.add_polygons(list(shapes), edge_width=2)
        widget._shapes_layer_selection.setCurrentText(layer.name)
        layer.selected_data = set(range(len(shapes)))
        return widget, layer

    def test_merge_makes_one_shape_with_a_hole(self, make_napari_viewer):
        widget, layer = self.setup_widget(
            make_napari_viewer, [self.OUTER, self.INNER]
        )
        widget._merge_selected_to_holes()

        assert len(layer.data) == 1
        assert len(polygon_to_rings(layer.data[0])) == 2
        assert triangles_in_hole(layer.data[0], (50, 50), 10) == 0

    def test_merge_keeps_the_enclosing_shape_class(self, make_napari_viewer):
        widget, layer = self.setup_widget(
            make_napari_viewer, [self.OUTER, self.INNER]
        )
        layer.features.loc[0, "class"] = "outer-class"
        layer.features.loc[1, "class"] = "inner-class"
        widget._merge_selected_to_holes()

        assert layer.features.iloc[0]["class"] == "outer-class"

    def test_merge_refuses_shapes_that_are_not_nested(
        self, make_napari_viewer
    ):
        widget, layer = self.setup_widget(
            make_napari_viewer, [self.OUTER, self.APART]
        )
        widget._merge_selected_to_holes()
        assert len(layer.data) == 2

    def test_merge_needs_two_shapes(self, make_napari_viewer):
        widget, layer = self.setup_widget(make_napari_viewer, [self.OUTER])
        widget._merge_selected_to_holes()
        assert len(layer.data) == 1

    def test_split_restores_one_shape_per_ring(self, make_napari_viewer):
        widget, layer = self.setup_widget(
            make_napari_viewer, [self.OUTER, self.INNER]
        )
        widget._merge_selected_to_holes()
        layer.selected_data = {0}
        widget._split_selected_rings()

        assert len(layer.data) == 2
        assert all(len(polygon_to_rings(p)) == 1 for p in layer.data)

    def test_split_leaves_a_plain_shape_alone(self, make_napari_viewer):
        widget, layer = self.setup_widget(make_napari_viewer, [self.OUTER])
        layer.selected_data = {0}
        widget._split_selected_rings()
        assert len(layer.data) == 1


class TestClosureVertexRecovery:
    """Each ring's first vertex appears twice, and napari makes both
    copies clickable. Moving only one desynchronizes the pair."""

    def octagon_donut(self):
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        outer = np.stack(
            [50 + 40 * np.sin(angles), 50 + 40 * np.cos(angles)], 1
        )
        inner = np.array([[45, 45], [55, 45], [55, 55], [45, 55]], float)
        return np.concatenate([outer, outer[:1], inner, inner[:1]])

    def test_ordinary_vertex_edits_keep_the_rings(self):
        donut = self.octagon_donut()
        moved = donut.copy()
        moved[1] += [5, 5]
        assert len(polygon_to_rings(moved)) == 2
        assert len(polygon_to_rings(np.delete(donut, 1, axis=0))) == 2
        assert (
            len(polygon_to_rings(np.insert(donut, 1, [[10, 50]], axis=0))) == 2
        )

    def test_desynced_closure_vertex_defeats_the_split(self):
        donut = self.octagon_donut()
        broken = donut.copy()
        broken[0] = [12, 50]  # its twin at index 8 stays put
        assert len(polygon_to_rings(broken)) == 1
        # the hole still renders, which is what makes this easy to miss
        assert triangles_in_hole(broken, (50, 50), 5) == 0

    def test_split_recovers_from_a_desynced_closure_vertex(
        self, make_napari_viewer
    ):
        from napari_sam4is import SAMWidget

        viewer = make_napari_viewer()
        viewer.add_image(np.zeros((100, 100), np.uint8))
        widget = SAMWidget(viewer)
        layer = widget._accepted_layer
        widget._ensure_features_columns(layer)

        broken = self.octagon_donut()
        broken[0] = [12, 50]
        layer.add_polygons([broken], edge_width=2)
        widget._shapes_layer_selection.setCurrentText(layer.name)
        layer.selected_data = {0}
        widget._split_selected_rings()

        assert len(layer.data) > 1


def band(start, end, width=6, size=200):
    """A thick line segment, as an elongated diagonal object."""
    from skimage.draw import polygon as draw_polygon

    direction = np.array(end, float) - np.array(start, float)
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]]) * width
    corners = np.array(
        [
            start + normal,
            end + normal,
            end - normal,
            start - normal,
        ]
    )
    mask = np.zeros((size, size), bool)
    mask[draw_polygon(corners[:, 0], corners[:, 1], (size, size))] = True
    return mask


def entry(mask=None, polygon=None, bbox_of=None):
    rows, cols = np.where(bbox_of if bbox_of is not None else mask)
    return {
        "bbox": (
            int(rows.min()),
            int(cols.min()),
            int(rows.max()),
            int(cols.max()),
        ),
        "polygon": polygon,
        "mask": mask,
    }


def is_duplicate(mask, existing, threshold=0.5):
    from napari_sam4is._widget import SAMWidget

    rows, cols = np.where(mask)
    bbox = (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )
    return SAMWidget._is_duplicate_mask(mask, bbox, existing, threshold)


class TestDuplicateDetection:
    """Detect All dedup compares masks, not bounding boxes."""

    def test_crossing_diagonals_are_not_duplicates(self):
        a = band((20, 20), (180, 180))
        b = band((180, 20), (20, 180))
        # their bounding boxes are nearly identical, their masks are not
        assert not is_duplicate(b, [entry(mask=a)])

    def test_near_identical_masks_are_duplicates(self):
        a = np.zeros((200, 200), bool)
        a[disk((100, 100), 40)] = True
        b = np.zeros((200, 200), bool)
        b[disk((102, 100), 40)] = True
        assert is_duplicate(b, [entry(mask=a)])

    def test_object_inside_a_donut_hole_is_not_a_duplicate(self):
        donut = np.zeros((200, 200), bool)
        donut[disk((100, 100), 40)] = True
        donut[disk((100, 100), 25)] = False
        core = np.zeros((200, 200), bool)
        core[disk((100, 100), 10)] = True
        assert not is_duplicate(core, [entry(mask=donut)])

    def test_existing_polygons_are_rasterized_on_demand(self):
        a = np.zeros((200, 200), bool)
        a[disk((100, 100), 40)] = True
        pending = entry(polygon=label2polygon(a)[0], bbox_of=a)
        assert pending["mask"] is None
        assert is_duplicate(a, [pending])
        assert pending["mask"] is not None

    def test_disjoint_boxes_skip_rasterization(self):
        a = np.zeros((200, 200), bool)
        a[disk((30, 30), 12)] = True
        b = np.zeros((200, 200), bool)
        b[disk((160, 160), 12)] = True
        pending = entry(polygon=label2polygon(a)[0], bbox_of=a)
        assert not is_duplicate(b, [pending])
        assert pending["mask"] is None  # never needed


class TestHoleButtonFeedback:
    """napari is often launched without a visible terminal, so the
    buttons must show when they are usable and say what happened."""

    def build(self, make_napari_viewer):
        from napari_sam4is import SAMWidget

        viewer = make_napari_viewer()
        viewer.add_image(np.zeros((200, 200, 3), np.uint8))
        widget = SAMWidget(viewer)
        layer = widget._accepted_layer
        widget._ensure_features_columns(layer)
        layer.add_polygons(
            [
                np.array([[10, 10], [90, 10], [90, 90], [10, 90]], float),
                np.array([[40, 40], [60, 40], [60, 60], [40, 60]], float),
            ],
            edge_width=2,
        )
        widget._shapes_layer_selection.setCurrentText(layer.name)
        return widget, layer, viewer

    def test_buttons_track_the_selection(self, make_napari_viewer):
        widget, layer, _ = self.build(make_napari_viewer)

        layer.selected_data = set()
        assert not widget._merge_holes_btn.isEnabled()
        assert not widget._split_rings_btn.isEnabled()

        layer.selected_data = {0}
        assert not widget._merge_holes_btn.isEnabled()
        assert widget._split_rings_btn.isEnabled()

        layer.selected_data = {0, 1}
        assert widget._merge_holes_btn.isEnabled()
        assert not widget._split_rings_btn.isEnabled()

    def test_result_reaches_the_status_bar(self, make_napari_viewer):
        widget, layer, viewer = self.build(make_napari_viewer)
        layer.selected_data = {0, 1}
        widget._merge_selected_to_holes()
        assert "マージ" in viewer.status

    def test_refusal_reaches_the_status_bar(self, make_napari_viewer):
        widget, layer, viewer = self.build(make_napari_viewer)
        layer.selected_data = {0}
        widget._merge_selected_to_holes()
        assert "2つ以上" in viewer.status


class TestMultipleRings:
    """Rings past the second are where the bridge cancellation broke:
    chaining them head to tail leaves every bridge in place, so the
    holes stay filled and the triangulator can give up entirely."""

    OUTER = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    H1 = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], float)
    H2 = np.array([[60, 60], [80, 60], [80, 80], [60, 80]], float)
    H3 = np.array([[10, 60], [30, 60], [30, 80], [10, 80]], float)

    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_every_bridge_cancels(self, count):
        holes = [self.H1, self.H2, self.H3][:count]
        polygon = rings_to_polygon(self.OUTER, holes)
        assert bridges_removed(polygon) == 2 * count
        assert len(polygon_to_rings(polygon)) == count + 1

    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_every_hole_stays_open(self, count):
        holes = [self.H1, self.H2, self.H3][:count]
        canvas = rasterize(rings_to_polygon(self.OUTER, holes))
        for hole in holes:
            centre = hole.mean(axis=0)
            rows, cols = disk(centre, 6, shape=canvas.shape)
            assert canvas[rows, cols].sum() == 0

    def test_components_each_keep_their_hole(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[disk((60, 60), 40)] = 1
        mask[disk((60, 60), 20)] = 0
        mask[disk((150, 150), 35)] = 1
        mask[disk((150, 150), 15)] = 0

        polygon = label2polygon(mask)[0]
        assert len(polygon_to_rings(polygon)) == 4
        assert bridges_removed(polygon) == 6
        assert triangles_in_hole(polygon, (60, 60), 20) == 0
        assert triangles_in_hole(polygon, (150, 150), 15) == 0


class TestBorderTouchingMasks:
    """A region running into the edge of the image gives an open
    contour. Closing it with a straight chord cuts through the other
    vertices on that edge, which used to crash the triangulator."""

    def ragged_border_mask(self):
        mask = np.zeros((120, 400), np.uint8)
        mask[0:100, 182:338] = 1
        for col in range(182, 338, 12):  # notches cutting the top edge
            mask[0 : (col % 5) + 1, col : col + 5] = 0
        for centre in [(40, 220), (55, 270), (35, 310), (75, 250)]:
            mask[disk(centre, 12, shape=mask.shape)] = 0
        return mask

    def test_ragged_border_with_holes_triangulates(self):
        from napari.layers.shapes._shapes_models import Polygon

        polygon = label2polygon(self.ragged_border_mask())[0]
        Polygon(polygon.astype(np.float32))  # must not raise

    def test_border_contact_does_not_fragment_the_outline(self):
        # padding keeps the notched top edge as one closed contour
        rings = polygon_to_rings(label2polygon(self.ragged_border_mask())[0])
        assert len(rings) == 5  # one outer boundary, four holes

    def test_holes_survive_border_contact(self):
        polygon = label2polygon(self.ragged_border_mask())[0]
        for centre in [(40, 220), (55, 270), (35, 310), (75, 250)]:
            assert triangles_in_hole(polygon, centre, 12) == 0
