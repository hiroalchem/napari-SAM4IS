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


def triangles_in_hole(polygon, center, radius):
    """Count rendered face triangles falling inside the hole."""
    from napari.layers.shapes._shapes_models import Polygon

    shape = Polygon(np.asarray(polygon, dtype=float))
    centroids = shape._face_vertices[shape._face_triangles].mean(axis=1)
    dist = np.hypot(centroids[:, 0] - center[0], centroids[:, 1] - center[1])
    return int((dist < radius * 0.8).sum())


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
