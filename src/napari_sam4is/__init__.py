# Monkey patch for napari 0.6.6 Selection.replace_selection bug
# napari 0.6.6's Shapes.selected_data setter calls replace_selection,
# but Selection class is missing this method. We patch both:
# 1. Add replace_selection to Selection class
# 2. Replace Shapes.selected_data setter to handle missing method
#
# Note: This patch is applied conditionally based on the presence of the
# replace_selection method, making it compatible with both affected versions
# (e.g., 0.6.6) and future versions where the bug may be fixed.

# --- CJK font patch ---
# vispy's default OpenSans font does not contain CJK glyphs, causing Japanese
# class names to render as tofu squares on the Shapes layer. We register the
# bundled NotoSansJP font as a vispy-internal font and wrap _load_glyph so that
# CJK characters (U+3000 and above) are rendered with NotoSansJP while all
# other characters continue to use the default OpenSans font.
import os as _os
import sys as _sys

_FONTS_DIR = _os.path.join(_os.path.dirname(__file__), "fonts")
_NOTO_FACE = "NotoSansJP"
_NOTO_TTF = _os.path.join(_FONTS_DIR, "NotoSansJP-Regular.ttf")


def _patch_vispy_cjk_font():
    """Register NotoSansJP with vispy and wrap _load_glyph for CJK fallback."""
    try:
        import sys as _sys2

        import vispy.util.fonts._triage  # noqa: F401

        # Import modules so they appear in sys.modules, then access via sys.modules
        import vispy.util.fonts._vispy_fonts  # noqa: F401
        _vf_mod = _sys2.modules["vispy.util.fonts._vispy_fonts"]
        _triage_mod = _sys2.modules["vispy.util.fonts._triage"]

        # Register NotoSansJP as a vispy built-in font so _load_font can find it
        _orig_vispy_fonts = _vf_mod._vispy_fonts
        if _NOTO_FACE not in _orig_vispy_fonts:
            _vf_mod._vispy_fonts = _orig_vispy_fonts + (_NOTO_FACE,)

        # Override _get_vispy_font_filename to return our bundled TTF for NotoSansJP
        _orig_get_filename = _vf_mod._get_vispy_font_filename

        def _patched_get_filename(face, bold, italic):
            if face == _NOTO_FACE:
                return _NOTO_TTF
            return _orig_get_filename(face, bold, italic)

        _vf_mod._get_vispy_font_filename = _patched_get_filename

        # Also patch the platform-specific module's reference to _get_vispy_font_filename
        if _sys.platform == "darwin":
            import vispy.util.fonts._quartz  # noqa: F401
            _platform_mod = _sys2.modules["vispy.util.fonts._quartz"]
        else:
            import vispy.util.fonts._freetype  # noqa: F401
            _platform_mod = _sys2.modules["vispy.util.fonts._freetype"]

        _platform_mod._get_vispy_font_filename = _patched_get_filename
        _platform_mod._vispy_fonts = _vf_mod._vispy_fonts
        _orig_load_glyph = _platform_mod._load_glyph

        def _cjk_load_glyph(f, char, glyphs_dict):
            """Use NotoSansJP for CJK characters, default font otherwise."""
            if ord(char) >= 0x3000:
                f = dict(f, face=_NOTO_FACE)
            _orig_load_glyph(f, char, glyphs_dict)

        # Patch the triage module (public API used by TextureFont)
        _triage_mod._load_glyph = _cjk_load_glyph
        # Patch the reference in the text visual module (used by TextureFont._load_char)
        try:
            import vispy.visuals.text.text  # noqa: F401
            _text_mod = _sys2.modules["vispy.visuals.text.text"]
            _text_mod._load_glyph = _cjk_load_glyph
        except (ImportError, KeyError, AttributeError):
            pass
    except (ImportError, ModuleNotFoundError, KeyError, AttributeError, TypeError) as _e:
        import warnings
        warnings.warn(
            f"napari-SAM4IS: CJK font patch failed ({_e}). "
            "Japanese class names may not display correctly on layers.",
            stacklevel=2,
        )


if _os.path.isfile(_NOTO_TTF):
    _patch_vispy_cjk_font()

# ----------------------

from collections.abc import Collection

from napari.layers.shapes.shapes import Shapes
from napari.utils.events.containers import Selection

# Add replace_selection to Selection class if missing (version guard)
if not hasattr(Selection, "replace_selection"):

    def _replace_selection(self, items):
        """Replace current selection with items."""
        self.clear()
        self.update(items)

    Selection.replace_selection = _replace_selection


def _patched_selected_data_setter(
    self, selected_data: Collection[int]
) -> None:
    """Patched version that handles missing replace_selection gracefully.

    This reimplements napari's Shapes.selected_data setter logic with a
    fallback for missing Selection.replace_selection method. We copy the
    entire setter implementation rather than wrapping the original because
    the original would fail before we could intercept the error.
    """
    # Use replace_selection if available, otherwise fall back
    if hasattr(self._selected_data, "replace_selection"):
        self._selected_data.replace_selection(selected_data)
    else:
        self._selected_data.clear()
        self._selected_data.update(selected_data)

    # Call the rest of the original setter logic
    # Import here to avoid circular dependencies
    import numpy as np
    from napari.layers.shapes.shapes import _unique_element

    self._selected_box = self.interaction_box(self._selected_data)

    if len(selected_data) > 0:
        selected_data_indices = list(selected_data)
        selected_face_colors = self._data_view._face_color[
            selected_data_indices
        ]
        if (
            unique_face_color := _unique_element(selected_face_colors)
        ) is not None:
            with self.block_update_properties():
                self.current_face_color = unique_face_color

        selected_edge_colors = self._data_view._edge_color[
            selected_data_indices
        ]
        if (
            unique_edge_color := _unique_element(selected_edge_colors)
        ) is not None:
            with self.block_update_properties():
                self.current_edge_color = unique_edge_color

        unique_edge_width = _unique_element(
            np.array(
                [self._data_view.shapes[i].edge_width for i in selected_data]
            )
        )
        if unique_edge_width is not None:
            with self.block_update_properties():
                self.current_edge_width = unique_edge_width

        unique_properties = {}
        for k, v in self.properties.items():
            unique_properties[k] = _unique_element(v[selected_data_indices])

        if all(p is not None for p in unique_properties.values()):
            with self.block_update_properties():
                self.current_properties = unique_properties

    self._set_highlight()


# Replace the property
Shapes.selected_data = property(
    Shapes.selected_data.fget, _patched_selected_data_setter
)

__all__ = ("SAMWidget",)


def __getattr__(name):
    if name == "SAMWidget":
        from ._widget import SAMWidget

        globals()["SAMWidget"] = SAMWidget
        return SAMWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
