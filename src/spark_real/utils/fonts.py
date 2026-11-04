"""Monospace PIL font for image annotation, with the PIL default as fallback.

DejaVu Sans Mono comes from matplotlib's bundled fonts, so the annotated
pixels do not depend on which system font packages are installed.
"""

from PIL import ImageFont


def load_font(size: int = 14):
    try:
        from matplotlib import font_manager

        path = font_manager.findfont("DejaVu Sans Mono", fallback_to_default=False)
        return ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001 - no matplotlib or font: PIL default
        return ImageFont.load_default()
