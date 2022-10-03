"""
This import statement allows users to import submodules from display more directly.

`from bcipy.display import init_display_window` vs. `from bcipy.display.main import init_display_window`
"""
from bcipy.config import BCIPY_LOGO_PATH
from .main import (
    Display,
    InformationProperties,
    init_display_window,
    PreviewInquiryProperties,
    StimuliProperties,
    VEPStimuliProperties,
    TaskDisplayProperties,
)

__all__ = [
    'Display',
    'init_display_window',
    'BCIPY_LOGO_PATH',
    'StimuliProperties',
    'VEPStimuliProperties',
    'InformationProperties',
    'TaskDisplayProperties',
    'PreviewInquiryProperties'
]
