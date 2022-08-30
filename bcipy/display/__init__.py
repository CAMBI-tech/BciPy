"""
This import statement allows users to import submodules from display more directly.

`from bcipy.display import init_display_window` vs. `from bcipy.display.main import init_display_window`
"""
from .main import (
    BCIPY_LOGO_PATH,
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
