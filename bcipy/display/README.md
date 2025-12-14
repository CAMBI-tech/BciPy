# Display Module

The Display module is a core component of BciPy that handles all visual presentation logic for BCI tasks. It provides a flexible and configurable framework for creating and managing visual stimuli, with precise timing control and synchronization capabilities. Please note that the VEPDisplay is still in progress. Please use with discretion!

## Structure

The module is organized into several key components:

- `main/`: Core display initialization and window management
- `paradigm/`: BCI paradigm-specific display implementations
  - `rsvp/`: RSVP Keyboard display components
  - `matrix/`: Matrix Speller display components
  - `vep/`:  *WIP* Visual Evoked Potetinal display components.
- `components/`: Reusable display components
- `tests/`: Unit and integration tests
- `demo/`: Example implementations and usage

## Core Concepts

### Display System

The display system is built on several key abstractions:

#### 1. Display Base Class

The base `Display` class provides core functionality for all displays. It is defined in `main.py`.

- Window management and initialization
- Stimulus presentation timing
- Task bar and information display
- Trigger handling and calibration

#### 2. Stimuli Properties

Configure visual properties of stimuli:

```python
from bcipy.display import StimuliProperties

properties = StimuliProperties(
    stim_font='Arial',
    stim_height=32,
    stim_pos=(0, 0)
)
```

#### 3. Information Properties

Manage task information display:

```python
from bcipy.display import InformationProperties

info = InformationProperties(
    info_color='white',
    info_text="Task Progress",
    info_font='Consolas',
    info_height=24,
    info_pos=(0, 0)
)
```

### Key Features

#### Window Management

- PsychoPy window initialization
- Screen resolution handling
- Fullscreen/windowed modes
- Refresh rate synchronization

#### Stimulus Presentation

- Precise timing control
- Animation and transitions
- Trigger synchronization
- Event logging

#### Task Display

- Progress tracking
- User feedback
- Custom UI elements
- Dynamic updates

#### Layout System

- Flexible positioning
- Responsive design
- Component alignment
- Screen space optimization

## Supported Paradigms

### RSVP Keyboard

The RSVP Keyboard is an EEG-based typing system that presents symbols sequentially at a single location. Users select symbols by attending to their target and eliciting a P300 response.

Key features:

- Single-location presentation
- Temporal separation of stimuli
- P300-based selection
- Configurable timing

### Matrix Speller

The Matrix Speller presents symbols in a grid layout, highlighting subsets of symbols to elicit P300 responses for selection.

Key features:

- Grid-based layout
- P300-based selection
- Configurable matrix size

## Development Guidelines

1. **Adding New Paradigms**
   - Create a new submodule in `paradigm/`
   - Inherit from base `Display` class
   - Implement required interface methods
   - Add comprehensive tests

2. **Timing Considerations**
   - Test timing with actual hardware
   - Account for refresh rate variations
   - Log timing events for analysis
   - Use PsychoPy's timing functions

3. **Best Practices**
   - Follow PsychoPy guidelines
   - Document timing parameters
   - Include example configurations
   - Add unit tests for timing

## References

1. RSVP Keyboard:

```text
Orhan, U., et al. (2012). RSVP Keyboard: An EEG Based Typing Interface. 
IEEE International Conference on Acoustics, Speech, and Signal Processing.
```

1. Matrix Speller:

```text
Farwell, L. A., & Donchin, E. (1988). Talking off the top of your head: 
toward a mental prosthesis utilizing event-related brain potentials.
Electroencephalography and clinical Neurophysiology, 70(6), 510-523.
```

## Support

For issues, questions, or contributions:

- Open an issue on GitHub
- Check existing documentation
- Review test examples
- Consult the demo implementations
