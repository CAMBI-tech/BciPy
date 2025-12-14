# BciPy Feedback Module

The feedback module provides a flexible framework for implementing real-time feedback mechanisms in BCI experiments. It supports both visual and auditory feedback, allowing researchers to create customized feedback paradigms for their specific needs.

## Overview

Feedback in BCI systems is crucial for providing users with information about their brain activity in real-time. This module implements a robust feedback system that can be easily extended and customized for different experimental paradigms.

## Core Components

### Base Classes

- `Feedback`: Abstract base class that defines the interface for all feedback mechanisms
  - Provides common functionality for feedback administration
  - Defines abstract methods that must be implemented by subclasses
  - Manages feedback type registration and logging

### Feedback Types

The module supports two main types of feedback:

1. **Visual Feedback** (`VisualFeedback`)
   - Displays text or image stimuli on screen
   - Supports customizable positioning, timing, and appearance
   - Provides precise timing control for stimulus presentation
   - Features:
     - Text and image stimulus support
     - Configurable font, size, and color
     - Position control
     - Timing synchronization

2. **Auditory Feedback** (`AuditoryFeedback`)
   - Plays sound stimuli through the system's audio output
   - Supports various audio formats and sampling rates
   - Provides timing control for audio presentation
   - Features:
     - Sound playback
     - Configurable audio parameters
     - Timing synchronization

## Usage Examples

### Visual Feedback

```python
from bcipy.feedback.visual.visual_feedback import VisualFeedback
from psychopy import visual
from bcipy.helpers.clock import Clock

# Initialize display window
window = visual.Window(size=[800, 600])

# Configure parameters
parameters = {
    'feedback_font': 'Arial',
    'feedback_stim_height': 0.1,
    'feedback_pos_x': 0,
    'feedback_pos_y': 0,
    'feedback_duration': 1.0,
    'feedback_color': 'white'
}

# Create feedback instance
clock = Clock()
feedback = VisualFeedback(window, parameters, clock)

# Administer feedback
timing = feedback.administer("Hello World", StimuliType.TEXT)
```

### Auditory Feedback

```python
from bcipy.feedback.sound.auditory_feedback import AuditoryFeedback
from psychopy import core
import numpy as np

# Configure parameters
parameters = {
    'sound_buffer_time': 1.0
}

# Create feedback instance
clock = core.Clock()
feedback = AuditoryFeedback(parameters, clock)

# Generate a simple tone
fs = 44100  # sampling frequency
t = np.linspace(0, 1, fs)  # 1 second duration
sound = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

# Administer feedback
timing = feedback.administer(sound, fs)
```

## Configuration

Both feedback types can be configured through a parameters dictionary. Common parameters include:

- Timing parameters (duration, intervals)
- Display parameters (position, size, color)
- Stimulus-specific parameters (font, audio format)

## Timing Control

The feedback module provides precise timing control through:

- Clock synchronization
- Timestamp recording
- Buffer time management

## Extending the Module

To create a new feedback type:

1. Create a new class inheriting from `Feedback`
2. Implement the required abstract methods:
   - `configure()`
   - `administer()`
3. Register the new feedback type in `FeedbackType` enum

## Best Practices

1. Always use the provided timing mechanisms for synchronization
2. Handle exceptions appropriately in feedback administration
3. Clean up resources after feedback presentation
4. Use appropriate buffer times for smooth presentation
5. Test feedback timing in your specific experimental setup

## References

- PsychoPy documentation for visual stimulus presentation
- SoundDevice documentation for audio playback
- BciPy documentation for integration with other modules
