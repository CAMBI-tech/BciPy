# BciPy GUI Module

The GUI module provides the graphical user interface components for BciPy, enabling users to interact with the BCI system through a user-friendly interface. This module is essential for running experiments, managing parameters, and controlling the BCI workflow.

## Overview

The GUI module consists of several key components:

1. **Main Interface (`BCInterface.py`)**
   - Primary interface for running BCI experiments
   - User management and experiment selection
   - Parameter configuration and task execution
   - Offline analysis capabilities

2. **Base UI Components (`bciui.py`)**
   - Core UI building blocks and utilities
   - Common functionality for all BciPy interfaces
   - Layout management and styling
   - Dynamic list and item management

3. **Experiment Management**
   - `ExperimentRegistry.py`: Interface for registering and managing experiments
   - `ExperimentField.py`: Form for collecting experiment-specific data
   - Field management and validation

4. **Alert System (`alert.py`)**
   - User notifications and confirmations
   - Error handling and system messages

5. **Task Transitions (`intertask_gui.py`)**
   - Progress tracking between tasks
   - Experiment flow control
   - User feedback during transitions

## Getting Started

### Running the GUI

To start the BciPy GUI:

```bash
python bcipy/gui/BCInterface.py
```

Or using Make (if installed):

```bash
make bci-gui
```

### Basic Usage

1. **User Management**
   - Enter or select a user ID
   - User IDs must be alphanumeric and meet length requirements

2. **Experiment Selection**
   - Choose between running a specific task or a complete experiment
   - Tasks are individual BCI operations (e.g., calibration)
   - Experiments are predefined sequences of tasks

3. **Parameter Configuration**
   - Load existing parameter files
   - Edit parameters through the parameter editor
   - Save custom configurations

4. **Task Execution**
   - Start sessions with selected configurations
   - Monitor progress through the intertask interface
   - View results and system feedback

5. **Signal Viewer**
   - Monitor real-time EEG signals during experiments
   - View and analyze recorded data from previous sessions
   - Toggle channel visibility and apply montages
   - Control display duration and filtering options
   - Pause/resume signal visualization
   - Support for multiple monitor configurations
   - See [Signal Viewer Documentation](gui/viewer/README.md) for more details

## Key Features

### Experiment Registry

- Create and manage experiment protocols
- Define task sequences and parameters
- Configure experiment-specific fields
- Save and load experiment configurations

### Parameter Management

- Load default or custom parameter files
- Edit parameters through a user-friendly interface
- Save parameter configurations
- Validate parameter settings

### Task Control

- Start and stop BCI tasks
- Monitor task progress
- Handle transitions between tasks
- Manage experiment flow

### User Interface

- Clean, intuitive design
- Consistent styling across components
- Responsive feedback
- Error handling and notifications

## Development

### Adding New Components

When extending the GUI:

1. Inherit from `BCIUI` for new interfaces
2. Use the provided UI utilities and components
3. Follow the established styling guidelines
4. Implement proper error handling
5. Add appropriate documentation

### Styling

The GUI uses a consistent styling system:

- CSS-based styling through `bcipy_stylesheet.css`
- Common UI elements and layouts
- Responsive design principles
- Accessibility considerations

## Best Practices

1. **Error Handling**
   - Use the alert system for user notifications
   - Validate inputs before processing
   - Provide clear error messages
   - Handle edge cases gracefully

2. **User Experience**
   - Maintain consistent interface behavior
   - Provide clear feedback for actions
   - Use appropriate timeouts and delays
   - Implement proper state management

3. **Performance**
   - Minimize UI blocking operations
   - Use appropriate threading for long operations
   - Optimize resource usage
   - Handle cleanup properly

## Troubleshooting

Common issues and solutions:

1. **GUI Not Starting**
   - Check Python and dependency versions
   - Verify file permissions
   - Check for conflicting processes

2. **Parameter Issues**
   - Validate parameter file format
   - Check file paths and permissions
   - Verify parameter values

3. **Task Execution Problems**
   - Check system requirements
   - Verify device connections
   - Review error logs
