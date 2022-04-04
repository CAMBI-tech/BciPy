# Display Module

The Display module defines the visual presentation logic needed for any tasks in BciPy. Most displays take in a large number of user-configured parameters (including text size and font) and handle
	passing back timestamps from their stimuli presentations for classification. 

### Structure
`display`
	`main`: Initializes a display window and contains useful display objects
		`paradigm`: top level module holding all bcipy related display objects
			`rsvp`:  RSVP related display objects and functions.
				`mode`: defines task specific displays
			`matrix`:  matrix related display objects and functions. Currently, only single character presentation is available.
				`mode`: defines task specific displays
		`tests`: tests for display module
		`demo`: demo code for the display module

### Guidelines

- Add new modes in their own submodule
- Inherit base classes defined in display.py where possible
- Test timing between your code and the devices you're using
	- consult psychopy (or other display codebase) for best practices for your OS
