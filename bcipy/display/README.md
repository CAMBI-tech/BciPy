# Display Module

This is the Display module for the BCI Suite

### Structure
`display`
	`main`: Initializes a display window and contains useful display objects
		`rsvp`:  RSVP related display objects and functions.
			`mode`: defines task specific displays
		`tests`: tests for display module
		`demo`: demo code for the display module

### Guidelines

- Add new modes in their own directory
- Test timing between your code and the devices you're using
	- consult psychopy (or other display codebase) for best practices for your OS
