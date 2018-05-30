# Display Module

This is the Display module for the BCI Suite

### Structure
`display`
	`display_main`: Initializes a display window and contains useful display objects
		`rsvp`:  RSVP related display objects and functions.
		`matrix`: Matrix relates display objects and functions
		`tests`: tests for display module

### Guidelines

- Add new modes in their own folder
- Test timing between your code and the devices you're using
	- consult psychopy (or other display codebase) for best practices
- Document the function and how to use it with demo scripts and inline comments.