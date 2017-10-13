# -*- coding: utf-8 -*-

def _write_triggers_from_sequence_calibration(array, file):

	x=0
	for i in array:

		# extract the letter and timing from the array
		(letter, time) = i

		# determine what the trigger are
		if x == 0:
			targetness = 'first_pres_target'
			target_letter = letter
		elif x == 1:
			targetness = 'fixation'
		elif x > 1 and target_letter == letter:
			targetness = 'target'
		else:
			targetness = 'nontarget'

		# write to the file
		file.write('%s %s %s' % (letter, targetness, time) + "\n")

		x += 1
 
	return file


def _write_triggers_from_sequence_copy_phrase(array, file, copy_text, typed_text):

	length = len(typed_text)
	x=0
	for i in array:

		# extract the letter and timing from the array
		(letter, time) = i

		# # determine what the trigger are
		# if copy_text[length + 1] == letter:
		# 	targetness = 'target'
		# elif letter == '+':
		# 	targetness = 'fixation'
		# else:
		# 	targetness = 'nontarget'

		# write to the file
		file.write('%s %s' % (letter, time) + "\n")

		x += 1
 
	return file
