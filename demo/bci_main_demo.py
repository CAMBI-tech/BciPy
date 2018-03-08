# This is a demo of the main bci system. It will run the task defined here
#  using the parameters file passed to it.

def main():
	import bci_main
	from helpers.load import load_json_parameters

	# Load a parameters file
	parameters = load_json_parameters('parameters/parameters.json')

	# Mode: ex. RSVP, Shuffle, Matrix
	test_mode = 'RSVP'

	# Test Type: ex. RSVP Calibration = 1, Copy Phrase = 2
	test_type = 1

	# Define a user
	user = 'bci_main_demo_user'

	# Try and intialize with bci main
	try:
	    bci_main.bci_main(parameters, user, test_type, test_mode)
	except Exception as e:
	    print("BCI MAIN Fail. Exiting. Error: \n")
	    print(e)

if __name__ == "__main__":
    main()
