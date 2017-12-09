# This is a demo of the main bci system. It will run whatever task defined here
#  with code downstream using the parameters file passed to it.

import bci_main
from helpers.load import load_json_parameters

# Load a parameters file
parameters = load_json_parameters('parameters/parameters.json')

# RSVP mode
test_mode = 'RSVP'

# Test Type: ERP Calibration = 1, Copy Phrase = 2
test_type = 2

# Define a user
user = 'demo_user'

# Try and intialize bci main
try:
    bci_main.bci_main(parameters, user, test_type, test_mode)
except Exception as e:
    print e
    print "BCI MAIN Fail. Exiting."
