from codecs import open as codecsopen
from json import load as jsonload


def load_json_parameters(path):
	# loads in json parameters and turns it into a dictionary 

	with codecsopen(path, 'r', encoding='utf-8') as f:
	    parameters = []
	    try:
	        parameters = jsonload(f)
	    except ValueError as error:
	        warn("Parameters file is formatted incorrectly!", Warning)
	        raise error

	f.close()

	return parameters

def load_experimental_data():

	return

def load_classifier():

	return

def read_csv_data():

	return