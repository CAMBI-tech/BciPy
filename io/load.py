from codecs import open as codecsopen
from json import load as jsonload
from Tkinter import Tk
from tkFileDialog import askopenfilename



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

	# use python's internal gui to call file explorers and get the filename
	Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
	filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

	return filename

def load_classifier():

	return

def read_csv_data():

	return
