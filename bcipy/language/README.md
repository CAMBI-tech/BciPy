# Language

BciPy Language module provides an interface for word and character level predictions. 

The core methods of any `LanguageModel` include:

> `predict` - given typing evidence input, return a prediction (character or word).

> `load` - load a pre-trained model given a path (currently BciPy does not support training language models!)

> `update` - update internal state of your model.

You may of course define other methods, however all integrated BciPy experiments using your model will require those to be defined!

The pretrained GPT2 language model is saved in [this folder on Google Drive] (https://drive.google.com/drive/folders/1pkvwHA8SR7awxf7fj7Ds4FhY6SGxeGtX?usp=sharing). Download the files in the folder and put them in a local directory. Then use the path to the local directory to load the model.

