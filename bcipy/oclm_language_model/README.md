# Language Model server recipe

###Step 1: Build or Retrieve the LM docker image
See https://bitbucket.org/cogsyslab/langmodelpy to see instructions on building a docker container!

 -or-

Contact LM Team (Shiran Dudy, Steven Bedrick) or BCI Codebase maintainers for a pre-built docker image.


###LMWrapper Module

This assumes you have a working Docker image on your local computer!

language\_model module provides the LangModel class from which requests are sent to the language model server and itself serves as the client. 

The current implementation uses logging, custom error handling and requires having docker environment to be up and running. 

Each method in LangModel class contains explanations corresponding to the purpose they were made for. 

The current Wrapper has 3 different methods: 

1.init that initializes a class of lm

2.reset that clears lm history

3.state update that gets a symbol decision and sends in return the appropriate prior distribution for letters and words

In addition, demo.py provides a show case for how to call langModel class methods from the language\_model module.
If there are problems in the process there are different error messages that can be raised: connection error, status code of response, correctness of input. 
