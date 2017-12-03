# Language Model server recipe
1. Install docker.

2. Install docker-machine (found in [docker for windows](https://docs.docker.com/engine/installation/windows/)).

3. Pull the latest version of langModelPy from bitbucket.

4. Make sure you have the following file hirarchy and you are here:
 + Dokerfile
 + specializer (all)
 + ebitweight (all)
 + lm
     - requirements.txt
     - lm_server.py
     - util.py
     - server.py
     - bitweight.py
 +  specialfiles
     - basictypes.pdx
     - fst.pxd
     - ios.pxd
     - memory.pxd

    
5. Read dockerRun.txt file and start a machine as described in 1.

6. Build an Image (a one time build) as described in 2 on a machine you defined in 1

Congrats! You built the lmImage :-)

7. Open use_LMWRAPPER.py and find the path on your machine to the fst file (e.g. c:\Users\shaobinx\langModelPy\brown\_closure.n5.kn.fst. [The current language model](https://bitbucket.org/cogsyslab/langmodelpy/src/34e35c06d0f4/lm/?at=master).

8. Run use_LMWrapper.py from its current directory (to be able to import the LM_Wrapper module)

9. After finishing running the RSVP code you can execute 3 in runDocker.txt

###LMWrapper Module

LMWrapper module provides the LangModel class from which requests are sent to the language model server. 
The current implementation uses logging, custom error handling and requires having docker environment to be up and running. 
Each method in LangModel class contains explanations corresponding to the purpose they were made for. 

The current Wrapper has 3 different methods: 

1.init that initializes a class of lm

2.reset that clears lm history

3.state update that gets a symbol decision and sends in return the appropriate prior distribution

In addition, use_LMWrapper.pt provides a show case for how to call langModel class methods from the LMWrapper module.
If there are problems in the process there are different error messages that can be raised: connection error, status code of response, correctness of input. 
