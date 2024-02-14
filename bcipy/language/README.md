# Language

BciPy Language module provides an interface for word and character level predictions.

The core methods of any `LanguageModel` include:

> `predict` - given typing evidence input, return a prediction (character or word).

> `load` - load a pre-trained model given a path (currently BciPy does not support training language models!)

> `update` - update internal state of your model.

You may of course define other methods, however all integrated BciPy experiments using your model will require those to be defined!

The language module has the following structure:

> `demo` - Demo scripts utilizing the different models.

> `lms` - The default location for the model resources.

> `model` - The python classes for each LanguageModel subclass. Detailed descriptions of each can be found below.

> `sets` - Different phrase sets that can be used to test the language model classes.

> `tests` - Unit test cases for each language model class.

<!-- The pretrained GPT2 language model is saved in [this folder on Google Drive](https://drive.google.com/drive/folders/1pkvwHA8SR7awxf7fj7Ds4FhY6SGxeGtX?usp=sharing). Download the files in the folder and put them in a local directory. Then use the path to the local directory to load the model. (Alternatively, just pass in the model name like "gpt2" as the language model path and the pretrained language model will be downloaded and stored in local cache) -->

## Uniform Model

The UniformLanguageModel provides equal probabilities for all symbols in the symbol set. This model is useful for evaluating other aspects of the system, such as EEG signal quality, without any influence from a language model.

## KenLM Model
The KenLMLanguageModel utilizes a pretrained n-gram language model to generate probabilities for all symbols in the symbol set. N-gram models use frequencies of different character sequences to generate their predictions. Models trained on AAC-like data can be found [here](https://imagineville.org/software/lm/dec19_char/). For faster load times, it is recommended to use the binary models located at the bottom of the page. The default parameters file utilizes `lm_dec19_char_large_12gram.kenlm`. If you have issues accessing, please reach out to us on GitHub or via email at `cambi_support@googlegroups.com`.

For models that import the kenlm module, this must be manually installed using `pip install kenlm==0.1 --global-option="max_order=12"`.

## Causal Model
The CausalLanguageModel class can use any causal language model from Huggingface, though it has only been tested with gpt2, facebook/opt, and distilgpt2 families of models. Causal language models predict the next token in a sequence of tokens. For the many of these models, byte-pair encoding (BPE) is used for tokenization. The main idea of BPE is to create a fixed-size vocabulary that contains common English subword units. Then a less common word would be broken down into several subword units in the vocabulary. For example, the tokenization of character sequence `peanut_butter_and_jel` would be:
> *['pe', 'anut', '_butter', '_and', '_j', 'el']*

Therefore, in order to generate a predictive distribution on the next character, we need to examine all the possibilities that could complete the final subword tokens in the input sequences. We must remove at least one token from the end of the context to allow the model the option of extending it, as opposed to only adding a new token. Removing more tokens allows the model more flexibility and may lead to better predictions, but at the cost of a higher prediction time. In this model we remove all of the subword tokens in the current (partially-typed) word to allow it the most flexibility. We then ask the model to estimate the likelihood of the next token and evaluate each token that matches our context. For efficiency, we only track a certain number of hypotheses at a time, known as the beam width, and each hypothesis until it surpasses the context. We can then store the likelihood for each final prediction in a list based on the character that directly follows the context. Once we have no more hypotheses to extend, we can sum the likelihoods stored for each character in our symbol set and normalize so they sum to 1, giving us our final distribution.


## Mixture Model
The MixtureLanguageModel class allows for the combination of two or more supported models. The selected models are mixed according to the provided weights, which can be tuned using the Bcipy/scripts/python/mixture_tuning.py script. It is not recommended to use more than one "heavy-weight" model with long prediction times (the CausalLanguageModel) since this model will query each component model and parallelization is not currently supported.

# Contact Information

For language model related questions, please contact Dylan Gaines (dcgaines [[at](https://en.wikipedia.org/wiki/At_sign)] mtu.edu) or create an issue.


