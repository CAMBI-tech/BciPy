# Language

BciPy Language module provides an interface for word and character level predictions.

The core methods of any `LanguageModel` include:

> `predict` - given typing evidence input, return a prediction (character or word).

> `load` - load a pre-trained model given a path (currently BciPy does not support training language models!)

> `update` - update internal state of your model.

You may of course define other methods, however all integrated BciPy experiments using your model will require those to be defined!

The pretrained GPT2 language model is saved in [this folder on Google Drive](https://drive.google.com/drive/folders/1pkvwHA8SR7awxf7fj7Ds4FhY6SGxeGtX?usp=sharing). Download the files in the folder and put them in a local directory. Then use the path to the local directory to load the model. (Alternatively, just pass in the model name like "gpt2" as the language model path and the pretrained language model will be downloaded and stored in local cache)

## Uniform Model

The UniformLanguageModel provides equal probabilities for all symbols in the symbol set. This model is useful for evaluating other aspects of the system, such as EEG signal quality, without strong influence from a language model.

## GPT2 Model

The GPT2LanguageModel utilizes Huggingface's pretrained GPT2 language model and the beam search algorithm to generate probabilities for all symbols in the symbol set. This is the model we are currently using to incorporate language model predictions into BciPy experiments.

For the GPT2 model, byte-pair encoding (BPE) is used for tokenization. The main idea of BPE is to create a fixed-size vocabulary that contains common English subword units. Then a less common word would be broken down into several subword units in the vocabulary. For example, the tokenization of character sequence `peanut_butter_and_jel` would be:
> *['pe', 'anut', '_butter', '_and', '_j', 'el']*

Therefore, in order to generate a predictive distribution on the next character, we need to examine all the possibilities that could complete the final subword units in the input sequences. We simply take the input and remove the last (partially-typed) subword unit and make the model predict it over the entire vocabulary. Then we select the subword units with prefixes matching the partially-typed last subword from the input (in the example case above, the selected subword units should start with *'el'*). Then we renormalize the probability distribution over the selected subword units and marginalize over the first character after the matched prefix. The resulting character-level distribution would be a prediction of the next character given current input.

However, predicting only the final subword units may limit the space of possible continuations of the input. To remedy that, we need to look beyond one subword unit. In such occasion, we would utilize beam search to generate multiple subword units. Beam search is a heuristic search algorithm that expands the most promising hypotheses in a selected set. In our case, the most promising hypotheses would be the top ranked subword units on the current beam. For each of these hypotheses, we would make the language model further predict the next subword unit over the entire vocabulary. Then the top ranked hypotheses returned would again be retained for further expansion. The renormalization and marginalization would be performed at the end of the beam search for all candidates on the beam. Note that the beam size and the beam search depth are hyperparameters that can be adjusted.

Furthermore, we use a unigram language model trained on the ALS Phraseset to interpolate with GPT2 for smoothing purpose, since the distributions given by GPT2 can be very "sharp", where most of the probability mass falls on the top few letters. Therefore, interpolating with a unigram model could give more probability to the lower ranked letters.

## Word Level Prediction

Currently we think it is a good idea to extract the suggestions given by the GPT2 model to complete the last partial subword unit from the user input, without including the ones with further word piece predictions. In other words, we only need the GPT2 model to perform a beam search for subword units with a depth of 1.

For example, if the user input is `peanut_butter_and_jel`, the last partial subword unit would be `_el`. Then we would like the GPT2 model to make a prediction from `peanut_butter_and` and include the top ranked subword units returned that start with `_el`. 

The current code has these candidates with their log likelihood: at line 215 of language/model/gpt2.py, we populate these candidates to all_beam_candidates. We can use this data structure as a starting point to generate word level predictions. For instance, we can perform a softmax on the log likehood of the candidates and obtain a probability distribution over subword units.

# Contact Information

For GPT2 related questions, please contact Shijia Liu (liu.shij [[at](https://en.wikipedia.org/wiki/At_sign)] northeastern.edu)

For other questions related to language modeling, please create an issue.


