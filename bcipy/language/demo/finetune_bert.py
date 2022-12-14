# Exploring how to finetune an encoder model for doing classification.
# This code works for a variety of the BERT-like models on Hugging Face.
#
# Started with BERT example from:
# https://huggingface.co/docs/transformers/training

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

from timeit import default_timer as timer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

if __name__ == "__main__":

    #model_name = "google/bert_uncased_L-2_H-128_A-2"           # Smallest BERT model
    #model_name = "google/electra-small-discriminator"
    model_name = "microsoft/mpnet-base"

    dataset = load_dataset("yelp_review_full")

    # Take only a subset of the data
    train_small = dataset["train"].shuffle(seed=42).select(range(1000))
    eval_small = dataset["test"].shuffle(seed=42).select(range(1000))

    #print(f"{train_small[0]}")
    #print(f"{eval_small[0]}")

    # Tokenize the subsets
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)   # Compared to example, had to add model_max_length
    train_small_tokenized = train_small.map(tokenize_function, batched=True)
    eval_small_tokenized = eval_small.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    training_args = TrainingArguments(seed=42,
                                      save_strategy="epoch",
                                      evaluation_strategy="epoch",
                                      output_dir="out_bert",
                                      resume_from_checkpoint="out_bert")

    metric = evaluate.load("accuracy")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_small_tokenized,
        eval_dataset=eval_small_tokenized,
        compute_metrics=compute_metrics,
    )

    trainer.train()
#    results = trainer.evaluate()
#    print(f"{results}")
