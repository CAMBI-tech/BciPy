#!/usr/bin/env python
#
# Example of fine-tuning GPT2 style casual language model on some other text source.
#
# Based on example from:
# https://huggingface.co/docs/transformers/tasks/language_modeling

from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import math
import argparse
from timeit import default_timer as stopwatch
import random
from transformers import EarlyStoppingCallback, IntervalStrategy
import sys


# In ELI, each training example has an answers.text that is an array of different answers.
# We join these answers together into a single string separated by spaces.
# Then we tokenize each training example into a sequence of subword tokens and a parallel array of attention masks.
# Returns a dictionary with input_ids and attention_mask arrays.
def preprocess_function(examples):
    # NOTE: Unsure about what return_special_tokens_mask does.
    return tokenizer([" ".join(x) for x in examples["answers.text"]], truncation=True, return_special_tokens_mask=False)


# Concatenate all the text.
# Split the concatenated text into smaller chunks defined by block_size.
def group_texts(examples):
    # Not sure what this block size
    block_size = 128

    # Examples is a KeysView object, dictionary with the keys 'input_ids' and 'attention_mask'.
    # Values are the list of sequences for each training/eval sample.
    # The following line areas a dictionary with the same keys, but the list of lists becomes a single list.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    # Find the length of the first key's list
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # This will round the length down to the closest equal multiple of block_size
    total_length = (total_length // block_size) * block_size

    # Take the single sequence and create a list where each element in the list if of block_size elements.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # For casual language modeling, the label is just the input token IDs.
    # Apparently somebody else handles shifting it so the neural network's job is to predict the next subword token.
    result["labels"] = result["input_ids"].copy()
    return result


if __name__ == "__main__":
    start = stopwatch()

    parser = argparse.ArgumentParser(description='fine-tune casual language model on some text.')
    parser.add_argument("--model-name", help="encoder model name", default="distilgpt2")
    parser.add_argument("--batch-size", type=int, help="training batch size", default=32)
    parser.add_argument("--eval-batch-size", type=int, help="evaluation batch size", default=32)
    parser.add_argument("--warmup-steps", type=int, help="warmup linear steps", default=100)
    parser.add_argument("--learning-rate", type=float, help="optimizer initial learning rate", default=2e-5)
    parser.add_argument("--adam-epsilon", type=float, help="optimizer epsilon hyperparameter", default=1e-8)
    parser.add_argument("--adam-beta1", type=float, help="optimizer beta1 hyperparameter", default=0.9)
    parser.add_argument("--adam-beta2", type=float, help="optimizer beta2 hyperparameter", default=0.999)
    parser.add_argument("--weight-decay", type=float, help="optimizer weight decay hyperparameter", default=0.01)
    parser.add_argument("--epochs", type=int, help="training epochs", default=3)
    parser.add_argument("--seed", type=int, help="random number seed", default=42)
    parser.add_argument("--out-dir", help="output directory for model(s)", default="out_tune")
    parser.add_argument("--logging-steps", type=int, help="how often trainer logs things", default=500)
    parser.add_argument("--load-dir", help="load previously trained model from this directory")
    parser.add_argument("--epoch-eval", help="evaluate after each epoch", action="store_true")
    parser.add_argument("--use-mps", help="use mps device on Apple silicon", action="store_true")
    parser.add_argument("--use-ipex", help="use Intel extension for PyTorch if available", action="store_true")
    parser.add_argument("--no-cuda", help="disable use of CUDA", action="store_true")
    parser.add_argument("--fp16", help="16-bit floating point training on CUDA", action="store_true")
    parser.add_argument("--fp16-eval", help="16-bit floating point evaluation on CUDA", action="store_true")
    parser.add_argument("--early-stop", help="early stopping using evaluation loss", action="store_true")
    parser.add_argument("--early-steps", type=int, help="steps between evaluation for early stopping", default=200)
    parser.add_argument("--early-patience", type=int, help="number of worse evals before early stopping", default=3)

    args = parser.parse_args()

    if args.early_stop and args.epoch_eval:
        print(f"ERROR: --early-stop and --epoch-eval can't both be set!")
        sys.exit(0)

    random.seed(args.seed)

    eli5 = load_dataset("eli5", split="train_asks[:100]")
    eli5 = eli5.train_test_split(test_size=0.2)
    eli5 = eli5.flatten()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    physical_cores = multiprocessing.cpu_count() // 2

    # First step is to tokenize the text in the sentence for each training/eval sample
    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=physical_cores,
        remove_columns=eli5["train"].column_names,
    )

    # Create sequences that are of a specific block size by merging the sequences and then splitting
    lm_dataset = tokenized_eli5.map(
        group_texts,
        batched=True,
        num_proc=physical_cores,
    )

    # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print(f"data collator: {data_collator}")

    # Optionally we load a previously trained model
    load_name = args.model_name
    if args.load_dir:
        load_name = args.load_dir

    eval_strategy = IntervalStrategy.NO
    if args.epoch_eval:
        eval_strategy = IntervalStrategy.EPOCH
    elif args.early_stop:
        eval_strategy = IntervalStrategy.STEPS

    # Set up arguments for early stopping
    eval_steps = None
    save_steps = None
    load_best_model_at_end = False
    save_total_limit = None
    metric_for_best_model = None
    callbacks = None
    save_strategy = IntervalStrategy.NO
    if args.early_stop:
        # Note: Both these have to be set in order for early stopping to work
        eval_steps = args.early_steps
        save_steps = args.early_steps
        load_best_model_at_end = True
        save_total_limit = args.early_patience + 1
        # Require increase in evaluation loss so many times in a row
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_patience)]
        save_strategy = IntervalStrategy.STEPS

    model = AutoModelForCausalLM.from_pretrained(load_name)

    training_args = TrainingArguments(
        seed=args.seed,
        save_strategy=save_strategy,
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=load_best_model_at_end,
        save_total_limit=save_total_limit,
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        use_mps_device=args.use_mps,
        use_ipex=args.use_ipex,
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        fp16_full_eval=args.fp16_eval,
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        callbacks=callbacks,
    )

    eval_results = trainer.evaluate()
    print(f"BEFORE Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"AFTER Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    print(f"TOTAL TIME: {stopwatch() - start:.2f}")
