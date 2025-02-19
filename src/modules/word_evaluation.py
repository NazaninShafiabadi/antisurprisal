"""
Credits to Chang and Bergen (2022b)

Evaluate language models (surprisal, antisurprisal, accuracy, rank of correct prediction) 
for certain tokens.

Sample usage:

python3 src/modules/word_evaluation.py \
--tokenizer="google/multiberts-seed_0" \
--wordbank_file="data/processed/wordbank.jsonl" \
--examples_file="data/processed/wikitext103_tokenized.txt" \
--max_samples=512 \
--batch_size=256 \
--output_file="results/surp-antisurp.txt" \
--model="google/multiberts-seed_0" --model_type="bert" \
--save_samples="data/processed/contexts.pickle" \
"""
import os
import sys
import pandas as pd
import pickle
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
import codecs
import random
from transformers import (
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="")
    # currently only supporting bert
    parser.add_argument('--model_type', default="bert")
    # should be the same as the model
    parser.add_argument('--tokenizer', default="")
    parser.add_argument('--output_file', default="surprisals.txt")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--wordbank_file', default="wordbank.tsv")
    # Examples should already be tokenized. Each line should be a
    # space-separated list of integer token ids.
    parser.add_argument('--examples_file', default="")
    # The minimum number of sample sentences to evaluate a token.
    parser.add_argument('--min_samples', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=512)
    # The minimum sequence length to evaluate a token in a sentence.
    # For unidirectional models, this only counts context before the target token.
    parser.add_argument('--min_seq_len', type=int, default=8)
    # Load token data (sample sentences for each token) from file.
    # If file does not exist, saves the token data to this file.
    parser.add_argument('--save_samples', default="")
    parser.add_argument('--save_indiv_surprisals', default="")
    return parser


def get_sample_sentences(tokenizer, wordbank_file, tokenized_examples_file,
                         max_seq_len, min_seq_len, max_samples, bidirectional=True):
    # Each entry of token data is a tuple of token, token_id, masked_sample_sentences, masked_negative_sample_sentences.
    token_data = []
    # Load words.
    df = pd.read_csv(wordbank_file, sep='\t').dropna().reset_index(drop=True)
    wordbank_tokens = df.token.unique().tolist()
    # Get token ids.
    for token in wordbank_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            token_data.append(tuple([token, token_id, [], []]))
    # Load sentences.
    print(f"Loading sentences from {tokenized_examples_file}.")
    infile = codecs.open(tokenized_examples_file, 'rb', encoding='utf-8')
    for line_count, line in enumerate(infile):
        if line_count % 100000 == 0:
            print("Finished line {}.".format(line_count))
        example_string = line.strip()
        example = [int(token_id) for token_id in example_string.split()]
        # Use the pair of sentences (instead of individual sentences), to have
        # longer sequences. Also more similar to training.
        if len(example) < min_seq_len:
            continue
        if len(example) > max_seq_len:
            example = example[:max_seq_len]
        for token, token_id, positive_samples, negative_samples in token_data:
            if len(positive_samples) >= max_samples:
                # This token already has enough sentences.
                continue
            token_indices = [index for index, curr_id in enumerate(example) if curr_id == token_id]
            # Warning: in bidirectional contexts, the mask can be in the first or last position,
            # which can cause no mask prediction to be made for the biLSTM.
            if not bidirectional:
                # The token must have enough unidirectional context.
                # The sequence length (including the target token) must be at least min_seq_len.
                token_indices = [index for index in token_indices if index >= min_seq_len-1]
            if len(token_indices) > 0:
                positive_example = example.copy()
                mask_idx = random.choice(token_indices)
                positive_example[mask_idx] = tokenizer.mask_token_id
                positive_samples.append(positive_example)
                # For every positive sample, we also save a negative sample.
                other_indices = [index for index, curr_id in enumerate(example) if curr_id != token_id]
                if len(other_indices) > 0:
                    negative_example = example.copy()
                    neg_mask_idx = random.choice(other_indices)
                    negative_example[neg_mask_idx] = tokenizer.mask_token_id # Masking another random token within the sequence
                    negative_samples.append(negative_example)
    infile.close()
    # Logging.
    for token, token_id, positive_samples, _ in token_data:
        print("{0} ({1}): {2} sentences.".format(token, token_id, len(positive_samples)))
    return token_data


# Convert a list of integer token_id lists into input_ids, attention_mask, and labels tensors.
# Inputs should already include CLS and SEP tokens.
# All sequences will be padded to the length of the longest example, so this
# should be called per batch.
# Note that the mask token will remain masked in the labels as well.
def prepare_tokenized_examples(tokenized_examples, tokenizer):
    # Convert into a tensor.
    tensor_examples = [torch.tensor(e, dtype=torch.long) for e in tokenized_examples]
    input_ids = pad_sequence(tensor_examples, 
                             batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    labels = input_ids.clone().detach()
    labels[labels == tokenizer.pad_token_id] = -100
    attention_mask = input_ids != tokenizer.pad_token_id
    inputs = {"input_ids": input_ids.to(DEVICE), "attention_mask": attention_mask.to(DEVICE), "labels": labels.to(DEVICE)}
    return inputs


"""
Output the logits (tensor shape: n_examples, vocab_size) given examples
(lists of token_ids). Assumes one mask token per example. Only outputs logits
for the masked token. Handles batching and example tensorizing.
The tokenizer should be loaded as in the main() function.
The model_type is bert.
The model can be loaded using the load_single_model() function.
"""
def run_model(model, examples, batch_size, tokenizer):
    # Create batches.
    batches = []
    i = 0
    while i+batch_size <= len(examples):
        batches.append(examples[i:i+batch_size])
        i += batch_size
    if len(examples) % batch_size != 0:
        batches.append(examples[i:])
        # Huggingface Transformers already handles batch sizes that are not
        # divisible by n_gpus.

    # Run evaluation.
    model.eval()
    with torch.no_grad():
        eval_logits = []
        model_outputs = []
        for batch_i in range(len(batches)):
            inputs = prepare_tokenized_examples(batches[batch_i], tokenizer)
            # Run model.
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            labels=inputs["labels"],
                            return_dict=True)
            logits = outputs['logits'].detach()
            # Now, logits correspond to labels.
            target_indices = inputs["labels"] == tokenizer.mask_token_id
            # Get logits at the target indices.
            # Initial logits had shape: batch_size x seq_len x vocab_size.
            # Output shape: n_masks x vocab_size. n_masks should equal batch_size.
            mask_logits = logits[target_indices, :]
            eval_logits.append(mask_logits.detach().cpu()) # Send to CPU so not all need to be held on GPU.
    # Logits output shape: num_examples x vocab_size.
    all_eval_logits = torch.cat(eval_logits, dim=0)
    if all_eval_logits.shape[0] != len(examples):
        # This can happen if there is not exactly one mask per example.
        # For example, if the last token in the sequence is masked, then the bidirectional LSTM
        # does not make a prediction for the masked token.
        print("WARNING: length of logits {0} does not equal number of examples {1}!!".format(
            all_eval_logits.shape[0], len(examples)
        ))
    return all_eval_logits, model_outputs


# Run token evaluations for a single model.
def evaluate_tokens(model, token_data, tokenizer, outfile,
                    curr_step, batch_size, min_samples):
    token_count = 0
    for token, token_id, sample_sents, negative_samples in token_data:
        print("\nEvaluation token: {}".format(token))
        token_count += 1
        print("{0} / {1} tokens".format(token_count, len(token_data)))
        print("CHECKPOINT STEP: {}".format(curr_step))
        num_examples = len(sample_sents)
        print("Num examples: {}".format(num_examples))
        if num_examples < min_samples:
            print("Not enough examples; skipped.")
            continue
        # Get logits with shape: num_examples x vocab_size.
        logits = run_model(model, sample_sents, batch_size, tokenizer)
        probs = torch.nn.Softmax(dim=-1)(logits)
        # Get median rank of correct token.
        rankings = torch.argsort(probs, axis=-1, descending=True)
        ranks = torch.nonzero(rankings == token_id) # Each output row is an index (sentence_i, token_rank).
        ranks = ranks[:, 1] # For each example, only interested in the rank (not the sentence index).
        median_rank = torch.median(ranks).item()
        # Get accuracy.
        predictions = rankings[:, 0] # The highest rank token_ids.
        accuracy = torch.mean((predictions == token_id).float()).item()
        # Get mean/stdev surprisal.
        token_probs = probs[:, token_id]    # shape: [num_examples]
        token_probs += 0.000000001 # Smooth with (1e-9).
        surprisals = -1.0*torch.log2(token_probs)    # shape: [num_examples]
        mean_surprisal = torch.mean(surprisals).item()
        std_surprisal = torch.std(surprisals).item()
        # Surprisals for negative samples
        neg_logits = run_model(model, negative_samples, batch_size, tokenizer)
        neg_probs = torch.nn.Softmax(dim=-1)(neg_logits)
        neg_token_probs = neg_probs[:, token_id]
        neg_token_probs += 0.000000001 # Smooth with (1e-9).
        neg_surprisals = -1.0*torch.log2(neg_token_probs)
        mean_neg_surprisal = torch.mean(neg_surprisals).item()
        std_neg_surprisal = torch.std(neg_surprisals).item()
        # Logging.
        print("Median rank: {}".format(median_rank))
        print("Mean surprisal: {}".format(mean_surprisal))
        print("Stdev surprisal: {}".format(std_surprisal))
        print("Accuracy: {}".format(accuracy))
        print("Mean surprisal of negative samples: {}".format(mean_neg_surprisal))
        outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(
            curr_step, token, median_rank, mean_surprisal, std_surprisal, 
            mean_neg_surprisal, std_neg_surprisal, accuracy, num_examples))
        # Save individual surprisals and antisurprisals if requested
        if args.save_indiv_surprisals != "":
            surprisals_list = surprisals.tolist()
            neg_surprisals_list = neg_surprisals.tolist()
            if len(surprisals_list) != len(neg_surprisals_list):
                max_len = max(len(surprisals_list), len(neg_surprisals_list))
                surprisals_list.extend([None] * (max_len - len(surprisals_list)))
                neg_surprisals_list.extend([None] * (max_len - len(neg_surprisals_list)))
            indiv_surps_df = pd.DataFrame(
                {'Steps': [curr_step] * len(surprisals_list),
                 'Token': [token] * len(surprisals_list),
                 'Context': sample_sents,
                 'Surprisal': surprisals_list,
                 'Antisurprisal': neg_surprisals_list
                 })
            # Append created DataFrame to the file
            indiv_surps_df.to_csv(args.save_indiv_surprisals, mode='a', header=False, index=False, sep='\t')
    return


def load_single_model(single_model_dir, config, tokenizer, model_type='bert'):
    print("Loading from: {}".format(single_model_dir))
    if model_type == "bert": # BertForMaskedLM.
        model = AutoModelForMaskedLM.from_pretrained(
            single_model_dir,
            config=config,
        ).to(DEVICE)
        model.resize_token_embeddings(len(tokenizer))
    else:
        sys.exit('Currently only supporting bert-type models.')

    return model


def main(args):
    config_path = args.model
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(config_path)
    bidirectional = True

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Overwrite special token ids in the configs.
    config.pad_token_id = tokenizer.pad_token_id
    max_seq_len = config.max_position_embeddings

    # Get the tokens to consider, and the corresponding sample sentences.
    print("Getting sample sentences for tokens.")
    if args.save_samples != "" and os.path.isfile(args.save_samples):
        print(f"Loading sample sentences from {args.save_samples}.")
        token_data = pickle.load(open(args.save_samples, "rb"))
    else: # save_samples is empty or file does not exist.
        print(f"Getting sample sentences from {args.wordbank_file}.")
        token_data = get_sample_sentences(
            tokenizer, args.wordbank_file, args.examples_file, max_seq_len, 
            args.min_seq_len, args.max_samples, bidirectional=bidirectional,
            inflections=args.inflections)
        if args.save_samples != "":
            pickle.dump(token_data, open(args.save_samples, "wb"))

    # Prepare for evaluation.
    outfile = codecs.open(args.output_file, 'w', encoding='utf-8')
    # File header.
    outfile.write("Steps\tToken\tMedianRank\tMeanSurprisal\tStdevSurprisal\tMeanAntisurprisal\tStdevAntisurprisal\tAccuracy\tNumExamples\n")

    if args.save_indiv_surprisals != "":
        indiv_surps = pd.DataFrame(columns=['Steps', 'Token', 'Context', 'Surprisal', 'NegSurprisal'])
        indiv_surps.to_csv(args.save_indiv_surprisals, index=False, sep='\t')
    
    # Get checkpoints & Run evaluation.
    steps = list(range(0, 200_000, 20_000)) + list(range(200_000, 2_100_000, 100_000))
    for step in steps:
        checkpoint = args.model + f"-step_{step//1000}k"
        model = load_single_model(checkpoint, config, tokenizer, args.model_type)
        evaluate_tokens(model, token_data, tokenizer, outfile,
                        step, args.batch_size, args.min_samples)

    outfile.close()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)