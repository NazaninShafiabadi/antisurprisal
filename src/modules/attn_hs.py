"""
Adapted from Chang and Bergen (2022b)

Evaluate language models (surprisal, antisurprisal, accuracy, rank of correct prediction) 
for certain tokens.

Sample usage:

python3 src/modules/attn_hs.py \
--tokenizer="google/multiberts-seed_0" \
--wordbank_file="data/processed/wordbank_attn_hs.jsonl" \
--examples_file="data/processed/sent_pairs.txt" \
--max_samples=512 \
--batch_size=32 \
--output_file="results/attn_hs/surp_antisurp.txt" \
--model="google/multiberts-seed_0" --model_type="bert" \
--save_samples="results/attn_hs/contexts.pickle" \
--attn_hs_dir="results/attn_hs"
"""
import argparse
import codecs
import gzip
import os
import pickle
import random
import string
import sys

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
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
    parser.add_argument('--save_indiv_surprisals', default="", help="Save individual surprisals and antisurprisals to a file.")
    # Save attention and hidden states for each token. 
    parser.add_argument('--attn_hs_dir', default="", help="Directory to save attention and hidden states.")
    return parser


def read_data_to_df(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t')
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.jsonl'):
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return df.dropna().reset_index(drop=True)


def get_sample_sentences(tokenizer, wordbank_file, examples_file,
                         max_seq_len, min_seq_len, max_samples, bidirectional):
    """
    examples_file contains untokenized sentences, max_segments per line, with CLS and SEP tokens.
    """
    # Each entry of token data is a tuple of token, token_id, masked_sample_sentences, masked_negative_sample_sentences.
    token_data = []
    # The masked token in the negative samples should not be one of these special tokens
    special_tokens = ['[CLS]', '[SEP]'] + list(string.punctuation)
    # Load words.
    df = read_data_to_df(wordbank_file)
    wordbank_tokens = df.token.unique().tolist()
    # Get token ids.
    for token in wordbank_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            token_data.append(tuple([token, token_id, [], []]))
    # Load sentences.
    print(f"Loading sentences from {examples_file}.")
    infile = codecs.open(examples_file, 'rb', encoding='utf-8')
    for line_count, line in enumerate(infile):
        if line_count % 100000 == 0:
            print("Finished line {}.".format(line_count))
        example_string = line.strip()
        # example = [int(token_id) for token_id in example_string.split()]
        example_tokens = example_string.split()
        # Use the pair of sentences (instead of individual sentences), to have
        # longer sequences. Also more similar to training.
        if len(example_tokens) < min_seq_len:
            continue
        if len(example_tokens) > max_seq_len:
            example_tokens = example_tokens[:max_seq_len]
        for token, token_id, positive_samples, negative_samples in token_data:
            if len(positive_samples) >= max_samples:
                # This token already has enough sentences.
                continue
            token_indices = [index for index, curr in enumerate(example_tokens) if curr.lower() == token]
            # Warning: in bidirectional contexts, the mask can be in the first or last position,
            # which can cause no mask prediction to be made for the biLSTM.
            if not bidirectional:
                # The token must have enough unidirectional context.
                # The sequence length (including the target token) must be at least min_seq_len.
                token_indices = [index for index in token_indices if index >= min_seq_len-1]
            if len(token_indices) > 0:
                positive_example = example_tokens.copy()
                mask_idx = random.choice(token_indices)
                positive_example[mask_idx] = tokenizer.mask_token
                positive_example_string = ' '.join(positive_example)
                positive_example_tokenized = tokenizer.encode(positive_example_string, add_special_tokens=False)
                # For every positive sample, we also save a negative sample.
                other_indices = [index for index, curr in enumerate(example_tokens) if curr.lower() != token and not curr in special_tokens]
                if len(other_indices) > 0:
                    # Save the positive sample only if there can also be a corresponding negative sample
                    positive_samples.append(positive_example_tokenized)
                    negative_example = example_tokens.copy()
                    neg_mask_idx = random.choice(other_indices)
                    negative_example[neg_mask_idx] = tokenizer.mask_token # Masking another random token within the sequence
                    negative_example_string = ' '.join(negative_example)
                    negative_example_tokenized = tokenizer.encode(negative_example_string, add_special_tokens=False)
                    negative_samples.append(negative_example_tokenized)
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
def run_model(model, examples, batch_size, tokenizer, return_attn_hs=False):
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
        eval_attentions = []
        eval_hidden_states = []
        eval_token_ids = []
        for batch_i in range(len(batches)):
            inputs = prepare_tokenized_examples(batches[batch_i], tokenizer)
            # Run model.         
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            labels=inputs["labels"],
                            output_attentions=return_attn_hs,
                            output_hidden_states=return_attn_hs,
                            return_dict=True)
            
            logits = outputs['logits'].detach()
            # Now, logits correspond to labels.
            target_indices = inputs["labels"] == tokenizer.mask_token_id    # shape: [batch_size, seq_len]
            # Get logits at the target indices.
            # Initial logits had shape: batch_size x seq_len x vocab_size.
            # Output shape: n_masks x vocab_size. n_masks should equal batch_size.
            mask_logits = logits[target_indices, :]
            eval_logits.append(mask_logits.detach().cpu()) # Send to CPU so not all need to be held on GPU.

            if return_attn_hs:
                # Get batch indices for each mask token (assumes one [MASK] per example)
                batch_size = target_indices.size(0)
                batch_indices = torch.arange(batch_size)
                # Get position of the [MASK] token in each sequence
                mask_positions = torch.nonzero(target_indices, as_tuple=True)[1]    # Shape: [batch_size]
                
                # Extract attentions at the target positions from the last layer
                # Each layer's attention is a tensor of shape [batch_size, num_heads, seq_len, seq_len]
                target_attentions = (outputs['attentions'][-1][batch_indices, :, mask_positions, :]
                                     .detach()  # Detach from the computation graph
                                     .cpu()     # Move to CPU to save GPU memory
                                    )           # Shape: [batch_size, num_heads, batch_size, seq_len] -> [batch_size, num_heads, seq_len]
                
                # Quantize and sparsify attention values (to reduce memory and disk usage)
                sparse_attn = torch.round(target_attentions * 10000).to(torch.int16).to_sparse()
                # Dequantize and densify later with sparse_tensor.to_dense().to(torch.float32) / 10000
                eval_attentions.append(sparse_attn)
            
                # Extract hidden states at the target positions from the last layer
                # Each layer's hidden states is a tensor of shape [batch_size, seq_len, hidden_size]
                target_hidden_states = outputs['hidden_states'][-1][batch_indices, mask_positions, :].detach().cpu().half()    # Shape: [batch_size, batch_size, hidden_size]
                eval_hidden_states.append(target_hidden_states)

                # Extract unpadded input_ids for future attention visualization
                trimmed_input_ids = [
                    ids[:mask.sum().item()].tolist()
                    for ids, mask in zip(inputs["input_ids"], inputs["attention_mask"])
                ]   # Shape: [batch_size, unpadded seq_len]
                eval_token_ids.extend(trimmed_input_ids)

            # Free memory after each batch
            del outputs
            torch.cuda.empty_cache()

    all_eval_logits = torch.cat(eval_logits, dim=0) # Shape: [num_examples, vocab_size]
    if all_eval_logits.shape[0] != len(examples):
        # This can happen if there is not exactly one mask per example.
        print("WARNING: length of logits {0} does not equal number of examples {1}!!".format(
            all_eval_logits.shape[0], len(examples)
        ))
    
    return all_eval_logits, eval_attentions, eval_hidden_states, eval_token_ids


# Run token evaluations for a single model.
def evaluate_tokens(model, token_data, tokenizer, outfile, indiv_surprisals_file,
                    curr_step, batch_size, min_samples, attn_hs_dir):
    token_count = 0
    for token, token_id, positive_samples, negative_samples in token_data:
        print("\nEvaluation token: {}".format(token))
        token_count += 1
        print("{0} / {1} tokens".format(token_count, len(token_data)))
        print("CHECKPOINT STEP: {}".format(curr_step))
        num_examples = len(positive_samples)
        print("Num examples: {}".format(num_examples))
        if num_examples < min_samples:
            print("Not enough examples; skipped.")
            continue
        
        # Sort the positive and negative samples by length to minimize padding, while keeping the pairs together.
        paired_samples = list(zip(positive_samples, negative_samples))
        paired_samples = sorted(paired_samples, key=lambda x: len(x[0]))    # sort by length of positive samples
        positive_samples, negative_samples = map(list, zip(*paired_samples))

        # Get logits with shape: num_examples x vocab_size.
        return_attn_hs = True if attn_hs_dir else False
        logits, attentions, hidden_states, token_id_lists = run_model(model, positive_samples, batch_size, tokenizer, return_attn_hs)
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
        std_surprisal = torch.std(surprisals).item() if surprisals.numel() > 1 else 0.0
        
        # Surprisals for negative samples (Antisurprisals)
        neg_logits, neg_attentions, neg_hidden_states, neg_token_id_lists = run_model(model, negative_samples, batch_size, tokenizer)
        neg_probs = torch.nn.Softmax(dim=-1)(neg_logits)
        neg_token_probs = neg_probs[:, token_id]
        neg_token_probs += 0.000000001 # Smooth with (1e-9).
        neg_surprisals = -1.0*torch.log2(neg_token_probs)
        mean_neg_surprisal = torch.mean(neg_surprisals).item()
        std_neg_surprisal = torch.std(neg_surprisals).item() if neg_surprisals.numel() > 1 else 0.0
        
        # Logging.
        print("Median rank: {}".format(median_rank))
        print("Mean surprisal: {}".format(mean_surprisal))
        print("Stdev surprisal: {}".format(std_surprisal))
        print("Accuracy: {}".format(accuracy))
        print("Mean antisurprisal: {}".format(mean_neg_surprisal))
        outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(
            curr_step, token, median_rank, mean_surprisal, std_surprisal, 
            mean_neg_surprisal, std_neg_surprisal, accuracy, num_examples))
        
        # Save individual surprisals and antisurprisals if requested
        if indiv_surprisals_file:
            ...
            # pickle.dump({
            #     'Step': curr_step,
            #     'Token': token,
            #     'TokenID': token_id,
            #     'Surprisals': surprisals.tolist(),
            #     'Antisurprisals': neg_surprisals.tolist(),
            #     'NumExamples': num_examples
            # }, indiv_surprisals_file, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save attention and hidden states if requested
        if attn_hs_dir:
            torch.save({'Step': curr_step, 
                        'Token': token, 
                        'TokenIDs': token_id_lists,
                        'Attentions': attentions, 
                        'HiddenStates': hidden_states
                        }, os.path.join(attn_hs_dir, f"token_{token}_step_{curr_step}.bin"))
    return


def load_single_model(single_model_dir, config, tokenizer, model_type='bert'):
    print("Loading from: {}".format(single_model_dir))
    if model_type == "bert": # BertForMaskedLM.
        model = AutoModelForMaskedLM.from_pretrained(
            single_model_dir,
            config=config,
            attn_implementation="eager"
        ).to(DEVICE)
        model.resize_token_embeddings(len(tokenizer))
    elif model_type == "gpt":
        model = AutoModelForCausalLM.from_pretrained(
            single_model_dir,
            config=config
        ).to(DEVICE)
        model.resize_token_embeddings(len(tokenizer))
    else:
        sys.exit('Currently only supporting bert-type and gpt-type models.')

    return model


def main(args):
    args.model_type = args.model_type.lower()
    if args.model_type != "bert" and args.model_type != "gpt":
        sys.exit("Currently only supporting bert-type and gpt-type models.")
    # Load model config.
    config_path = args.model
    config = AutoConfig.from_pretrained(config_path)
    bidirectional = True if args.model_type == "bert" else False

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Overwrite special token ids in the configs.
    config.pad_token_id = tokenizer.pad_token_id
    max_seq_len = config.max_position_embeddings

    # Get tokens and corresponding sample sentences.
    print("Getting sample sentences for tokens.")
    if args.save_samples != "" and os.path.isfile(args.save_samples):
        print(f"Loading sample sentences from {args.save_samples}.")
        token_data = pickle.load(open(args.save_samples, "rb"))
    else: # save_samples is empty or file does not exist.
        print(f"Getting sample sentences from {args.examples_file}.")
        token_data = get_sample_sentences(
            tokenizer, args.wordbank_file, args.examples_file, max_seq_len, 
            args.min_seq_len, args.max_samples, bidirectional=bidirectional)
        if args.save_samples != "":
            try:
                pickle.dump(token_data, open(args.save_samples, "wb"))
            except FileNotFoundError:
                os.makedirs(os.path.dirname(args.save_samples))
                pickle.dump(token_data, open(args.save_samples, "wb"))

    # Prepare for evaluation.
    outfile = codecs.open(args.output_file, 'w', encoding='utf-8')
    # File header.
    outfile.write("Step\tToken\tMedianRank\tMeanSurprisal\tStdSurprisal\tMeanAntisurprisal\tStdAntisurprisal\tAccuracy\tNumExamples\n")

    if args.save_indiv_surprisals != "":
        # Ensure the file extension is correct
        indiv_surprisals_file = args.save_indiv_surprisals.split('.')[0] + '.pkl.gz' \
                                if not args.save_indiv_surprisals.endswith('.pkl.gz') \
                                else args.save_indiv_surprisals
        # If the file already exists, delete it.
        if os.path.isfile(indiv_surprisals_file):
            os.remove(indiv_surprisals_file)
        # Open the file in append mode
        indiv_surprisals_file = gzip.open(indiv_surprisals_file, 'ab')
    else:
        indiv_surprisals_file = None

    # Check that the attn_hs_dir exists
    if args.attn_hs_dir and not os.path.exists(args.attn_hs_dir):
        os.makedirs(args.attn_hs_dir)
    
    # Get checkpoints & Run evaluation.
    if args.model_type == "bert":
        steps = list(range(0, 200_000, 20_000)) + list(range(200_000, 2_100_000, 100_000))
        for step in steps:
            checkpoint = args.model + f"-step_{step//1000}k"
            model = load_single_model(checkpoint, config, tokenizer, args.model_type)
            evaluate_tokens(model, token_data, tokenizer, outfile, indiv_surprisals_file,
                            step, args.batch_size, args.min_samples, args.attn_hs_dir)
            del model
            torch.cuda.empty_cache()
    else:
        sys.exit("Currently only supporting bert-type models.")
    
    outfile.close()
    if indiv_surprisals_file:
        indiv_surprisals_file.close()
    return


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)