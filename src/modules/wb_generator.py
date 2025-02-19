"""
Generates a word bank from a text file and (optionally) a file 
containing the filtered words along with the reasons for their exclusion.

The word bank is saved as a JSON Lines file with the following fields:
    - token: the word token in lowercase
    - token_id: the token ID as assigned by the tokenizer
    - count: the number of occurrences of the token in the text
    - POS: a list of Part-of-Speech (POS) tags assigned to the token

Words are filtered out based on the following criteria:
    - Words containing non-alphabetic characters
    - Words containing non-ASCII characters
    - Multi-token words (if specified by the user)
    - Proper nouns

Sample usage:    
python src/modules/wb_generator.py \
--input_file data/raw/wikitext103_test.txt \
--output_file data/processed/wordbank.jsonl \
--model_name google/multiberts-seed_0 \
--pos_tagging_mode sentence \
--filtered_words_file data/processed/filtered_words.tsv \
--single-tokens-only
"""
import argparse
import spacy
import pandas as pd
from collections import Counter, defaultdict
from transformers import AutoTokenizer


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='The input file to generate the word bank from.')
    parser.add_argument('--output_file', type=str, help='The output file to write the word bank to.')
    parser.add_argument('--model_name', type=str, help='The name of the pre-trained model for tokenization.')
    parser.add_argument('--pos_tagging_mode', type=str, default='word', help='The POS tagging granularity (word or sentence).')
    parser.add_argument('--filtered_words_file', type=str, default='',  help='The file to write the filtered words to.')
    parser.add_argument('--single-tokens-only', action='store_true', help='Keep only words that are tokenized into single tokens.')
    return parser


def in_context_POS_tagging(text, tagger, wordbank):
    """
    Performs POS tagging using the full text context
    """
    tagger.max_length = 1500000
    doc = tagger(text)

    for token in doc:
        word = token.text.lower()
        if word in wordbank:
            wordbank[word]['POS'].add(token.pos_)

    return wordbank


def generate_word_bank(args):
    """
    Generate a word bank from a list of words, including tokenization and POS tagging information.
    
    Args: 
        input_file (str): Path to the input file containing a string of text.
        output_file (str): Path to the output file to save the word bank (in JSON Lines format).
        model_name (str): Name of the pre-trained model for tokenization.
        pos_tagging_mode (str): POS tagging granularity ('word' or 'sentence').
        single_tokens_only (bool): Whether to include all or only single-token words
    """
    # Load SpaCy model and tokenizer
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Only need POS tagging
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Read input words
    with open(args.input_file, 'r', encoding="utf-8") as f:
        text = f.read()

    word_counts = Counter(text.split())
    word_bank = {}  # {word: {"token": str, "count": int, "POS": set}}
    filtered_words = defaultdict(dict)

    for doc in nlp.pipe(word_counts.keys(), batch_size=50):
        token = doc[0]  # (single-token input)
        word = token.text
        word_lower = word.lower()

        # Skip non-alphabetic, non-ASCII and multi-token (if required) words
        tokens = tokenizer.tokenize(word)
        if not word.isalpha() or not word.isascii() or (args.single_tokens_only and len(tokens) != 1):  
            if args.filtered_words_file != '':
                filtered_words[word]['word'] = word
                filtered_words[word]['isalpha'] = word.isalpha()
                filtered_words[word]['isascii'] = word.isascii()
                filtered_words[word]['len_tokens'] = len(tokens)
                filtered_words[word]['isPROPN'] = token.pos_ == 'PROPN'
            continue

        # Initialize word entry if not present
        entry = word_bank.setdefault(word_lower, {
            "token": word_lower,
            "count": 0,
            "POS": set()
        })
        entry["count"] += word_counts[word]

        if args.pos_tagging_mode == 'word':  # Out-of-context POS tagging
            entry["POS"].add(token.pos_) 

    if args.pos_tagging_mode == 'sentence':  # In-context POS tagging (takes longer)
        word_bank = in_context_POS_tagging(text, nlp, word_bank)

    # Convert word bank to DataFrame
    df = pd.DataFrame.from_dict(word_bank, orient='index').reset_index(drop=True)

    # Remove proper nouns
    wb_df = df[~df['POS'].apply(lambda x: x == {'PROPN'})].copy()

    if args.filtered_words_file != '':
        for w in df[df['POS'].apply(lambda x: x == {'PROPN'})]['token']:
            filtered_words[w]['word'] = w
            filtered_words[w]['isalpha'] = False
            filtered_words[w]['isascii'] = False
            filtered_words[w]['len_tokens'] = 1
            filtered_words[w]['isPROPN'] = True
        
        filtered_df = pd.DataFrame.from_dict(filtered_words, orient='index').reset_index(drop=True)
        filtered_df.to_csv(args.filtered_words_file, sep='\t', index=False)
    
    # Convert POS sets to lists for JSON serialization
    wb_df['POS'] = wb_df['POS'].apply(list)
    
    # Add token IDs
    wb_df['token_id'] = tokenizer.convert_tokens_to_ids(wb_df['token'].tolist())

    # Save word bank to a JSON Lines file
    wb_df = wb_df[wb_df['count'] > 0][['token', 'token_id', 'count', 'POS']]    # Remove zero-count words & reorder columns
    wb_df.to_json(args.output_file, orient='records', lines=True)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Verify input arguments
    if args.pos_tagging_mode not in ['word', 'sentence']:
        raise ValueError("pos_tagging_mode must be either 'word' or 'sentence'.")
    if not args.single_tokens_only:
        raise NotImplementedError("Multi-token words are not supported yet.")
    
    generate_word_bank(args)