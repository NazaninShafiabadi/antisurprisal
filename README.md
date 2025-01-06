# Anti-surprisal
A complementary metric to $surprisal$, measuring a model's capacity to refrain from using words in inappropriate or unexpected contexts.

## Introduction
Traditionally, lexical skill acquisition has been assessed using learning curves, which track metrics such as accuracy, perplexity, or surprisal to measure a model’s ability to predict words in context over time or across training iterations. 
However, we believe that this traditional approach has limitations, as it provides only a partial view of a model’s ability to “learn” a word: while existing learning curves effectively capture whether a model can use a word in an appropriate context, they do not account for whether the model has also learned when not to use the word — a dimension of learning that is equally important yet often overlooked.
In this work, we demonstrate that examining the dual evolution of a model’s ability to use words correctly in context while avoiding misapplication offers deeper insights than those gained by focusing solely on the probability of correct word usage, thereby capturing a fuller picture of the model’s linguistic capabilities.

## Setup
The `requirements.txt` file includes all necessary dependencies to run the code of this repository.
```
pip install -r requirements.txt
```

## Tokenization
Tokenize a raw text file for evaluation. In this example, we concatenate each pair of lines.
```
python3 src/tokenize_dataset.py \
--tokenizer="google/multiberts-seed_0" \
--input_file="data/raw/wikitext103_test.txt" \
--output_file="data/processed/wikitext103_tokenized.txt" \
--max_segments=2 --max_seq_len=-1
```

## Word/token evaluation
To collect surprisals and anti-surprisals for individual tokens at each checkpoint:
```
python3 src/modules/word_evaluation.py \
--tokenizer="google/multiberts-seed_0" \
--wordbank_file="data/processed/wikitext_wordbank.tsv" \
--examples_file="data/processed/wikitext103_tokenized.txt" \
--max_samples=512 \
--batch_size=256 \
--output_file="results/surp-antisurp.txt" \
--model="google/multiberts-seed_0" --model_type="bert" \
--save_samples="data/wikitext/sample_sents.pickle"
```
Currently, only BERT-type models are supported. The saved samples store the occurrences of target tokens in the dataset. 

The code in the `tokenize_dataset.py` and `word_evaluation.py` files are largely adapted from [Chang and Bergen (2022b)](https://github.com/tylerachang/word-acquisition-language-models).

## Analyses
Aggregated at the corpus level, surprisal decreases smoothly as learning progresses, while anti-surprisal increases. This indicates that, on average, the probability of using words in appropriate contexts rises over the course of training, while the probability of inappropriate word use declines.

<img src="https://github.com/NazaninShafiabadi/antisurprisal/blob/main/img/corpus_surprisal.jpg" width="300" height="auto">

On the other hand, analyzing learning curves for individual words reveals that the tested model has highly variable learning dynamics from one word to the next. Examples of four carefully selected words, each with distinct learning trajectories representing all possible combinations of increasing or decreasing surprisal and anti-surprisal curves are presented below. 

<img src="https://github.com/NazaninShafiabadi/antisurprisal/blob/main/img/trend_category_examples.jpg" width="500" height="auto">

A summary of these trends is presented in the following Table, showing how often (in the tested wordbank) both curves move in the same direction, as well as cases where one curve increases while the other decreases.

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2";">Surprisal</th>
    </tr>
    <tr>
      <th>Anti-surprisal</th>
      <th>Increasing</th>
      <th>Decreasing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Increasing</th>
      <td>2.7%</td>
      <td>91.5%</td>
    </tr>
    <tr>
      <th>Decreasing</th>
      <td>0.4%</td>
      <td>5.4%</td>
    </tr>
  </tbody>
</table>

The data reveals that the vast majority of words fall into the category of decreasing surprisal and increasing anti-surprisal, which is the expected behavior during training. However, a notable minority (8.5% of the words in our word bank) exhibit different trends, underscoring the complexity of lexical acquisition beyond what corpus-level or surprisal-focused analyses might suggest.

We show that combining surprisal and anti-surprisal metrics offers a richer understanding of lexical skill acquisition in LLMs. These findings pave the way for more refined training and evaluation strategies in future research.

## Citation
...









