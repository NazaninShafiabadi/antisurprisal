from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import spacy


def get_tags(doc_path):
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 2000000
    pos_dict = {}
    with open(doc_path, 'r', encoding='utf-8') as file:
        text = file.read()
        doc = nlp(text)
        for token in doc:
            if token.text.lower() in pos_dict and not token.pos_ in pos_dict[token.text.lower()]:
                pos_dict[token.text.lower()].append(token.pos_)
            else:
                pos_dict[token.text.lower()] = [token.pos_]
        
    return pd.DataFrame(list(pos_dict.items()), columns=['Token', 'POS'])


def plot_avg(dfs: List[pd.DataFrame]):
    plt.style.use('ggplot')
    plt.figure(figsize=(3, 3))
    max_y = 0
    min_y = float('inf')
    for df in dfs:
        avg = (df.groupby('Steps')
                 .agg({'MeanSurprisal': 'mean', 'POS': 'first'})
                 .reset_index()
                 .assign(Diffs=lambda x: x['MeanSurprisal'].diff().fillna(0)))

        plt.plot(avg['Steps'], avg['MeanSurprisal'], label=f"{avg['POS'].values[0][0]}")
        max_y = avg['MeanSurprisal'].max() if avg['MeanSurprisal'].max() > max_y else max_y
        min_y = avg['MeanSurprisal'].min() if avg['MeanSurprisal'].min() < min_y else min_y

    # plt.ylim(max_y, min_y - 1)
    plt.xlabel('Steps')
    plt.ylabel('Mean Surprisal')
    # plt.xscale('log')
    plt.legend()
    plt.show()


def plot_all_in_one(words:List[str], surprisals_df, neg_samples=False):
    plt.style.use('ggplot')
    plt.figure(figsize=(3, 3))

    colormap = plt.get_cmap('tab20')
    colors = [colormap(i / len(words)) for i in range(len(words))]

    for i, word in enumerate(words):   
        word_data = surprisals_df[surprisals_df['Token'] == word]
        if word_data.empty:
            print(f'No data found for the word "{word}"')
            continue
        
        plt.plot(word_data['Steps'], word_data['MeanSurprisal'], marker='o', 
                 color=colors[i], label=f'{word} (surprisal)')

        if neg_samples:
            plt.plot(word_data['Steps'], word_data['MeanNegSurprisal'], marker='o', 
                     alpha=0.5, color=colors[i], label=f'{word} (antisurprisal)')
    
    plt.xlabel('Steps')
    plt.ylabel('Mean surprisal')
    # plt.gca().invert_yaxis()  # Invert the y-axis
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()