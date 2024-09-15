""" Contains the functions needed for analyzing learning curves. """

import math
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



def fit_linear(X:np.ndarray, y:np.ndarray, first_step=True) -> Dict[str, float]:
    if not first_step:
        X, y = X[1:], y[1:]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    alpha = model.coef_[0]  # slope
    beta = model.intercept_
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return X.flatten(), y_pred, {'model': 'Linear Regression', 'alpha': alpha, 'beta': beta, 'R2': r2, 'MSE': mse}


def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def polynomial_model(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def linear_model(x, a, b):
    return a * x + b

def fit_model(model, x, y, initial_guess, bounds=(-np.inf, np.inf)):
    popt, _ = curve_fit(model, x, y, p0=initial_guess, bounds=bounds)
    y_fit = model(x, *popt)
    return popt, y_fit

def select_best_model(x, y):
    models = [
        (exponential_decay, [y.max(), 1e-6, y.min()], 'Exponential Decay'),
        (polynomial_model, [0, 0, 0, y.mean()], 'Polynomial Model'),
        (linear_model, [0, y.mean()], 'Linear Model')
    ]
    
    best_model = None
    best_fit = None
    best_score = np.inf
    best_name = ""
    
    for model, initial_guess, name in models:
        try:
            popt, y_fit = fit_model(model, x, y, initial_guess)
            mse = mean_squared_error(y, y_fit)
            if mse < best_score:
                r2 = r2_score(y, y_fit)
                best_score = mse
                best_model = model
                best_fit = y_fit
                best_name = name
        except RuntimeError as e:
            # print(f"Error fitting {name}: {e}")
            continue
    
    return best_model, best_fit, best_name, best_score, r2


def plot_surprisals(
        words:List[str], surprisals_df, show_error_interval=False, neg_samples=False, 
        first_step=True, fit_line=False, fit_curve=False, convergence=False, 
        print_slope=False, legend=True, return_outputs=False, save_as=None):
    """ 
    If first_step is set to False, neither the correlations nor the linear model will consider 
    the first step, but the first step will still be shown on the plot.
    """
    num_words = len(words)
    cols = min(num_words, 3)
    rows = math.ceil(num_words / cols)

    plt.style.use('ggplot')
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axs = np.atleast_2d(axs)

    correlations = {}
    all_metrics = {}

    for i, word in enumerate(words):   
        word_data = surprisals_df[surprisals_df['Token'] == word].reset_index(drop=True)
        if word_data.empty:
            print(f'No data found for the word "{word}"')
            continue

        ax = axs[i//cols, i%cols]

        metrics = {}
        
        x = word_data['Steps'].values
        y_pos = word_data['MeanSurprisal'].values
        
        # Plot positive surprisals
        if show_error_interval:
            try:
                sns.lineplot(data=word_data, x='Steps', y='Surprisal', marker='o', color='darkseagreen', 
                             errorbar=('ci', 100), label='Positive Samples' if i == num_words - 1 else None, 
                             ax=ax, legend=(i == num_words - 1))
            except Exception as e:
                print(e)
                ax.plot(x, y_pos, marker='o', color='darkseagreen', label='Positive Samples')
        else:
            ax.plot(x, y_pos, marker='o', color='darkseagreen', label='Positive Samples')

        if convergence:
            lowest_surprisal = min(y_pos)
            lowest_step = x[np.argmin(y_pos)]
            ax.axhline(lowest_surprisal, color='#043927', linestyle=':')
            ax.axvline(lowest_step, color='#043927', linestyle=':')

        if fit_line:
            # Fit linear model for positive samples
            X_flat, y_pred_pos, metrics['positive'] = fit_linear(x.reshape(-1, 1), y_pos, first_step=first_step)
            ax.plot(X_flat, y_pred_pos, linestyle='--', color='#043927', label='Fitted Line (+)')

            if print_slope:
                alpha = metrics['positive']['alpha']
                ax.text(x[-1], y_pred_pos[-1] + 1 if alpha > 0 else y_pred_pos[-1] + 3, 
                        f"α⁺ = {alpha:.2e}", color='#043927', fontsize=8, ha='right')

        if fit_curve:
            # Fit and plot the best curve for positive samples
            best_model, best_fit, best_name, best_score, r2 = select_best_model(x, y_pos)
            if best_model is not None:
                ax.plot(x, best_fit, label=f'Positive Best Fit', color='#043927')
                metrics['positive'] = {'model': best_name, 'R2': r2, 'MSE': best_score}
            else:
                print(f"Could not fit any model for the positive samples of word '{word}'")

        if neg_samples:
            y_neg = word_data['MeanNegSurprisal'].values
            
            # Plot negative surprisals
            if show_error_interval:
                try:
                    sns.lineplot(data=word_data, x='Steps', y='NegSurprisal', marker='o', color='indianred', 
                                 errorbar=('ci', 100), label='Negative Samples'if i == num_words - 1 else None, 
                                 ax=ax, legend=(i == num_words - 1))   
                except Exception as e:
                    print(e)         
                    ax.plot(word_data['Steps'], word_data['MeanNegSurprisal'], marker='o', color='indianred', label='Negative Samples')
            else:
                ax.plot(word_data['Steps'], word_data['MeanNegSurprisal'], marker='o', color='indianred', label='Negative Samples')
            
            if convergence:
                highest_antisurprisal = max(y_neg)
                highest_step = x[np.argmax(y_neg)]
                ax.axhline(highest_antisurprisal, color='#8D021F', linestyle=':')
                ax.axvline(highest_step, color='#8D021F', linestyle=':')
            
            if fit_line:
                # Fit linear model for negative samples
                X_flat, y_pred_neg, metrics['negative'] = fit_linear(x.reshape(-1, 1), y_neg, first_step=first_step)
                ax.plot(X_flat, y_pred_neg, linestyle='--', color='#8D021F', label='Fitted Line (-)')
                
                if print_slope:
                    alpha = metrics['negative']['alpha']
                    ax.text(x[-1], y_pred_neg[-1] - 2 if alpha < 0 else y_pred_neg[-1] - 4, 
                            f"α⁻ = {alpha:.2e}", color='#8D021F', fontsize=8, ha='right')
            
            if fit_curve:
                # Fit and plot the best curve for positive samples
                best_model, best_fit, best_name, best_score, r2 = select_best_model(x, y_neg)
                if best_model is not None:
                    ax.plot(x, best_fit, label=f'Negative Best Fit', color='#8D021F')
                    metrics['negative'] = {'model': best_name, 'R2': r2, 'MSE': best_score}
                else:
                    print(f"Could not fit any model for the negative samples of word '{word}'")

            # Calculate correlation
            if not first_step:
                word_data = word_data[word_data['Steps'] != 0]                
            corr, _ = pearsonr(word_data['MeanSurprisal'], word_data['MeanNegSurprisal'])
            correlations[word] = corr
            ax.set_title(f'"{word}"', pad=18)
            ax.text(0.5, 1.02, f'Pos-Neg Correlation: {corr:.2f}', fontsize=10, ha='center', transform=ax.transAxes)

        else:
            ax.set_title(f'"{word}"')
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Mean surprisal')

        all_metrics[word] = metrics
    
    # Remove empty subplots
    for j in range(i+1, rows*cols):
        fig.delaxes(axs.flatten()[j])
    
    plt.tight_layout()

    # Legend
    last_ax = axs.flatten()[i]
    handles, labels = last_ax.get_legend_handles_labels()
    last_ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, (i % cols) / cols + 0.5 / cols), title="Legend")

    if not legend:
        for ax in axs.flatten():
            if ax.get_legend():
                ax.get_legend().remove()

    if save_as:
        plt.savefig(save_as, format='pdf', bbox_inches='tight')
        print(f"Figure saved to {save_as}")   
    
    plt.show()

    if return_outputs:
        return correlations, all_metrics    
    

def plot_all_in_one(words:List[str], surprisals_df):
    plt.style.use('ggplot')
    plt.figure(figsize=(4, 3.5))

    max_x = 0

    for i, word in enumerate(words):   
        word_data = surprisals_df[surprisals_df['Token'] == word]
        if word_data.empty:
            print(f'No data found for the word "{word}"')
            continue

        line, = plt.plot(word_data['Steps'], word_data['MeanSurprisal'], marker='o', alpha=0.7)

        # annotate the end of the line
        x = word_data['Steps'].iloc[-1]
        y = word_data['MeanSurprisal'].iloc[-1]
        plt.annotate(word, (x, y), textcoords='offset points', xytext=(+10,+0), color=line.get_color())

        max_x = max(max_x, x)

    xlim = plt.gca().get_xlim()
    plt.gca().set_xlim(xlim[0], max_x * 1.3)  # increase the maximum x value by 30%

    # plt.title('All Words')
    plt.xlabel('Steps')
    plt.ylabel('Mean surprisal')
    # plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def get_avg_df(dfs: List[pd.DataFrame], column: str):
    avg_dfs = []
    for df in dfs:
        avg = (df.groupby('Steps')
                 .agg({column: 'mean'})
                 .reset_index()
                 .assign(Diffs=lambda x: x[column].diff().fillna(0)))
        avg_dfs.append(avg)
    return avg_dfs


def plot_avg_pos_neg(positives, negatives, save_as=None):
    plt.style.use('ggplot')
    plt.figure(figsize=(3, 3))

    pos_labels = ['Full Corpus (surprisal)', 'Frequent Words (surprisal)', 'Infrequent Words (surprisal)']
    neg_labels = ['Full Corpus (antisurprisal)', 'Frequent Words (antisurprisal)', 'Infrequent Words (antisurprisal)']
    colors = ['purple', 'green', 'red']

    for i, df in enumerate(positives):
        plt.plot(df.Steps, df['MeanSurprisal'], marker='o', color=colors[i] if len(positives) > 1 else 'green', label=pos_labels[i])
    
    for i, df in enumerate(negatives):
        plt.plot(df.Steps, df['MeanNegSurprisal'], marker='o', color=colors[i] if len(negatives) > 1 else 'red', alpha= 0.3, label=neg_labels[i])

    # plt.gca().invert_yaxis()
    # plt.xscale('log')
    plt.xlabel('Steps')
    plt.ylabel('Mean Surprisal')

    if save_as:
        plt.savefig(save_as, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {save_as}")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()