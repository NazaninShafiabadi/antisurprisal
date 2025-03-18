""" Contains the functions needed for analyzing learning curves. """

import math
from itertools import cycle
from typing import List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import h5py

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_token_data(filepath, token):
    """
    Load all step data for a given token from the HDF5 file.
    
    Parameters:
        filepath (str): Path to the HDF5 file.
        token (str): Token to load data for.
    
    Returns:
        dict: Dictionary containing step data.
    """
    token_data = {}

    with h5py.File(filepath, 'r') as f:
        token_group = f.get(f"{token}")
        if token_group is None:
            print(f"No data found for token '{token}'")
            return token_data

        attentions = defaultdict(list)
        hidden_states = defaultdict(list)
        
        for step in token_group:
            step_group = token_group[step]
            attentions[step].append(np.array(step_group["attentions"][:]))
            hidden_states[step].append(np.array(step_group["hidden_states"][:]))

    return attentions, hidden_states


####################################################################################################

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
        words:List[str], surprisals_df, show_error_interval=False, plot_antisurprisal=True, 
        first_step=True, fit_line=False, fit_curve=False, convergence=False, 
        print_slope=False, legend=True, return_outputs=False, save_as:str=None):
    """ 
    Plots surprisal curves for a list of words based on the given dataframe. 
    Optionally, fits linear models, best curves, and marks convergence points.

    Parameters:
    - words (List[str]): List of words to plot.
    - surprisals_df (pd.DataFrame): DataFrame containing surprisal values with columns ['Token', 'Steps', 'MeanSurprisal', 'MeanAntisurprisal'].
    - show_error_interval (bool): If True, displays confidence intervals around the curves.
    - plot_antisurprisal (bool): If True, plots antisurprisal curve in addition to surprisal.
    - first_step (bool): If False, excludes the first step from correlation and linear fitting calculations but still shows it on the plot.
    - fit_line (bool): If True, fits a linear regression line to the surprisal and/or antisurprisal curves.
    - fit_curve (bool): If True, fits the best nonlinear curve to the surprisal and/or antisurprisal curves.
    - convergence (bool): If True, marks the step where surprisal reaches its lowest value and antisurprisal its highest.
    - print_slope (bool): If True, displays the slope (α) of the fitted linear model.
    - legend (bool): If True, includes a legend in the plot.
    - return_outputs (bool): If True, returns correlation values and fitting metrics.
    - save_as (str, optional): If provided, saves the plot as a PDF file with the given filename.

    Returns:
    - dict: If return_outputs=True, returns a dictionary of correlation values and fitting metrics for each word.
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
        
        x = word_data['Step'].values
        y_pos = word_data['MeanSurprisal'].values
        
        # Plot positive surprisals
        if show_error_interval:
            pos_err = word_data['StdSurprisal'].values
            ax.plot(x, y_pos, marker='o', color='darkseagreen', label='Positive Samples' if i == num_words - 1 else None)
            ax.fill_between(x, y_pos - pos_err, y_pos + pos_err, alpha=0.2, color='darkseagreen')
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

        if plot_antisurprisal:
            y_neg = word_data['MeanAntisurprisal'].values
            
            # Plot negative surprisals
            if show_error_interval:
                neg_err = word_data['StdAntisurprisal'].values
                ax.plot(x, y_neg, marker='o', color='indianred', label='Negative Samples' if i == num_words - 1 else None)
                ax.fill_between(x, y_neg - neg_err, y_neg + neg_err, alpha=0.2, color='indianred')   
            else:
                ax.plot(x, y_neg, marker='o', color='indianred', label='Negative Samples')
            
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
                word_data = word_data[word_data['Step'] != 0]                
            corr, _ = pearsonr(word_data['MeanSurprisal'], word_data['MeanAntisurprisal'])
            correlations[word] = corr
            ax.set_title(f'"{word}"', pad=18)
            ax.text(0.5, 1.02, f'Pos-Neg Correlation: {corr:.2f}', fontsize=10, ha='center', transform=ax.transAxes)

        else:
            ax.set_title(f'"{word}"')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg. surprisal')

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

####################################################################################################

def compare(words: List[str], surprisals_df, compute_corr: bool = False, plot_learning_curves: bool = False, 
            show_error_interval: bool = False, plot_differences: bool = False, save_as: str = None):
    """
    Plots surprisal and antisurprisal curves for given words, with options for correlation 
    computation and pairwise difference visualization.

    Parameters:
    -----------
    words : List[str]
        A list of words to plot surprisal and antisurprisal curves for.
    
    surprisals_df : DataFrame
        A pandas DataFrame containing columns:
        - 'Token': the word,
        - 'Step': the x-axis values,
        - 'MeanSurprisal': surprisal values,
        - 'MeanAntisurprisal': antisurprisal values.
    
    compute_corr : bool, optional
        If True, computes, displays and returns the mean Pearson correlation between surprisal and 
        antisurprisal curves. Default is False.

    plot_learning_curves : bool, optional
        If True, plots the surprisal and antisurprisal curves. Default is False.

    show_error_interval : bool, optional
        If True, shows the error interval around the mean surprisal and antisurprisal values. Default is False.
    
    plot_differences : bool, optional
        If True, plots pairwise differences between surprisal and antisurprisal curves. Default is False.
    
    save_as : str, optional
        If provided, saves the figure as a PDF with the specified filename. Default is None.
    """
    plt.style.use('ggplot')
    
    if plot_learning_curves and plot_differences:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.5))
    elif plot_learning_curves:
        fig, ax1 = plt.subplots(figsize=(5.3, 3.5))
    elif plot_differences:
        fig, ax2 = plt.subplots(figsize=(5.3, 3.5))
    
    num_words = len(words)
    surprisal_colors = cm.Greens(np.linspace(0.8, 1, num_words))  # Shades of green
    antisurprisal_colors = cm.Reds(np.linspace(0.8, 1, num_words))  # Shades of red

    markers = cycle(['o', 'x', '*', 'O', 'X', '<', '>', '~'])

    # Store surprisal and antisurprisal values for correlation computation
    surprisal_curves = []
    antisurprisal_curves = []

    for i, word in enumerate(words):   
        marker = next(markers)

        word_data = surprisals_df[surprisals_df['Token'] == word]
        if word_data.empty:
            print(f'No data found for the word "{word}"')
            continue

        if compute_corr:
            # Store data for correlation computation
            surprisal_curves.append(word_data['MeanSurprisal'].values)
            antisurprisal_curves.append(word_data['MeanAntisurprisal'].values)

        if plot_learning_curves:
            x = word_data['Step']
            y_pos = word_data['MeanSurprisal']
            y_neg = word_data['MeanAntisurprisal']
            pos_err = word_data['StdSurprisal']
            neg_err = word_data['StdAntisurprisal']

            # Plot surprisal and antisurprisal curves
            ax1.plot(x, y_pos, marker=marker, alpha=0.7, color=surprisal_colors[i], label=f'{word} (S)')
            ax1.plot(x, y_neg, marker=marker, alpha=0.7, color=antisurprisal_colors[i], label=f'{word} (AS)')
            if show_error_interval:
                ax1.fill_between(x, y_pos - pos_err, y_pos + pos_err, color=surprisal_colors[i], alpha=0.2)
                ax1.fill_between(x, y_neg - neg_err, y_neg + neg_err, color=antisurprisal_colors[i], alpha=0.2)
            legend = ax1.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
            legend_bg_color = legend.get_frame().get_facecolor()
            legend_edge_color = legend.get_frame().get_edgecolor()
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Avg. Surprisal / Antisurprisal')

    # Compute correlation and/or difference curves (if enabled)
    if (plot_differences or compute_corr) and len(surprisal_curves) > 1:
        surprisal_matrix = np.array(surprisal_curves)
        antisurprisal_matrix = np.array(antisurprisal_curves)

        steps = surprisals_df['Step'].unique()

        surprisal_corrs = []
        antisurprisal_corrs = []
        
        for i in range(len(surprisal_curves)):
            for j in range(i + 1, len(surprisal_curves)):
                
                if plot_differences:
                    # Plot pairwise differences between surprisal & antisurprisal curves
                    surprisal_diff = np.abs(surprisal_matrix[i] - surprisal_matrix[j])
                    antisurprisal_diff = np.abs(antisurprisal_matrix[i] - antisurprisal_matrix[j])
                    ax2.plot(steps, surprisal_diff, linestyle="-", color="green", alpha=0.6, label=f'Surprisal diff')
                    ax2.plot(steps, antisurprisal_diff, linestyle="-", color="red", alpha=0.6, label=f'Antisurprisal diff')
                    legend_text = f"Words:\n{words[i]}, {words[j]}"
                    handles, labels = ax2.get_legend_handles_labels()
                    handles.insert(0, plt.Line2D([0], [0], color='none', label=legend_text))
                    ax2.set_xlabel('Step')
                    ax2.set_ylabel('Differences')
                    ax2.legend(handles=handles, fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left') 

                if compute_corr:
                    # Correlation between surprisal curves and antisurprisal curves
                    surprisal_corrs.append(pearsonr(surprisal_matrix[i], surprisal_matrix[j])[0])
                    antisurprisal_corrs.append(pearsonr(antisurprisal_matrix[i], antisurprisal_matrix[j])[0])

        if compute_corr:
            mean_surprisal_corr = np.mean(surprisal_corrs)
            mean_antisurprisal_corr = np.mean(antisurprisal_corrs)
            if plot_learning_curves:
                ax1.text(1.09, 0.5, f"Corr (S) = {mean_surprisal_corr:.2f}\nCorr (AS) = {mean_antisurprisal_corr:.2f}", 
                         transform=ax1.transAxes, fontsize=9, 
                         bbox=dict(facecolor=legend_bg_color, edgecolor=legend_edge_color, boxstyle='round,pad=0.5'))

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, format='pdf', bbox_inches='tight')
        print(f"Figure saved to {save_as}") 

    if plot_learning_curves or plot_differences:
        plt.show()

    if compute_corr:
        return mean_surprisal_corr, mean_antisurprisal_corr

####################################################################################################

def get_avg_df(dfs: List[pd.DataFrame], column: str):
    avg_dfs = []
    for df in dfs:
        avg = (df.groupby('Steps')
                 .agg({column: 'mean'})
                 .reset_index()
                 .assign(Diffs=lambda x: x[column].diff().fillna(0)))
        avg_dfs.append(avg)
    return avg_dfs

####################################################################################################

def plot_avg_pos_neg(positives, negatives, save_as=None):
    plt.style.use('ggplot')
    plt.figure(figsize=(3, 3))

    pos_labels = ['Full Corpus (surprisal)', 'Frequent Words (surprisal)', 'Infrequent Words (surprisal)']
    neg_labels = ['Full Corpus (antisurprisal)', 'Frequent Words (antisurprisal)', 'Infrequent Words (antisurprisal)']
    colors = ['purple', 'green', 'red']

    for i, df in enumerate(positives):
        plt.plot(df.Steps, df['MeanSurprisal'], marker='o', color=colors[i] if len(positives) > 1 else 'green', label=pos_labels[i])
    
    for i, df in enumerate(negatives):
        plt.plot(df.Steps, df['MeanAntisurprisal'], marker='o', color=colors[i] if len(negatives) > 1 else 'red', alpha= 0.3, label=neg_labels[i])

    # plt.gca().invert_yaxis()
    # plt.xscale('log')
    plt.xlabel('Steps')
    plt.ylabel('Mean Surprisal')

    if save_as:
        plt.savefig(save_as, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {save_as}")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()