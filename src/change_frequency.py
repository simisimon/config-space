import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_change_frequency_options(df_options: pd.DataFrame):
    
    pivot_df = df_options.pivot_table(index="Option", columns="Concept", values="Internal Changes", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, len(pivot_df) * 0.3))
    sns.heatmap(pivot_df, annot=True, cmap="Reds", cbar=True, linewidths=.5, ax=ax)
    return fig