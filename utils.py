import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def correlation(df: pd.DataFrame, drop_columns=None, save_path="correlation.png", title="Correlation Heatmap", cmap="coolwarm", annot=True):
    """Generate and save a correlation heatmap."""
    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')

    corr_matrix = df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        corr_matrix, 
        annot=annot, 
        mask=mask, 
        cmap=cmap, 
        cbar=False, 
        vmin=-1, 
        vmax=1, 
        linewidths=0.5
    )

    # Set the title
    heatmap.set_title(title, fontdict={'fontsize': 14}, pad=12)

    # Save the plot
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

    return corr_matrix


def plot_num_feat(df: pd.DataFrame, columns=None, fig_size=(12, 5), colors=None, save_plots=True):
    """Plot numeric feature distributions."""
    if columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns
        columns = [col for col in numeric_columns if col != 'count']
    else:
        columns = [col for col in columns if col in df.columns]

    if 'count' not in df.columns:
        raise ValueError("'count' column is required in the DataFrame.")

    # Set up colors
    if colors is None:
        colors = ['r', 'g', 'b', 'y', 'c']  # Default color cycle

    # Create subplots
    num_plots = len(columns)
    fig, axes = plt.subplots(ncols=num_plots, figsize=fig_size, constrained_layout=True)

    # Handle case when there's only one column to plot
    if num_plots == 1:
        axes = [axes]

    # Plot histograms
    for ax, column, color in zip(axes, columns, colors):
        sns.histplot(x=column, y="count", data=df, ax=ax, color=color)
        ax.set_title(f'{column} vs Count') 

    if save_plots:
        # Save each plot individually
        for col in columns:
            plt.savefig(f"{col}_vs_count.png", dpi=300)
        plt.show()
        plt.close(fig)

        
def extract_day_time_fe(df: pd.DataFrame, datetime: str) -> pd.DataFrame:
    """Extract day and time-based features from datetime column."""
    df["datetime"] = pd.to_datetime(df[datetime])

    df[f"{datetime}_year"] = df[datetime].dt.year
    df[f"{datetime}_hour"] = df[datetime].dt.hour
    df[f"{datetime}_month"] = df[datetime].dt.month
    df[f"{datetime}_month_name"] = df[datetime].dt.month_name()
    df[f"{datetime}_day"] = df[datetime].dt.day
    df[f"{datetime}_day_name"] = df[datetime].dt.day_name()
    df.set_index(["datetime"], inplace=True)

    return df


def plot_rolling_median(df: pd.DataFrame, window_size=1000, column='count', save_path="rolling.png", legend_loc='upper left', legend_fancybox=True, alpha=0.9):
    """Plot rolling median for a given column."""
    df.rolling(window_size)[column].median().plot(y=column)

    # Customizing the legend
    legend = plt.legend(frameon=True, loc=legend_loc, fancybox=legend_fancybox, edgecolor='gray')
    legend.get_frame().set_alpha(alpha)

    # Clean up the plot by removing the top and right spines
    sns.despine()

    plt.title(f'Rolling Median ({window_size}-window) of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)

    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def plotting(df: pd.DataFrame, column='datetime_year', save_path="total_yearly.png", legend_loc='upper left', legend_fancybox=True, alpha=0.9):
    """Plot the yearly user count."""
    years = df[column].unique()
    for year in years:
        # Filter the DataFrame for the specific year
        df_year = df[df[column] == year]

        # Group by month, count the occurrences
        df_plot = df_year.groupby('datetime_month')['count'].size()

        # Plot the data
        df_plot.plot(kind='line', marker='o', title=f"Year: {year}")
        plt.xlabel("Month")
        plt.ylabel("Total Count")

        legend = plt.legend(frameon=True, loc=legend_loc, fancybox=legend_fancybox, edgecolor='gray')
        legend.get_frame().set_alpha(alpha)
        sns.despine()

        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()


def plot_yearly_user_activity(df: pd.DataFrame, year_column: str = 'datetime_year', month_column: str = 'datetime_month', registered_col: str = 'registered', casual_col: str = 'casual', holiday_col: str = 'holiday'):
    """Plot yearly user activity by month."""
    years = df[year_column].unique()
    for year in years:
        df_year = df[df[year_column] == year]

        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        sns.lineplot(x=month_column, y=registered_col, data=df_year, hue=holiday_col, ax=axes[0], marker='o', label="Registered", errorbar=None)
        axes[0].set_title(f"Year: {year} - Registered Users")
        axes[0].set_ylabel("Registered Count")
        axes[0].legend(frameon=True, title="Holiday", loc="upper left", labels=["No Holiday", "Holiday"], fancybox=True, edgecolor='gray')

        sns.lineplot(x=month_column, y=casual_col, data=df_year, hue=holiday_col, ax=axes[1], linestyle='--', label="Casual", errorbar=None)
        axes[1].set_title(f"Year: {year} - Casual Users")
        axes[1].set_xlabel("Month", fontsize=12)
        axes[1].set_ylabel("Casual Count")
        axes[1].legend(frameon=True, title="Holiday", loc="upper left", labels=["No Holiday", "Holiday"], fancybox=True, edgecolor='gray')

        sns.despine()
        plt.tight_layout()
        plt.show()