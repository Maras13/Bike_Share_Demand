import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV file into a pandas DataFrame."""
    df = pd.read_csv(filepath,  parse_dates=True)
    return df

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

def correlation(df, drop_columns=None, save_path="correlation.png", title="Correlation Heatmap", cmap="coolwarm", annot=True):
    
    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')

    corr_matrix = df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(
        corr_matrix, 
        annot=annot, 
        mask=mask, 
        cmap=cmap, 
        cbar=True, 
        vmin=-1, 
        vmax=1, 
        linewidths=0.7,
        linecolor='white',
        annot_kws={"size": 10},
    )

    # Title and formatting
    heatmap.set_title(title, fontdict={'fontsize': 18, 'fontweight': 'bold'}, pad=20)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return corr_matrix

def plot_num_feat(df, columns=None, fig_size=(15, 6), colors=None, save_path="num_features.png"):
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    else:
        columns = [col for col in columns if col in df.columns]

    if not columns:
        raise ValueError("No numeric columns to plot.")

    # Set up color palette
    colors = colors or sns.color_palette("Set2", len(columns))

    # Create subplots
    num_plots = len(columns)
    fig, axes = plt.subplots(1, num_plots, figsize=fig_size, constrained_layout=True)

    # Handle single column case
    if num_plots == 1:
        axes = [axes]

    # Plot each feature
    for ax, column, color in zip(axes, columns, colors):
        sns.histplot(data=df, x=column, ax=ax, color=color, kde=True, bins=30)
        ax.set_title(f"Distribution: {column}", fontsize=14, pad=10)
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



def plot_rolling_median(df, window_size=100, column='count', save_path="rolling.png"):
   
    rolling_series = df[column].rolling(window=window_size).median()
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_series, label=f"Rolling Median (Window={window_size})", color="dodgerblue")

    # Enhance the plot
    plt.title(f'Rolling Median of {column}', fontsize=16, pad=15)
    plt.xlabel("Index", fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, edgecolor="gray")
    plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
    sns.despine()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



def plotting(df: pd.DataFrame, column='datetime_year', save_path="total_yearly.png", legend_loc='upper left', alpha=0.9):
  
    years = df[column].unique()
    plt.figure(figsize=(12, 8))  # Set figure size

    # Loop through each year to plot
    for year in sorted(years):
        # Filter the DataFrame for the specific year
        df_year = df[df[column] == year]

        # Group by month and calculate the sum of counts
        monthly_counts = df_year.groupby('datetime_month')['count'].sum()

        # Plot the data with Seaborn
        sns.lineplot(
            x=monthly_counts.index, 
            y=monthly_counts.values, 
            marker='o', 
            label=f"Year: {year}"
        )

    # Enhance plot aesthetics
    plt.title("Yearly User Count by Month", fontsize=18, pad=15)
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("Total Count", fontsize=14)
    plt.xticks(ticks=range(1, 13), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=12)
    plt.yticks(fontsize=12)

    # Customize the legend
    legend = plt.legend(
        title="Years", 
        loc=legend_loc, 
        frameon=True, 
        fancybox=True, 
        edgecolor='gray', 
        fontsize=12
    )
    legend.get_frame().set_alpha(alpha)

    # Add gridlines for better readability
    plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Clean up the plot
    sns.despine()

    # Save and show the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_yearly_user_activity(df, year_column='datetime_year', month_column='datetime_month', 
                              registered_col='registered', casual_col='casual', holiday_col='holiday', save_path="yearly_user_activity.png"):

    years = df[year_column].unique()
    
    # Loop through each year and create plots
    for year in years:
        df_year = df[df[year_column] == year]

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'hspace': 0.3})

        # Registered users
        sns.lineplot(data=df_year, x=month_column, y=registered_col, hue=holiday_col, ax=axes[0], marker='o')
        axes[0].set_title(f"Year {year}: Registered Users", fontsize=14, pad=10)
        axes[0].set_ylabel("Registered Count", fontsize=12)
        axes[0].legend(title="Holiday", frameon=True, loc="upper right", fancybox=True, fontsize=10)

        # Casual users
        sns.lineplot(data=df_year, x=month_column, y=casual_col, hue=holiday_col, ax=axes[1], marker='o', linestyle="--")
        axes[1].set_title(f"Year {year}: Casual Users", fontsize=14, pad=10)
        axes[1].set_xlabel("Month", fontsize=12)
        axes[1].set_ylabel("Casual Count", fontsize=12)
        axes[1].legend(title="Holiday", frameon=True, loc="upper right", fancybox=True, fontsize=10)

        for ax in axes:
            ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
            sns.despine()

        # Save and show the plot for the current year
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_{year}.png", dpi=300, bbox_inches='tight')  # Save with year in filename
            plt.close(fig)  # Close the figure to avoid memory issues

    # If you want to display all plots at once (after loop finishes)
    plt.show()




def plot_line_with_legend(df, x, y, hue, title, xlabel, ylabel, save_path=True, figsize=(12, 6)):
    """Create a polished line plot with a detailed legend."""
    plt.figure(figsize=figsize)

    sns.lineplot(data=df, x=x, y=y, hue=hue, marker='o', palette="Set2", linewidth=2)

    # Add title and labels
    plt.title(title, fontsize=18, pad=15)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Customize legend
    legend = plt.legend(title=hue.replace('_', ' ').title(), frameon=True, fancybox=True, edgecolor='gray', fontsize=10)
    legend.get_frame().set_alpha(0.9)

    plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
    sns.despine()

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_hue_subplots(df, x, y, hues, figsize=(15, 10), sharex=True, palette="Set2", markers="o", save_path="hue_subplots.png"):
   
    # Create subplots: one for each hue
    fig, axes = plt.subplots(len(hues), 1, figsize=figsize, sharex=sharex)
    axes = np.atleast_1d(axes)  # Ensure axes are iterable even for one subplot

    # Loop through each hue and plot on the corresponding axis
    for i, hue in enumerate(hues):
        sns.pointplot(
            x=x, 
            y=y, 
            data=df, 
            hue=hue, 
            ax=axes[i], 
            palette=palette, 
            markers=markers, 
            linestyles="-"
        )
        # Add title and improve readability
        axes[i].set_title(f"{hue.capitalize()} vs {y}", fontsize=14, pad=10)
        axes[i].set_xlabel(x.capitalize(), fontsize=12)
        axes[i].set_ylabel(y.capitalize(), fontsize=12)

        # Improve legend styling
        legend = axes[i].legend(
            title=hue.replace("_", " ").capitalize(), 
            frameon=True, 
            fancybox=True, 
            edgecolor='gray', 
            fontsize=10, 
            title_fontsize=11
        )
        legend.get_frame().set_alpha(0.8)

    # Adjust overall layout
    sns.despine()
    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plots
    plt.show()

