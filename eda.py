import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import correlation, plot_num_feat, extract_day_time_fe, plot_rolling_median, plotting, plot_yearly_user_activity, load_data, plot_hue_subplots, plot_line_with_legend
# Call the functions from utils.py

plt.style.use(['ggplot', 'seaborn-darkgrid'])


plt.rcParams['axes.facecolor'] = '#f0f0f0'
plt.rcParams['grid.color'] = 'grey'
plt.rcParams['grid.linestyle'] = '--'




# Load dataset
df = load_data(filepath = 'data/train.csv')



# Correlation heatmap
correlation_matrix = correlation(
    df, 
    drop_columns="datetime", 
    save_path="custom_correlation.png", 
    title="Custom Correlation Heatmap", 
    cmap="viridis"
)

# Plot numeric features
plot_num_feat(df, columns=["temp", "windspeed", "humidity"])

# Extract day and time-based features
df = extract_day_time_fe(df, "datetime")

# Plot rolling median
plot_rolling_median(df, window_size=100, column='count', save_path="rolling.png")


# Plot yearly user activity
plot_yearly_user_activity(df, year_column='datetime_year', month_column='datetime_month', 
                           registered_col='registered', casual_col='casual', holiday_col='holiday')

# Plot user count by year
plotting(df, column='datetime_year', legend_loc='lower right')


hues = ['workingday', 'weather', 'season']
plot_hue_subplots(df=df, x='datetime_hour', y='count', hues=hues, figsize=(15, 10))

plot_line_with_legend(
    df=df,
    x='datetime_hour',
    y='count',
    hue='datetime_day_name',
    title="Hourly Count of Users by Day of the Week",
    xlabel="Hour of the Day",
    ylabel="User Count",
    save_path="hourly_count_by_day.png"
)

