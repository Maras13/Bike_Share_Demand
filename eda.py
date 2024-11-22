import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import correlation, plot_num_feat, extract_day_time_fe, plot_rolling_median, plotting, plot_yearly_user_activity, load_data
# Call the functions from utils.py



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
plot_rolling_median(df, window_size=1000, column='count', legend_loc='lower right')

# Plot yearly user activity
plot_yearly_user_activity(df, year_column='datetime_year', month_column='datetime_month', 
                           registered_col='registered', casual_col='casual', holiday_col='holiday')

# Plot user count by year
plotting(df, column='datetime_year', legend_loc='lower right')
