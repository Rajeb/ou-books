import seaborn as sns
import matplotlib.pyplot as plt

# List of weather variables to plot
weather_columns = weather_df.columns.drop('date')

# Plot histograms and KDEs for each weather variable
plt.figure(figsize=(15, 10))
for i, col in enumerate(weather_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(weather_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()




weather_df['date'] = pd.to_datetime(weather_df['date'])

# Extract relevant time components (year, month, season)
weather_df['year'] = weather_df['date'].dt.year
weather_df['month'] = weather_df['date'].dt.month
weather_df['day_of_year'] = weather_df['date'].dt.dayofyear



plt.figure(figsize=(15, 8))
for col in weather_df.columns.drop(['date', 'year', 'month', 'season', 'day_of_year']):
    plt.plot(daily_weather['day_of_year'], daily_weather[col], label=col)
plt.title('Daily Changes of Weather Variables Over the Years')
plt.xlabel('Day of Year')
plt.ylabel('Average Weather Value')
plt.legend(loc='upper right')
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Assuming these columns represent the aggregated time scales:
# - 'day_of_year' for daily data
# - 'month' for monthly data
# - 'season' for seasonal data

# List of weather variables (excluding the date, year, month, day_of_year, and season columns)
weather_columns = weather_df.columns.drop(['date', 'year', 'month', 'season', 'day_of_year'])

# Number of weather variables
n_vars = len(weather_columns)

### Subplots for Daily, Monthly, and Seasonal Data ###

fig, axes = plt.subplots(n_vars, 3, figsize=(18, 4 * n_vars))  # 3 subplots per row (daily, monthly, seasonal)

# Plotting for each weather variable
for i, col in enumerate(weather_columns):
    
    # Daily Changes Plot
    sns.lineplot(x='day_of_year', y=col, hue='year', data=weather_df, ax=axes[i, 0], palette='coolwarm')
    axes[i, 0].set_title(f'Daily Changes of {col} by Year')
    axes[i, 0].set_xlabel('Day of Year')
    axes[i, 0].set_ylabel(f'{col} Value')
    
    # Monthly Changes Plot
    sns.lineplot(x='month', y=col, hue='year', data=weather_df, ax=axes[i, 1], palette='coolwarm')
    axes[i, 1].set_title(f'Monthly Changes of {col} by Year')
    axes[i, 1].set_xlabel('Month')
    axes[i, 1].set_ylabel(f'{col} Value')
    
    # Seasonal Changes Plot
    sns.lineplot(x='season', y=col, hue='year', data=weather_df, ax=axes[i, 2], palette='coolwarm')
    axes[i, 2].set_title(f'Seasonal Changes of {col} by Year')
    axes[i, 2].set_xlabel('Season')
    axes[i, 2].set_ylabel(f'{col} Value')

# Adjust layout for better readability
plt.tight_layout()
plt.show()




import pandas as pd
import matplotlib.pyplot as plt

# Ensure 'date' is in datetime format
weather_df['date'] = pd.to_datetime(weather_df['date'])

# Extract year from the date
weather_df['year'] = weather_df['date'].dt.year

# Count the number of events per year
events_per_year = weather_df.groupby('year').size().reset_index(name='total_events')

# Plot the total event count for each year
plt.figure(figsize=(10, 6))
plt.bar(events_per_year['year'], events_per_year['total_events'], color='coral')
plt.title('Total Event Count by Year')
plt.xlabel('Year')
plt.ylabel('Total Event Count')
plt.xticks(events_per_year['year'])  # Ensure all years are displayed on x-axis
plt.grid(axis='y')
plt.tight_layout()
plt.show()
