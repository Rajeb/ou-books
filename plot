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
