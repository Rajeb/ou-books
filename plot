________________________________ yearly boxplot__________________
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Generating new data to cover 2015-2024
data = {
    'outage_date': pd.date_range(start='2015-01-01', periods=1000, freq='M').tolist(),
    'min_distance': np.random.rand(1000).tolist()
}

# Create a new dataframe and add year and month columns
df_full = pd.DataFrame(data)
df_full['outage_date'] = pd.to_datetime(df_full['outage_date'])
df_full['year'] = df_full['outage_date'].dt.year
df_full['month'] = df_full['outage_date'].dt.month

# Data for 2022 and 2024 separately
df_2022_2024 = df_full[df_full['year'].isin([2022, 2024])]

# Step 1: Overall Plot (2015-2024)
fig = px.box(df_full, x='month', y='min_distance', title='Overall Box Plot of Minimum Distance (2015-2024)')
# Calculate and overlay the mean distance
mean_distances = df_full.groupby('month')['min_distance'].mean().reset_index()
fig.add_trace(go.Scatter(x=mean_distances['month'], y=mean_distances['min_distance'], mode='markers+lines', 
                         name='Mean Distance', marker=dict(color='red', size=8)))

fig.update_layout(xaxis_title='Month', yaxis_title='Minimum Distance')
fig.show()

# Step 2: Plot for 2022 and 2024
fig2 = px.box(df_2022_2024, x='month', y='min_distance', color='year', title='Box Plot of Minimum Distance (2022 and 2024)')
# Calculate and overlay the mean distance for each year
mean_distances_2022_2024 = df_2022_2024.groupby(['month', 'year'])['min_distance'].mean().reset_index()

# Overlay mean distance for 2022 and 2024
for year in [2022, 2024]:
    mean_data = mean_distances_2022_2024[mean_distances_2022_2024['year'] == year]
    fig2.add_trace(go.Scatter(x=mean_data['month'], y=mean_data['min_distance'], mode='markers+lines', 
                              name=f'Mean Distance {year}', marker=dict(size=8)))

fig2.update_layout(xaxis_title='Month', yaxis_title='Minimum Distance')
fig2.show()
___________________________________________________














import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

def plot_correlation_multiple(df_list, location_names, x_col, y_col):
    fig = px.scatter(title="Correlation Between Minimum Distance and Event Occurrence")

    # Iterate through each DataFrame and location name
    for df, location_name in zip(df_list, location_names):
        # Prepare data for regression
        X = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values

        # Perform linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Predict values for regression line
        y_pred = model.predict(X)

        # Calculate correlation
        correlation = df[x_col].corr(df[y_col])

        # Scatter plot for this location
        scatter = px.scatter(df, x=x_col, y=y_col, labels={x_col: "Minimum Distance", y_col: "Event Occurrence"},
                             title=f"Correlation: {correlation:.2f} ({location_name})")
        
        # Add traces to the main figure
        for trace in scatter.data:
            trace.name = location_name  # Add location name as the trace name
            trace.showlegend = True  # Show legend for different locations
            fig.add_trace(trace)
        
        # Add regression line for this location
        fig.add_traces(px.line(x=df[x_col], y=y_pred, name=f'Regression Line ({location_name})').data)

    # Update plot layout
    fig.update_layout(
        xaxis_title="Minimum Distance",
        yaxis_title="Event Occurrence",
        showlegend=True
    )

    fig.show()

# Example usage with 3 different DataFrames for 3 locations
df1 = pd.DataFrame({'minimum_distance': [1.2, 2.5, 3.0, 0.8, 1.9], 'event_occurrence': [5, 10, 15, 6, 8]})
df2 = pd.DataFrame({'minimum_distance': [1.5, 2.0, 3.5, 0.9, 1.7], 'event_occurrence': [6, 11, 14, 5, 9]})
df3 = pd.DataFrame({'minimum_distance': [2.1, 2.8, 3.2, 1.0, 2.0], 'event_occurrence': [4, 12, 13, 7, 10]})

# List of DataFrames and location names
df_list = [df1, df2, df3]
location_names = ['Location 1', 'Location 2', 'Location 3']

# Plot the correlations for all three locations
plot_correlation_multiple(df_list, location_names, x_col='minimum_distance', y_col='event_occurrence')







import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample DataFrame with 'minimum_distance' and 'event_occurrence' columns
data = {
    'minimum_distance': [1.2, 2.5, 3.0, 0.8, 1.9],
    'event_occurrence': [5, 10, 15, 6, 8]
}
df = pd.DataFrame(data)

# Prepare data for regression
X = df['minimum_distance'].values.reshape(-1, 1)
y = df['event_occurrence'].values

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Predict values for regression line
y_pred = model.predict(X)

# Calculate correlation
correlation = df['minimum_distance'].corr(df['event_occurrence'])

# Plot with Plotly
fig = px.scatter(df, x='minimum_distance', y='event_occurrence', title=f'Correlation: {correlation:.2f}',
                 labels={'minimum_distance': 'Minimum Distance', 'event_occurrence': 'Event Occurrence'})

# Add regression line
fig.add_traces(px.line(x=df['minimum_distance'], y=y_pred, name='Regression Line').data)

# Customize plot
fig.update_layout(
    xaxis_title="Minimum Distance",
    yaxis_title="Event Occurrence",
    annotations=[
        dict(
            x=max(df['minimum_distance']), y=max(y_pred),
            text=f"R² = {model.score(X, y):.2f}",
            showarrow=False,
            font=dict(size=14)
        )
    ]
)

# Show the plot
fig.show()








import pandas as pd

# Define the range for the bins
start = 1
end = 500
bin_size = 10

# Generate bins from 1 to 500 with size 10
bins = list(range(start, end + bin_size, bin_size))

# Create labels for each bin
labels = [f'{i}-{i + bin_size - 1}' for i in bins[:-1]]

# Example DataFrame
df = pd.DataFrame({
    'values': [5, 12, 23, 34, 45, 56, 67, 78, 89, 100, 150, 200, 250, 300, 400, 450, 500]
})

# Categorize 'values' into bins and assign labels
df['binned'] = pd.cut(df['values'], bins=bins, labels=labels, right=False)

# Display the DataFrame with the binned values
print(df)


import plotly.graph_objects as go
import pandas as pd

# Example DataFrames (replace these with your actual DataFrames)
df1 = pd.DataFrame({
    'min_distance_to_water': [10, 60, 110, 150, 210, 250, 270, 310, 350, 400, 420, 460, 500, 520, 540]
})

df2 = pd.DataFrame({
    'min_distance_to_water': [5, 55, 105, 155, 205, 255, 305, 355, 405, 455, 505, 515, 525]
})

# Define bins and labels
bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 500, 550]
labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-500', '>500']

# Function to process DataFrame and generate event counts
def process_df(df, label):
    df['distance_bin'] = pd.cut(df['min_distance_to_water'], bins=bins, labels=labels, right=False)
    event_counts = df['distance_bin'].value_counts().sort_index()
    return pd.DataFrame({
        'distance_bin': event_counts.index.astype(str),
        'Number of Events': event_counts.values,
        'Dataset': label
    })

# Process each DataFrame
df1_processed = process_df(df1, 'Dataset 1')
df2_processed = process_df(df2, 'Dataset 2')

# Create traces
trace1 = go.Scatter(
    x=df1_processed['distance_bin'],
    y=df1_processed['Number of Events'],
    mode='lines+markers',
    name='Dataset 1',
    yaxis='y2'  # Secondary y-axis
)

trace2 = go.Scatter(
    x=df2_processed['distance_bin'],
    y=df2_processed['Number of Events'],
    mode='lines+markers',
    name='Dataset 2',
    yaxis='y'
)

# Create the figure
fig = go.Figure()

# Add traces
fig.add_trace(trace1)
fig.add_trace(trace2)

# Update layout
fig.update_layout(
    title="Occurrence of Events Relative to Distance to Water",
    xaxis_title="Distance from Water",
    yaxis_title="Number of Events (Dataset 2)",
    yaxis2=dict(
        title="Number of Events (Dataset 1)",
        overlaying='y',
        side='right'
    ),
    showlegend=True
)

# Show the plot
fig.show()



















outage and ignition plot into one
import plotly.express as px
import pandas as pd

# Example DataFrames (replace these with your actual DataFrames)
df1 = pd.DataFrame({
    'min_distance_to_water': [10, 60, 110, 150, 210, 250, 270, 310, 350, 400, 420, 460, 500, 520, 540]
})

df2 = pd.DataFrame({
    'min_distance_to_water': [5, 55, 105, 155, 205, 255, 305, 355, 405, 455, 505, 515, 525]
})

# Define bins and labels
bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 500, 550]
labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-500', '>500']

# Function to process DataFrame and generate event counts
def process_df(df, label):
    df['distance_bin'] = pd.cut(df['min_distance_to_water'], bins=bins, labels=labels, right=False)
    event_counts = df['distance_bin'].value_counts().sort_index()
    return pd.DataFrame({
        'distance_bin': event_counts.index.astype(str),
        'Number of Events': event_counts.values,
        'Dataset': label
    })

# Process each DataFrame
df1_processed = process_df(df1, 'Dataset 1')
df2_processed = process_df(df2, 'Dataset 2')

# Combine processed DataFrames
combined_df = pd.concat([df1_processed, df2_processed])

# Plot using Plotly Express
fig = px.line(
    combined_df,
    x='distance_bin',
    y='Number of Events',
    color='Dataset',
    markers=True,
    title="Occurrence of Events Relative to Distance to Water"
)

# Update layout
fig.update_traces(mode="lines+markers")
fig.update_layout(
    xaxis_title="Distance from Water",
    yaxis_title="Number of Events",
    showlegend=True
)

# Show the plot
fig.show()













### event occurrence by distance
bins =[0, 50, 100, 150, 200, 250, 300,350, 400, 500, 550]
labels =['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', "400-450", ">450" ]
minimum_distance_outages['distance_bin'] = pd.cut(minimum_distance_outages['min_distance_to_water'], bins=bins, labels =labels, right=False)
event_counts = minimum_distance_outages['distance_bin'].value_counts().sort_index()

## plot results 

fig = px.line(event_counts,
             x=event_counts.index.astype(str),
             y=event_counts.values,
             labels={'x': "Distance from water", 'y':"Number of events"}, 
             title= "Occurrence of events relative to distance to water ")

fig.update_traces(mode ="lines+markers")
fig.show()

##################
add range of distance
import pandas as pd
import plotly.express as px

# Assuming minimum_distance_outages DataFrame and bins/labels are already defined

# Assign the distances to bins
minimum_distance_outages['distance_bin'] = pd.cut(minimum_distance_outages['min_distance_to_water'], bins=bins, labels=labels, right=False)

# Calculate event counts for each bin
event_counts = minimum_distance_outages['distance_bin'].value_counts().sort_index()

# Calculate the min and max distance for each bin
distance_range = minimum_distance_outages.groupby('distance_bin')['min_distance_to_water'].agg([min, max])

# Create text labels for the plot that include event counts and distance ranges
event_text = [f"{label}: {count} events (range: {dist_min:.1f}-{dist_max:.1f})"
              for label, count, dist_min, dist_max in zip(event_counts.index.astype(str), 
                                                         event_counts.values, 
                                                         distance_range['min'], 
                                                         distance_range['max'])]

# Plot the results with event count and distance range as text
fig = px.line(
    event_counts,
    x=event_counts.index.astype(str),
    y=event_counts.values,
    labels={'x': "Distance from water", 'y': "Number of events"},
    title="Occurrence of events relative to distance to water",
    text=event_text  # Add the event count and range as text for each point
)

# Update the trace to display markers, lines, and the text permanently
fig.update_traces(mode="lines+markers+text", textposition="top center")

# Show the plot
fig.show()

------------------------
import plotly.graph_objects as go
import pandas as pd

# Assuming minimum_distance_outages DataFrame and bins/labels are already defined

# Assign the distances to bins
minimum_distance_outages['distance_bin'] = pd.cut(minimum_distance_outages['min_distance_to_water'], bins=bins, labels=labels, right=False)

# Calculate event counts for each bin
event_counts = minimum_distance_outages['distance_bin'].value_counts().sort_index()

# Calculate the min, max, and median distance for each bin
distance_stats = minimum_distance_outages.groupby('distance_bin')['min_distance_to_water'].agg(['min', 'max', 'median'])

# Create the figure using error bars for the min and max
fig = go.Figure()

# Add trace for the median distances with error bars showing min and max
fig.add_trace(go.Scatter(
    x=event_counts.index.astype(str),  # Bin labels
    y=distance_stats['median'],  # Median distances
    error_y=dict(
        type='data',  # Data-based error bars
        symmetric=False,  # Asymmetric (different min and max)
        array=distance_stats['max'] - distance_stats['median'],  # Distance from median to max
        arrayminus=distance_stats['median'] - distance_stats['min'],  # Distance from median to min
        visible=True
    ),
    mode='markers+lines',  # Lines and markers
    marker=dict(size=10, color='blue'),
    line=dict(dash='dash'),  # Dashed lines to connect points
    name='Distance range (min-max)',
    hovertemplate='<b>Distance Bin: %{x}</b><br>Median: %{y}<br>Min: %{error_y.arrayminus:.1f}<br>Max: %{error_y.array:.1f}<extra></extra>',
))

# Customize the layout
fig.update_layout(
    title="Distance from Water with Min-Max Range for Each Bin",
    xaxis_title="Distance Bin",
    yaxis_title="Distance (min-max range)",
    showlegend=True
)

# Show the plot
fig.show()




plot minimum distance


import pandas as pd
import plotly.express as px

# Load the final CSV file with event locations and minimum distances
event_df = pd.read_csv('event_min_distances.csv')

# Plotly scatter plot of event locations with distance as color scale
fig = px.scatter_geo(
    event_df,
    lat='event_lat',
    lon='event_long',
    color='min_distance_to_water',  # Use the minimum distance as the color scale
    color_continuous_scale='Viridis',  # Choose a color scale
    size='min_distance_to_water',  # Optional: size of markers based on distance
    hover_name='min_distance_to_water',  # Show the distance on hover
    projection="natural earth",  # Use the natural earth projection for global maps
    title="Minimum Distance to Water Occurrence from Event Locations"
)

# Update layout for better visualization
fig.update_layout(
    geo=dict(
        showland=True,
        landcolor="rgb(217, 217, 217)",
        showcountries=True,
        countrycolor="rgb(204, 204, 204)",
    ),
    margin={"r":0,"t":30,"l":0,"b":0}
)

# Show the plot
fig.show()

############ use buffer

import os
import pandas as pd
from geopy.distance import great_circle
from scipy.spatial import KDTree

# Constants
MILES_TO_KM = 1.60934
BUFFER_RADIUS_MILES = 50
BUFFER_RADIUS_KM = BUFFER_RADIUS_MILES * MILES_TO_KM

# Define the directory containing CSV files for water occurrence
csv_directory = 'path_to_directory_with_csv_files'
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Event DataFrame (make sure you have lat/long for events)
df_events = pd.DataFrame({
    'event_id': [1, 2, 3],  # Example event IDs
    'latitude': [34.1, 35.2, 36.5],
    'longitude': [-118.5, -119.6, -120.7]
})

# Function to calculate distance between two lat/lon points
def calculate_distance(event_lat, event_lon, point_lat, point_lon):
    return great_circle((event_lat, event_lon), (point_lat, point_lon)).miles

# Process each CSV file
for csv_file in csv_files:
    file_path = os.path.join(csv_directory, csv_file)
    
    # Read the CSV file in chunks
    for chunk in pd.read_csv(file_path, chunksize=10000):
        # Filter out rows with values of 0 or no-data
        chunk = chunk[(chunk['value'] != 0) & (chunk['value'] != -9999)]
        
        # Loop through each event
        for event_idx, event_row in df_events.iterrows():
            event_lat, event_lon = event_row['latitude'], event_row['longitude']
            
            # Filter water points within a 50-mile buffer
            chunk['distance'] = chunk.apply(
                lambda row: calculate_distance(event_lat, event_lon, row['latitude'], row['longitude']),
                axis=1
            )
            filtered_chunk = chunk[chunk['distance'] <= BUFFER_RADIUS_MILES]

            # If there are points within the buffer, calculate the minimum distance
            if not filtered_chunk.empty:
                min_distance = filtered_chunk['distance'].min()
                print(f"Event {event_row['event_id']} minimum distance to water: {min_distance:.2f} miles")
            else:
                print(f"Event {event_row['event_id']} has no water occurrences within 50 miles.")

















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











# Create an empty DataFrame to store results
results = []

for weather_var in weather_vars:
    # Aggregate weather data to get the yearly mean
    yearly_weather = weather_df.groupby('year')[weather_var].mean().reset_index()
    
    # Merge the total event count with the yearly weather data
    combined_df = pd.merge(events_per_year, yearly_weather, on='year')
    
    # Calculate the correlation
    correlation = combined_df['event_count'].corr(combined_df[weather_var])
    
    # Append results
    results.append({'weather_var': weather_var, 'correlation': correlation})
    
    # Plot the correlation
    plt.figure(figsize=(8, 6))
    plt.scatter(combined_df[weather_var], combined_df['event_count'], color='coral', edgecolor='k')
    plt.title(f'Event Count vs. {weather_var}')
    plt.xlabel(f'{weather_var}')
    plt.ylabel('Total Event Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results)

# Display correlation results
print(results_df)

