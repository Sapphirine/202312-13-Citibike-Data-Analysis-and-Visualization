from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, count, lit, datediff, to_date, col
import seaborn as sns
import pandas as pd
import pickle
import os
import re
import matplotlib.dates as mdates
from datetime import datetime, timedelta

app = Flask(__name__)

# Initialize Spark
spark = SparkSession.builder.config("spark.driver.host", "localhost").appName("CitiBikeApp").getOrCreate()

#data loading & cleaning
def load_data():
    return spark.read.csv("citibike.csv", header=True, inferSchema=True)

def clean_data(df):
    # Filter out rides where start and end stations are the same
    cleaned_df = df.filter(col('start_station_name') != col('end_station_name'))
    return cleaned_df

df = load_data()
cleaned_df = clean_data(df)

#Busiest Stations
@app.route('/busiest_stations', methods=['POST'])
def busiest_stations():
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    num_stations = int(request.form.get('num_stations', 10))
    top_stations = get_busiest_stations(cleaned_df, start_date, end_date, num_stations)
    busiest_stations_plot = plot_busiest_stations(top_stations, num_stations, start_date, end_date)
    return render_template('index.html', busiest_stations_plot=busiest_stations_plot, active_section='busiestStations')

def get_busiest_stations(cleaned_df, start_date, end_date, num_stations):
    filtered_df = cleaned_df.filter((col('started_at') >= start_date) & (col('ended_at') <= end_date))
    start_counts = filtered_df.groupBy('start_station_name').count().withColumnRenamed('count', 'start_count').withColumnRenamed('start_station_name', 'station_name')
    end_counts = filtered_df.groupBy('end_station_name').count().withColumnRenamed('count', 'end_count').withColumnRenamed('end_station_name', 'station_name')
    total_counts = start_counts.join(end_counts, 'station_name', 'outer')
    total_counts = total_counts.fillna(0)
    total_counts = total_counts.withColumn('total_count', col('start_count') + col('end_count'))
    station_counts = total_counts.orderBy('total_count', ascending=False).limit(num_stations)
    return station_counts.toPandas()


def plot_busiest_stations(station_df, num_stations, start_date, end_date):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))
    bars = plt.bar(station_df['station_name'], station_df['total_count'], color=sns.color_palette("muted", len(station_df)))

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=10)

    plt.xlabel('Station Name', fontsize=12)
    plt.ylabel('Total Rides (Starts + Ends)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f'Top {num_stations} Busiest Stations from {start_date} to {end_date}', fontsize=14)

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    busiest_stations_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return busiest_stations_plot

#Popular Routes
@app.route('/popular_routes', methods=['POST'])
def popular_routes():
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    num_routes = int(request.form.get('num_routes', 10))
    popular_routes_df = get_popular_routes(cleaned_df, start_date, end_date, num_routes)
    popular_routes_plot = plot_popular_routes(popular_routes_df, num_routes, start_date, end_date)
    return render_template('index.html', popular_routes_plot=popular_routes_plot, active_section='popularRoutes')

def get_popular_routes(cleaned_df, start_date, end_date, num_routes):
    filtered_df = cleaned_df.filter((col('started_at') >= start_date) & (col('ended_at') <= end_date))
    route_counts = filtered_df.groupBy('start_station_name', 'end_station_name').count().orderBy('count', ascending=False).limit(num_routes)
    return route_counts.toPandas()

def plot_popular_routes(routes_df, num_routes, start_date, end_date):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))

    route_names = routes_df['start_station_name'].astype(str) + " to " + routes_df['end_station_name'].astype(str)
    bars = plt.bar(route_names, routes_df['count'], color=sns.color_palette("muted", len(routes_df)))

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=10)

    plt.xlabel('Route', fontsize=12)
    plt.ylabel('Number of Trips', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f'Top {num_routes} Popular Routes from {start_date} to {end_date}', fontsize=14)

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    popular_routes_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return popular_routes_plot

#Station info
@app.route('/station_info', methods=['POST'])
def station_info():
    station_name = request.form.get('station_name')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    station_exists = cleaned_df.filter(
        (cleaned_df['start_station_name'] == station_name) | 
        (cleaned_df['end_station_name'] == station_name)
    ).count() > 0

    if not station_exists:
        error_message = f"Station '{station_name}' does not exist."
        return render_template('index.html', error_message=error_message, active_section='stationInfo')
    
    filtered_df = cleaned_df.filter(
        ((cleaned_df['start_station_name'] == station_name) | (cleaned_df['end_station_name'] == station_name)) &
        (cleaned_df['started_at'] >= start_date) & 
        (cleaned_df['ended_at'] <= end_date)
    )

    hourly_plot = plot_hourly_activity(filtered_df, station_name, start_date, end_date)
    bike_type_plot = plot_bike_type_usage(filtered_df, station_name, start_date, end_date)
    member_casual_plot = plot_member_casual_comparison(filtered_df, station_name, start_date, end_date)

    return render_template('index.html', hourly_plot=hourly_plot, bike_type_plot=bike_type_plot, member_casual_plot=member_casual_plot, active_section='stationInfo')

def plot_hourly_activity(filtered_df, station_name, start_date, end_date):
    num_days = datediff(to_date(lit(end_date)), to_date(lit(start_date))) + 1

    hourly_starts = filtered_df.withColumn('hour', hour('started_at')).groupBy('hour').agg(count(lit(1)).alias('starts_count'))
    hourly_starts = hourly_starts.withColumn('starts_avg', hourly_starts['starts_count'] / num_days)
    hourly_starts_data = hourly_starts.toPandas()

    hourly_ends = filtered_df.withColumn('hour', hour('ended_at')).groupBy('hour').agg(count(lit(1)).alias('ends_count'))
    hourly_ends = hourly_ends.withColumn('ends_avg', hourly_ends['ends_count'] / num_days)
    hourly_ends_data = hourly_ends.toPandas()

    hourly_data = pd.merge(hourly_starts_data, hourly_ends_data, on='hour', how='outer').fillna(0)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hour', y='starts_avg', data=hourly_data, marker='o', label='Average Starts')
    sns.lineplot(x='hour', y='ends_avg', data=hourly_data, marker='o', label='Average Ends')
    plt.title(f'Average Hourly Activity for {station_name} from {start_date} to {end_date}')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Number of Rides')
    plt.legend()

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

import matplotlib.pyplot as plt
from io import BytesIO
import base64

def plot_bike_type_usage(filtered_df, station_name, start_date, end_date):
    bike_type_usage = filtered_df.groupBy('rideable_type').count()
    bike_type_data = bike_type_usage.toPandas()
    
    total_rides = bike_type_data['count'].sum()
    bike_type_data['percentage'] = (bike_type_data['count'] / total_rides) * 100

    plt.figure(figsize=(8, 8))
    plt.pie(bike_type_data['count'], labels=bike_type_data['rideable_type'], autopct=lambda p: '{:.1f}%\n({:.0f} rides)'.format(p, p * total_rides / 100), startangle=140)
    plt.title(f'Bike Type Usage for {station_name} from {start_date} to {end_date}')

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


def plot_member_casual_comparison(filtered_df, station_name, start_date, end_date):
    member_type = filtered_df.groupBy('member_casual').count().orderBy('count', ascending=False)
    member_type_data = member_type.toPandas()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='member_casual', y='count', data=member_type_data)
    plt.title(f'Member vs Casual Comparison for {station_name} from {start_date} to {end_date}')
    plt.xlabel('Member Type')
    plt.ylabel('Number of Rides')

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

from pyspark.sql.types import FloatType
from pyspark.sql.functions import col, coalesce, sum, lit


# Top Member
@app.route('/top_member', methods=['POST'])
def top_member():
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    num_stations = int(request.form.get('num_stations', 10))

    # Get top member stations and plot
    top_member_stations = get_top_member_stations(cleaned_df, start_date, end_date, num_stations)
    top_member_stations_plot = plot_top_member_stations(top_member_stations, num_stations, start_date, end_date)

    # Get top stations with highest member-to-casual ratio and plot
    top_ratio_stations = get_top_ratio_stations(cleaned_df, start_date, end_date, num_stations)
    top_ratio_stations_plot = plot_top_ratio_stations(top_ratio_stations, num_stations, start_date, end_date)

    return render_template('index.html', top_member_stations_plot=top_member_stations_plot, top_ratio_stations_plot=top_ratio_stations_plot, active_section='topMember')

#top member stations
def get_top_member_stations(cleaned_df, start_date, end_date, num_stations):
    # Filter the data based on user input
    filtered_df = cleaned_df.filter((col('started_at') >= start_date) & (col('ended_at') <= end_date) & (col('member_casual') == 'member'))

    # Group by start station and count the number of rides
    station_counts = filtered_df.groupBy('start_station_name').count().withColumnRenamed('count', 'member_count').withColumnRenamed('start_station_name', 'station_name')

    # Order by the member count and limit to the specified number of stations
    top_member_stations = station_counts.orderBy('member_count', ascending=False).limit(num_stations)

    return top_member_stations.toPandas()

# Function to plot the top member stations
def plot_top_member_stations(station_df, num_stations, start_date, end_date):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))
    bars = plt.bar(station_df['station_name'], station_df['member_count'], color=sns.color_palette("muted", len(station_df)))

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=10)

    plt.xlabel('Station Name', fontsize=12)
    plt.ylabel('Number of Member Rides', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f'Top {num_stations} Stations with Most Members from {start_date} to {end_date}', fontsize=14)

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    top_member_stations_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return top_member_stations_plot

# Function to get the top stations with the highest member-to-casual ratio
def get_top_ratio_stations(cleaned_df, start_date, end_date, num_stations):
    # Filter the data based on user input
    filtered_df = cleaned_df.filter((col('started_at') >= start_date) & (col('ended_at') <= end_date))

    # Group by start station and count the number of rides for each member type
    station_counts = filtered_df.groupBy('start_station_name', 'member_casual').count()

    # Pivot the table to have separate columns for member and casual counts
    pivoted_counts = station_counts.groupBy('start_station_name').pivot('member_casual').agg(coalesce(sum('count').cast(FloatType()), lit(0)))

    # Calculate the ratio between member and casual rides
    pivoted_counts = pivoted_counts.withColumn('ratio', coalesce(col('member') / col('casual'), lit(0)))

    # Order by the ratio and limit to the specified number of stations
    top_ratio_stations = pivoted_counts.orderBy('ratio', ascending=False).limit(num_stations)

    return top_ratio_stations.toPandas()

# Function to plot the top stations with the highest member-to-casual ratio
def plot_top_ratio_stations(station_df, num_stations, start_date, end_date):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))
    bars = plt.bar(station_df['start_station_name'], station_df['ratio'], color=sns.color_palette("muted", len(station_df)))

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", va='bottom', ha='center', fontsize=10)

    plt.xlabel('Station Name', fontsize=12)
    plt.ylabel('Member-to-Casual Ratio', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f'Top {num_stations} Stations with Highest Member-to-Casual Ratio from {start_date} to {end_date}', fontsize=14)

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    top_ratio_stations_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return top_ratio_stations_plot

# Predict Hourly Activity
@app.route('/predict_hourly_activity', methods=['POST'])
def predict_hourly_activity():
    station_name = request.form.get('station_name')
    start_date = request.form.get('start_date')  # Get the start date from the form
    end_date = request.form.get('end_date')      # Get the end date from the form
    
    station_exists = cleaned_df.filter(
        (cleaned_df['start_station_name'] == station_name) | 
        (cleaned_df['end_station_name'] == station_name)
    ).count() > 0

    if not station_exists:
        error_message = f"Station '{station_name}' does not exist."
        return render_template('index.html', error_message=error_message, active_section='predictHourly')
    
    def safe_filename(name):
        return re.sub(r'\W+', '_', name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'prophet_models')
    
    model_file_name = f'prophet_model_{safe_filename(station_name)}.pkl'
    model_file_path = os.path.join(model_dir, model_file_name)

    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)

        future = pd.DataFrame({'ds': pd.date_range(start=start_date, end=end_date, freq='H')})
        forecast = model.predict(future)

    plt.figure(figsize=(15, 6))
    plt.subplots_adjust(bottom=0.2)  
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.5, label='Uncertainty interval')

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))  
    plt.xticks(rotation=90) 

    plt.xlabel('Date and Hour')
    plt.ylabel('Ride Volume')

    formatted_start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
    formatted_end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d')

    plt.title(f'Forecasted Hourly Activity from {formatted_start_date} to {formatted_end_date} at {station_name}')

    plt.legend()

    plt.grid(True)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return render_template('index.html', prediction_plot=plot_url, active_section='predictHourly')

# Predict top-station of the day
@app.route('/top_station_by_date', methods=['POST'])
def top_station_by_date():
    date = None
    top_station = None
    date = request.form.get('date')
    
    if date:
        dates = []
        dates.append(date)
        df = pd.DataFrame(dates, columns=['Date'])

        # Convert the 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'top_station_model')
        
        model_file_name = 'top-station-model.pkl'
        model_file_path = os.path.join(model_dir, model_file_name)

        label_encoder_name = 'label-encoder.pkl'
        label_encoder_path = os.path.join(model_dir, label_encoder_name)

        # load trained model 
        with open(model_file_path, 'rb') as file:
            clf = pickle.load(file)
        with open(label_encoder_path, 'rb') as file:
            le = pickle.load(file)
        
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month

        X = df[['day_of_week', 'month']]  # Features

        predictions = clf.predict(X)

        # Convert predictions back to station names
        predicted_stations = le.inverse_transform(predictions)

        top_station = str(predicted_stations[0])

    return render_template("index.html", top_station=top_station, active_section='topStationPredict')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', active_section='busiestStations')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
