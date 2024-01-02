import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import copy

def encode(X_train, X_test, y_train, y_test):
    new_X_train = copy.copy(X_train)
    new_X_test = copy.copy(X_test)
    new_y_train = copy.copy(y_train)
    new_y_test = copy.copy(y_test)

    day_of_week_enc = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }

    traffic_sistuation = {
        'low': 0,
        'normal': 1,
        'high': 2, 
        'heavy':3
    }

    enc = OrdinalEncoder()

    new_X_train['Day of the week'] = new_X_train['Day of the week'].replace(day_of_week_enc)
    new_X_train['Time'] =  enc.fit_transform(np.array([new_X_train['Time']]).reshape(-1, 1)).reshape(-1)
    new_y_train['Traffic Situation'] = new_y_train['Traffic Situation'].replace(traffic_sistuation)

    new_X_test['Day of the week'] = new_X_test['Day of the week'].replace(day_of_week_enc)
    new_X_test['Time'] =  enc.transform(np.array([new_X_test['Time']]).reshape(-1, 1)).reshape(-1)
    new_y_test['Traffic Situation'] = new_y_test['Traffic Situation'].replace(traffic_sistuation)

    return new_X_train, new_X_test, new_y_train, new_y_test

def get_features_target(Train, Test):
    X_train = Train.drop(['Traffic Situation'], axis=1).values
    y_train = Train['Traffic Situation'].values

    X_test = Test.drop(['Traffic Situation'], axis=1).values
    y_test = Test['Traffic Situation'].values

    return X_train, X_test, y_train, y_test

def plot_grouped_barchart(categories, values, models, xlabel, ylabel, titile):
    num_groups = len(values)
    num_categories = len(categories)
    width = 0.20  # the width of the bars
    space = 0.20  # space between the groups

    fig, ax = plt.subplots(figsize=(16, 8))

    for i in range(num_categories):
        val = [row[i] for row in values]
        x = np.arange(num_groups) * (num_categories * width + space) + i * width
        ax.bar(x, val, width, label=categories[i])

    ax.set_title(titile)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(num_groups) * (num_categories * width + space) + width * num_categories / 2 - width / 2)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()

    plt.show()

def plot_bar_chart(data):
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    ax.bar(data.keys(), data.values(), color=colors)

    ax.set_title('Accuracy among models')
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracies')
    ax.set_xticklabels(data.keys(), rotation=45)

    plt.show()


def generate_datetime(input_time, input_date, input_day_of_week, year=2023, month=12):
    # Mapping of weekday to integer
    weekday_to_int = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    # Convert time string to datetime object
    time_object = datetime.strptime(input_time, '%I:%M:%S %p')

    # Create a datetime object with the input date and time
    dt = datetime(year=year, month=month, day=input_date, hour=time_object.hour, minute=time_object.minute, second=time_object.second)

    # Calculate the difference between the desired and current weekday
    weekday_diff = weekday_to_int[input_day_of_week] - dt.weekday()
    if weekday_diff < 0:
        # If the desired weekday is before the current date, move to the next week
        weekday_diff += 7

    # Create the final datetime object
    final_dt = dt + timedelta(days=weekday_diff)

    return final_dt


def plot_Loss_Val(history):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], 'b', label='Training loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_Accuracy(history):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], 'b', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def load_data():
    df = pd.read_csv('./TrafficTwoMonth.csv')
    X = df.drop(['Traffic Situation'], axis=1)
    y = df[['Traffic Situation']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2022)
    X_train, X_test, y_train, y_test = encode(X_train, X_test, y_train, y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.values, y_test.values

def load_data_RNN_case2():
    df = pd.read_csv('./TrafficTwoMonth.csv')

    traffic_sistuation = {
        'low': 0,
        'normal': 1,
        'high': 2, 
        'heavy':3
    }

    new_df = copy.copy(df)
    new_df = new_df.drop(["CarCount", "BikeCount", "BusCount", "TruckCount", "Total"], axis=1)
    new_df2 = copy.copy(new_df)
    new_df2['Datetime'] = new_df2.apply(lambda x: generate_datetime(x['Time'], x['Date'], x['Day of the week']), axis=1) 
    new_df3 = new_df2.set_index('Datetime')
    new_df4 = new_df3.drop(['Time', 'Date', 'Day of the week'], axis=1)
    new_df4['Traffic Situation'] = new_df4['Traffic Situation'].replace(traffic_sistuation)
    data = pd.get_dummies(new_df4.to_numpy().reshape(-1), dtype=int).values
    
    timestep = 30

    X = []
    y = []

    n = len(data)

    for i in range(n - (timestep)):
        X.append(data[i:i+timestep])
        y.append(data[i+timestep])

    X = np.asanyarray(X)
    y = np.asanyarray(y)


    index_test = 5000
    X_train = X[:index_test,:,:]
    X_test = X[index_test:,:,:]
    y_train = y[:index_test]    
    y_test= y[index_test:]

    return X_train, X_test, y_train, y_test
