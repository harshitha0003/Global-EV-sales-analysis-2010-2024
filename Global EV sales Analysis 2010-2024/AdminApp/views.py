from django.shortcuts import render
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,RandomForestClassifier,AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Create your views here.
def index(request):
    return render(request, 'AdminApp/index.html')

def login(request):
    return render(request, 'AdminApp/Admin.html')

def LogAction(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    if username == 'Admin' and password == 'Admin':      
        return render(request, 'AdminApp/AdminHome.html')
    else:
        context = {'data': 'Login Failed ....!!'}
        return render(request, 'AdminApp/Admin.html', context)

def home(request):
    return render(request, 'AdminApp/AdminHome.html')

# Global variable to store dataset and models
global df, X_train, X_test, y_train, y_test, rfc, model, ranacc, adacc, dt, dtacc
df = None

def LoadData(request):
    global df
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(BASE_DIR + "\\dataset\\Global_EV_Data.csv")  # Replace with your dataset path

    # Handle missing values and infinite values
    df.fillna(0, inplace=True)  # Replace NaNs with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
    df.fillna(0, inplace=True)  # Replace NaNs (after infinities replaced) with 0

    # Clip large values (optional: can be adjusted based on the domain)
    #df['value'] = np.clip(df['value'], -1e10, 1e10)  # Clip values to the range [-1e10, 1e10]
    
    context = {'data': "Dataset Loaded\n"}
    return render(request, 'AdminApp/AdminHome.html', context)

def split(request):
    global X_train, X_test, y_train, y_test
    global df

    # Encoding categorical variables into numeric values using LabelEncoder
    label_encoder = LabelEncoder()

    # Region encoding (to handle all possible region names)
    df['region'] = df['region'].map({
        'Australia': 0, 'Austria': 1, 'Belgium': 2, 'Brazil': 3, 'Bulgaria': 4, 'Canada': 5,
        'Chile': 6, 'China': 7, 'Colombia': 8, 'Costa Rica': 9, 'Croatia': 10, 'Cyprus': 11,
        'Czech Republic': 12, 'Denmark': 13, 'Estonia': 14, 'EU27': 15, 'Europe': 16, 'Finland': 17,
        'France': 18, 'Germany': 19, 'Greece': 20, 'Hungary': 21, 'Iceland': 22, 'India': 23,
        'Indonesia': 24, 'Ireland': 25, 'Israel': 26, 'Italy': 27, 'Japan': 28, 'Korea': 29,
        'Latvia': 30, 'Lithuania': 31, 'Luxembourg': 32, 'Mexico': 33, 'Netherlands': 34,
        'New Zealand': 35, 'Norway': 36, 'Poland': 37, 'Portugal': 38, 'Rest of the world': 39,
        'Romania': 40, 'Seychelles': 41, 'Slovakia': 42, 'Slovenia': 43, 'South Africa': 44,
        'Spain': 45, 'Sweden': 46, 'Switzerland': 47, 'Thailand': 48, 'Turkiye': 49,
        'United Arab Emirates': 50, 'United Kingdom': 51, 'USA': 52, 'World': 53
    })

    # Category encoding
    df['category'] = df['category'].map({'Historical': 0, 'Projection-STEPS': 1, 'Projection-APS': 2})

    # Parameter encoding
    df['parameter'] = df['parameter'].map({
        'EV stock share': 0, 'EV sales share': 1, 'EV sales': 2, 'EV stock': 3,
        'EV charging points': 4, 'Electricity demand': 5, 'Oil displacement Mbd': 6,
        'Oil displacement, million lge': 7
    })

    # Mode encoding
    df['mode'] = df['mode'].map({'Cars': 0, 'EV': 1, 'Buses': 2, 'Vans': 3, 'Trucks': 4})

    # Powertrain encoding
    df['powertrain'] = df['powertrain'].map({
        'EV': 0, 'BEV': 1, 'PHEV': 2, 'Publicly available fast': 3,
        'Publicly available slow': 4, 'FCEV': 5
    })

    # Unit encoding
    df['unit'] = df['unit'].map({
        'percent': 0, 'Vehicles': 1, 'charging points': 2, 'GWh': 3,
        'Million barrels per day': 4, 'Oil displacement, million lge': 5
    })

    # Ensure numerical columns are treated correctly
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    #df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['value']=label_encoder.fit_transform(df['value'])
    df['percentage']=label_encoder.fit_transform(df['percentage'])
    # Now, handle the data preparation for training
    # Select features (X) and target variable (y)
    X = df.drop(columns=['value','percentage','year'])  # Drop target column 'Value' from features
    y = df['percentage']  # Target variable 'Value'

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    context = {"data": "Preprocessing Done"}
    return render(request, 'AdminApp/AdminHome.html', context)

# Train Random Forest Regressor
def runRandomForest(request):
    global ranacc, rfc, X_train, y_train, X_test, y_test

    if X_train is None or y_train is None:
        # Make sure data is split first
        context = {'data': "Please load and preprocess data first."}
        return render(request, 'AdminApp/AdminHome.html', context)
    
    rfc = RandomForestClassifier(n_estimators=100)  # Initialize Random Forest Regressor

    # Clean data before training 
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train the model
    rfc.fit(X_train_clean, y_train)

    # Make predictions
    prediction = rfc.predict(X_test_clean)
    ranacc = accuracy_score(y_test, prediction) * 100  # Calculate accuracy

    context = {"data": f"RandomForest Accuracy: {ranacc}"}
    return render(request, 'AdminApp/AdminHome.html', context)

# Train AdaBoost Regressor
def runAdaboost(request):
    global adacc, model, X_train, y_train, X_test, y_test
    if X_train is None or y_train is None:
        context = {'data': "Please load and preprocess data first."}
        return render(request, 'AdminApp/AdminHome.html', context)
    
    model = AdaBoostClassifier()  # Initialize AdaBoost Regressor
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train the model
    model.fit(X_train_clean, y_train)

    # Make predictions
    prediction = model.predict(X_test_clean)  # Predict on test data
    adacc = accuracy_score(y_test, prediction) * 100  # Calculate accuracy
    context = {"data": f"AdaBoost Accuracy: {adacc}"}
    return render(request, 'AdminApp/AdminHome.html', context)

# Train DecisionTree Regressor
def runDecisiontree(request):
    global ranacc, rfc, X_train, y_train, X_test, y_test, dtacc, dt

    if X_train is None or y_train is None:
        context = {'data': "Please load and preprocess data first."}
        return render(request, 'AdminApp/AdminHome.html', context)
    
    dt = DecisionTreeClassifier()  # Initialize DecisionTreeRegressor

    # Clean data before training 
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train the model
    dt.fit(X_train_clean, y_train)

    # Make predictions
    prediction = dt.predict(X_test_clean)
    dtacc = accuracy_score(y_test, prediction) * 100  # Calculate accuracy

    context = {"data": f"DecisionTree Accuracy: {dtacc}"}
    return render(request, 'AdminApp/AdminHome.html', context)
global ranacc, adacc,dtacc,X_train,y_train,X_test,y_test
def runComparision(request):
    global ranacc, adacc,dtacc,X_train,y_train,X_test,y_test
    if ranacc is None or adacc is None:
        # Check if models are trained before plotting
        context = {'data': "Please run the models first."}
        return render(request, 'AdminApp/AdminHome.html', context)

    bars = ['RandomForest Accuracy', 'AdaBoost Accuracy','DecisionTree Accuracy']
    height = [ranacc, adacc,dtacc]
    y_pos = np.arange(len(bars))

    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.show()
    
    return render(request, 'AdminApp/AdminHome.html')

def predict(request):
    return render(request, 'AdminApp/Prediction.html')
global model, dt, rfc, ranacc, adacc, dtacc
def PredAction(request):
    global model, dt, rfc, ranacc, adacc, dtacc

    # Ensure that models are trained before making predictions
    if model is None or dt is None or rfc is None:
        context = {'data': "Models have not been trained yet."}
        return render(request, 'AdminApp/AdminHome.html', context)

    # Get the input data from the form
    region = request.POST.get('region')
    category = request.POST.get('category')
    parameter = request.POST.get('parameter')
    mode = request.POST.get('mode')
    powertrain = request.POST.get('powertrain')
    unit = request.POST.get('unit')

    # Encoding for each feature as before
    region_mapping = {
        'Australia': 0, 'Austria': 1, 'Belgium': 2, 'Brazil': 3, 'Bulgaria': 4, 'Canada': 5,
        'Chile': 6, 'China': 7, 'Colombia': 8, 'Costa Rica': 9, 'Croatia': 10, 'Cyprus': 11,
        'Czech Republic': 12, 'Denmark': 13, 'Estonia': 14, 'EU27': 15, 'Europe': 16, 'Finland': 17,
        'France': 18, 'Germany': 19, 'Greece': 20, 'Hungary': 21, 'Iceland': 22, 'India': 23,
        'Indonesia': 24, 'Ireland': 25, 'Israel': 26, 'Italy': 27, 'Japan': 28, 'Korea': 29,
        'Latvia': 30, 'Lithuania': 31, 'Luxembourg': 32, 'Mexico': 33, 'Netherlands': 34,
        'New Zealand': 35, 'Norway': 36, 'Poland': 37, 'Portugal': 38, 'Rest of the world': 39,
        'Romania': 40, 'Seychelles': 41, 'Slovakia': 42, 'Slovenia': 43, 'South Africa': 44,
        'Spain': 45, 'Sweden': 46, 'Switzerland': 47, 'Thailand': 48, 'Turkiye': 49,
        'United Arab Emirates': 50, 'United Kingdom': 51, 'USA': 52, 'World': 53
    }
    category_mapping = {'Historical': 0, 'Projection-STEPS': 1, 'Projection-APS': 2}
    parameter_mapping = {'EV stock share': 0, 'EV sales share': 1, 'EV sales': 2, 'EV stock': 3,
                         'EV charging points': 4, 'Electricity demand': 5, 'Oil displacement Mbd': 6,
                         'Oil displacement, million lge': 7}
    mode_mapping = {'Cars': 0, 'EV': 1, 'Buses': 2, 'Vans': 3, 'Trucks': 4}
    powertrain_mapping = {'EV': 0, 'BEV': 1, 'PHEV': 2, 'Publicly available fast': 3,
                          'Publicly available slow': 4, 'FCEV': 5}
    unit_mapping = {'percent': 0, 'Vehicles': 1, 'charging points': 2, 'GWh': 3,
                    'Million barrels per day': 4, 'Oil displacement, million lge': 5}

    # Convert the inputs using mappings
    region = region_mapping.get(region, -1)
    category = category_mapping.get(category, -1)
    parameter = parameter_mapping.get(parameter, -1)
    mode = mode_mapping.get(mode, -1)
    powertrain = powertrain_mapping.get(powertrain, -1)
    unit = unit_mapping.get(unit, -1)

    # Prepare the input data for prediction
    input_data = pd.DataFrame([[region, category, parameter, mode, powertrain, unit]],
                              columns=['region', 'category', 'parameter', 'mode', 'powertrain', 'unit'])

    # Handle invalid data (NaN, Inf)
    input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    input_data.fillna(0, inplace=True)

    

    # Make predictions using the best model
    pred = model.predict(input_data)

    # Display the predicted value with a percentage sign
    context = {'data': f"Predicted Value: {pred[0] * 100:.2f}%"}

    return render(request, 'AdminApp/PredictedData.html', context)

