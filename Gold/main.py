from data_scrapper import generate_synthetic_data
from datetime import datetime
import json
import train
from LSTMCell import GPP
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np


lstm = GPP()

def calculate_days_between(date_str1, date_str2):
    # Parse the date strings into datetime objects
    date_format = "%Y-%m-%d"
    d1 = datetime.strptime(date_str1, date_format)
    d2 = datetime.strptime(date_str2, date_format)
    
    # Calculate the difference
    delta = d2 - d1
    print(f"Predicting {delta.days} days into the future...")    
    # Return the absolute number of days (so order doesn't matter)
    return abs(delta.days)

generate_synthetic_data()

with open("real_db.json", "r") as f:
    data = json.load(f)
    
dates = data["dates"]
prices_list = data["prices"]



scaler = MinMaxScaler(feature_range=(0, 1)) #
prices_scaled = scaler.fit_transform(np.array(prices_list).reshape(-1, 1)).flatten().tolist() #

train.train_model(lstm,prices_scaled)





# NEW PHASE THE UI
print("-"*50)
input_date = input("System ready to take a date input to guess a gold price (YYYY-MM-DD):")
days_between = calculate_days_between(dates[-1],input_date)
current_sequence = prices_scaled[-29:]

for day in range(days_between):
    
    x_t = torch.tensor(current_sequence[-29:]).float().view(1, 29, 1)
    

    with torch.no_grad(): #
        prediction = lstm(x_t)
        pred_scalar = prediction.item() #

    current_sequence.append(pred_scalar)
    real_price = scaler.inverse_transform([[pred_scalar]])[0][0] #
    print(f"After day {day+1}:{real_price}")
    
print(f"\nPredicted gold price on {input_date}: ${real_price:.4f}")



    
    
    