# This code is using a machine learning model to predict the market direction of a 
# given cryptocurrency. The model is a neural network created using the MLPClassifier 
# class from the scikit-learn library. The code also uses the GridSearchCV class 
# to tune the hyperparameters of the model and find the best combination. 
# The code then updates the model with the best hyperparameters and trains it using 
# historical data. The trained model is then used to predict the market direction of 
# the cryptocurrency. The code also includes a trading script that uses the predicted 
# market direction to buy or sell the cryptocurrency on an exchange.
# If the market is bullish it uses a 0 if it is bearish it uses a 1

# The code includes a trading script that places orders on an exchange based 
# on the predicted market direction. The script places a buy order if the market 
# is predicted to be bullish and a sell order if the market is predicted to be bearish.

# These are the time frame you can use for KuCoin
# {'1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min', 
# '1h': '1hour', '2h': '2hour', '4h': '4hour', '6h': '6hour', '8h': '8hour', 
# '12h': '12hour', '1d': '1day', '1w': '1week'}

## Things to improve on: 
# 1) use bayes for chosing hyperparamters or evolutions algorithm
# 2) deep neural networks and vector alogrithms could do better than a regular neural network
# 3) maybe add a stop loss (the script does this for us currently by chosing when to sell)
# 4) long time frame mean more data which means better decisions (training on 15m every 5m)

import ccxt
from time import sleep
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pickle
import sys

# Replace EXCHANGE_NAME with the name of the exchange you want to use
exchange_name = 'kucoin'

# Instantiate the Exchange class
exchange = getattr(ccxt, exchange_name)()

# Set sandbox mode to True or False
exchange.set_sandbox_mode(enabled=False)

# Set your API keys
exchange.apiKey = '63976e0702c544000126a8ca'
exchange.secret = '9655f65f-16f0-4790-9610-cfd61c908101'
exchange.password = 'meowmeow'

# Set the symbol you want to trade on Kucoin
symbol = 'BTC/USDT'

# KuCoin fee per transcation
fee = .001

# Set the premium for the sell order
#print('# Set the premium for the sell order')
premium = 0.003 + fee

## Start the trading script
while True:
    try:
        # Batch streaming data from KuCoin it uses a pair
        # and a window time frame
        data = exchange.fetch_ohlcv('BTC/USDT', '15m')

        # get the model.pkl file from the learn.py session
        model_file = '15mincheck15minochlvmodel.pkl'

        # predict if bullish or bearish
        def predict_market_direction(data, model_file):
            # extract the features and the target variable from the data (IMPROTRANT: chagned > to <)
            features = np.array([d[1:5] for d in data])
            target = np.array([1 if d[4] < d[1] else 0 for d in data])

            # load the model from the model file
            with open(model_file, 'rb') as f:
                mlp = pickle.load(f)

            # update the model with the new data using partial_fit
            mlp.partial_fit(features, target)

            # save the updated model to the model file it was wb i changed to ab
            with open(model_file, 'ab') as f:
                pickle.dump(mlp, f)

            # predict the market direction using the updated model
            # returns 0 for bullish
            # returns 1 for bearish
            # Use the reshape() method to convert the features[-1] array to a 2D shape with 4 features
            features_2d = np.array(features[-1]).reshape(-1, 4)

            # Use the predict() method with the reshaped array
            prediction = mlp.predict(features_2d)
            return prediction

        # Create an infinite loop to trade continuously
        while True:
            # Fetch the current ticker information for the symbol
            print('# Fetch the current ticker information for the symbol')
            ticker = exchange.fetch_ticker(symbol)

            # Check the current bid and ask prices
            print('# Check the current bid and ask prices')
            bid = ticker['bid']
            ask = ticker['ask']

            # Calculate the midpoint of the bid and ask prices
            print('# Calculate the midpoint of the bid and ask prices')
            midpoint = (bid + ask) / 2

            # Set the amount of BTC you want to trade or chose dollar amount to spend divided by the midpoint
            # round to the 3rd decimle palce
            # amount = .005
            amount = round(90 / midpoint, 3)

            # Get balance for selling
            balance = exchange.fetch_balance()
            btc_balance = balance['BTC']['free']
            usdt_balance = balance['USDT']['free']
            print(btc_balance)
            print(usdt_balance)
            
            # Market Data Print
            current_time = datetime.now()
            print('# Market Data Print: Bullish [0] vs Bearish [1]')

            print("The market is ---> {}".format(predict_market_direction(data, model_file)))

            print(current_time.strftime("%B %d, %Y %I:%M %p"))
            # Market Data Print

            # Check if there are any open orders
            print('# Check if there are any open orders')
            try:
                open_orders = exchange.fetch_open_orders(symbol)
            except:
                sleep(60)
                open_orders = exchange.fetch_open_orders(symbol)

            if not open_orders:
                print('# Place a limit buy order at the midpoint price')
                try:
                    # Check if it is bullish 1 or bearish 0 before buying
                    if predict_market_direction(data, model_file).tolist()[0] == 0:
                        # Place a limit buy order at the midpoint price
                        order_id = exchange.create_order(symbol, 'limit', 'buy', amount, ask)
                except:
                    # We must own bitcoin and we want to sell it if the script
                    # tries to buy more bitcoin and has insufficent funds
                    if not open_orders:
                        # Check if it is bullish 0 or bearish 1 before buying
                        if predict_market_direction(data, model_file).tolist()[0] == 1:
                            # Place a limit sell order at the midpoint price plus the premium which includes the fee
                            #order_id = exchange.create_order(symbol, 'limit', 'sell', btc_balance, midpoint * (1 + premium))
                            order_id = exchange.create_order(symbol, 'limit', 'sell', btc_balance, midpoint)
                #else:
                    ##Always run script even if error restart
                    #auto_start = True

            # Pause for a few seconds and check the status of the open orders
            print('# Pause for a few seconds and check the status of the open orders')
            sleep(5)
            open_orders = exchange.fetch_open_orders(symbol)

            # Check if there are any open orders
            print('# Check if there are any open orders')
            if not open_orders:
                # Place a limit sell order at the midpoint price plus the premium
                try:
                    if predict_market_direction(data, model_file).tolist()[0] == 1:
                        #order_id = exchange.create_order(symbol, 'limit', 'sell', btc_balance, midpoint * (1 + premium))
                        order_id = exchange.create_order(symbol, 'limit', 'sell', btc_balance, midpoint)
                except:
                    # Place a limit buy order at the midpoint price
                    # If for some reason the script doesnt have anything to sell
                    # It'll just buy it
                    if predict_market_direction(data, model_file).tolist()[0] == 0:
                        order_id = exchange.create_order(symbol, 'limit', 'buy', amount, ask)


            # Pause for a few seconds and check the status of the open orders XYZ
            # The logic behind this is after it purchased and then turned around 
            # and sold or vice versa it'll wait so many seconds before it does it again.
            print('# Pause for a few seconds and check the status of the open orders - checks every 15min')
            sleep(60 * 15)
            try:
                open_orders = exchange.fetch_open_orders(symbol)
            except:
                sleep(5)
                open_orders = exchange.fetch_open_orders(symbol)
    except:
        # The logic for this sleeping is if the script fails for some rate limit error 
        # or other issue it'll wait one minute before restarting the script again.
        sleep(60)
        continue

#In this code, the target variable is being defined as an array containing a series of 0s and 1s. T
# he values in the target array are determined by evaluating the following expression for each element 
# d in the data list:

# 1 if d[4] > d[1] else 0
# This expression uses an if-else statement to determine the value of the target array element 
# based on the values of d[4] and d[1]. If d[4] is greater than d[1], the element is set to 1. 
# If d[4] is less than or equal to d[1], the element is set to 0.

# In the context of the list of lists you provided earlier, each element in the list represents 
# a time period and contains the following values:

#d[0]: Timestamp (in milliseconds)
#d[1]: Open price
#d[2]: Highest price
#d[3]: Lowest price
#d[4]: Closing price
#d[5]: Volume (in terms of the base currency)

# So, in this context, d[4] represents the closing price and d[1] represents the open price for a 
# given time period. The expression 1 if d[4] > d[1] else 0 is setting the value of the target array 
# element to 1 if the closing price is greater than the open price, and to 0 if the closing price 
# is less than or equal to the open price.