import numpy as np
import pickle
import streamlit as st
import importlib.util
import sys

# Check if xgboost is available before loading the model
if importlib.util.find_spec("xgboost") is not None:
    import xgboost  # noqa: F401
    model = pickle.load(open("AllFeatureHousePriceModel.pkl", "rb"))
else:
    sys.exit("❌ xgboost is not installed. Please run: pip install xgboost")

# Load the trained model
model = pickle.load(open("AllFeatureHousePriceModel.pkl", "rb"))

# ------------ Take input from user ------------
try:
    car = int(input("Enter number of car spaces: "))
    landsize = float(input("Enter land size (in square meters): "))
    building_area = float(input("Enter building area (in square meters): "))
    year_built = int(input("Enter year built (e.g. 1995): "))

    # Prepare input for prediction
    input_data = np.array([[car, landsize, building_area, year_built]])

    # Predict the price
    predicted_price = model.predict(input_data)[0]

    # Show the result
    print(f"\n🏷️ Predicted House Price: ${predicted_price:,.2f}")

except ValueError:
    print("\n❌ Invalid input. Please enter numeric values only.")
except Exception as e:
    print(f"\n❌ Error: {e}")
