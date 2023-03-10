import pandas as pd
from datetime import datetime
import datetime as dt

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np


df = pd.read_csv("DXY5.csv")

df.columns = ["date", "time", "open", "high", "low", "close", "volume"]
df["datetime"] = df["date"] + " " + df["time"]
df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M:%S.%f")
df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
df["time"] = pd.to_datetime(
    df["time"].str.replace(":00.000", ""), format="%H:%M"
).dt.time
print(df)


# read data from CSV file into a DataFrame


# convert the 'date' column to a datetime type

# create a new column with the day of the week
df["day_of_week"] = df["date"].dt.day_name()

# delete rows with Sunday
df = df[df["day_of_week"] != "Sunday"]
df = df[df["day_of_week"] != "Saturday"]
# save the modified DataFrame to a new CSV file
df.to_csv("cleanedDXY5.csv", index=False)
