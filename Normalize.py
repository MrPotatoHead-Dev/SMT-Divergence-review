import pandas as pd
from datetime import datetime
import datetime as dt
import time
import scipy.stats
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import warnings
from collections import Counter


warnings.filterwarnings("ignore")

# Load the data into a pandas DataFrame
df_EG = pd.read_csv("EURGBP5.csv")
df_GU = pd.read_csv("GBPUSD5.csv")
df_EU = pd.read_csv("EURUSD5.csv")
df_dxy = pd.read_csv("dxy5.csv")  # missing data/ Not useable
df_dxy["datetime"] = df_dxy["date"] + " " + df_dxy["time"]
df_dxy["date"] = pd.to_datetime(df_dxy["date"], format="%Y-%m-%d")


def structureMT4Data(df):  # df cleaning and formatting
    df.columns = ["date", "time", "open", "high", "low", "close", "volume"]
    df["datetime"] = df["date"] + " " + df["time"]
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H:%M")
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m/%d")
    df["time"] = pd.to_datetime(df["time"], format="%H:%M").dt.time
    return df


def exportDate(df, start_date, end_date=None):
    if end_date == None:
        datesDf = df["date"] == start_date
    else:
        datesDf = (df["date"] >= start_date) & (df["date"] <= end_date)
    structuredDf = df[datesDf]

    return structuredDf


# def hlc3(df):
#     df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
#     return df["hlc3"]

# def hl2(df):
#     df["hl2"] = (df["high"] + df["low"]) / 2
#     return df["hl2"]


def hlc3(row):
    return (row["high"] + row["low"] + row["close"]) / 3


def hl2(row):
    return (row["high"] + row["low"]) / 2


def pivotid(df1, l, n1, n2):  # n1 n2 before and after candle l
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0

    pividlow = 1
    pividhigh = 1
    for i in range(l - n1, l + n2 + 1):
        if df1.low[l] > df1.low[i]:
            pividlow = 0
        if df1.high[l] < df1.high[i]:
            pividhigh = 0
    if pividlow and pividhigh:
        return 3
    elif pividlow:
        return 1
    elif pividhigh:
        return 2
    else:
        return 0


def pointpos(x):
    if x["pivot"] == 1:
        return x["low"] - 1e-4
    elif x["pivot"] == 2:
        return x["high"] + 1e-4
    else:
        return np.nan


def normalize(df):
    stage1 = (df - df.min()) / (df.max() - df.min())
    stage2 = stage1.to_frame()
    stage3 = [i for i in range(len(stage2))]
    normalize = stage2.set_index(pd.Index(stage3))
    return normalize


def divergence(df1, df2):  # returns dates, times
    merged_dates = df1.merge(df2, on=["date", "time"], suffixes=("_1", "_2"))
    divergence_dates = set(
        merged_dates.loc[
            (merged_dates["trend highs_1"] != merged_dates["trend highs_2"])
            | (merged_dates["trend lows_1"] != merged_dates["trend lows_2"]),
            "date",
        ]
    )
    merged_times = df1.merge(df2, on="time", suffixes=("_1", "_2"))
    divergence_times = merged_times.loc[
        (merged_times["trend highs_1"] != merged_times["trend highs_2"])
        | (merged_times["trend lows_1"] != merged_times["trend lows_2"]),
        "time",
    ]

    return (divergence_dates), (divergence_times)


def slopeSignal(df):  # finding trend using pivots (HH, HL...)
    slopeLow = []
    slopeHigh = []
    lowPoints = []
    highPoints = []
    trendHighs = [np.nan] * len(df)  # initialize with np.nan values
    trendLows = [np.nan] * len(df)  # initialize with np.nan values

    for i in range(1, len(df) - 1):
        if df["pivot"][i] == 2:
            highPoints.append(df["high"][i])
            if len(highPoints) > 1:
                slopeHigh.append(highPoints[-1] - highPoints[-2])
                if len(slopeHigh) > 1:
                    if slopeHigh[-1] > 0:
                        trendHighs[i] = True
                    if slopeHigh[-1] < 0:
                        trendHighs[i] = False
        if df["pivot"][i] == 1:
            lowPoints.append(df["low"][i])
            if len(lowPoints) > 1:
                slopeLow.append(lowPoints[-1] - lowPoints[-2])
                if len(slopeLow) > 1:
                    if slopeLow[-1] > 0:
                        trendLows[i] = True
                    if slopeLow[-1] < 0:
                        trendLows[i] = False

    for i in range(1, len(df)):
        if np.isnan(trendHighs[i]):
            trendHighs[i] = trendHighs[i - 1]
        if np.isnan(trendLows[i]):
            trendLows[i] = trendLows[i - 1]

    df["trend highs"] = trendHighs
    df["trend lows"] = trendLows
    return df["trend highs"], df["trend lows"]


dfEG = structureMT4Data(df_EG)
dfGU = structureMT4Data(df_GU)
dfEU = structureMT4Data(df_EU)

# dfEG = exportDate(dfEG, "2022-02-01", "2022-11-30")
# dfGU = exportDate(dfGU, "2022-02-01", "2022-11-30")
# dfEU = exportDate(dfEU, "2022-02-01", "2022-11-30")
# dfDXY = exportDate(df_dxy, "2022-02-01", "2022-11-30")


dfEG = exportDate(dfEG, "2022-07-29")
dfGU = exportDate(dfGU, "2022-07-29")
dfEU = exportDate(dfEU, "2022-07-29")
dfDXY = exportDate(df_dxy, "2022-07-29")

# ------------------------------------------ Line graph methods ------------------------------------------#

dfEG["hlc3"] = dfEG.apply(hlc3, axis=1)
dfGU["hlc3"] = dfGU.apply(hlc3, axis=1)
dfEU["hlc3"] = dfEU.apply(hlc3, axis=1)
dfDXY["hlc3"] = dfDXY.apply(hlc3, axis=1)

dfEG["hl2"] = dfEG.apply(hl2, axis=1)
dfGU["hl2"] = dfGU.apply(hl2, axis=1)
dfEU["hl2"] = dfEU.apply(hl2, axis=1)
dfDXY["hl2"] = dfDXY.apply(hl2, axis=1)

# ---------------------------------------------- Normalize ----------------------------------------------#
normEG = normalize(dfEG["hl2"])
normGU = normalize(dfGU["hl2"])
normEU = normalize(dfEU["hl2"])
normDXY = normalize(dfDXY["hl2"])

# ---------------------------------------------- SMT review ----------------------------------------------#


# reset pandas indexing
dfGU = dfGU.reset_index(drop=True)
dfEU = dfEU.reset_index(drop=True)
dfDXY = dfDXY.reset_index(drop=True)

dfGU["pivot"] = dfGU.apply(lambda x: pivotid(dfGU, x.name, 10, 10), axis=1)
dfEU["pivot"] = dfEU.apply(lambda x: pivotid(dfEU, x.name, 10, 10), axis=1)
dfDXY["pivot"] = dfDXY.apply(lambda x: pivotid(dfDXY, x.name, 7, 7), axis=1)

dfEU["pointpos"] = dfEU.apply(lambda row: pointpos(row), axis=1)
dfGU["pointpos"] = dfGU.apply(lambda row: pointpos(row), axis=1)
dfDXY["pointpos"] = dfDXY.apply(lambda row: pointpos(row), axis=1)

dfGU["trend highs"], dfGU["trend lows"] = slopeSignal(dfGU)
dfEU["trend highs"], dfEU["trend lows"] = slopeSignal(dfEU)
dfDXY["trend highs"], dfDXY["trend lows"] = slopeSignal(dfDXY)

# ------------------------------------------- Filter trade times ------------------------------------------#

dfDXY["date"] = dfDXY["date"].dt.date
dfDXY["date"] = pd.to_datetime(dfDXY["date"])
dfDXY["time"] = pd.to_datetime(dfDXY["time"], format="%H:%M:%S")
dfDXY_filtered = dfDXY[(dfDXY["time"].dt.hour >= 2) & (dfDXY["time"].dt.hour < 23)]

dfEU["date"] = pd.to_datetime(dfEU["date"])
dfEU["time"] = pd.to_datetime(dfEU["time"], format="%H:%M:%S")
dfEU_filtered = dfEU[(dfEU["time"].dt.hour >= 2) & (dfEU["time"].dt.hour < 23)]

dfGU["date"] = pd.to_datetime(dfGU["date"])
dfGU["time"] = pd.to_datetime(dfGU["time"], format="%H:%M:%S")
dfGU_filtered = dfGU[(dfGU["time"].dt.hour >= 2) & (dfGU["time"].dt.hour < 23)]


# -------------------------------------------- Merge dataframes -------------------------------------------#
merged_df_eu_gu = pd.merge(dfEU, dfGU, on=["date", "time"], suffixes=("_1", "_2"))
merged_df_eu_dxy = pd.merge(
    dfEU_filtered, dfDXY_filtered, on=["date", "time"], suffixes=("_1", "_2")
)
merged_df_gu_dxy = pd.merge(
    dfGU_filtered, dfDXY_filtered, on=["date", "time"], suffixes=("_1", "_2")
)


smtDates, smtTimes = divergence(dfGU_filtered, dfDXY_filtered)


# ----------------------------------------- Corrolation coefficient ----------------------------------------#
######################################## Corrolation anchored per day #######################################
# def correlationPDay(df_merged):
#     corr_anchored = []

#     for i in range(1, len(df_merged)):
#         corr = np.corrcoef(
#             df_merged["hl2_1"].iloc[:i],
#             df_merged["hl2_2"].iloc[:i],
#         )[0, 1]

#         corr_anchored.append((round(corr, 2)))
#     return corr_anchored


# corr_indicator_eu_dxy = correlationPDay(merged_df_eu_dxy)
# corr_indicator_eu_gu = correlationPDay(merged_df_gu_dxy)

############################################ Corrolation per day ############################################
# def correlationPDay(df_merged):
#     corr_per_day = []
#     df_merged["date_shifted"] = df_merged["date"].shift()
#     for i in range(1, len(df_merged)):
#         if (
#             df_merged["date"][i] != df_merged["date_shifted"][i]
#             and df_merged["date_shifted"][i]
#         ):
#             corr = np.corrcoef(
#                 df_merged["hl2_1"].iloc[i - 288 : i],
#                 df_merged["hl2_2"].iloc[i - 288 : i],
#             )[0, 1]

#             day_of_week = df_merged["date"].dt.day_name().iloc[i]
#             corr_per_day.append((day_of_week, corr))

#     mon_corrs = [corr for day, corr in corr_per_day if day == "Monday"]
#     tue_corrs = [corr for day, corr in corr_per_day if day == "Tuesday"]
#     wed_corrs = [corr for day, corr in corr_per_day if day == "Wednesday"]
#     thu_corrs = [corr for day, corr in corr_per_day if day == "Thursday"]
#     fri_corrs = [corr for day, corr in corr_per_day if day == "Friday"]
#     return mon_corrs, tue_corrs, wed_corrs, thu_corrs, fri_corrs


# mon, tue, wed, thu, fri = correlationPDay(merged_df_eu_gu)
# mon1, tue1, wed1, thu1, fri1 = correlationPDay(merged_df_eu_dxy)
# mon2, tue2, wed2, thu2, fri2 = correlationPDay(merged_df_gu_dxy)

# print(
#     f"EU & GU, Mon: {round(np.mean(mon), 2)}, Tue: {round(np.mean(tue), 2)}, Wed: {round(np.mean(wed), 2)}, Thu: {round(np.mean(thu), 2)}, Fri: {round(np.mean(fri), 2)} "
# )
# print(
#     f"EU & DXY, Mon: {round(np.mean(mon1), 2)}, Tue: {round(np.mean(tue1), 2)}, Wed: {round(np.mean(wed1), 2)}, Thu: {round(np.mean(thu1), 2)}, Fri: {round(np.mean(fri1), 2)} "
# )
# print(
#     f"GU & DXY, Mon: {round(np.mean(mon2), 2)}, Tue: {round(np.mean(tue2), 2)}, Wed: {round(np.mean(wed2), 2)}, Thu: {round(np.mean(thu2), 2)}, Fri: {round(np.mean(fri2), 2)} "
# )

##################################### Corrolation over entire dataframe #####################################

# corr_hl2_eu_gu = np.corrcoef(merged_df_eu_gu["hl2_1"], merged_df_eu_gu["hl2_2"])[0, 1]
# corr_hl2_eu_eg = np.corrcoef(merged_df_eu_eg["hl2_1"], merged_df_eu_eg["hl2_2"])[0, 1]
# corr_hl2_gu_eg = np.corrcoef(merged_df_gu_eg["hl2_1"], merged_df_gu_eg["hl2_2"])[0, 1]

# print(f"EU & GU {corr_hl2_eu_gu}")
# print(f"EU & EG {corr_hl2_eu_eg}")
# print(f"GU & EG {corr_hl2_gu_eg}")
# # corr_hlc3 = np.corrcoef(dfEU["hlc3"], dfEG["hlc3"])[0, 1]

# dfEU.set_index("datetime", inplace=True)
# dfGU.set_index("datetime", inplace=True)
# # Group the data into hourly blocks
# dfEU_hourly = dfEU.groupby(pd.Grouper(freq="H")).last()
# dfGU_hourly = dfGU.groupby(pd.Grouper(freq="H")).last()


# print(dfEU_hourly)
# print(smtTimes)


# ---------------------------------------------- Plotting OHCL ---------------------------------------------#

smtDates = list(smtDates)
split_gu = pd.DataFrame()
split_gu["date"] = merged_df_eu_gu["date"]
split_gu["open"] = merged_df_eu_gu["open_2"]
split_gu["high"] = merged_df_eu_gu["high_2"]
split_gu["low"] = merged_df_eu_gu["low_2"]
split_gu["close"] = merged_df_eu_gu["close_2"]
split_gu["volume"] = merged_df_eu_gu["volume_2"]
split_gu["datetime"] = merged_df_eu_gu["datetime_2"]
split_gu["pointpos"] = merged_df_eu_gu["pointpos_2"]


split_eu = pd.DataFrame()
split_eu["date"] = merged_df_eu_gu["date"]
split_eu["open"] = merged_df_eu_gu["open_1"]
split_eu["high"] = merged_df_eu_gu["high_1"]
split_eu["low"] = merged_df_eu_gu["low_1"]
split_eu["close"] = merged_df_eu_gu["close_1"]
split_eu["volume"] = merged_df_eu_gu["volume_1"]
split_eu["datetime"] = merged_df_eu_gu["datetime_1"]
split_eu["pointpos"] = merged_df_eu_gu["pointpos_1"]
split_eu.set_index("datetime", inplace=True)
split_gu.set_index("datetime", inplace=True)

for i in range(len(smtDates)):
    specific_date = pd.to_datetime(smtDates[i])
    div_day_eu = split_eu[split_gu["date"] == specific_date]
    div_day_gu = split_gu[split_gu["date"] == specific_date]

    day_of_week = div_day_gu["date"].dt.day_name().iloc[0]
    ap = [
        mpf.make_addplot(div_day_gu["pointpos"], type="scatter"),
        mpf.make_addplot(
            div_day_eu,
            panel=1,
            type="candle",
            ylabel="EU Price",
        ),
        mpf.make_addplot(div_day_eu["pointpos"], panel=1, type="scatter"),
    ]
    mpf.plot(
        div_day_gu,
        type="candle",
        style="yahoo",
        addplot=ap,
        panel_ratios=(10, 10),
        figratio=(10, 5),
        ylabel="GU Price",
        title=day_of_week,
        savefig=dict(fname=f"SMT_eg{i}.png", dpi=500, pad_inches=60),
    )


# ------------------------------------------------ Plottting -----------------------------------------------#


# ------------------------------------------------ Testing -----------------------------------------------#

# dfAll = pd.concat([normEG, normGU, normEU], axis=1)
# missing_valuesEG = normEG.isna().sum().sum()
# missing_valuesGU = normGU.isna().sum().sum()
# missing_valuesEU = normEU.isna().sum().sum()
# print(missing_valuesEG, missing_valuesGU, missing_valuesEU)
# print(len(normEG), len(normEU), len(normGU))
# print(dfEG.head())
# print(dfGU.head())


# ---------------------------------------------- Individual plotting ----------------------------------------------#

# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True, sharey=True)


# normDXY.plot(ax=axes[0], title="DXY")
# normGU.plot(ax=axes[1], title="GU")
# normEU.plot(ax=axes[2], title="EU")

# plt.tight_layout()
# plt.show()

# all on the same plot
# ---------------------------------------- All on the same plot ---------------------------------------#
# fig, ax = plt.subplots(figsize=(12, 6))

# normDXY.plot(ax=ax, label="DXY")
# normGU.plot(ax=ax, label="GU")
# normEU.plot(ax=ax, label="EU")

# ax.set_xlabel("Index")
# ax.set_ylabel("Normalized values")
# ax.set_title("Normalized values of DXY, GU, and EU")
# ax.legend(["DXY", "GU", "EU"])


# plt.axvline(x=174, color="grey", label="axvline - full height")
# plt.axvline(x=120, color="red", label="axvline - full height")

# while x < len(normEG):
#     x = x + 288
#     ny = x + 180
#     if ny > len(normEG):
#         ny = 0
#     plt.axvline(x=60, color="black", linestyle="--", label="axvline - dashed")
#     plt.axvline(x=120, color="red", linestyle="--", label="axvline - dashed")
#     plt.axvline(x=0, color="grey", label="axvline - full height")
#     plt.axvline(x=174, color="grey", label="axvline - full height")

# plt.show()
# ---------------------------------------- Correlation indicator ---------------------------------------#


# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
# # normEU.plot(ax=axes[0], label="EU")
# # normDXY.plot(ax=axes[0], label="DXY")

# axes[0].plot(normGU, label="GU")
# axes[0].plot(normDXY, label="DXY")
# axes[1].plot(corr_indicator_eu_gu, label="Correlation GU DXY")

# axes[0].legend()
# axes[1].legend()
# plt.tight_layout()

# # ax.set_xlabel("Index")
# # ax.set_ylabel("Normalized values")
# # ax.set_title("Correlation indicator")
# # ax.legend(["DXY", "EU"])
# plt.show()
