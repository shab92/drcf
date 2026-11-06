#!/usr/bin/env python3
"""UCI Individual Household Electric Power Consumption (Hebrail & Berard) -> hourly CSV (time,kw).
Hourly mean of Global_active_power; the 1.25% missing minutes are skipped, empty hours are interpolated.
Usage: python3 examples/prepare_uci_household.py household_power_consumption.txt > uci_hourly.csv"""
import sys
import pandas as pd
x = pd.read_csv(sys.argv[1], sep=";", na_values="?", usecols=[0, 1, 2], low_memory=False)
t = pd.to_datetime(x.Date + " " + x.Time, format="%d/%m/%Y %H:%M:%S")
s = pd.Series(x.Global_active_power.values, index=t).resample("h").mean().interpolate(limit_direction="both")
s.to_csv(sys.stdout, header=["kw"], index_label="time")
