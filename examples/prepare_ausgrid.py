#!/usr/bin/env python3
"""Ausgrid labelled data (Jaiswal et al.) -> one CSV per customer with columns ts, kwh, anomaly.
Usage: python3 examples/prepare_ausgrid.py "Anomaly labelled dataset.csv" [customer-id] > ausgrid_7.csv"""
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
cust = int(sys.argv[2]) if len(sys.argv) > 2 else 7
d = df[df.costumer == cust]
ts = pd.to_datetime(dict(year=d.year, month=d.monthofyear, day=d.dayofmonth, hour=d.hourofday, minute=d.minuteofhour))
pd.DataFrame({"ts": ts, "kwh": d.GC.values, "anomaly": d.Anomaly.values}).to_csv(sys.stdout, index=False)
