#!/usr/bin/env python3
"""Polish day-ahead market data -> CSV with columns date, price, demand, wind.
Usage: python3 examples/prepare_polish.py <repo>/data/data.xlsx [end-date] > polish.csv
end-date (optional, e.g. 2019-03-05) cuts the series so the last week is the test week."""
import sys
import pandas as pd
x = pd.read_excel(sys.argv[1])
x["date"] = pd.to_datetime(x["date"])
if len(sys.argv) > 2: x = x[x["date"] < pd.Timestamp(sys.argv[2])]
out = pd.DataFrame({"date": x["date"], "price": x["electricity_price (PLN/MWh)"], "demand": x["energy_demand (MW)"],
                    "wind": x["energy_from_wind_sources (MW)"]})
out.to_csv(sys.stdout, index=False)
