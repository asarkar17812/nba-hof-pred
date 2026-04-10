import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df_data = pd.read_excel("/Users/ayushsarkar/nba-hof-pred/nba-hof-pred/source/NBA ALL STAR DATA.xlsx", sheet_name=1)
df_team_abbr = pd.read_excel("/Users/ayushsarkar/nba-hof-pred/nba-hof-pred/source/NBA ALL STAR DATA.xlsx", sheet_name=7)
print(df_data)