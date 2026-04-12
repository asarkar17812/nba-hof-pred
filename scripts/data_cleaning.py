import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# df_data = pd.read_excel("/Users/ayushsarkar/nba-hof-pred/nba-hof-pred/source/NBA ALL STAR DATA.xlsx", sheet_name=1)
# df_team_abbr = pd.read_excel("/Users/ayushsarkar/nba-hof-pred/nba-hof-pred/source/NBA ALL STAR DATA.xlsx", sheet_name=7)
# print(df_data)

script_dir = Path(__file__).parent
data_path = script_dir / ".." / "source" / "uncleaned" / "NBA ALL STAR DATA.xlsx"

df_data = pd.read_excel(data_path, sheet_name=1)
df_team_abbr = pd.read_excel(data_path, sheet_name=7)

# print(df_data)
df_data.columns = df_data.columns.str.strip()
df_data = df_data.drop(['Games Started', 'Unnamed: 0'], axis=1)

def convert_height(height_str):
    try:
        if pd.isna(height_str) or not isinstance(height_str, str):
            return 0
        # This handles both "6-11" and "6'11"
        parts = height_str.replace("'", "-").split('-')
        
        feet = int(parts[0])
        inches = int(parts[1]) if len(parts) > 1 else 0
        
        return (feet * 12) + inches
    except:
        return 0

df_data['Height_Inches'] = df_data['Height'].apply(convert_height)
df_data = df_data.drop('Height', axis=1)

idx_to_keep = df_data.groupby(['Player', 'Season Ending Year'])['Games'].idxmax()
df_data = df_data.loc[idx_to_keep]
df_data = df_data.sort_index()

df_clean = df_data.dropna(subset=['Player'])

# print(df_clean)
# confirms the rest of the nulls are within field goal % related stats
# print(df_clean.isnull().sum()) 

# setting these to 0, players didn't shoot certain shots (can't divide by 0 --> just set to 0)
df_clean = df_clean.fillna(0)
# print(df_clean.isnull().sum()) # just checking

target = df_clean.pop('All Star')
# 2. Put it back in at the end
df_clean['All Star'] = target
# print(df_clean)

# ------------------------------------------------------------------
# PREPROCESSING for training
# ------------------------------------------------------------------

''' List of features: 
Player, Season Ending Year

- era based stats:
FGA per game, 2PA per game, 3PA per game, FTA per game            
ORB per game, DRB per game, TRB per game, AST per game, STL per game, BLK per game, TOV per game, PF per game, PTS per game             
FG%, 2P%, 3P%, FT%, eFG%   

- other stats:
Age, Games, Minutes per game, # Team Games, Team Win %          
Height_Inches, Weight, Prev All Stars

- non numerical:
Pos, Team

Target:
All Star 
'''
# want to use other for data exploration
df_preproc = df_clean.copy(deep=True)

# these columns are stats that are affected by era
era_stats = [
    "FGA per game", "2PA per game", "3PA per game", "FTA per game", "ORB per game", "DRB per game", 
    "TRB per game", "AST per game", "STL per game", "BLK per game", "TOV per game", "PF per game", "PTS per game", 
    "FG%", "2P%", "3P%","FT%", "eFG%"
]

for col in era_stats:
    df_preproc[col] = df_preproc.groupby('Season Ending Year')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x
        # this is equivalent to standard scaler
    )

other_stats = [
    "Age", "Games", "Minutes per game", "# Team Games", 
    "Team Win %", "Height_Inches", "Weight", "Prev All Stars"
]
global_scaler = StandardScaler()
df_preproc[other_stats] = global_scaler.fit_transform(df_preproc[other_stats])

print(df_preproc)

# 

categorical_cols = ['Pos', 'Team']

# One-Hot Encoding
df_preproc = pd.get_dummies(df_preproc, columns=categorical_cols, drop_first=True)

# verify new columns
print(f"New shape of dataframe: {df_preproc.shape}")
print(df_preproc.columns) 


# ------------------------------------------------------------------
# SPLITTING DATA
# ------------------------------------------------------------------

y = df_preproc['All Star']
X = df_preproc.drop(columns=['All Star', 'Player', 'Season Ending Year'])

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# should be 60%, 20%, 20%
print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
