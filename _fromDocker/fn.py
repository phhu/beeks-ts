from sklearn.preprocessing import MinMaxScaler
from toolz import curry
import pandas as pd
import numpy as np
import glob
import os

@curry
def addExogenous(marketOpenField,df):
  # for i in range(7):
  #   df[f'day{i}'] = 1 * (pd.to_datetime(df["ds"]).dt.weekday == i)
  df["marketOpen"] = ( 1-df[marketOpenField].isna() ).astype(int)
  #df["random"] = np.random.randint(0,5, size=len(df))
  #df["random2"] = np.random.randint(0,5, size=len(df))
  return df

readFile = lambda fn: pd.read_csv(fn,header=1)  

# addDateTime
@curry
def addDateTime(srcCol,unit,df):
  df.insert(0,"ds",pd.to_datetime(df[srcCol], unit=unit))
  # df['ds'] = pd.to_datetime(df[srcCol], unit=unit)
  return df
  

def addDateTimeAuto(df):
  # print("*** cols", df.columns)
  actualCol = "time" if "time" in df.columns else "time".capitalize()
  unit = "ns" if df[actualCol][1] > 171242214000000 else "ms"
  df.insert(0,"ds",pd.to_datetime(df[actualCol], unit=unit))
  #df['ds'] = pd.to_datetime(df[actualCol], unit=unit)
  return df

scaler = MinMaxScaler(feature_range=(0, 1))

@curry
def rescale (col, newCol, df):
  df[newCol] = scaler.fit_transform(df[col].values.reshape(-1,1))
  return df


@curry
def splitByDate(splitDate, df):
  return (
    df.loc[df['ds'] <= splitDate],
    df.loc[df['ds'] > splitDate]
  )
keep = lambda cols: lambda df: df[cols]
dropSeriesFromTestData = lambda y: lambda df: (df[0], df[1].drop(y,axis=1))

@curry
def logTransform (y,df):
  df[y] = df[y].replace({'0':0.00001, 0:0.00001})
  df[y] = np.log(df[y])
  return df

@curry
def expTransform (df):
  cols_to_transform = [col for col in df if col not in ['unique_id', 'ds']]
  for col in cols_to_transform:
      df[col] = np.exp(df[col])
  return df

def readAllFiles(fileSpec2):
  return pd.concat(
    pd.read_csv(f, skiprows=1) for f in glob.glob(fileSpec2)
  )   #,ignore_index=True
