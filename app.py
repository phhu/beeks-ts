from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import plotly.express as px
import pandas as pd
import fn
from toolz.functoolz import pipe
import glob
from sklearn.preprocessing import MinMaxScaler 
from utilsforecast.plotting import plot_series

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
# from neuralforecast.utils import AirPassengersDF

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.linear_model import LinearRegression

from nixtla import NixtlaClient
nixtla_client = NixtlaClient(api_key = 'nixtla-tok-aCYd8dJ44XStxX5iyAxWmaBivQ1ustAC8dpj1C4ZEPI8wM1VuQ9KDyoCIibC3oX6XaUG4Q5Q9l1jQ7io')

from prophet import Prophet
from prophet.plot import plot_plotly

from neuralprophet import NeuralProphet

from sklearn.model_selection import train_test_split

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

def loadDf(filename):
  print("loadDf: " + filename)
  return pipe(
    #fn.readAllFiles('./data/beeksai/historical_market_data_stats/*.csv') 
    #,fn.readFile("./data/TripleWitching-2024-05-06 15_56_54.csv")
    # pd.read_csv('./data/beeksai/skew.csv'),  lambda df: df.rename(columns={'mean_adapters_0_timeSyncTimeSkewNs': 'y'})
    pd.read_csv(filename,skiprows=1),
    #pd.read_csv('./data/TripleWitching-2024-05-06 15_56_54.csv',skiprows=1), lambda df: df.rename(columns={'FINRA_tdds_A': 'y'})
    lambda df:df._get_numeric_data(),
    fn.addDateTimeAuto,
    #,fn.rescale("y","ys")
    #,fn.keep(["ds","y","ys"])
    # fn.addExogenous("y"),
    lambda df: df.fillna(0),
    #,fn.logTransform("y")
  )

# https://stephenallwright.com/scale-columns-traing-score/
def scale_columns(df, columns, scalers):
  if scalers is None:
    scalers = {}
    for col in columns:
      try:
        scaler = MinMaxScaler().fit(df[[col]])
        df[col] = scaler.transform(df[[col]])
        scalers[col] = scaler
      except:
       print("error scaling column: " + col)
  else:
    for col in columns:
      scaler = scalers.get(col)
      df[col] = scaler.transform(df[[col]])
  return df, scalers

def getColumns(df):
    # print("getting columns: " + ", ".join(list(df)))
    return  [c for c in list(df) if c not in ["Time", "time", "ds","name"]]

def filterDf(df, column):
    return df[['ds', column]]

app = Dash()

files = glob.glob("./data/beeksai/historical_market_data_stats/*.csv") + \
glob.glob("./data/beeksai/*.csv") + \
glob.glob("./data/*.csv")

defaultFile = "./data/TripleWitching-2024-05-06 15_56_54.csv"   # files[0]

def update(file):
    df = loadDf(file)
    df_scaled,scalers = scale_columns(df.copy(),columns=getColumns(df),scalers=None)
    columns = getColumns(df)
    return df, df_scaled, columns

df, df_scaled, columns = update(defaultFile)
defaultColumn = "NASDAQ_Canadian_Chix_A" # columns[0]

scaling = ["No scaling","0 to 1"]

app.layout = [
    html.H1(children='Timeseries charts', style={'textAlign':'center'}),
    #dcc.Dropdown(["y","ys"], "y", id='sel-y', style={"width":"40%"}),
    dcc.Dropdown(files, defaultFile, id='sel-file'),   
    dcc.Dropdown(columns, defaultColumn, id='sel-column'), 
    dcc.Dropdown(scaling, scaling[0], id='sel-scaling'),   
    
    html.Button('Update data', id='button-update', n_clicks=0),     # , style={"display": "none"}
    #dcc.Input(id="check-rescale", type="checkbox", value="resscale"), 
    dcc.Graph(id='plot-timeseries' ), 
    
    html.Button('Update nixtla', id='button-update-forecasts', n_clicks=0),
    dcc.Markdown('## Anomolies against Nixtla TimeGPT forecast', id="plot-nixtla-anomoly-title"),
    dcc.Graph(id='plot-nixtla-anomoly' ),
    dcc.Markdown('## Nixtla TimeGPT forecast', id="plot-nixtla-forecast-title"),
    dcc.Graph(id='plot-nixtla-forecast' ),

   # html.Div([
    html.Div([
      html.Div([html.Button('Update Prophet', id='button-update-prophet', n_clicks=0)]),    
      dcc.Markdown('## Prophet Forecast', id="plot-prophet-title"),
      dcc.Graph(id='plot-prophet' ),
    ]),
    
    html.Div([
      html.Div([html.Button('Update NP', id='button-update-np', n_clicks=0)]),    
      dcc.Markdown('## Neural Prophet Forecast', id="plot-neuralprophet-title"),
      dcc.Graph(id='plot-neuralprophet' ),
    ]),   
   #],style = {"columnCount":2}),
  
    html.Div([
      html.Div([html.Button('Update neuralforecast NBEATS', id='button-update-neuralforecast_nbeats', n_clicks=0)]),    
      dcc.Markdown('#### NBEATS neural forecast', id="title-neuralforecast_nbeats"),
      dcc.Graph(id='plot-neuralforecast_nbeats' ),
    ]),   
    
    html.Div([
      html.Div([html.Button('Update mlforecast linreg', id='button-update-mlforecast', n_clicks=0)]),    
      dcc.Markdown('#### MLforecast', id="title-mlforecast"),
      dcc.Graph(id='plot-mlforecast' ),
    ]),      
    
    html.Div([
      dcc.Graph(id='plot-empdist'),
      dcc.Graph(id='plot-empcdf' ),
    ], style = {"columnCount":2}),
    dash_table.DataTable(
        data=df.to_dict('records'), 
        page_size=10,
        sort_action="native",
        sort_mode="multi",
        id="datatable"
    ),
    html.Pre("Head:\n" + df.head().to_string(),id="data-head"),
    html.Pre("Tail:\n" + df.tail().to_string(),id="data-tail"),
]

@callback( 
    Output('sel-column', 'options'), 
    Input('sel-file', 'value'),
)
def updateColumns(file):
    global columns
    global df
    global df_scaled
    # df = loadDf(file)
    # columns = getColumns(df)
    df, df_scaled, columns = update(file)
    # print("updateColumns: " + ",".join(columns))
    return columns


def getDf (scaling):
    return df_scaled if scaling == "0 to 1" else df

# DATA SUMMARY
@callback(
    [
      Output('plot-timeseries', 'figure'),
      Output('plot-empdist', 'figure'),
      Output('plot-empcdf', 'figure'),
      Output('data-head', 'children'),
      Output('data-tail', 'children'),
      Output('datatable', 'data'),
      #Output('plot-nixtla-anomoly', 'figure')
    ],[
      Input('button-update', 'n_clicks'),
      Input('sel-column', 'value'), 
      # Input('sel-file', 'value'), 
      Input('sel-scaling','value') ,
    ],[
     #State('sel-column', 'value'), 
     State('sel-file', 'value'), 
     #State('sel-scaling','value') 
    ],
)
def update_output(n_clicks, column, file, scaling):
    print ("updateOutput", column, n_clicks , file, scaling)
    ds = getDf (scaling)
    d = pipe(
      ds,
      lambda df: df.rename(columns={column: 'y'}),
      fn.keep(["ds","y"]),
      #lambda d: d.fillna(0)
    )    
    d0 = d.fillna(0)
    d_droppedNa = d0.query('y!=0')    # for distributions, assume zero not meaningful (needs UI switch)
    return [
      px.line(ds, x='ds', y=column, title="Timeseries: " + column + " from " + file),    #timeseries
      px.histogram(d_droppedNa, x="y", title="Empirical distribution: " + column),     # plot-empdist
      px.ecdf(d_droppedNa, x="y", title="Empirical cumulative distribution:" + column)  ,      # plot-empcdf
      "Head: " + column + " " + file + "\n" + df.head().to_string(),
      "Tail: " + column + " " + file + "\n" + df.tail().to_string(),
      filterDf(df, column).to_dict('records'), # datatable
      # nixtla_client.plot(d, nixtla_client.detect_anomalies(d.fillna(0), freq='h'),engine='plotly') , 
    ] 
    
@callback(
    [
      Output('plot-neuralprophet-title', 'children'),
      Output('plot-neuralprophet', 'figure'),
    ],[
      Input('button-update-np', 'n_clicks'),
    ],[
     State('sel-column', 'value'), 
     State('sel-file', 'value'), 
     State('sel-scaling','value'),
    ],
    prevent_initial_call=True
)
def update_forcast_np(n_clicks, column, file, scaling):
    print ("updateForecast_np", column, n_clicks , file, scaling)
    ds = getDf (scaling)
    d = pipe(
      ds,
      lambda df: df.rename(columns={column: 'y'}),
      fn.keep(["ds","y"]),    # column
      #lambda d: d.fillna(0)
    )    
    d0 = d.fillna(0)
    #d_droppedNa = d0.query('y!=0')    # for distributions, assume zero not meaningful (needs UI switch)

    mNeuralProphet = NeuralProphet()   
    mNeuralProphet.set_plotting_backend("plotly")     
    mNeuralProphet.fit(d0, freq="h", epochs=20)
    future = mNeuralProphet.make_future_dataframe(periods=24*60, df=d0)
    npForecast = mNeuralProphet.predict(future)
 
    return [
      '#### Neural Prophet forecast: '  + column + "  -  " + file,
      mNeuralProphet.plot(npForecast),
    ]      

# ** neuralforecast_nbeats
@callback(
    [
      Output('title-neuralforecast_nbeats', 'children'),
      Output('plot-neuralforecast_nbeats', 'figure'),
    ],[
      Input('button-update-neuralforecast_nbeats', 'n_clicks'),
    ],[
     State('sel-column', 'value'), 
     State('sel-file', 'value'), 
     State('sel-scaling','value'),
    ],
    prevent_initial_call=True
)
def update_neuralforecast_nbeats(n_clicks, column, file, scaling):
    print ("update neuralforecast_nbeats", column, n_clicks , file, scaling)
    ds = getDf (scaling)
    d = pipe(
      ds,
      lambda df: df.rename(columns={column: 'y'}),
      fn.keep(["ds","y"]),    # column
      #lambda d: d.fillna(0)
    )    
    d0 = d.fillna(0)
    d0['unique_id'] = 1
  
    nf = NeuralForecast(
        models = [NBEATS(input_size=24, h=400, max_steps=100)],
        freq = 'h'
    )
    nf.fit(df=d0)
    pred = nf.predict()
    combinedDf =  pd.concat([d0, pred])

    return [
      '#### NBEATS neural forecast: '  + column + "  -  " + file,
      px.line(combinedDf, x="ds", y=["y","NBEATS"])
      # px.line(pred),
    ]      

# mlforecast
@callback(
    [
      Output('title-mlforecast', 'children'),
      Output('plot-mlforecast', 'figure'),
    ],[
      Input('button-update-mlforecast', 'n_clicks'),
    ],[
     State('sel-column', 'value'), 
     State('sel-file', 'value'), 
     State('sel-scaling','value'),
    ],
    prevent_initial_call=True
)
def update_mlforecast(n_clicks, column, file, scaling):
    print ("update mlforecast", column, n_clicks , file, scaling)
    ds = getDf (scaling)
    d = pipe(
      ds,
      lambda df: df.rename(columns={column: 'y'}),
      fn.keep(["ds","y"]),    # column
      #lambda d: d.fillna(0)
    )    
    d0 = d.fillna(0)
    d0['unique_id'] = 1
  
    fcst = MLForecast(
        models=LinearRegression(),
        freq='h',  # our serie has a monthly frequency
        lags=[24*7,24*7*2,24*7*3,24*7*4],
        target_transforms=[Differences([24*7])],
    )
    fcst.fit(d0)
    preds = fcst.predict(24*7*3)
    fig = plot_series(d0, preds, engine="plotly")

    return [
      '#### mlforecast linreg: '  + column + "  -  " + file,
      fig
    ]      

# ** PROPHET
@callback(
    [
      Output('plot-prophet-title', 'children'),
      Output('plot-prophet', 'figure'),
    ],[
      Input('button-update-prophet', 'n_clicks'),
    ],[
     State('sel-column', 'value'), 
     State('sel-file', 'value'), 
     State('sel-scaling','value'),
    ],
    prevent_initial_call=True
)
def update_forcasts(n_clicks, column, file, scaling):
    print ("updateForecast", column, n_clicks , file, scaling)
    ds = getDf (scaling)
    d = pipe(
      ds,
      lambda df: df.rename(columns={column: 'y'}),
      fn.keep(["ds","y"]),    # column
      #lambda d: d.fillna(0)
    )    
    d0 = d.fillna(0)
    #d_droppedNa = d0.query('y!=0')    # for distributions, assume zero not meaningful (needs UI switch)


    mProphet = Prophet(changepoint_prior_scale=0.01)
    mProphet.add_seasonality(name='hourly', period=1/24, fourier_order=5)    
    mProphet.fit(d)
    future = mProphet.make_future_dataframe(periods=24*60,freq='h')
    pForecast = mProphet.predict(future)

    
    return [
      '#### Prophet forecast: '  + column + "  -  " + file,
      plot_plotly(mProphet,pForecast),
    ]      

# ** NIXTLA
@callback(
    [
      Output('plot-nixtla-anomoly-title', 'children'),
      Output('plot-nixtla-anomoly', 'figure'),
      Output('plot-nixtla-forecast-title', 'children'),
      Output('plot-nixtla-forecast', 'figure'),
    ],[
      Input('button-update-forecasts', 'n_clicks'),
    ],[
     State('sel-column', 'value'), 
     State('sel-file', 'value'), 
     State('sel-scaling','value'),
    ],
    prevent_initial_call=True
)
def update_forcasts(n_clicks, column, file, scaling):
    print ("updateForecast", column, n_clicks , file, scaling)
    ds = getDf (scaling)
    d = pipe(
      ds,
      lambda df: df.rename(columns={column: 'y'}),
      fn.keep(["ds","y"]),    # column
      #lambda d: d.fillna(0)
    )    
    d0 = d.fillna(0)
    #d_droppedNa = d0.query('y!=0')    # for distributions, assume zero not meaningful (needs UI switch)

    (train,test) = pipe(d
      ,fn.addExogenous("y")
      ,lambda df: df.fillna(0)
      ,lambda df: train_test_split(df, test_size=0.2, shuffle=False)
      #,fn.splitByDate("2024-04-15 00:00:00")
      ,fn.dropSeriesFromTestData("y")
    )
    
    print("train:")
    print(train.head(8))
    print("test:")
    print(test.head(8))
        
    fcst_df = pipe((train,test)
      ,lambda dp: nixtla_client.forecast(
        dp[0]
        ,X_df=dp[1]
        ,h=test.shape[0]    # number of rows in dpx
        ,model='timegpt-1-long-horizon'
        #,time_col="ds"
        #,target_col="NASDAQ_Canadian_Chix_A"
        ,level=[80,90]
        ,finetune_steps=30
        #,finetune_loss='mae', 
      )
      ,lambda df: print(df.head()) or df
      #,fn.expTransform
    )    

    return [
      '#### Anomolies against Nixtla TimeGPT forecast: ' + column + "  -  " + file,
      nixtla_client.plot(d, nixtla_client.detect_anomalies(d.fillna(0), freq='h', target_col="y",),engine='plotly') ,
      '#### Nixtla TimeGPT forecast with exogenous marketOpen data: ' + column + "  -  " + file,
      nixtla_client.plot(d, fcst_df, level=[80,90] ,engine='plotly'),
    ]      
    
 

# app.css.append_css({
#     'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# })

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")      # host 0.0.0.0 for ease of docker