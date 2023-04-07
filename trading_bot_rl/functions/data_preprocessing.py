from __future__ import annotations
import datetime
from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from trading_bot_rl.functions.yahoodownloader import YahooDownloader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# ----------------------------------------------------------------------------------------------------------------------------------

def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] <= end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data

# ----------------------------------------------------------------------------------------------------------------------------------

def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)
    
    
# ----------------------------------------------------------------------------------------------------------------------------------
def perform_date_cyclic_encoding(dataset):
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset = pd.concat([dataset.drop('date', axis=1), 
                         pd.DataFrame({'date_sin': np.sin(2 * np.pi * dataset['date'].dt.dayofyear / 365),
                                       'date_cos': np.cos(2 * np.pi * dataset['date'].dt.dayofyear / 365)})],
                       axis=1)
    current_order = dataset.columns.tolist()
    new_order = current_order[-2:] + current_order[:-2]
    dataset = dataset.reindex(columns=new_order)
    return dataset
                                 
# ----------------------------------------------------------------------------------------------------------------------------------

def numerical_columns_scaling(scaler_name, train, trade, valid = None):
    columns_to_scale = train.select_dtypes(include=['int', 'float']).columns.tolist()
    if scaler_name == 'MinMax':
        scaler = MinMaxScaler()
    elif scaler_name == 'Standard':
        scaler = StandardScaler()
    for col in columns_to_scale:
        train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))
        trade[col] = scaler.transform(trade[col].values.reshape(-1, 1))
        if valid != None: valid[col] = scaler.transform(valid[col].values.reshape(-1, 1))
    return train, trade, valid

# ----------------------------------------------------------------------------------------------------------------------------------                                 
                                 
def BOOL_TO_INT_FUNCTION(dataset):
    for i in range(len(dataset.columns)):
            if dataset[dataset.columns[i]].dtype == 'bool':
                dataset = dataset.astype({dataset.columns[i]:'int'})
    return dataset

# ----------------------------------------------------------------------------------------------------------------------------------    

def data_loading(df_file, tic_name):
    df = pd.read_csv(df_file)
    if 'tic' not in df.columns:
        df['tic'] = tic_name
    return df

# ----------------------------------------------------------------------------------------------------------------------------------    
    
def adding_vix_tirbulence_sort(fe, df_main):
    # Adding 'vix', 'turbulence' (if True)
    df_main = fe.preprocess_data(df_main)

    # Sorting columns
    f = (df_main.columns.tolist())
    for x in (['date', 'tic', 'open','high','low','close','volume']):
        f.remove(x)
    df_main = df_main[(['date', 'tic', 'open','high','low','close','volume']) + f]
    return df_main

# ----------------------------------------------------------------------------------------------------------------------------------

class FeatureEngineer:
    """Provides methods for preprocessing the stock price data
    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from neofinrl_config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            use user defined features or not
    Methods
    -------
    preprocess_data()
        main method to do the feature engineering
    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # clean data
        df = self.clean_data(df)

        # add technical indicators using stockstats
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add vix for multiple stock
        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def clean_data(self, data):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        # df = data.copy()
        # list_ticker = df["tic"].unique().tolist()
        # only apply to daily level data, need to fix for minute level
        # list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
        # combination = list(itertools.product(list_date,list_ticker))

        # df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
        # df_full = df_full[df_full['date'].isin(df['date'])]
        # df_full = df_full.sort_values(['date','tic'])
        # df_full = df_full.fillna(0)
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                        "date"
                    ].to_list()
                    # indicator_df = indicator_df.append(
                    #     temp_indicator, ignore_index=True
                    # )
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )
        df = df.sort_values(by=["date", "tic"])
        return df
        # df = data.set_index(['date','tic']).sort_index()
        # df = df.join(df.groupby(level=0, group_keys=False).apply(lambda x, y: Sdf.retype(x)[y], y=self.tech_indicator_list))
        # return df.reset_index()

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df

    def add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df_vix = YahooDownloader(
            start_date=df.date.min(), end_date=df.date.max(), ticker_list=["^VIX"]
        ).fetch_data()
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index

# ----------------------------------------------------------------------------------------------------------------------------------

def data_date_pct_split(df, test_and_valid_pct, valid_split):
    if valid_split:
        valid_number = int(test_and_valid_pct * (len(df)))
        test_number = valid_number
        train_number = len(df) - (valid_number + test_number)

        df_dates = df['date'].reset_index()
        df_dates = df_dates.drop('index', axis=1)

        df_dates_train = df_dates[0:train_number].reset_index().drop('index', axis=1)
        df_dates_valid = df_dates[train_number : train_number + valid_number].reset_index().drop('index', axis=1)
        df_dates_test = df_dates[-test_number:].reset_index().drop('index', axis=1)

        TRAIN_START_DATE = df_dates_train['date'][0]
        TRAIN_END_DATE = df_dates_train['date'][-1:].reset_index()['date'][0]

        TRADE_START_DATE = df_dates_test['date'][0]
        TRADE_END_DATE = df_dates_test['date'][-1:].reset_index()['date'][0]

        VALID_START_DATE = df_dates_valid['date'][0]
        VALID_END_DATE = df_dates_valid['date'][-1:].reset_index()['date'][0]
        
        return TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TRADE_START_DATE, TRADE_END_DATE
        
    else:
        test_number = int(test_and_valid_pct * (len(df)))
        train_number = len(df) - test_number

        df_dates = df['date'].reset_index()
        df_dates = df_dates.drop('index', axis=1)

        df_dates_train = df_dates[0:train_number].reset_index().drop('index', axis=1)
        df_dates_test = df_dates[-test_number:].reset_index().drop('index', axis=1)

        TRAIN_START_DATE = df_dates_train['date'][0]
        TRAIN_END_DATE = df_dates_train['date'][-1:].reset_index()['date'][0]

        TRADE_START_DATE = df_dates_test['date'][0]
        TRADE_END_DATE = df_dates_test['date'][-1:].reset_index()['date'][0]
        
        return TRAIN_START_DATE, TRAIN_END_DATE, None, None, TRADE_START_DATE, TRADE_END_DATE


# ----------------------------------------------------------------------------------------------------------------------------------

# df_main, df_forecasts, test_and_valid_pct, {'tic_name': 'SPY', 'use_vix': 'True/False','use_turbulence': 'True/False','user_defined_feature': 'True/False', 'tech_indicators_usage': 'True/False'}
def data_read_preprocessing_singleTIC(df_main_file,
                                      df_forecasts_file=None,
                                      test_and_valid_pct=0.1,
                                      tech_indicators_usage=False,
                                      use_vix=False,
                                      use_turbulence=False,
                                      user_defined_feature=False,
                                      tic_name='',
                                      valid_split=False,
                                      BOOL_TO_INT=True):
    
    fe = FeatureEngineer(
                        use_technical_indicator=tech_indicators_usage,
                        use_vix=use_vix,
                        use_turbulence=use_turbulence,
                        user_defined_feature = user_defined_feature)
    
    
    # Loading Data
    if df_forecasts_file != None:  df_raw_forecasts = data_loading(df_forecasts_file, tic_name)
    df_raw = data_loading(df_main_file, tic_name)
        
    # Sorting columns + adding 'vix', 'turbulence' (if true)
    if df_forecasts_file != None:
        df_main = adding_vix_tirbulence_sort(fe, df_raw_forecasts)
    else:
        df_main = adding_vix_tirbulence_sort(fe, df_raw)
                                 
     # Splitting by pct 
    if valid_split:
        TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TRADE_START_DATE, TRADE_END_DATE = data_date_pct_split(df_main, test_and_valid_pct, valid_split)
    else:
        TRAIN_START_DATE, TRAIN_END_DATE, _, _, TRADE_START_DATE, TRADE_END_DATE = data_date_pct_split(df_main, test_and_valid_pct, valid_split)

    #--------
    print('train ', TRAIN_START_DATE, ' ', TRAIN_END_DATE)
    if valid_split: print('valid ', VALID_START_DATE, ' ', VALID_END_DATE)
    print('trade ', TRADE_START_DATE, ' ', TRADE_END_DATE)
    #--------
    
    # Split data by 'date'
    train = data_split(df_main, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(df_main, TRADE_START_DATE, TRADE_END_DATE)
    if valid_split: valid = data_split(df_main, VALID_START_DATE, VALID_END_DATE)

    # Bools to 'int'
    if BOOL_TO_INT:
        train = BOOL_TO_INT_FUNCTION(train)
        trade = BOOL_TO_INT_FUNCTION(trade)
        if valid_split: valid = BOOL_TO_INT_FUNCTION(valid)     
        

    if df_forecasts_file != None:
        # then first dataset was forecasting -> saving results first
        train_forecasts = train
        trade_forecasts = trade
        if valid_split: valid_forecasts = valid
        
        # Sorting columns + adding 'vix', 'turbulence' (if true)
        df_main = adding_vix_tirbulence_sort(fe, df_raw) 
        
        #--------
        print('train ', TRAIN_START_DATE, ' ', TRAIN_END_DATE)
        if valid_split: print('valid ', VALID_START_DATE, ' ', VALID_END_DATE)
        print('trade ', TRADE_START_DATE, ' ', TRADE_END_DATE)
        #--------
        
        # Splitting data
        train = data_split(df_main, TRAIN_START_DATE, TRAIN_END_DATE)
        trade = data_split(df_main, TRADE_START_DATE, TRADE_END_DATE)
        if valid_split: valid = data_split(df_main, VALID_START_DATE, VALID_END_DATE)

        # Bools to 'int'
        if BOOL_TO_INT:
            train = BOOL_TO_INT_FUNCTION(train)
            trade = BOOL_TO_INT_FUNCTION(trade)
            if valid_split: valid = BOOL_TO_INT_FUNCTION(valid)    
    
    if df_forecasts_file != None:
        if valid_split: 
            return train, valid, trade, train_forecasts, valid_forecasts, trade_forecasts
        else: 
            return train, None, trade, train_forecasts, None, trade_forecasts
    else:
        if valid_split: 
            return train, valid, trade, None, None, None
        else: 
            return train, None, trade, None, None, None
                                 
# ----------------------------------------------------------------------------------------------------------------------------------                                 

# df_main, df_forecasts, test_and_valid_pct, {'tic_name': 'SPY', 'use_vix': 'True/False','use_turbulence': 'True/False','user_defined_feature': 'True/False', 'tech_indicators_usage': 'True/False'}
def data_read_preprocessing_singleTIC_normalized_encoded(df_main_file,
                                      df_forecasts_file=None,
                                      test_and_valid_pct=0.1,
                                      tech_indicators_usage=False,
                                      use_vix=False,
                                      use_turbulence=False,
                                      user_defined_feature=False,
                                      tic_name='',
                                      valid_split=False,
                                      BOOL_TO_INT=True,
                                      scaler='MinMax'):
    
    fe = FeatureEngineer(
                    use_technical_indicator=tech_indicators_usage,
                    use_vix=use_vix,
                    use_turbulence=use_turbulence,
                    user_defined_feature = user_defined_feature)
    
    
    # Loading Data
    if df_forecasts_file != None:  df_raw_forecasts = data_loading(df_forecasts_file, tic_name)
    df_raw = data_loading(df_main_file, tic_name)
        
    # Sorting columns + adding 'vix', 'turbulence' (if true)
    if df_forecasts_file != None:
        df_main = adding_vix_tirbulence_sort(fe, df_raw_forecasts)
    else:
        df_main = adding_vix_tirbulence_sort(fe, df_raw)
                                 
     # Splitting by pct 
    if valid_split:
        TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TRADE_START_DATE, TRADE_END_DATE = data_date_pct_split(df_main, test_and_valid_pct, valid_split)
    else:
        TRAIN_START_DATE, TRAIN_END_DATE, _, _, TRADE_START_DATE, TRADE_END_DATE = data_date_pct_split(df_main, test_and_valid_pct, valid_split)

    #--------
    print('train ', TRAIN_START_DATE, ' ', TRAIN_END_DATE)
    if valid_split: print('valid ', VALID_START_DATE, ' ', VALID_END_DATE)
    print('trade ', TRADE_START_DATE, ' ', TRADE_END_DATE)
    #--------
    
    # Split data by 'date'
    train = data_split(df_main, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(df_main, TRADE_START_DATE, TRADE_END_DATE)
    if valid_split: valid = data_split(df_main, VALID_START_DATE, VALID_END_DATE)

    # Scaling + Encoding
    train, trade, valid = numerical_columns_scaling(scaler, train, trade, valid) if valid_split else numerical_columns_scaling(scaler, train, trade)

    # Cyclic encoding of 'date' column
    train = perform_date_cyclic_encoding(train)
    trade = perform_date_cyclic_encoding(trade)
    if valid_split: valid = perform_date_cyclic_encoding(valid)

    # Bools to 'int'
    if BOOL_TO_INT:
        train = BOOL_TO_INT_FUNCTION(train)
        trade = BOOL_TO_INT_FUNCTION(trade)
        if valid_split: valid = BOOL_TO_INT_FUNCTION(valid)   

    if df_forecasts_file != None:
        # Saving data (if forecasted -> it was first)
        train_forecasts = train
        trade_forecasts = trade
        if valid_split: valid_forecasts = valid
        
        # Sorting columns + adding 'vix', 'turbulence' (if true)
        df_main = adding_vix_tirbulence_sort(fe, df_raw)
        
        #--------
        print('train ', TRAIN_START_DATE, ' ', TRAIN_END_DATE)
        if valid_split: print('valid ', VALID_START_DATE, ' ', VALID_END_DATE)
        print('trade ', TRADE_START_DATE, ' ', TRADE_END_DATE)
        #--------

        # Split data by 'date'
        train = data_split(df_main, TRAIN_START_DATE, TRAIN_END_DATE)
        trade = data_split(df_main, TRADE_START_DATE, TRADE_END_DATE)
        if valid_split: valid = data_split(df_main, VALID_START_DATE, VALID_END_DATE)

        # Scaling + Encoding
        train, trade, valid = numerical_columns_scaling(scaler, train, trade, valid) if valid_split else numerical_columns_scaling(scaler, train, trade)

        # Cyclic encoding of 'date' column
        train = perform_date_cyclic_encoding(train)
        trade = perform_date_cyclic_encoding(trade)
        if valid_split: valid = perform_date_cyclic_encoding(valid)

        # Bools to 'int'
        if BOOL_TO_INT:
            train = BOOL_TO_INT_FUNCTION(train)
            trade = BOOL_TO_INT_FUNCTION(trade)
            if valid_split: valid = BOOL_TO_INT_FUNCTION(valid)     
                                 
  
    if df_forecasts_file != None:
        if valid_split: 
            return train, valid, trade, train_forecasts, valid_forecasts, trade_forecasts
        else: 
            return train, None, trade, train_forecasts, None, trade_forecasts
    else:
        if valid_split: 
            return train, valid, trade, None, None, None
        else: 
            return train, None, trade, None, None, None
