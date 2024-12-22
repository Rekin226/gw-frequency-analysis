import sys
import os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import chisquare
import dateutil
import math
import jfft
import matplotlib.pyplot as plt
import pickle
import datetime
import gw_subroutine as sub

class SuccessFittingEnd(Exception):
    '''
    loop to try different bound assignment for fitting
    the exception is raised when the fitting can be finished 
    '''
    pass

def argv_phrase(argv: List):
    """
    Phrase your argv
    """
    assert isinstance(argv, list)

    argv_params: Dict = {}
    for content in argv:
        sepline = content.split("=")
        flag = sepline[0].lower()
        argv_params[flag] = sepline[1]

        if flag in ["tank_size"]:
            argv_params[flag] = int(argv_params[flag])
        if flag in ["recalibration", "daily"]:
            argv_params[flag] = True
    if "daily" is not argv_params:
        argv_params["daily"] = False     
    return argv_params


def data_preparation(argv_params):
    """
    Rainfall  & groundwater --> Merge
    PRS calculation
    """
    df_rainfall = pd.read_csv(argv_params["rainfall"])
    df_rainfall = df_rainfall.set_index("Date Time")
    df_rainfall.index = pd.to_datetime(df_rainfall.index)
    print('the length of df_rf', len(df_rainfall))
 

    # Groundwater
    # use absolute path
        # Use absolute path
    csv_file_path = os.path.abspath("../data/all_well_imputation_cleaned.csv")
    df_gw_all = pd.read_csv(csv_file_path)
    # Convert the 'date time' column to datetime and set it as the index
    df_gw_all['date time'] = pd.to_datetime(df_gw_all['date time'])
    df_gw_all.set_index('date time', inplace=True)

    # Start reading the data from '2012-01-01 00:00'
    df_gw_all = df_gw_all.loc['2012-01-01 00:00':'2020-12-31 23:00']

    # Remove columns with NaN values
    df_gw_all = df_gw_all.dropna(axis=1)

    # Drop ST_NO column with column head starting from 08 and 10
    df_gw_all = df_gw_all.loc[:, ~df_gw_all.columns.str.startswith('08')]
    df_gw_all = df_gw_all.loc[:, ~df_gw_all.columns.str.startswith('10')]

    # remove the first 0 for all columns in df_gw_all
    df_gw_all.columns = df_gw_all.columns.str.lstrip('0')
    
    #print(df_gw_all)


    # Read the active stations in df_gw_all using arv_params and ensure the header of the column and index are shown
    df_gw = df_gw_all.loc[:, [argv_params['groundwater']]]
    df_gw.columns = ['GWL (m)']
    print(df_gw)

    # AMP
    fs = 24  # 24 data points per day
    y = np.array(df_gw.loc[:, "GWL (m)"].values)
    rng = pd.date_range(
        start="2012-01-01 00:00",
        freq="1h",
        periods=int(len(df_gw)),
    )
    df = pd.DataFrame({"gwl (m)": y}, index=rng)
    # STFT
    mystft = jfft.STFT_Obj(y, fs=fs, framesz=30, hop=5, dt_list=rng)
    # get the index of 1cpd
    xfindex = mystft.find_xf_index(1)  # daily for 1.
    stft_result = mystft.get_yf()
    _stft_xf = mystft.get_xf()
    stft_timeval = mystft.get_timeval()

    df_prs = pd.DataFrame(stft_result[:, xfindex], index=stft_timeval)
    df_pumping = df_prs.resample("1h").interpolate()
    df_pumping.columns = ["AMP"]

    # AMT
    xfindex1 = mystft.find_xf_index(1.93)  # daily for 1.
    stft_result1 = mystft.get_yf()
    _stft_xf = mystft.get_xf()
    stft_timeval = mystft.get_timeval()       
    
    df1_tides = pd.DataFrame(stft_result1[:, xfindex1], index=stft_timeval)
    #print(df1_tides)
    df_tides = df1_tides.resample("1h").interpolate()
    df_tides.columns = ["AMT"]

    #df_merge0 = pd.concat(
    #    [df_groundwater, df_pumping],
    #    axis=1,
    #)
    #df_pumping = df_pumping.fillna(df_pumping.mean())
    if argv_params["daily"]:
        #df_merge = df_merge.resample("1d").mean()
        #print(df_merge)
        #df_rainfall = df_rainfall.resample("1d").sum()
        #print(df_rainfall)
        df_gw = df_gw.resample("1d").mean()
        print('the length of df_gw is : ', len(df_gw))
        df_pumping = df_pumping.resample("1d").mean()
        df_tides = df_tides.resample("1d").mean()
        #print(df_merge0)
        df_merge = pd.concat(
                    [df_rainfall, df_gw, df_pumping, df_tides],
                     axis=1,
                    )
        df_merge = df_merge.fillna(df_pumping.mean())
        df_merge = df_merge.fillna(df_tides.mean())
        print(df_merge)
    return df_merge

def model_fit_preprocess(df_merge: pd.DataFrame, argv_params: Dict) -> Tuple:
    """
    Pre-process of model fit
    """
    #print(df_merge)
    x = np.ones((df_merge.shape[0], 5))

    x[:, 0] = np.array(df_merge.loc[:, "Rainfall (mm)"].values)
    x[:, 1] = 0  # Qwell 1 = tides = 0
    x[:, 2] = 0  # Qwell 2 = tides = 0
    x[:, 3] = np.array(df_merge.loc[:, "AMP"].values)   #Qwell
    x[:, 4] = np.array(df_merge.loc[:, "AMT"].values)     #tides

    y = np.array(df_merge.loc[:, "GWL (m)"].values)
    h_init = np.zeros(argv_params["tank_size"])
    h_init[-1] = y[0]  # initial condition
    return x, y, h_init

bounds_try = [
    # try 1
    [
    ([0, 0, -40] , [1.9, 1.9, 110]),
    [0,18],
    [0,19],
    [-15,25]
    ],
    # try 2
    [
    ([0, 0, -40] , [1.9, 1.9, 100]),
    [0,14],
    [0,19],
    [-15,23]
    ],
    # try 3
    [
    ([0, 0, -30] , [1.8, 1.8, 100]),
    [0,13],
    [0,14],
    [-15,15]
    ],
    # try 4
    [
    ([0, 0, -20] , [1.8, 1.5, 100]),
    [0,10],
    [0,14],
    [-15,11]
    ],
    # try 5
    [
    ([0, 0, -28] , [1.5, 1.5, 100]),
    [0,12],
    [0,14],
    [-15,14]
    ],  
    # try 6
    [
    ([0, 0, -20] , [1.5, 1.5, 100]),
    [0,10],
    [0,14],
    [-15,12]
    ],  
    # try 7
    [
    ([0, 0, -20] , [1, 1, 100]),
    [0,10],
    [0,14],
    [-15,13]
    ],
    # try 8
    [
    ([0, 0, -20] , [1, 1, 100]),
    [0,10],
    [0,14],
    [-15,11]
    ],
    # try 9
    [
    ([0, 0, -20] , [0.8, 0.8, 100]),
    [0,10],
    [0,14],
    [-15,12]
    ],
    # try 10
    [
    ([0, 0, -20] , [0.7, 0.7, 100]),
    [0,10],
    [0,14],
    [-15,12]
    ],
    # try 11
    [
    ([0, 0, -15] , [0.7, 0.7, 100]),
    [0,10],
    [0,14],
    [-15,11]
    ],
    # try 12
    [
    ([0, 0, -20] , [0.5, 0.5, 100]),
    [0,10],
    [0,14],
    [-15,11]
    ],
    # try 13
    [
    ([0, 0, -20] , [0.2, 0.2, 100]),
    [0,10],
    [0,5],
    [-15,11]
    ]
    ]

def prepare_curve_fit(argv_params: Dict, try_index : int = 0):
    '''
    prepare bounds and p0 (initial guess) for curve fitting 
    '''
    #(a,b,z,c,d,z_tides)
    bounds_range = bounds_try[try_index][0] #a,b,z
    bounds_c=bounds_try[try_index][1]
    bounds_d=bounds_try[try_index][2]
    bounds_z_tides=bounds_try[try_index][3]
    #c,d,z_tides

    k=0
    tank_size = argv_params["tank_size"]
    variable_size = tank_size*len(bounds_range[0])+3                #3個水桶總共參數3+3+6=12             
    p0 = np.random.rand(variable_size)      #12個參數隨機預測            
    bounds = np.zeros((2, variable_size))   #2行，每行12個，每次歸零重新計算
    for i in range(len(bounds_range[0])):
        for j in range(tank_size):    
            p0[k] = p0[k] * ((bounds_range[1][i] - bounds_range[0][i]) + bounds_range[0][i])#在範圍內隨機預測(不確定)
            bounds[:, k] = [
                            bounds_range[0][i],
                            bounds_range[1][i]
                                ]               #參數預測
            k += 1 
    #c、d、z_tides_bounds
    p0[k] = p0[k] * ((bounds_c[1] - bounds_c[0]) + bounds_c[0])
    bounds[:, k] = [
                    bounds_c[0],
                    bounds_c[1]       
                    ]                  
    k += 1 
    p0[k] = p0[k] * ((bounds_d[1] - bounds_d[0]) + bounds_d[0])
    bounds[:, k] = [
                    bounds_d[0],
                    bounds_d[1]       
                    ]               
    k += 1 
    p0[k] = p0[k] * ((bounds_z_tides[1] - bounds_z_tides[0]) + bounds_z_tides[0])
    bounds[:, k] = [
                    bounds_z_tides[0],
                    bounds_z_tides[1]       
                    ]                   #第三個tank的三個參數預測
    k += 1 

    #print(bounds)
    return p0, bounds

def model_fit_predict(df_merge, argv_params):
    df_merge = df_merge.interpolate()
    #df_merge = df_merge.head(30)
    #print(df_merge)
    
    # prepare the four inputs columns x, include upstream, tides, rainfall and pumping
    dt1 = datetime.datetime.now()
    x, y, h_init = model_fit_preprocess(df_merge, argv_params)
    dt = 1
    if argv_params['daily']:
        dt = 24
    def MLR_hinit(
        x,
        *parameters
        
    ):
        """
        to assign the initial condition of each tanks
        分配每個水筒初始條件
        """
        #print(parameters)
        return sub.gw_model_shell(
            x, *parameters, h_init=h_init, dt=dt
        )
    # parameters estimation
    # curve fit

    p0, bounds = prepare_curve_fit(argv_params)
    if "p0" in argv_params:
        p0 = argv_params["p0"]

    #print("the bounds list is: {}".format(bounds))
    #print("Parameter searching bounds (upper): {}".format(bounds[1, :]))
    #print("                            (lower): {}".format(bounds[0, :]))
    #print("Initial point: {}".format(p0))
    #print(df_merge)
    try:
        params, _covs = curve_fit(MLR_hinit, x, y, p0=p0, bounds=bounds)
    except (RuntimeError, ValueError) as e:
            # random, re-create p0
            try:
                # params for bounds assignment
                for try_index in range(len(bounds_try)):
                    try:
                        print("Try index: {}".format(try_index))
                        p0, bounds = prepare_curve_fit(argv_params, try_index=try_index)
                        params, _covs = curve_fit(
                            MLR_hinit, x, y, p0=p0, bounds=bounds
                        )
                        # finishing fitting
                        raise SuccessFittingEnd
                    except (RuntimeError, ValueError):
                        pass
            except SuccessFittingEnd:
                pass
  

    print("Optimal params are: {}".format (params))

    ypred = sub.gw_model_shell(x, *params, h_init=h_init, dt=dt)
    #print(ypred)
    df_merge.loc[:, "prediction"] = ypred
    #print(df_merge)
    dt2 = datetime.datetime.now()
    #print('The time consumed is : ', dt2-dt1)

    return df_merge, params, p0

def show_calibrated_params(params_pack: Dict, tanksize: int):
    """
    Calibrated parameters demonstration
        export formated parameters
    """
    params = {}
    pname_list = [
        "a1", "a2", "a3",
        "b1", "b2", "b3",
        "z1", "z2", "z3",
        "c",
        "d",
        "z_tides",
    ]
    params_val = np.asarray(params_pack["calibrated_params"])

    for i in range(len(params_val)):
        params[pname_list[i]] = params_val[i]

    return params

def rmse(val1, val2) -> float:
    return np.power(np.mean(np.power(val1 - val2, 2)), 0.5)


# Absolute mean error
def mae(val1, val2) -> float:
    return np.mean(np.absolute(val1 - val2))


def me(val1, val2) -> float:
    return np.mean(val1 - val2)

def post_performance_calculation(df_merge: pd.DataFrame, params) -> Dict:
    """
    1. determine r2, RSME, MAE, ME
    2. pack params
    """
    # Post analysis
    r2 = round(
        r2_score(df_merge["GWL (m)"], df_merge["prediction"]),
        2,
    )

    Rmse = round(
        rmse(
            df_merge["GWL (m)"],
            df_merge["prediction"],
        ),
        4,
    )

    MAE = round(
        mae(df_merge["GWL (m)"], 
            df_merge["prediction"]
        ),
        3,
    )

    ME = round(
        me(df_merge["GWL (m)"],
           df_merge["prediction"]
        ),
        3,
    )


    params_pack = {
        "calibrated_params": params,
        "r2": r2,
        "RMSE": Rmse,
        "MAE": MAE,
        "ME": ME,
    }
    return params_pack

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def model_process(argv_params):
    #df_merge = data_preparation(argv_params)
    #df_merge = df_merge.head(30)

    print(
        {
            flag: argv_params[flag]
            for flag in [
                "groundwater",
                "rainfall",
                "recalibration",
                "daily",
            ]
            if flag in argv_params
        }
    )

    # determining station name for groundwater and rainfall from the filename
    name1 = os.path.basename(argv_params["groundwater"]).split(".")[0]
    name2 = os.path.basename(argv_params["rainfall"]).split(".")[0]
    print("  --> {} / {}".format(name1, name2))
    parameter_fname = (
        os.path.join(  # filename for the calibrated parameter file
            "tides_stations", "{}-{}-tides.pickle".format(name1, name2)
        )
    )

    # 內部 daily 改為日資料
    df_merge = data_preparation(argv_params)
    #df_merge = df_merge.head(92)
    #df_merge = df_merge.loc['2017-01-01':'2017-04-01']
    #print(df_merge)
    #sys.exit()

    # save df_merge to csv as the station name
    df_merge.to_csv('tides_stations/df_tides-{}.csv'.format(name1))

    param_previous: Dict = {}
    params = ()
    p0 = ()
    dt = 1
    if argv_params["daily"]:
        dt = 24
    if os.path.exists(parameter_fname):
        # load calibrated parameter
        with open(parameter_fname, "rb") as f:
            param_previous = pickle.load(f)
        if argv_params.get("recalibration", False):
            #####################################################
            # Model fitting
            df_merge, params, p0 = model_fit_predict(df_merge, argv_params)
        else:
            params = param_previous["calibrated_params"]

        # predict
        x, _y, h_init = model_fit_preprocess(df_merge, argv_params)
        df_merge.loc[:, "prediction"] = sub.gw_model_shell(
            x, *params, h_init=h_init, dt=dt
        )
    else:
        #####################################################
        # Model fitting
        df_merge, params, p0 = model_fit_predict(df_merge, argv_params)

        # predict
        x, _y, h_init = model_fit_preprocess(df_merge, argv_params)
        df_merge.loc[:, "prediction"] = sub.gw_model_shell(
            x, *params, h_init=h_init, dt=dt
        )

    # Post analysis
    params_pack = post_performance_calculation(df_merge, params)

    # save to file
    # use pickle dump for dict type data
    # https://clay-atlas.com/blog/2020/03/28/使用-pickle-模組保存-python-資料/
    if params_pack["r2"] >= param_previous.get("r2", -999999999):
        # 輸出
        perform_list = {
            "r2": params_pack["r2"],
            "RMSE": params_pack["RMSE"],
            "MAE": params_pack["MAE"],
            "ME": params_pack["ME"],
        }
        best_params = show_calibrated_params(params_pack, 1)
        best_perform = Merge(perform_list, best_params)
        message = "# {} / {}\n".format(name1, name2)
        message += "  Calibrated parameters: {}, r2: {}".format(
            show_calibrated_params(params_pack, 1),
            params_pack["r2"],
        )
        print(message)
        try:
            with open(parameter_fname, "wb") as f:
                pickle.dump(params_pack, f)
        except:
            # if the folder 'Results' does not exist, create the folder
            os.makedirs("tides_stations", exist_ok=True)
            with open(parameter_fname, "wb") as f:
                pickle.dump(params_pack, f)
    else:
        perform_list = {
            "r2": param_previous["r2"],
            "RMSE": param_previous["RMSE"],
            "MAE": param_previous["MAE"],
            "ME": param_previous["ME"],
        }
        best_params = show_calibrated_params(param_previous, 1)
        best_perform = Merge(perform_list, best_params)
        message = "# {} / {}\n".format(name1, name2)
        message += "  Calibrated parameters (previous): {}, r2: {}".format(
            show_calibrated_params(param_previous, 1),
            param_previous["r2"],
        )
        print(message)

    # print('---> the best performance is :', best_perform)
    best_params_list = [i for i in best_params.values()]
    # print('--->the best params list', best_params_list)

    # use best performance parameters
    df_merge.loc[:, "best pred"] = sub.gw_model_shell(
        x, *best_params_list, h_init=h_init, dt=dt
    )

    best_perform.update(
        {
            "mean gw": df_merge["GWL (m)"].mean(),
            "mean gw_pred": df_merge["best pred"].mean(),
            #"mean rf": df_merge["Rainfall (mm)"].mean(),
            "mean AMP": df_merge["AMP"].mean(),
            "mean AMT": df_merge["AMT"].mean(),
        }
    )

  
    #df_merge, params, p0 = model_fit_predict(df_merge, argv_params)

    
    dt1 = datetime.datetime.now()
     #plot--------------------------------------------   


    _fig, ax = plt.subplots(4, figsize=(12, 9), sharex=True)

        #set figure as gwl and rainfall name 
    name_gwl = os.path.basename(argv_params["groundwater"]).split(".")[0]
    name_rain = os.path.basename(argv_params["rainfall"]).split("_")[0]

        # plot rainfall at ax[0] ; plot GWL ax ax[1], obs. & pred.
    df_merge["GWL (m)"].plot(ax=ax[0], grid=True)
    df_merge["best pred"].plot(ax=ax[0], grid=True)
    df_merge["Rainfall (mm)"].plot(ax=ax[1], grid=True)
    df_merge["AMP"].plot(ax=ax[2],grid=True)
    df_merge["AMT"].plot(ax=ax[3],grid=True)
    

    R2 = round(r2_score(df_merge['GWL (m)'],df_merge['best pred']),3)
    RMSE = round(rmse(df_merge['GWL (m)'], df_merge['best pred']), 3)
    MAE = round(mae(df_merge['GWL (m)'], df_merge['best pred']), 3)
    ME = round(me(df_merge["GWL (m)"], df_merge["best pred"]), 3)
    

           
    #print('r2 = {}\nrmse = {}\nmae = {}\nme = {}'.format(R2, RMSE, MAE, ME))
    legend = ["Observation"]
    legend.append("Prediction")

    ax[0].set_title("{}-{}".format(name_gwl, name_rain))
    ax[0].set_title("r2={},RMSE={},MAE={}".format(R2, RMSE, MAE), loc="left",fontsize=10)
    ax[3].set_xlabel("Time")
    ax[0].set_ylabel(r"GWL & prediction $m$")
    ax[1].set_ylabel(r"Rainfall $mm$")
    ax[2].set_ylabel(r"AMP(t) ")
    ax[3].set_ylabel(r"AMT(t) ")
    ax[0].legend(
        legend, loc="best", fancybox=True, shadow=True
    )
    plt.savefig('tides_stations/{}-{}.png'.format(name_gwl, name_rain))
    plt.show()
    dt2 = datetime.datetime.now()
    print('The time consumed is : ', dt2-dt1)
    return best_perform

if __name__ == '__main__':
    argv_params = argv_phrase(sys.argv[1:])
    #print(argv_params)
    #df_merge = data_preparation(argv_params)
    #print(df_merge)
    #df_merge, params, p0 = model_fit_predict (df_merge, argv_params)
    #print(df_merge["prediction"])
    model_process(argv_params)
   