"""
Parallize GW linear reservoir model 

1. Parallelize (done)
2. create a new system for different situation
    a. calibrate parameters and save the parameter(time consume)
        when the saved parameter file does not exist
    b. use the saved parameter file and plot (fast)
        when the saved parameter file exists


    c. re-calibrate and re-save
        if the previus results is worse then the newer one, save the new version
        if the  .....             better then pass

        force to re-calibrate
"""
# https://anaconda.org/conda-forge/py-cpuinfo
# https://anaconda.org/conda-forge/multiprocess
from cpuinfo import get_cpu_info
from multiprocessing import Pool
import sys
from typing import List, Dict
import datetime
import pandas as pd
import gw_model_shell as glrs
#import test_val_test as vm

log_parallel = True  # False means sequence mode, True means parallel mode


def argv_phrase(argv: List):
    """
    Phrase your argv
    """
    assert isinstance(argv, list)

    argv_params: Dict = {}
    for content in argv:
        sepline = content.split("=")
        flag = sepline[0].lower()
        try:
            argv_params[flag] = sepline[1]
        except IndexError:
            argv_params[flag] = flag

        if flag in ["tank_size", "parallel_size"]:
            # str to int
            argv_params[flag] = int(argv_params[flag])
        elif flag in ["groundwater", "rainfall"]:
            argv_params[flag] = argv_params[flag].split(",")
        
        if flag in ["recalibration", "daily"]:
            argv_params[flag] = True
    if "daily" not in argv_params:
        argv_params["daily"] = False
    return argv_params


if __name__ == "__main__":
    argv_params = argv_phrase(sys.argv[1:])
    print('===>', argv_params)

    cpu_item = get_cpu_info()
    # 60% of CPU power or define by our self
    print("CPU core: {}".format(cpu_item["count"]))
    process_size = argv_params.get(
        "parallel_size",
        max(
            int(float(cpu_item["count"]) * 0.6), 1
        ),  # 60% of CPU cores, default value
    )
    #print("Process size (in parallel process): {}".format(process_size))

    # check before running
    for flag in ["groundwater", "rainfall"]:
        assert isinstance(argv_params[flag], list)
    assert len(argv_params["groundwater"]) == len(argv_params["rainfall"])

    # prepare the arguments for each
    flags = []
    for i in range(len(argv_params["groundwater"])):
        flags.append(
            {
                "groundwater": argv_params["groundwater"][i],
                "rainfall": argv_params["rainfall"][i],
                **{
                    flag: argv_params[flag]
                    for flag in ["tank_size", "recalibration", "daily"]
                    if flag in argv_params
                },
            }
        )
    print('---> the flags list is :', flags)

    result = []
    if not log_parallel:
        # Sequence mode
        for flag in flags:
            result.append(glrs.validation_process(flag))
    else:
        # Parallel mode
        with Pool(processes=process_size) as pool:
            # you can google multiprocess & pool
            result = pool.map(glrs.model_process, flags)
            pool.close()  # Close the pool to not create new process
            pool.join()  # Make main process to wait for the pool
    #print('---> the result layout', result)
    
    mat = []
    for i in range(len(flags)):   
        mat.append(
            list(flags[i].values()) + 
            [
                result[i][item]
                for item in [
                                "r2", "RMSE", "ME", "MAE",
                                "a1", "a2", "a3",
                                "b1", "b2", "b3",
                                "z1", "z2", "z3",
                                "c", 
                                "d",
                                "z_tides",
                                "mean gw",
                                "mean gw_pred",
                                #"mean rf",
                                "mean AMP",
                                "mean AMT"
                ]
            ]
        )
    print("---> mat", mat)

    columns=[
        "gw_file",
        "rn_file",
        "tank_size",
        "recalibration",
                        "daily",
                        "r2","RMSE", "ME", "MAE",
                        "a1", "a2", "a3",
                        "b1", "b2", "b3",
                        "z1", "z2", "z3",
                        "c", 
                        "d",
                        "z_tides",
                        "mean gw",
                        "mean gw_pred",
                        #"mean rf",
                        "mean AMP",
                        "mean AMT",
                        
    ]

    # Check if the lengths match
    if len(mat[0]) != len(columns):
        print(f"Length of mat: {len(mat[0])}")
        print(f"Length of columns: {len(columns)}")
        print("mat:", mat[0])
        print("columns:", columns)
    else:
        df_result = pd.DataFrame(mat, columns=columns)
        print("==> ", df_result)

    df_result = pd.DataFrame(mat, columns=[
        "gw_file",
        "rn_file",
        "tank_size",
        "recalibration",
                        #"daily",
                        "r2","RMSE", "ME", "MAE",
                        "a1", "a2", "a3",
                        "b1", "b2", "b3",
                        "z1", "z2", "z3",
                        "c", 
                        "d",
                        "z_tides",
                        "mean gw",
                        "mean gw_pred",
                        #"mean rf",
                        "mean AMP",
                        "mean AMT",
                        
    ])
    print("==> ", df_result)
    '''
q1_list=[]    
    for i in range(len(df_result)):

        # Q1
        q1_list.append(
            df_result.loc[df_result.index[i], 'a1']
            *
            (
                df_result.loc[df_result.index[i], 'mean gw'] - df_result.loc[df_result.index[i], 'z1']
            )
        )
    q2_list=[]    
    for i in range(len(df_result)):

        # Q2
        q2_list.append(
            df_result.loc[df_result.index[i], 'a2']
            *
            (
                df_result.loc[df_result.index[i], 'mean gw'] - df_result.loc[df_result.index[i], 'z2']
            )
        )
    q3_list=[]    
    for i in range(len(df_result)):

        # Q3
        q3_list.append(
            df_result.loc[df_result.index[i], 'a3']
            *
            (
                df_result.loc[df_result.index[i], 'mean gw'] - df_result.loc[df_result.index[i], 'z3']
            )
        )

    p1_list=[]    
    for i in range(len(df_result)):

        # P1
        p1_list.append(
            df_result.loc[df_result.index[i], 'b1']*df_result.loc[df_result.index[i], 'mean gw'])
    p2_list=[]    
    for i in range(len(df_result)):

        # P2
        p2_list.append(
            df_result.loc[df_result.index[i], 'b2'] * df_result.loc[df_result.index[i], 'mean gw']
            )
    p3_list=[]    
    for i in range(len(df_result)):

        # P3
        p3_list.append(
            df_result.loc[df_result.index[i], 'b3'] * df_result.loc[df_result.index[i], 'mean gw']
            )

    pumping_list=[]    
    for i in range(len(df_result)):

        # Pumping
        pumping_list.append(
            df_result.loc[df_result.index[i], 'c'] * df_result.loc[df_result.index[i], 'mean AMP']
            )
        
    tides_list=[]    
    for i in range(len(df_result)):

        # Tides
        tides_list.append(
            df_result.loc[df_result.index[i], 'd']
            *
            (
                df_result.loc[df_result.index[i], 'mean tides'] - df_result.loc[df_result.index[i], 'z_tides']
            )
        )

    new_df = pd.DataFrame(
        {
        'q1_list': q1_list, 'q2_list': q2_list,'q3_list':q3_list,
        'p1_list':p1_list,'p2_list':p2_list,'p3_list':p3_list,
        'pumping_list':pumping_list,'tides_list':tides_list
        }
        )

    # 原始DataFrame新的DataFrame連接新的
    df_result = pd.concat([df_result, new_df], axis=1)    
    '''
    

    #df_result.to_csv('Prof Ni Request/export_result.csv')
    
    