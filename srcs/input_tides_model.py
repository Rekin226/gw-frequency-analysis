"""
1. GROUNDWATER, RAINFALL, tank_size & freq_type are read from a CSV file
2. call gw_linear_reservoir_parallel.py
3. use the other argvs (without sys.argv[1])
"""

import sys
import os
import pandas as pd

if __name__ == "__main__":
    csv_fname = sys.argv[1]
    # reading CSV file
    # Active: 1 --> True
    #         0 --> False
    df_csv = pd.read_csv(csv_fname)
    print(df_csv)

    set_model_type = list(
        set(
            (
                df_csv.loc[index, "tank_size"],
                df_csv.loc[index, "freq_type"],
            )
            for index in df_csv.index
        )
    )

    command_list = []
    for smt in set_model_type:
        # Create a command to call gw_linear_reservoir_parallel.py
        #   for the same tank size and frequency types such as "HOURLY" or "DAILY"
        (tank_size, freq_type) = smt
        # print (tank_size, freq_type)

        command = "python ../srcs/gw_model_parallel.py"
        flist = ""
        for index in df_csv.index:
            if df_csv.loc[index, "active"] == 1:
                flist += "{},".format(df_csv.loc[index, "GROUNDWATER"])
        command += " GROUNDWATER={}".format(flist[:-1])

                # Prepare GROUNDWATER parameter
        flist = ""
        for index in df_csv.index:
            if df_csv.loc[index, "active"] == 1:
                flist += "{},".format(df_csv.loc[index, "GROUNDWATER"])
        command += " GW_ST_NO={}".format(flist[:-1])

        # Prepare TM_X97 parameter
        flist = ""
        for index in df_csv.index:
            if df_csv.loc[index, "active"] == 1:
                flist += "{},".format(df_csv.loc[index, "TM_X97"])

        command += " TM_X97={}".format(flist[:-1])

        # Prepare TM_Y97 parameter
        flist = ""
        for index in df_csv.index:
            if df_csv.loc[index, "active"] == 1:
                flist += "{},".format(df_csv.loc[index, "TM_Y97"])
        command += " TM_Y97={}".format(flist[:-1])

        flist = ""
        for index in df_csv.index:
            if df_csv.loc[index, "active"] == 1:
                flist += "{},".format(df_csv.loc[index, "RAINFALL"])
        command += " RAINFALL={}".format(flist[:-1])

        command += " tank_size={} {}".format(tank_size, freq_type)

        for argv in sys.argv[2:]:
            command += " {}".format(argv)
        print("# Command: {}".format(command))
        command_list.append(command)
    print(command_list)

    os.system("rm -rf logging")
    for command in command_list:
        os.system(command)