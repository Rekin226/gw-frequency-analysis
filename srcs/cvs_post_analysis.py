"""
1. Read the csv_post_analysis.csv CSV file
2. create a command to call csv_post_analysis.py for sid,dfid,data_path, active
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
                df_csv.loc[index, "model_type"],
                df_csv.loc[index, "data_path"]
            )
            for index in df_csv.index
        )
    )

    #print('hello this is set model', set_model_type)
    command_list = []
    for smt in set_model_type:
        # Create a command to call gw_linear_reservoir_parallel.py
        #   for the same tank size and frequency types such as "HOURLY" or "DAILY"
        (model_type, data_path) = smt
        #print (model_type, data_path)

        command = "python ../srcs/post_analysis.py"
        flist = ""
        for index in df_csv.index:
            if df_csv.loc[index, "active"] == 1:
                flist += "{},".format(df_csv.loc[index, "sid"])
        command += " sid={}".format(flist[:-1])

        flist = ""
        for index in df_csv.index:
            if df_csv.loc[index, "active"] == 1:
                flist += "{},".format(df_csv.loc[index, "dfid"])
        command += " dfid={}".format(flist[:-1])

        command += " model_type={}".format(model_type)
        command += " data_path={}".format(data_path)

        for argv in sys.argv[2:]:
            command += " {}".format(argv)
        print("# Command: {}".format(command))
        command_list.append(command)
    print(command_list)
    
    os.system("rm -rf logging")
    for command in command_list:
        os.system(command)