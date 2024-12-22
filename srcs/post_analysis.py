"""
Input the well id
 1. search sid and dfid from the data_path
2. load the pickle file
3. load df_id
"""
import os
import pickle
import sys
from typing import List, Dict
import pandas as pd
import gw_subroutine as glrc
import gw_model_shell as glrs



'''
def argv_phrase(argv: List[str]) -> Dict[str, str]:
    """
    Phrase your argv
    """
    assert isinstance(argv, list)

    argv_params: Dict[str, str] = {}
    for content in argv:
        sepline = content.split("=")
        flag = sepline[0].lower()
        try:
            argv_params[flag] = sepline[1]
        except IndexError:
            argv_params[flag] = flag

        if flag in ["sid"]:
            # str to int
            argv_params[flag] = int(argv_params[flag])

    #print("Parsed argv_params:", argv_params)
    return argv_params
'''

def argv_phrasing(argvs: List) -> Dict:
    argv_params = {}
    for argv in argvs:
        for flag in ["sid", "dfid", "data_path"]:
            if argv.find("{}=".format(flag)) >= 0:
                argv_params[flag.lower()] = argv.replace("{}=".format(flag), "")
    return argv_params


def find_files_with_sid(data_path: str, sid: int) -> List[str]:
    """
    Find pickle files and csv file with sid
    """
    assert isinstance(data_path, str)
    assert isinstance(sid, int)

    all_files = os.listdir(data_path)
    sid_files = []
    for file in all_files:
        if str(sid) in file:
            sid_files.append(file)
    return sid_files

if __name__ == "__main__":
    argv_params = argv_phrase(sys.argv[1:])
    #print('===>', argv_params)
    data_path = argv_params.get("data_path", "data")
    sid = argv_params.get("sid", 1)
    #print('===>', data_path, sid)
    sid_files = find_files_with_sid(data_path, sid)
    # assert if the sid_files is not empty
    assert sid_files, f"sid {sid} not found in {data_path}"
    #print('===>', sid_files)

    # Check if the pickle file exists
    pickle_file = next((file for file in sid_files if file.endswith(".pickle")), None)
    if pickle_file:
        with open(os.path.join(data_path, pickle_file), "rb") as f:
            model_params = pickle.load(f)
            #print(model_params)
    else:
        print(f"No pickle file found for sid {sid} in {data_path}")

    # Read the csv file
    csv_file = next((file for file in sid_files if file.endswith(".csv")), None)
    if csv_file:
        print(f"Reading csv file {csv_file}")
    else:
        print(f"No csv file found for sid {sid} in {data_path}")

    # Load the csv file
    df_merge = pd.read_csv(os.path.join(data_path, csv_file))
    df_merge = df_merge.rename(
        columns={"Unnamed: 0": "Date time"}
        ).set_index("Date time")
    #print(df_merge.head())


    # load and re-calculate the model using glrs.model_fit_preprocess
    x, y, h_init = glrs.model_fit_preprocess(df_merge, argv_params)
    #print(x, y, h_init)
    
    dt = 24
    glrc.gw_model_shell(
        x,
        *model_params["calibrated_params"],
        h_init=h_init,
        dt=dt,
        subterm_fname=os.path.join(
            "subterm_analysis", "{}_subterm.csv".format(argv_params["sid"])
        ),
    )
    print("Done")

