
test1:
	python ../srcs/gw_model_parallel.py GROUNDWATER=../data/7020111.csv,../data/907211.csv RAINFALL=../data/C0G720_raf.csv,../data/C0K460_raf.csv TANK_SIZE=3 DAILY RECALIBRATION

test_with_csv:
	python ../srcs/input_csv_list.py 								\
		../data/input_data_list.csv RECALIBRATION

test_amt:
	python ../srcs/all_input_csv.py 									\
		../data/input_sea_tides.csv

test_tides:
	python ../srcs/input_tides_model.py 									\
		../data/input_sea_tides.csv


post_analysis:
	python ../srcs/post_analysis.py 									\
		sid=7050121 dfid=df_tides_7050121								\
		data_path=tides_stations