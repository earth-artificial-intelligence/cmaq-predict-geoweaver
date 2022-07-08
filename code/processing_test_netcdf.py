import xarray as xr
import pandas as pd
import glob, os
import numpy as np
from pathlib import Path
import datetime
from datetime import timedelta

base = datetime.datetime.today() - timedelta(days=2)
date_list = [base + timedelta(days=x) for x in range(2)]
days = [date.strftime('%Y%m%d') for date in date_list]

csvs_path = '/groups/ESS/aalnaim/cmaq/output_hourly'

for day in days:
	print("Opening: "+"/groups/ESS/share/projects/SWUS3km/data/cmaqdata/CCTMout/12km/POST/COMBINE3D_ACONC_v531_gcc_AQF5X_"+day+"_extracted.nc")
      
	df_cdf = xr.open_dataset("/groups/ESS/share/projects/SWUS3km/data/cmaqdata/CCTMout/12km/POST/COMBINE3D_ACONC_v531_gcc_AQF5X_"+day+"_extracted.nc")

	
	all_hourly_files = sorted(glob.glob(os.path.join(csvs_path, "prediction_rf_"+day+"*.csv")), key=lambda name: int(name[-6:-4]))
	df_from_each_hourly_file = (pd.read_csv(f) for f in all_hourly_files)
	print("read all hourly csv's")
    
	for df in df_from_each_hourly_file:
		print("reading df for day: "+day)

		# Getting hour value in df to match hour in cdf file for replacement of O3 value
		df['YYYYMMDDHH'] = df['YYYYMMDDHH'].astype(str)
		hour = df['YYYYMMDDHH'].str[8:10]
		print(hour.values[0])
        
		# Matching hours between df and cdf file for proper replacement of values.
		cdf_filt = df_cdf.sel(TSTEP=df_cdf.TSTEP == int(hour.values[0]))

		# Changing the shape of "prediction" column to match expected dim shape found in cdf file
		reshaped_prediction = np.atleast_3d(df['prediction']).reshape(-1, 265, 442)

		# Remove "LAY" Dimension in O3 variable already in nc file.
		reduced_dim = cdf_filt['O3'].sel(LAY=1, drop=True)
		# Swap values from original nc file with new prediction data
		reduced_dim.values = reshaped_prediction

		# Apply changes to data variable in nc file
		cdf_filt['O3'] = (['TSTEP', 'ROW', 'COL'], reshaped_prediction)

		cdf_filt.to_netcdf('/groups/ESS/aalnaim/cmaq/prediction_nc_files/COMBINE3D_ACONC_v531_gcc_AQF5X_'+day+hour.values[0]+'_Hourly_ML_extracted.nc')

		print('Saved updated netCDF file: /groups/ESS/aalnaim/cmaq/prediction_nc_files/COMBINE3D_ACONC_v531_gcc_AQF5X_'+day+hour.values[0]+'_Hourly_ML_extracted.nc')
