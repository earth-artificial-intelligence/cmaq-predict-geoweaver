
# Importing necessary libraries
import pandas as pd
import pickle
from pathlib import Path
from time import sleep
import glob
import os

# home directory
home = str(Path.home())

# load the model from disk
filename = '/groups/ESS/aalnaim/cmaq/models/rfOLD_Jun13.sav'
loaded_model = pickle.load(open(filename, 'rb'))
print("Loaded ML model!")
# importing data
# final=pd.read_csv("/groups/ESS/aalnaim/cmaq/input_hourly/testing.csv")
# print(final.head())

path = '/groups/ESS/aalnaim/cmaq/input_hourly'
all_hourly_files = sorted(glob.glob(os.path.join(path, "*.csv"))) 
df_from_each_hourly_file = (pd.read_csv(f) for f in all_hourly_files)
print("Read all hourly input csv's")

for file in df_from_each_hourly_file:
	
	print("Predicting data for: "+str(file['YYYYMMDDHH'].values[0]))
    
	X = file.drop(['YYYYMMDDHH','Latitude','Longitude',],axis=1)
    # making prediction
	pred = loaded_model.predict(X)
	# adding prediction values to test dataset
	file['prediction'] = pred.tolist()
	file = file[['Latitude', 'Longitude','YYYYMMDDHH','prediction']]
	# saving the dataset into local drive
	print("Prediction done for "+str(file['YYYYMMDDHH'].values[0]))
	file.to_csv('/groups/ESS/aalnaim/cmaq/output_hourly/prediction_rf_'+str(file['YYYYMMDDHH'].values[0])+'.csv',index=False)






