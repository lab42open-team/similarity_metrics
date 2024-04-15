import pandas as pd 
import pyarrow.parquet as pq
import h5py
import pickle

### TEST - Read files (parquet - hfd5 - pickle)
parquet_file = "/ccmri/similarity_metrics/data/test_dataset/lf_test_folder_super_table.parquet"
#table = pq.read_table(parquet_file)
df_parquet = pd.read_parquet(parquet_file)
print("Head Parquet file: ", df_parquet.head())

"""
hdf5_file = "/ccmri/similarity_metrics/data/SuperTable/long_format/lf_test_folder_super_table.h5"
with h5py.File(hdf5_file, "r") as f:
    df_counts = pd.DataFrame(f["Counts"][:])  
    df_sample = pd.DataFrame(f["Sample"][:])
    df_taxa = pd.DataFrame(f["Taxa"][:])
df_h5 = pd.concat([df_counts, df_sample, df_taxa], axis=1)
#print(df_h5.head())

pickle_file = "/ccmri/similarity_metrics/data/SuperTable/long_format/lf_test_folder_super_table.pkl"
with open(pickle_file, "rb") as f:
    data = pickle.load(f)
print(data.keys())    
for key, value in data.items():
    print("Length of {} : {}".format(key, len(value)))
"""    