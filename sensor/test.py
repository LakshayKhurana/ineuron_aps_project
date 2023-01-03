import pandas as pd
data1 = pd.read_csv('/config/workspace/aps_failure_training_set1.csv')
print(data1.shape)

data2 = pd.read_csv('/config/workspace/artifacts/01032023__202432/data_ingestion/feature_store/sensor.csv')
print(data2.shape)