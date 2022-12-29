import pymongo
import pandas as pd
import json

client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

FILE_PATH = '/config/workspace/aps_failure_training_set1.csv'
DATABASE_NAME = 'aps'
COLLECTION_NAME = 'sensor'

if __name__ == '__main__':
    df = pd.read_csv(FILE_PATH)
    # print('Rows & Columns of dataset:',df.shape)

# Convert dataframe to json in order to dump it into MongoDB

df.reset_index(drop=True,inplace=True)

json_format = list(json.loads(df.T.to_json()).values())
# print(json_format[0])

# Insert converted json records to MongoDB
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_format)
