from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# uniform resource indentifier
uri = "mongodb+srv://snshrivas:Snshrivas@cluster0.u46c4.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# create database name and collection name
DATABASE_NAME="pwskills"
COLLECTION_NAME="waferfault"

# read the data as a dataframe
df=pd.read_csv(r"C:\Study1\ALL_PROJECTS\wafer_fault_detection_pw\sensor-fault-detection\notebooks\wafer_23012020_041211.csv")
df=df.drop("Unnamed: 0",axis=1)

# Convert the data into json
json_record=list(json.loads(df.T.to_json()).values())

#now dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
