from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import boto3
from io import StringIO
import pandas as pd

# Load data & Split to train/test

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42
)

train_df = pd.DataFrame(X_train)
test_df = pd.DataFrame(X_test)

train_df['target'] = y_train

# Create script to upload to S3

def upload_df(df, bucket, location):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_response = s3_resource.Object(bucket, location).put(Body=csv_buffer.getvalue())
    csv_buffer.close()
    return s3_response

# Load into S3

s3_client = boto3.client('s3')

s3_bucket = 'censussmdemo'
s3_location_train = 'housingdemo/train/train.csv'
s3_location_test = 'housingdemo/test/test.csv'

s3_response_train = upload_df(pd.DataFrame(X_train), s3_bucket, s3_location_train)
s3_response_test = upload_df(pd.DataFrame(X_test), s3_bucket, s3_location_test)