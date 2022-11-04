import os
import boto3
from argparse import ArgumentParser

def parse_args():
    args = ArgumentParser()
    args.add_argument('--bucket_name', type=str, required=True, help='Bucket name')
    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    gcs_client  = boto3.client(
            "s3", # !just like that
            region_name="auto",
            endpoint_url="https://storage.googleapis.com",
            aws_access_key_id=os.environ['GCS_KEY'],
            aws_secret_access_key=os.environ['GCS_SECRET'],
        )


    # Call GCS to list objects in bucket_name
    response = gcs_client.list_objects(Bucket=args.bucket_name)

    # Print object names
    print("Objects:")
    for blob in response["Contents"]:
        print(blob["Key"])

