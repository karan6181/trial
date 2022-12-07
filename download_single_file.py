import os
import oci
import urllib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_file", type=str)
    parser.add_argument("--remote_file", type=str)
    args = parser.parse_args()

    config = oci.config.from_file()
    print(f'Before config: {config}')
    if os.environ['OCI_CLI_KEY_FILE']:
        config['key'] = os.environ['OCI_CLI_KEY_FILE']
    print(f'After config: {config}')

    client = oci.object_storage.ObjectStorageClient(
        config=config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
    namespace = client.get_namespace().data
    obj = urllib.parse.urlparse(args.remote_file)
    if obj.scheme != 'oci':
        raise ValueError(f'Expected obj.scheme to be "oci", got {obj.scheme} for remote={args.remote_file}')

    bucket_name = obj.netloc.split('@' + namespace)[0]
    # Remove leading and trailing forward slash from string
    object_path = obj.path.strip('/')
    print(f'namespace: {namespace} bucket_name: {bucket_name} object_path: {object_path} local: {args.local_file}')
    object_details = client.get_object(namespace, bucket_name, object_path)
    with open(args.local_file, 'wb') as f:
        for chunk in object_details.data.raw.stream(2048**2, decode_content=False):
            f.write(chunk)
    print("DONE")
