import boto3
import xmltodict

with open('.ssh/params.txt', 'r') as f:
    params = f.read().split('\n')
region_name = 'us-east-1'
aws_access_key_id = params[0]
aws_secret_access_key = params[1]

endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'


client = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)


print(client.get_account_balance()['AvailableBalance'])
