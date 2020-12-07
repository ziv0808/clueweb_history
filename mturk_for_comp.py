import boto3
import xmltodict

region_name = 'us-east-1'
aws_access_key_id = 'AKIAI4XYPCUMAVRODWBA'
aws_secret_access_key = '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'

endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'


client = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)


print(client.get_account_balance()['AvailableBalance'])
