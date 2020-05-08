#!/usr/bin/env python
# coding=utf-8

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

secret_id = ''  # 替换为用户的secret_id
secret_key = ''  # 替换为用户的secret_key
region = 'ap-shanghai'  # 替换为用户的region
token = None  # 使用临时密钥需要传入Token，默认为空,可不填
config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token)  # 获取配置对象
client = CosS3Client(config)

# 文件流 简单上传
file_name = 'data/submission.csv'
with open(file_name, 'rb') as fp:
    response = client.put_object(
        Bucket='txad-1252070910',  # Bucket由bucketname-appid组成
        Body=fp,
        Key=file_name,
        StorageClass='STANDARD',
        ContentType='text/html; charset=utf-8'
    )
    print(response['ETag'])