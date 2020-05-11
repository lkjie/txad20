#!/usr/bin/env python
# coding=utf-8

import logging
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from qcloud_cos.cos_auth import CosS3Auth
from qcloud_cos.cos_comm import check_object_content_length, mapped, get_content_md5


secret_id = ''  # 替换为用户的secret_id
secret_key = ''  # 替换为用户的secret_key
region = 'ap-shanghai'  # 替换为用户的region
token = None  # 使用临时密钥需要传入Token，默认为空,可不填
config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token)  # 获取配置对象
client = CosS3Client(config)

'''
url: 
'''


def put_object(client, Bucket, Body, Key, EnableMD5=False, **kwargs):
    """单文件上传接口，适用于小文件，最大不得超过5GB

    :param Bucket(string): 存储桶名称.
    :param Body(file|string): 上传的文件内容，类型为文件流或字节流.
    :param Key(string): COS路径.
    :param EnableMD5(bool): 是否需要SDK计算Content-MD5，打开此开关会增加上传耗时.
    :kwargs(dict): 设置上传的headers.
    :return(dict): 上传成功返回的结果，包含ETag等信息.

    .. code-block:: python

        config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token)  # 获取配置对象
        client = CosS3Client(config)
        # 上传本地文件到cos
        with open('test.txt', 'rb') as fp:
            response = client.put_object(
                Bucket='bucket',
                Body=fp,
                Key='test.txt'
            )
            print (response['ETag'])
    """
    if not isinstance(client, CosS3Client):
        raise Exception("client error!")
    check_object_content_length(Body)
    headers = mapped(kwargs)
    url = client._conf.uri(bucket=Bucket, path=Key)
    logging.info("put object, url=:{url} ,headers=:{headers}".format(
        url=url,
        headers=headers))
    if EnableMD5:
        md5_str = get_content_md5(Body)
        if md5_str:
            headers['Content-MD5'] = md5_str
    rt = client.send_request(
        method='PUT',
        url=url,
        bucket=Bucket,
        auth=CosS3Auth(client._conf, Key),
        data=Body,
        headers=headers)

    return rt.url


# 文件流 简单上传
file_name = 'submission.csv'
# file_name = 'cosupload.py'
with open(file_name, 'rb') as fp:
    url = put_object(client,
        Bucket='',  # Bucket由bucketname-appid组成，填自己的bucket号
        Body=fp,
        Key=file_name,
        StorageClass='STANDARD',
        ContentType='text/csv; charset=utf-8'
    )
    print(url)