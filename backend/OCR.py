#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from urllib.request import Request  # urllib2在python3.3之后就不再使用（install urllib3）
import json

import urllib.request
import base64
import ssl

ENCODING = 'utf-8'
APPCODE = '4eea32fcf5014eeba3d4fc7fdfbe3592'

def get_img_base64(img_file):
    with open(img_file, 'rb') as infile:
        s = infile.read()
        return base64.b64encode(s).decode(ENCODING)



def predict(url, appcode, img_base64, kv_configure):

    param = {}
    param['image'] = img_base64
    if kv_configure is not None:
        param['configure'] = json.dumps(kv_configure)
    body = json.dumps(param)
    data = bytes(body, "utf-8")

    request = urllib.request.Request(url = url,data = data)
    request.add_header('Authorization', 'APPCODE ' + appcode)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        response = urllib.request.urlopen(request,context=ctx)
        return response.code, response.headers, response.read()
    except urllib.request.HTTPError as e:
        return e.code, e.headers, e.read()


def ocr_getResult(img_path):
    host = 'https://tysbgpu.market.alicloudapi.com'
    path = '/api/predict/ocr_general'
    url = host + path
    appcode = APPCODE
    img_file = img_path
    configure = {'min_size':16,
                 'output_prob':True,
                 'output_keypoints':False,
                 'skip_detection':False,
                 'without_predicting_direction':False}
    #如果没有configure字段，configure设为None
    #configure = None

    img_base64data = get_img_base64(img_file)
    stat, header, content = predict( url, appcode, img_base64data, configure)
    if stat != 200:
        print('Http status code: ', stat)
        print('Error msg in header: ', header['x-ca-error-message'] if 'x-ca-error-message' in header else '')
        print('Error msg in body: ', content)
        exit()
    result_str = content

    result = json.loads(result_str)
    return result


ocr_getResult('E:\\PycharmProjects\\test1\\HeightLimitSign2021\\outputs\\test1.jpg')

