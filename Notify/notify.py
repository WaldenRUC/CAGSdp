import requests, time, datetime, hmac, hashlib, base64, urllib.parse, os, argparse
from pprint import pprint
curr_time = datetime.datetime.now()
time_str = datetime.datetime.strftime(curr_time,'%Y-%m-%d %H:%M:%S')
url="https://oapi.dingtalk.com/robot/send?access_token=ec031748442e3298ad82b25476438dcafe460255f28a92ab22c380e7de620d41&timestamp=%s&sign=%s"
def get_sign():
    timestamp = str(round(time.time() * 1000))
    secret = 'SEC429127893778a2f7ae256b8bbe0a1a7803a740d212aea103a96d6363547591d7' # 加签时候生成的秘钥
    secret_enc = secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return timestamp,sign
def dd_notify(text, atMobiles=['18005017312']):
    body={
        "msgtype": "markdown",
        "markdown": {
            "title": "模型训练完成",
            "text": text
        },
        "at": {
            "atMobiles": atMobiles,
            "isAtAll": False
        }
    }
    timestamp,sign=get_sign()
    url_sign=url % (timestamp, sign)
    res = requests.post(url_sign, json=body, timeout=2.0)
    print(res.text)
parser = argparse.ArgumentParser()
parser.add_argument(
    '--text',
    type=str,
    help='需要通知的文本内容'
)
args = parser.parse_args()
text = args.text
dd_notify(text = text)