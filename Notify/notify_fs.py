import requests, json, argparse, datetime
def notify(msg: str):
    """
    在外部程序调用此函数即可发送通知到飞书
    msg: 需要传递的消息文本内容
    例: notify("Hi")
    """
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%Y-%m-%d %H:%M:%S')
    url="https://open.feishu.cn/open-apis/bot/v2/hook/dec37940-e22a-459f-9dd8-1be648b859a5"
    #secret = "vt81f7EvBLrhStkuGCvxUg"
    send_post = {
        "msg_type": "text",
        "content":{
            "text": time_str + "\n" + msg
        }
    }
    headers = {
        "Content-Type": "application/json; charset=utf-8"
    }
    response = requests.post(url=url, headers=headers, data=json.dumps(send_post))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text',
        type=str,
        help='需要通知的文本内容'
    )
    args = parser.parse_args()
    text = args.text
    notify(msg=text)