import json
import requests


class LineNotifier:
    def __init__(self, access_token, user_id):
        self.webhook_url = "https://api.line.me/v2/bot/message/push"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {{{token}}}".format(token=access_token)
        }
        self.user_id = user_id

    def notify(self, date, name):
        message = "{date} {name} さんが入室しました。".format(date=date.strftime("%H:%M"), name=name)
        json_message = {
            'to' : [self.user_id],
            'messages' : [
                {
                    'type' : 'text',
                    'text' : message
                }
            ]
        }
        requests.post(self.webhook_url, headers=self.headers, data=json.dumps(json_message))
