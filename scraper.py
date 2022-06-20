import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
bearer_token = os.getenv("BEARER_TOKEN")

def search_twitter(query, tweet_fields, bearer_token = ""):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}

    url = "https://api.twitter.com/1.1/search/tweet.json?q={}&result_type={}".format(
        query, tweet_fields
    )
    print(url)
    response = requests.get(url)
    print(response.status_code)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.content()

