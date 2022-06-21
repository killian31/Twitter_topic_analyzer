import requests
import json
from dotenv import load_dotenv
import os
import datetime
import dateutil.relativedelta

load_dotenv()

def auth():
    return os.getenv('BEARER_TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url(keyword, start_date, end_date, max_results = 10):
    
    search_url = "https://api.twitter.com/2/tweets/search/recent"

    query_params = {'query': keyword,
                    'start_time': start_date + "T00:00:00.000Z",
                    'end_time': end_date + "T00:00:00.000Z",
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   
    response = requests.get(url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def scrap(topic, start_date = str(datetime.datetime.now() + dateutil.relativedelta.relativedelta(days=-6))[:10], end_date = str(datetime.datetime.now())[:10], max_results=15):
    bearer_token = auth()
    headers = create_headers(bearer_token)
    keyword = f"{topic} lang:fr"
    max_results = max_results

    url = create_url(keyword, start_date, end_date, max_results)
    json_response = connect_to_endpoint(url[0], headers, url[1])
    return json_response

def get_texts(topic, start_date = str(datetime.datetime.now() + dateutil.relativedelta.relativedelta(days=-6))[:10], end_date = str(datetime.datetime.now())[:10], max_results=15):
    '''max_results has to be between 10 and 150'''
    dic_json = scrap(topic, start_date, end_date, max_results)
    all_texts = [dic_json["data"][i]["text"] for i in range(max_results)]
    return all_texts

if __name__ == "__main__":
    get_texts(topic="piscine", max_results=100)
