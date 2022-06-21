from scraper import get_texts
import re
import string
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
import numpy as np
import pandas as pd
import dateutil.relativedelta
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def remove_emoji(list_of_texts):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    return [emoji_pattern.sub('', text) for text in list_of_texts]

def write_to_txt(list_of_texts, filename="exemple.txt"):
    texts = list_of_texts
    with open(filename, "w", encoding='utf-8') as file: 
        for text in texts:
            file.write(text + "\n")

def keep_only_words(list_of_texts):
    texts_no_emo = remove_emoji(list_of_texts)
    texts_no_gt = [text.replace("&gt;", "") for text in texts_no_emo]
    texts_no_lt = [text.replace("&lt;", "") for text in texts_no_gt]
    texts_no_rt = [text.replace("RT", "") for text in texts_no_lt]
    texts_no_id = []
    for text in texts_no_rt:
        list_words = text.split(" ")
        new_text = []
        for word in list_words:
            if word != "":
                if word[0] != "@" and "https://" not in word:
                    new_text.append(word.strip())
        texts_no_id.append(" ".join(new_text))
    return texts_no_id

def clean_1(list_of_texts):
    texts = [text.lower() for text in list_of_texts]
    texts = [text.replace("d'", "") for text in texts]
    texts = [text.replace("l'", "") for text in texts]
    texts = [text.replace("d’", "") for text in texts]
    texts = [text.replace("l’", "") for text in texts]
    texts = [text.replace("«", "") for text in texts]
    texts = [text.replace("»", "") for text in texts]
    texts = [text.replace("–", "") for text in texts]
    texts = [re.sub('[%s]' % re.escape(string.punctuation), '', text) for text in texts]
    texts = [re.sub('\w*\d\w*', '', text) for text in texts]
    return texts

def remove_duplicates(list_of_texts):
    return list(dict.fromkeys(list_of_texts))

def document_term(list_of_texts, stop_words=list(fr_stop)):
    cv = CountVectorizer(stop_words=stop_words)
    data_cv = cv.fit_transform(list_of_texts)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names(), index=[f"tweet_{i}" for i in range(1, len(list_of_texts)+1)])
    return data_dtm

def add_most_common_words_to_stop_list(document_term_matrix, max=30):
    data = document_term_matrix.transpose()
    top_dict = {}
    for c in list(data.columns):
        top = data[c].sort_values(ascending=False).head(max)
        top_dict[c] = list(zip(top.index, top.values))
    words = []
    for index in data.columns:
        top = [word for word, _ in top_dict[index]]
        for t in top:
            words.append(t)
    return [word for word, count in Counter(words).most_common() if count > len(data.columns)]

def main(topic, start_date=str(datetime.datetime.now() + dateutil.relativedelta.relativedelta(days=-6))[:10], end_date=str(datetime.datetime.now())[:10], max_results=35):
    list_of_texts = get_texts(topic=topic, start_date=start_date, end_date=end_date, max_results=max_results)
    list_of_texts = keep_only_words(list_of_texts)
    list_of_texts = clean_1(list_of_texts)
    list_of_texts = remove_duplicates(list_of_texts)
    list_of_texts = list(filter(None, list_of_texts))
    list_of_texts = [text.replace(topic, "") for text in list_of_texts]
    print("preprocessing ok")
    print("-"*50)
    data_dtm = document_term(list_of_texts)
    print("1st dtm ok")
    print("-"*50)
    add_stop_words = add_most_common_words_to_stop_list(data_dtm, max=30)
    print("2nd dtm ok")
    print("-"*50)
    final_stop_words = add_stop_words + list(fr_stop)
    final_data = " ".join(list_of_texts)
    wc = WordCloud(stopwords=final_stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)
    wc.generate(final_data)

    plt.figure(figsize=(8,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(topic)
    plt.savefig(f"wc_{topic}.jpg", dpi=350)    
    plt.show()

if __name__ == "__main__":
    main(input("Sujet à chercher : "))

