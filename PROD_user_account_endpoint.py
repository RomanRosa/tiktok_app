import pandas as pd
from selenium import webdriver


from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import os
import time 
import re
import plotly.express as px
import json
from io import BytesIO
from itertools import zip_longest

import requests
from urllib.request import urlopen

from wordcloud import WordCloud

from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


def define_stopwords():
    sw_es=stopwords.words('spanish')
    sw_pt=stopwords.words('portuguese')
    sw_en=stopwords.words('english')
    sw_es_title=[word.title() for word in sw_es]
    sw_pt_title=[word.title() for word in sw_pt]
    sw_en_title=[word.title() for word in sw_en]
    return sw_es, sw_pt, sw_en, sw_es_title, sw_pt_title, sw_en_title

additional_stopwords = ["pagina", "costo", "audiencia", "valoración", "intermedia", "00", "así", "intermediamexico", "puede", "mejor",
                        "si", "testigo", "costo", "positivofracción", "intermediamexico testigo valoración", "tras", "audiencia", "pue",
                        "tiraje", "autor", "neutro", "fracción", "opinion", "redacción", "neutrofracción", "sección", "debe", "además",
                        "incluso", "link", "https", "http", "va", "00audiencia", "00tiraje", "embargo", "caso", "podría", "aún", "paí",
                        "decir", "sino", "redacciónfecha", "debido", "pues", "http", "ello", "solo"]
stopwords_result = define_stopwords()
sw_es, sw_pt, sw_en, sw_es_title, sw_pt_title, sw_en_title = [stopwords_result[i] for i in range(0,6)]
sw_total = sw_es + sw_pt + sw_en +  sw_es_title +  sw_pt_title + sw_en_title + additional_stopwords

def clean_and_tokenize(texto):
    texto=str(texto).lower()
    texto=re.sub('[^\w\s]',' ',texto)
    tokenizer=RegexpTokenizer(r'\w+')
    texto=tokenizer.tokenize(texto)
    texto=[word for word in texto if word not in sw_pt]
    texto=[word for word in texto if word not in sw_es]
    texto=[word for word in texto if word not in sw_total]
    tokens=[word.strip() for word in texto if word is not None]
    tokens=[word.strip() for word in tokens if len(word)>1]
    return tokens

def clean_text_wt_v1(texto):
    clean_tokens=clean_and_tokenize(texto)
    texto = ' '.join(clean_tokens)
    return texto

def download_videos(df, column_name, hashtag):
    output_dir = f'videos_{hashtag}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, url in enumerate(df[column_name]):
      start_time = time.time()
      try:
        response = requests.get(url)
        with open(output_dir + f'video_{i}.mp4', 'wb') as f:
          f.write(response.content)
          elapsed_time = time.time() - start_time
          elapsed_mins = int(elapsed_time // 60)
          elapsed_secs = int(elapsed_time % 60)
          print(f"Video {i} downloaded successfully in {elapsed_mins} minutes and {elapsed_secs} seconds.")
      except:
        print(f"Error downloading video {i}.")


def extract_tiktok_attributes(link):
    url = "https://tiktok-videos-without-watermark.p.rapidapi.com/getVideo"

    querystring = {"url":f"{link}?is_from_webapp=1&sender_device=pc&web_id=7085812281682642434"}

    headers = {
        "X-RapidAPI-Key": "899d894352mshf4d1863c0bc7c08p104f19jsn1cf81b3bc134",
        "X-RapidAPI-Host": "tiktok-videos-without-watermark.p.rapidapi.com"
    }

    try:
        response = requests.request("GET", url, headers=headers, params=querystring)
        formatted_json = json.loads(response.text)

        id = formatted_json['item']['id']
        desc = formatted_json['item']['desc']
        uid = formatted_json['item']['author']['uid']
        nickname = formatted_json['item']['author']['nickname']
        music_title = formatted_json['item']['music']['title']
        music_author = formatted_json['item']['music']['author']
        play_url_uri_music = formatted_json['item']['music']['play_url']['uri']
        duration = formatted_json['item']['music']['duration']
        owner_nickname = formatted_json['item']['music']['owner_nickname']
        is_original = formatted_json['item']['music']['is_original']
        redirect = formatted_json['item']['music']['redirect']
        is_restricted = formatted_json['item']['music']['is_restricted']
        owner_handle = formatted_json['item']['music']['owner_handle']
        author_position = formatted_json['item']['music']['author_position']
        video_uri = formatted_json['item']['video']['play_addr']['uri']
        video_url_list_0 = formatted_json['item']['video']['play_addr']['url_list'][0]
        video_url_list_1 = formatted_json['item']['video']['play_addr']['url_list'][1]
        video_url_list_2 = formatted_json['item']['video']['play_addr']['url_list'][2]
        width = formatted_json['item']['video']['play_addr']['width']
        height = formatted_json['item']['video']['play_addr']['height']
        url_key = formatted_json['item']['video']['play_addr']['url_key']
        data_size = formatted_json['item']['video']['play_addr']['data_size']
        file_hash = formatted_json['item']['video']['play_addr']['file_hash']


        data = {'id': id,
                'desc': desc,
                'uid': uid,
                'nickname': nickname,
                'music_title': music_title,
                'music_author': music_author,
                'play_url_uri': play_url_uri_music,
                'duration': duration,
                'owner_nickname': owner_nickname,
                'is_original': is_original,
                'redirect': redirect,
                'is_restricted': is_restricted,
                'owner_handle': owner_handle,
                'author_position': author_position,
                'video_uri': video_uri,
                'video_url_list_0': video_url_list_0,
                'video_url_list_1': video_url_list_1,
                'video_url_list_2': video_url_list_2,
                'width': width,
                'height': height,
                'url_key': url_key,
                'data_size': data_size,
                'file_hash': file_hash
                }
        return pd.Series(data)

    except Exception as e:
      print(f"Error ocurred in link: {link}")
      print(e)
      return pd.Series()
    
def extract_emojis(text):
    pattern_es = re.compile("[\U0001F1E0-\U0001F1FF]+|[\U0001F600-\U0001F64F]+|[\U0001F300-\U0001F5FF]+|[\U0001F680-\U0001F6FF]+|[\U0001F190-\U0001F1FF]+")
    pattern_en = re.compile("[\U0001F600-\U0001F64F]+|[\U0001F300-\U0001F5FF]+|[\U0001F680-\U0001F6FF]+|[\U0001F1E0-\U0001F1FF]+")
    pattern_pt = re.compile("[\U0001F600-\U0001F64F]+|[\U0001F300-\U0001F5FF]+|[\U0001F680-\U0001F6FF]+|[\U0001F1E0-\U0001F1FF]+|[\U0001F1F5-\U0001F1F9]+|[\U0001F191-\U0001F1FF]+")

    if pattern_es.search(text):
        pattern = pattern_es
    elif pattern_en.search(text):
        pattern = pattern_en
    elif pattern_pt.search(text):
        pattern = pattern_pt
    else:
        return ""

    emojis = re.findall(pattern, text)
    return " ".join(emojis)

def search_useraccount_on_tiktok(url): # (By Username Or User Account)
    if "tiktok.com/@" not in url:
        print("Invalid TikTok user account URL.")
        return None
    
    browser_options = webdriver.ChromeOptions()
    browser_options.add_argument("--no-sandbox")
    browser_options.add_argument("--disable-infobars")
    browser_options.add_argument("--start-maximized")
    browser_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")
    browser_options.add_argument("--incognito")
    browser_options.add_argument("--disable-extensions")
    browser_options.add_argument("--disable-cookies")
    browser_options.add_argument('--lang=en-US')


    prefs = {"profile.default_content_setting_values.notifications" : 2}
    browser_options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=browser_options)

    driver.get(url)

    time.sleep(20)

    scroll_pause_time = 1
    screen_height = driver.execute_script("return window.screen.height;")
    i = 1

    while True:
      driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))
      i += 1
      time.sleep(scroll_pause_time)
      scroll_height = driver.execute_script("return document.body.scrollHeight;")
      if (screen_height) * i > scroll_height:
        break

# Data HTML page TikTok with BeutifulSoup4
    html = driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
    soup = BeautifulSoup(html, 'html.parser')
    
    # Scraping TikTok data
    description_tiktok, link_tiktok, username, followers, following, likes, title_tiktok, hashtag_tiktok, userbio, strongVideoCount=[], [], [], [], [], [], [], [], [], []

    # Get TikTok Video Title
    for title in soup.find_all('div', class_= "tiktok-5lnynx-DivTagCardDesc eih2qak1"):
        description_tiktok.append(title.text)

        searchTitle = title.text
        cleanTitle = re.sub("#[A-Za-z0-9_]+","", searchTitle)
        cleanTitle = cleanTitle.replace("  ", "")
        title_tiktok.append(cleanTitle)

        # Get TikTok Video Hashtags
        newHashtag = re.findall("#([a-zA-Z0-9_]{1,50})", searchTitle)
        newHashtag ='#' + ' #'.join(newHashtag)
        hashtag_tiktok.append(newHashtag)

        # Get TikTok Video Link
    for link in soup.find_all('div', class_='tiktok-x6y88p-DivItemContainerV2 e19c29qe7'):
        link_tiktok.append(link.a['href'])

        # Get TikTok Video Username
    for user in soup.find_all('h1', {'data-e2e': 'user-subtitle', 'class': 'tiktok-qpyus6-H1ShareSubTitle ekmpd5l6'}):
        username.append(user.text)
    username = username*len(link_tiktok)

    # Get Followers
    for follower in soup.find_all('strong', {'data-e2e': 'followers-count'}):
        followers.append(follower.text)
    followers = followers*len(link_tiktok)

        # Get Following
    for follow in soup.find_all('strong', {'data-e2e': 'following-count'}):
        following.append(follow.text)
    following = following*len(link_tiktok)

        # Get Likes
    for like in soup.find_all('strong', {'data-e2e': 'likes-count'}):
        likes.append(like.text)
    likes = likes*len(link_tiktok)

        # Get TikTok Video UserBio
    for user in soup.find_all('h2', {'data-e2e': 'user-bio', 'class': 'tiktok-1n8z9r7-H2ShareDesc e1457k4r4'}):
        userbio.append(user.text)
    userbio = userbio*len(link_tiktok)

  
    # Get TikTok Video StrongVideoCount
    for video in soup.find_all('strong', class_ = 'video-count tiktok-1nb981f-StrongVideoCount e148ts222'):
        strongVideoCount.append(video.text)

      # Save data in dictionary format
    listCols = ['link_tiktok',
    'description_tiktok',
    'title_tiktok',
    'hashtag_tiktok',
    'username',
    'followers',
    'following',
    'likes',
    'userbio',
    'strongVideoCount']

    dict_continued = dict(zip(listCols, (link_tiktok, description_tiktok, title_tiktok, hashtag_tiktok, username, followers, following, likes, userbio, strongVideoCount)))

    data = list(zip_longest(link_tiktok, username, followers, following, likes, userbio, description_tiktok, hashtag_tiktok, strongVideoCount))

    df = pd.DataFrame(data, columns=['Link', 'Username', 'Followers', 'Following', 'Likes', 'User Bio', 'Description TikTok', 'Hashtags', 'Video Views'])

    print("Retrieving API attributes. This may take a moment, please wait...")
    start_time = time.time()
    df_tiktok_attributes = df['Link'].apply(extract_tiktok_attributes)
    elapsed_time = time.time() - start_time
    elapsed_time_minutes = int(elapsed_time // 60)
    elapsed_time_seconds = int(elapsed_time % 60)
    print((f"Elapsed Time for Retrieving API attributes: {elapsed_time_minutes} minutes, {elapsed_time_seconds} seconds."))


    df_result = pd.concat([df, df_tiktok_attributes], axis=1)
    
    df_result = df
    columns_to_keep = ['Link',
    'Username',
    'Followers',
    'Following',
    'Likes',
    'User Bio',
    'Description TikTok',
    'Hashtags',
    'Video Views',
    'id',
    'desc',
    'uid',
    'nickname',
    'music_title',
    'music_author',
    'play_url_uri',
    'duration',
    'owner_nickname',
    'is_original',
    'redirect',
    'is_restricted',
    'owner_handle',
    'author_position',
    'video_uri',
    'video_url_list_0',
    'video_url_list_1',
    'video_url_list_2',
    'width',
    'height',
    'url_key',
    'data_size',
    'file_hash']

    df_result = df_result[columns_to_keep]

    df_result = df_result.astype(str)

    df_result['Emojis In Description'] = df_result['Description TikTok'].apply(extract_emojis)

    print(df_result.shape)
    df_result.head()

    print('Dataframe Shape:', df_result.shape)
    print('Dataframe Head:')
    df_result['Processed TikTok Description'] = df_result['Description TikTok'].apply(lambda row: clean_text_wt_v1(row))

    # ----- Most Frequent Hashtags ----- #
    hashtag_col = 'Hashtags'
    tags = df_result[hashtag_col].str.split(expand=True).unstack().dropna()
    tag_counts = tags.value_counts()
    top_tags = tag_counts[:35]
    fig = px.bar(
        top_tags, 
        x=top_tags.index, 
        y=top_tags.values,
        labels={'x': 'Hashtags', 'y': 'Count'}
    )
    most_common_hashtag = top_tags.idxmax()
    colors = ['blue' if x != most_common_hashtag else 'red' for x in top_tags.index]
    fig.update_traces(text=top_tags.values, textposition='auto', marker_color=colors)

    fig.update_layout(title='Most Popular Hashtags in the Conversation')
    fig.show()


    # ----- Most Frequent Emojis ----- #
    emoji_col = 'Emojis In Description'
    emojis = df_result[emoji_col].str.split(expand=True).unstack().dropna()
    emoji_counts = emojis.value_counts()
    top_emojis = emoji_counts[:35]
    fig = px.bar(
        top_emojis, 
       x=top_emojis.index, 
        y=top_emojis.values,
        labels={'x': 'Emojis', 'y': 'Count'}
    )
    most_common_emoji = top_emojis.idxmax()
    colors = ['blue' if x != most_common_emoji else 'red' for x in top_emojis.index]
    fig.update_traces(text=top_emojis.values, textposition='auto', marker_color=colors)

    fig.update_layout(title='Most Popular Emojis in the Conversation')
    fig.show()

    # ----- TikTok Description WordCloud ----- #
    df_result['Processed TikTok Description'] = \
    df_result['Processed TikTok Description'].map(lambda x: re.sub('[,\.!?]', '', x))
    df_result['Processed TikTok Description'] = \
    df_result['Processed TikTok Description'].map(lambda x: x.lower())
    df_result['Processed TikTok Description'].head()

    description_long_string = ','.join(list(df_result['Processed TikTok Description'].values))
    description_wordcloud = WordCloud(background_color="white", max_words=3000, contour_width=3, contour_color='steelblue',width=1200, height=600)
    description_wordcloud.generate(description_long_string)
    description_wordcloud.to_image()

    # ----- Downloading Videos ----- #
    print("Starting Downloading Videos...")
    username = url.replace("https://www.tiktok.com/@", "")
    download_videos(df_result, 'video_url_list_2', username)


    # ----- Save Excel File ----- #
    df_result.to_excel("scrapping_user_account_" + url.replace("https://www.tiktok.com/@", "") + ".xlsx", index=False)
    return df_result

if __name__ == "__main__":
    #search_useraccount_on_tiktok("https://www.tiktok.com/@como_cocino")
    search_useraccount_on_tiktok("https://www.tiktok.com/@ronaldarnez")