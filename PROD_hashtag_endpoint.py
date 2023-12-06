import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import os
import time 
import re
import plotly.express as px
import json
import warnings
import base64
from io import BytesIO
from itertools import zip_longest

import requests
from urllib.request import urlopen
from PIL import Image
import random

#from wordcloud import WordCloud

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


def get_mayusculas(texto):
    texto=str(texto)
    palabras_importantes=[]
    mayusculas=(r"([A-Z][a-zÁ-ÿ0-9]{1,20}\s?\,?\.?\s?)")
    texto=re.sub('[^\w\s]',' ',texto)
    texto=re.sub('[0-9]+', '', texto)  
    tokenizer=RegexpTokenizer(r'\w+')
    texto=tokenizer.tokenize(texto)
    texto=[word for word in texto if word not in sw_pt]
    texto=[word for word in texto if word not in sw_es]
    texto=[word for word in texto if word not in sw_en]
    texto=[word for word in texto if word not in sw_pt_title]
    texto=[word for word in texto if word not in sw_es_title]
    texto=[word for word in texto if word not in sw_en_title]
    tokens=[word.strip() for word in texto if word is not None]
    tokens=[word.strip() for word in tokens if len(word)>1]
    palabras_importantes=[word for word in tokens if word.istitle()]
    palabras_importantes=[word for word in tokens if re.match(mayusculas, word)]
    return palabras_importantes

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
    #st.write(querystring)

    #headers = {
    #    "X-RapidAPI-Key": "899d894352mshf4d1863c0bc7c08p104f19jsn1cf81b3bc134",
    #    "X-RapidAPI-Host": "tiktok-videos-without-watermark.p.rapidapi.com"
    #}
    
    # Load the API keys from the JSON file
    with open('api_keys.json', 'r') as file:
        api_keys = json.load(file)
    
    # Access the API keys using the key names
    x_rapidapi_key = api_keys['X-RapidAPI-Key']
    x_rapidapi_host = api_keys['X-RapidAPI-Host']
    
    # Create the headers dictionary with the API keys
    headers = {
        "X-RapidAPI-Key": x_rapidapi_key,
        "X-RapidAPI-Host": x_rapidapi_host
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

def download_videos(df, column_name, hashtag):
    output_dir = r'C:\github_repos\tiktok_app\project\output_files\videos_{}'.format(hashtag)  # Specify the output directory path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, url in enumerate(df[column_name]):
        start_time = time.time()
        try:
            response = requests.get(url)
            with open(os.path.join(output_dir, f'video_{i}.mp4'), 'wb') as f:  # Update the file path to include the output directory
                f.write(response.content)
                elapsed_time = time.time() - start_time
                elapsed_mins = int(elapsed_time // 60)
                elapsed_secs = int(elapsed_time % 60)
                print(f"Video {i} downloaded successfully in {elapsed_mins} minutes and {elapsed_secs} seconds.")
        except:
            print(f"Error downloading video {i}.")

def search_hashtag_on_tiktok(hashtag):
    tiktokSite = "https://www.tiktok.com"
    pathChrome = r"C:\github_repos\tiktok_app\chromedriver.exe"

    browser_options = webdriver.ChromeOptions()
    browser_options.add_argument("--no-sandbox")
    browser_options.add_argument("disable-notifications")
    browser_options.add_argument("--disable-infobars")
    browser_options.add_argument("--start-maximized")
    browser_options.add_argument("--incognito")
    browser_options.add_argument("--disable-extensions")
    browser_options.add_argument("--disable-cookies")
    browser_options.add_argument('--lang=en-US')

    driver = webdriver.Chrome(executable_path=pathChrome, options=browser_options)
    
    driver.get(tiktokSite)

    search_bar = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Search accounts and videos']"))
    )

    time.sleep(5)
    search_bar.send_keys(hashtag)
    search_bar.send_keys(Keys.ENTER)

    # Click "Load more" button until all videos are available
    start_time = time.time()
    while True:
        try:
            load_more_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button[@data-e2e='search-load-more']")))
            load_more_button.click()
            time.sleep(5)
        except:
            break

    end_time = time.time()
    elapsed_time = end_time - start_time

    minutes = int(elapsed_time / 60)
    seconds = int(elapsed_time % 60)

    print(f"Execution time: {minutes} minutes {seconds} seconds")
            
# Data HTML page TikTok with BeutifulSoup4
    html = driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
    soup = BeautifulSoup(html, 'html.parser')

# Scraping TikTok data
    description_tiktok, link_tiktok, username, title_tiktok, hashtag_tiktok, userid, strongVideoCount, published_date=[], [], [], [], [], [], [], []

# Get TikTok Video Title
    for title in soup.find_all('div', class_='tiktok-11cua35-DivContainer ejg0rhn0'):
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
    for link in soup.find_all('div', class_='tiktok-yz6ijl-DivWrapper e1cg0wnj1'):
        #print(link['href'])
        link_tiktok.append(link.a['href'])

# Get TikTok Video Username
    for user in soup.find_all('p', class_= 'tiktok-2zn17v-PUniqueId etrd4pu6'):
        username.append(user.text)

# Get TikTok Video UserId
    for id in soup.find_all('p', class_= 'tiktok-2zn17v-PUniqueId etrd4pu6'):
        userid.append(id.text)

# Get TikTok Video StrongVideoCount
    for video in soup.find_all('strong', class_ = 'tiktok-ws4x78-StrongVideoCount etrd4pu10'):
        strongVideoCount.append(video.text)

# Get TikTok Video Published Date
    for date in soup.find_all('div', class_ = 'tiktok-842lvj-DivTimeTag e19c29qe14'):
        published_date.append(date.text)

# Save data in dictionary format
    listCols = ['TikTok Url', 'TikTok Description', 'TikTok Title', 'TikTok Hashtags', 'Username', 'User Id', 'Video Views Count', 'Published Date']
    dict_continued = dict(zip(listCols, (link_tiktok, description_tiktok, title_tiktok, hashtag_tiktok, username, userid, strongVideoCount, published_date)))

# Create a Pandas DataFrame from the Dictionary
    df = pd.DataFrame(dict_continued)

# Extract emojis from TikTok Title
    df['Emojis In Title'] = df['TikTok Title'].apply(extract_emojis)

# Extract emojis from TikTok Description
    df['Emojis In Description'] = df['TikTok Description'].apply(extract_emojis)

    start_time = time.time()
    df_tiktok_attributes = df['TikTok Url'].apply(extract_tiktok_attributes)
    elapsed_time = time.time() - start_time
    elapsed_time_minutes = int(elapsed_time // 60)
    elapsed_time_seconds = int(elapsed_time % 60)
    print(f"Elapsed Time for Retrieving API attributes: {elapsed_time_minutes} minutes, {elapsed_time_seconds} seconds.")


# Concatenate the original DataFrame with the new one
    df_result = pd.concat([df, df_tiktok_attributes], axis=1)

# Specify the columns to keep in the final dataframe
    columns_to_keep = ['TikTok Url',
                    'TikTok Description',
                    'TikTok Title',
                    'TikTok Hashtags',
                    'Username',
                    'User Id',
                    'Video Views Count',
                    'Published Date',
                    'Emojis In Title',
                    'Emojis In Description',
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
# Create the final DataFrame with only the selected columns
    df_result = df_result[columns_to_keep]
    print('Dataframe Shape:', df_result.shape)
    print('Dataframe Head:')
    df_result['Processed TikTok Title'] = df_result['TikTok Title'].apply(lambda row: clean_text_wt_v1(row))
    df_result['Processed TikTok Description'] = df_result['TikTok Description'].apply(lambda row: clean_text_wt_v1(row))


    # ----- Most Frequent Hashtags ----- #
    hashtag_col = 'TikTok Hashtags'
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

    ##description_long_string = ','.join(list(df_result['Processed TikTok Description'].values))
    ##description_wordcloud = WordCloud(background_color="white", max_words=3000, contour_width=3, contour_color='steelblue',width=1200, height=600)
    ##description_wordcloud.generate(description_long_string)
    ##description_wordcloud.to_image()

    # ----- TikTok Title WordCloud ----- #
    df_result['Processed TikTok Title'] = \
    df_result['Processed TikTok Title'].map(lambda x: re.sub('[,\.!?]', '', x))
    df_result['Processed TikTok Title'] = \
    df_result['Processed TikTok Title'].map(lambda x: x.lower())
    df_result['Processed TikTok Title'].head()

    ##title_long_string = ','.join(list(df_result['Processed TikTok Title'].values))
    ##title_wordcloud = WordCloud(background_color="white", max_words=3000, contour_width=3, contour_color='steelblue',width=1200, height=600)
    ##title_wordcloud.generate(title_long_string)
    ##title_wordcloud.to_image()
    
    output_directory = r"C:\github_repos\tiktok_app\project\output_files"
    file_name = f"{hashtag}_tiktok_data"
    output_file_path = os.path.join(output_directory, "scrapping_" + file_name + "_top.xlsx")
    df_result.to_excel(output_file_path, index=False)
    
    print("Excel file saved successfully at:", output_file_path)
    df_result.to_excel(output_directory, "scrapping_" + file_name + "_top.xlsx", index=False)
    
       # ----- Downloading Videos ----- #
    print("Starting Downloading Videos...")
    download_videos(df_result, 'video_url_list_2', hashtag)
    
    return df_result

    # ----- Call the function ----- #
if __name__ == "__main__":
    hashtag = "#cocacolacolombia"
    df = search_hashtag_on_tiktok(hashtag)
    print(df.shape)
    df.head(10)