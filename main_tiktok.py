import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import os
import time 
import re
import plotly.express as px
import json
import streamlit as st
import streamlit.components.v1 as components
import warnings
import base64
from io import BytesIO
from itertools import zip_longest

import requests
from urllib.request import urlopen
from PIL import Image
#from textblob import TextBlob

#import gensim
#from gensim import corpora
import random
#import pyLDAvis
#from gensim.models import CoherenceModel
#from gensim.utils import simple_preprocess
#from pprint import pprint
#import pyLDAvis.gensim_models
#from helper_functions import open_html

#from wordcloud import WordCloud

from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


#st.sidebar.title('TikTok Scrapper Application:')

#option = st.sidebar.selectbox('Select an option:', ('By Hashtag Or Keyword', 'By Username Or User Account', 'By Extracting Historical Comments'))
# Set page config
#st.beta_set_page_config(page_title="My App", page_icon=":rocket:", layout="wide")

# Import necessary libraries

# Load the logo image
#logo = Image.open(r"C:\Users\roman\Downloads\Logo Principal_blanco_fondotrnsp.png")

# Display the logo on the sidebar
#st.sidebar.image(logo, width=300, use_column_width=False)


st.title("BIU TikTok Data Solution")
#@st.cache_data()
# Suppress warnings
def get_sentiment(description):
    '''
    Function to analyze the sentiment of a text using TextBlob.
    Input:
        - description: str, the text to analyze.
    Output:
        - str, the sentiment of the text: positive, negative, or neutral.
    '''
    # Detect the language of the text
    #lang = TextBlob(description).detect_language()
    
    # Translate the text to English if it's not already in English
    #if lang != 'en':
    #    description = TextBlob(description).translate(to='en')
    
    # Analyze the sentiment of the text
    #sentiment = TextBlob(description).sentiment.polarity
    
    # Classify the sentiment as positive, negative, or neutral
    #if sentiment > 0:
        #return 'positive'
    #elif sentiment < 0:
    #    return 'negative'
    #else:
    #    return 'neutral'

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
        with st.spinner(f"Downloading video {i}..."):
            start_time = time.time()
            try:
                response = requests.get(url)
                with open(output_dir + f'video_{i}.mp4', 'wb') as f:
                    f.write(response.content)
                elapsed_time = time.time() - start_time
                elapsed_mins = int(elapsed_time // 60)
                elapsed_secs = int(elapsed_time % 60)
                st.write(f"Video {i} downloaded successfully in {elapsed_mins} minutes and {elapsed_secs} seconds.")
            except:
                st.write(f"Error downloading video {i}.")    

def extract_tiktok_attributes(link):
    url = "https://tiktok-videos-without-watermark.p.rapidapi.com/getVideo"

    querystring = {"url":f"{link}?is_from_webapp=1&sender_device=pc&web_id=7085812281682642434"}
    #st.write(querystring)

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

#@st.cache_data()
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

#@st.cache_data()
def search_hashtag_on_tiktok(hashtag):
    tiktokSite = "https://www.tiktok.com"
    pathChrome = r"C:\github_repos\tiktok_app\chromedriver.exe"

    browser_options = webdriver.ChromeOptions()
    browser_options.add_argument("--no-sandbox")
    browser_options.add_argument("disable-notifications")
    browser_options.add_argument("--disable-infobars")
    browser_options.add_argument("--start-maximized")

    driver = webdriver.Chrome(executable_path=pathChrome, options = browser_options)

    driver.get(tiktokSite)

    search_bar = WebDriverWait(driver, 10).until(
      EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Search accounts and videos']"))
      )

    time.sleep(5)
    search_bar.send_keys(hashtag)
    search_bar.send_keys(Keys.ENTER)

    with st.spinner("Fetching TikTok videos. This may take a few minutes..."):
        start_time = time.time()
        ii = 0
        while ii < 1:
            try:
                driver.find_element(By.XPATH, "/html/body/div[2]/div[2]/div[2]/div[1]/div/div[1]/div[1]/div[1]").click()
                ii = 1
            except:
                ii = 0
                time.sleep(5)

        i = 0
        while i < 100:
            try:
                NextStory = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div[2]/div[2]/div[2]/div[2]/button")))
                NextStory.click()
                time.sleep(2)
            except:
                i = 100
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)
        st.write(f"Elapsed Time for Fetching TikTok Videos: {elapsed_minutes} minutes {elapsed_seconds} seconds")


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
    #print(df.shape)
    #df.head()

    # Apply the get_sentiment function to the 'description' column
    #df['Sentiment'] = df['TikTok Description'].apply(get_sentiment)


# Extract emojis from TikTok Title
    df['Emojis In Title'] = df['TikTok Title'].apply(extract_emojis)
    #print(df.shape)
    #df.head()

# Extract emojis from TikTok Description
    df['Emojis In Description'] = df['TikTok Description'].apply(extract_emojis)
    #print(df.shape)
    #df.head()

# Apply the function to the "link_tiktok" column and create a new DataFrame with the results
    #df = df.head(10)
    with st.spinner("Retrieving API attributes. This may take a moment, please wait..."):
        start_time = time.time()
        df_tiktok_attributes = df['TikTok Url'].apply(extract_tiktok_attributes)
        elapsed_time = time.time() - start_time
        elapsed_time_minutes = int(elapsed_time // 60)
        elapsed_time_seconds = int(elapsed_time % 60)
        st.write(f"Elapsed Time for Retrieving API attributes: {elapsed_time_minutes} minutes, {elapsed_time_seconds} seconds.")
    #print(df_tiktok_attributes.shape)
    #df_tiktok_attributes.head()


# Concatenate the original DataFrame with the new one
    df_result = pd.concat([df, df_tiktok_attributes], axis=1)
    #print(df_result.shape)
    #df_result.head()

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
    st.write('Dataframe Shape:', df_result.shape)
    st.write('Dataframe Head:')
    #st.write(df_result.head())
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
    st.plotly_chart(fig)

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
    st.plotly_chart(fig)

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

    st.write("TikTok Description Keywords in Conversation")
    st.image(description_wordcloud.to_image(), use_column_width=True)


    # ----- TikTok Title WordCloud ----- #
    df_result['Processed TikTok Title'] = \
    df_result['Processed TikTok Title'].map(lambda x: re.sub('[,\.!?]', '', x))
    df_result['Processed TikTok Title'] = \
    df_result['Processed TikTok Title'].map(lambda x: x.lower())
    df_result['Processed TikTok Title'].head()

    title_long_string = ','.join(list(df_result['Processed TikTok Title'].values))
    title_wordcloud = WordCloud(background_color="white", max_words=3000, contour_width=3, contour_color='steelblue',width=1200, height=600)
    title_wordcloud.generate(title_long_string)
    title_wordcloud.to_image()

    st.write("TikTok Title Keywords in Conversation")
    st.image(title_wordcloud.to_image(), use_column_width=True)

    # ----- PLDAVIS GRAPH ----- #
    def sent_to_words(sentences):
        for sentence in sentences:
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)
        
    def define_stopwords():
        sw_es=stopwords.words('spanish')
        sw_pt=stopwords.words('portuguese')
        sw_en=stopwords.words('english')
        sw_es_title=[word.title() for word in sw_es]
        sw_pt_title=[word.title() for word in sw_pt]
        sw_en_title=[word.title() for word in sw_en]
        return sw_es, sw_pt, sw_en, sw_es_title, sw_pt_title, sw_en_title


    stopwords_result = define_stopwords()
    sw_es, sw_pt, sw_en, sw_es_title, sw_pt_title, sw_en_title = [stopwords_result[i] for i in range(0,6)]
    sw_total = sw_es + sw_pt + sw_en +  sw_es_title +  sw_pt_title + sw_en_title

    def remove_stopwords(texts):
        stop_words = sw_total
        return [[word for word in simple_preprocess(str(doc))
             if word not in sw_total] for doc in texts]
    
    tiktok_description = df_result['Processed TikTok Description'].values.tolist()
    tiktok_words_content = list(sent_to_words(tiktok_description))
    tiktok_words_content = remove_stopwords(tiktok_words_content)
    #st.write(tiktok_words_content[:1][0][:30])
    #st.write(tiktok_words_content)
    
    # Create a DataFrame from the list of words
    df_tiktok_words = pd.DataFrame({'words': tiktok_words_content})

    # Display the DataFrame
    st.write(df_tiktok_words.head())



    id2word = corpora.Dictionary(tiktok_words_content)
    texts = tiktok_words_content
    corpus_description = [id2word.doc2bow(text) for text in texts]
    #st.write(corpus_description[:1][0][:10])
    #st.write(corpus_description)


    #num_topics = 10
    #lda_model = gensim.models.LdaMulticore(corpus=corpus_description,
    #                                    id2word=id2word,
    #                                    num_topics=num_topics)

    #pprint(lda_model.print_topics())
    #doc_lda = lda_model[corpus_description]

    #description_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus_description, id2word)
    #st.pydeck_chart(description_vis)


    # ----- Downloading Videos ----- #
    st.write("Starting Downloading Videos...")
    download_videos(df_result, 'video_url_list_2', hashtag)


    #pyLDAvis.save_html(description_vis, 'tiktok_description_vis.html')
        
    # display html page in streamlit
    # open file and decode it, then serve
    #description_vis = open("tiktok_description_vis.html", 'r', encoding='utf-8')
    #source_code = description_vis.read() 
    #components.html(source_code, height = 800, scrolling=True) 


    file_name = f"{hashtag}_tiktok_data"
    df_result.to_excel("scrapping_" + file_name + "_top.xlsx", index=False)
    return df_result


    #df_result.to_excel('scraping_' + hashtag + '_top.xlsx', index=False)
    
    # create a button to save the Excel file
    #if st.button('Save File'):
    #    with st.spinner('Saving file...'):
    #        df_result.to_excel('dataframe.xlsx', index=False)
    #        st.success('File saved successfully!')


# Print the shape of the final dataframe and the first few rows
    #print(df_final.shape)
    #df_final.head()

    #@st.cache(suppress_st_warning=True)
    #def download_button(df):
    #    output = BytesIO()
    #    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    #    df.to_excel(writer, sheet_name='Sheet1', index=False)
    ##    writer.save()
     #   processed_data = output.getvalue()
     ###   b64 = base64.b64encode(processed_data)
       # download_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64.decode()}" download="tiktok_data.xlsx">Download Excel File</a>'
       # return download_link


    #df_xlsx = to_excel(df_result)
    #st.download_button(label=' Download Current Result',
    #                                data=df_xlsx ,
    #                                file_name= 'scraping_' + hashtag + '_top.xlsx')

def search_useraccount_on_tiktok(url): # (By Username Or User Account)
    if "tiktok.com/@" not in url:
        st.write("Invalid TikTok user account URL.")
        return None
    #st.write("STEP 2: Open Chrome browser")
    driver = webdriver.Chrome()

    # Customize chrome display
    browser_options = webdriver.ChromeOptions()
    browser_options.add_argument("--no-sandbox")
    browser_options.add_argument("--disable-infobars")
    browser_options.add_argument("--start-maximized")
    browser_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")

    prefs = {"profile.default_content_setting_values.notifications" : 2}
    browser_options.add_experimental_option("prefs", prefs)

    # Open Chrome browser
    driver = webdriver.Chrome(options=browser_options)

    # Change the tiktok link
    #tiktok_url = st.text_input("Enter TikTok URL", "")
    driver.get(url)

    # IF YOU GET A TIKTOK CAPTCHA, CHANGE THE TIMEOUT HERE
    # to 60 seconds, just enough time for you to complete the captcha yourself.
    time.sleep(10)

    scroll_pause_time = 1
    screen_height = driver.execute_script("return window.screen.height;")
    i = 1

    #st.write("STEP 3: Scrolling page")
    with st.spinner("Fetching User Account TikTok Videos Information. This may take a few minutes..."):
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

    #df = df.head(15)

    with st.spinner("Retrieving API attributes. This may take a moment, please wait..."):
            start_time = time.time()
            df_tiktok_attributes = df['Link'].apply(extract_tiktok_attributes)
            elapsed_time = time.time() - start_time
            elapsed_time_minutes = int(elapsed_time // 60)
            elapsed_time_seconds = int(elapsed_time % 60)
            st.write(f"Elapsed Time for Retrieving API attributes: {elapsed_time_minutes} minutes, {elapsed_time_seconds} seconds.")

    df_result = pd.concat([df, df_tiktok_attributes], axis=1)

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

    # Create the final DataFrame with only the selected columns
    st.write('Dataframe Shape:', df_result.shape)
    st.write('Dataframe Head:')
    #st.write(df_final.head(10))
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
    st.plotly_chart(fig)

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
    st.plotly_chart(fig)

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

    st.write("TikTok Description Keywords in Conversation")
    st.image(description_wordcloud.to_image(), use_column_width=True)

    # ----- Downloading Videos ----- #
    st.write("Starting Downloading Videos...")
    username = url.replace("https://www.tiktok.com/@", "")
    download_videos(df_result, 'video_url_list_2', username)


    file_name = f"{url}_user_account"
    df_result.to_excel("scrapping_user_account_" + url.replace("https://www.tiktok.com/@", "") + ".xlsx", index=False)
    return df_result

def search_video_comments_on_tiktok(video_url):
    if "tiktok.com/@" not in video_url:
        st.write("Invalid TikTok video URL.")
        return None
    # Open Chrome browser
    driver = webdriver.Chrome()
    
    # Change the TikTok link
    driver.get(video_url)
    
    time.sleep(3)
    
    start_time = time.time()
    
    scroll_pause_time = 15
    screen_height = driver.execute_script("return window.screen.height;")
    i = 1
    
    #print("STEP 3: Scrolling page")
    with st.spinner("Fetching comments from TikTok video. This may take a few minutes..."):
        while True:
            driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))  
            i += 1
            time.sleep(scroll_pause_time)
            scroll_height = driver.execute_script("return document.body.scrollHeight;")  
            if (screen_height) * i > scroll_height:
                break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    minutes = int(elapsed_time / 60)
    seconds = int(elapsed_time % 60)
    
    st.write(f"Execution time: {minutes} minutes {seconds} seconds")
    
    # Get the current video URL
    current_url = driver.current_url

    # Extract the video ID from the URL
    video_id = current_url.split("/")[-2]
    
    # Data HTML page TikTok with BeutifulSoup4
    html = driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
    soup = BeautifulSoup(html, "html.parser")
    st.write(soup)
    
    # Get All TikTok Video Comments
    comments = []
    for comment in soup.find_all("p", class_= "tiktok-q9aj5z-PCommentText e1g2efjf6"):
        comments.append(comment.text)
    
    # Get Usernames/Nicknames
    usernames = []
    for username in soup.find_all("a", class_ = "e1g2efjf4 tiktok-1oblbwp-StyledLink-StyledUserLinkName er1vbsz0"):
        usernames.append(username.text)
    
    # Get Video Likes
    likes = []
    for like in soup.find_all("strong",{"data-e2e": "browse-like-count", "class": "tiktok-14xas1m-StrongText e1hk3hf92"}):
        likes.append(like.text)
    
    # Get Video Total Comments
    total_comments = []
    for total_comment in soup.find_all("strong", class_ = "tiktok-14xas1m-StrongText e1hk3hf92"):
        total_comments.append(total_comment.text)
    
    # Create a Pandas DataFrame to store the comments
    df = pd.DataFrame({"TikTok Video": [current_url]*len(comments), "Comments": comments, "Username":usernames, "Video Likes": [likes]*len(comments), "Video Total Comments": [total_comments]*len(comments)})
    print(df.shape)
    
    # Extract Emojis In Comments
    df['Emojis In Comments'] = df['Comments'].apply(extract_emojis)
    print(df.shape)

    # Create the final DataFrame with only the selected columns
    st.write('Dataframe Shape:', df.shape)
    st.write('Dataframe Head:')
    #st.write(df_final.head(10))
    filename = f"scrapping_user_video_{video_id}.xlsx"
    df.to_excel(filename, index=False)

    driver.quit()
    return df



# Main function
def main():
    warnings.filterwarnings("ignore")
    st.sidebar.title('Endpoints:')
    option = st.sidebar.selectbox('Select an option:', ('By Hashtag Or Keyword', 'By Username Or User Account', 'By Extracting Historical Comments'))

    if option == 'By Hashtag Or Keyword':
        hashtag = st.text_input('Enter a hashtag or keyword:')
        if st.button('Search'):
            df_result = search_hashtag_on_tiktok(hashtag)
            st.write(df_result)
            st.write("Search Completed!")
            st.write("Excel File Saved Successfully!")


    elif option == 'By Username Or User Account':
        url = st.text_input('Enter a TikTok user account URL:')
        if st.button('Search'):
            df_result = search_useraccount_on_tiktok(url)
            st.write(df_result)
            st.write("Search Completed!")
            st.write("Excel File Saved Successfully!")
    
    elif option == 'By Extracting Historical Comments':
        video_url = st.text_input('Enter a TikTok video URL:')
        if st.button('Search'):
            df = search_video_comments_on_tiktok(video_url)
            st.write(df)
            st.write("Search Completed!")
            st.write("Excel File Saved Successfully!")

if __name__ == '__main__':
    main()