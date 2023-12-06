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

def search_video_comments_on_tiktok(video_url):
    if "tiktok.com/@" not in video_url:
        print("Invalid TikTok video URL.")
        return None
    
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
    
    # Open Chrome browser
    driver = webdriver.Chrome()
    
    # Change the TikTok link
    driver.get(video_url)

    time.sleep(15)

    start_time = time.time()

    scroll_pause_time = 15
    screen_height = driver.execute_script("return window.screen.height;")
    i = 1

    # ----- Scrolling page to load all comments ----- #
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

    print(f"Execution time: {minutes} minutes {seconds} seconds")

    # Get the current video URL
    current_url = driver.current_url

    # Extract the video ID from the URL
    video_id = current_url.split("/")[-2]

    # Data HTML page TikTok with BeutifulSoup4
    html = driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
    soup = BeautifulSoup(html, "html.parser")

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
    print('Dataframe Shape:', df.shape)
    print('Dataframe Head:')
    filename = f"scrapping_user_video_{video_id}.xlsx"
    df.to_excel(filename, index=False)

    driver.quit()
    return df

# ----- Call the function to get the comments -----
if __name__ == "__main__":
    #video_url = "https://www.tiktok.com/@henryjimenezkerbox/video/7217698305608797482"
    #video_url= "https://www.tiktok.com/@esaidrdz_gg/video/7234646068120521990?q=rivers%20snickers&t=1685465696545"
    video_url= "https://www.tiktok.com/@blurry21_gg/video/7234670768217476358?q=rivers%20snickers&t=1685465696545"
    df = search_video_comments_on_tiktok(video_url)
    #print(df.shape)
    #df.head(10)