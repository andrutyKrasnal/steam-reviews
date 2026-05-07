# app.py

import os
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import quote_plus
from wordcloud import WordCloud

load_dotenv()
STEAM_API_KEY = os.getenv("STEAM_API_KEY")

st.set_page_config(page_title = "Analizator Sentymentu Steam", layout = "centered")


@st.cache_data
def get_app_list():
    apps = []
    last_appid = 0
    have_more_results = True
    
    while have_more_results:        
        url = f"https://api.steampowered.com/IStoreService/GetAppList/v1/?key={STEAM_API_KEY}&include_games=true&max_results=50000&last_appid={last_appid}"
        
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            if 'response' in data and 'apps' in data['response']:
                apps.extend(data['response']['apps'])
                
                have_more_results = data['response'].get('have_more_results', False)
                if have_more_results:
                    last_appid = data['response'].get('last_appid', 0)
            else:
                st.error("Otrzymano nieoczekiwany format danych z API.")
                break
                
        except requests.exceptions.RequestException as e:
            st.error(f"Błąd podczas połączenia z API Steam: {e}")
            break

    if apps:
        return pd.DataFrame(apps)
    else:
        return None


def fetch_english_reviews(appid, max_reviews = 500):

    reviews_data = []
    summary = {}
    cursor = '*'
    reviews_per_page = 100

    while len(reviews_data) < max_reviews:
        encoded_cursor = quote_plus(cursor)
        url = f"https://store.steampowered.com/appreviews/{appid}?json=1&filter=recent&language=english&num_per_page={reviews_per_page}&cursor={encoded_cursor}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('success') == 1:
                if not summary and 'query_summary' in data:
                    summary = data['query_summary']
                
                if 'reviews' in data and data['reviews']:
                    for review in data['reviews']:
                        reviews_data.append({
                            "review_text": review.get('review', ''),
                            "voted_up": review.get('voted_up', False)
                        })
                    
                    cursor = data.get('cursor')
                    if not cursor or len(data['reviews']) < reviews_per_page:
                        break
                else:
                    break
            else:
                break
        except requests.exceptions.RequestException:
            break

    return (reviews_data[:max_reviews], summary)

def analyse_sentiment_vader(reviews_data):

    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    positive_reviews_text = []
    negative_reviews_text = []
    reviews_with_scores = []

    for review in reviews_data:
        review_text = review.get('review_text', '')
        score = analyzer.polarity_scores(review_text)
        compound_score = score['compound']
        
        reviews_with_scores.append({
            "text": review_text, 
            "score": compound_score,
            "voted_up": review.get('voted_up', False)
        })
        
        if compound_score >= 0.05:
            sentiments.append('Pozytywny')
            positive_reviews_text.append(review_text)
        elif compound_score <= -0.05:
            sentiments.append('Negatywny')
            negative_reviews_text.append(review_text)
        else:
            sentiments.append('Neutralny')
            
    results = {
        "counts": pd.Series(sentiments).value_counts(),
        "positive_text": " ".join(positive_reviews_text),
        "negative_text": " ".join(negative_reviews_text),
        "reviews_with_scores": reviews_with_scores
    }
    return results


st.title("Analizator Sentymentu Recenzji Steam") 

apps_df = get_app_list()

if apps_df is not None:

    apps_df = apps_df.dropna(subset=['name'])
    
    game_options = apps_df['name'].tolist()

    selected_game_name = st.selectbox(
        "Wyszukaj i wybierz grę:",
        options=game_options,
        index=None,
        placeholder="Zacznij wpisywać nazwę gry...",
    )

    app_id = None
    if selected_game_name:
        app_id = int(apps_df[apps_df['name'] == selected_game_name].iloc[0]['appid'])
        st.caption(f"Wybrano: {selected_game_name} | AppID: {app_id}")
else:
    st.error("Nie udało się pobrać listy gier.")
    app_id = None

num_reviews = st.slider("Wybierz liczbę recenzji do analizy:", min_value=100, max_value=5000, value=1000, step=50)

if st.button("Analizuj wybraną grę"):
    if app_id:
        st.info(f"Wybrano grę: **{selected_game_name}** (AppID: {app_id}). Pobieranie i analiza recenzji...")
        
        with st.spinner("Pobieranie i analiza recenzji..."):
            reviews_list, summary = fetch_english_reviews(app_id, max_reviews=num_reviews)
            
            if reviews_list:
                analysis_results = analyse_sentiment_vader(reviews_list)
                sentiment_counts = analysis_results["counts"]
                
                st.success(f"Analiza zakończona! Przeanalizowano **{len(reviews_list)}** recenzji.")

                if summary:
                    st.subheader("Ogólna ocena ze Steam")
                    total_reviews = summary.get('total_reviews', 0)
                    pos_percent = 0
                    if total_reviews > 0:
                        pos_percent = summary.get('total_positive', 0) / total_reviews * 100
                    
                    cols = st.columns(3)
                    cols[0].metric("Opis oceny", summary.get('review_score_desc', 'Brak'))
                    cols[1].metric("Pozytywnych recenzji", f"{pos_percent:.1f}%")
                    cols[2].metric("Liczba wszystkich opinii", f"{total_reviews:,}")

                st.subheader("Podsumowanie sentymentu analizatora")
                fig, ax = plt.subplots()
                colors = {'Pozytywny': 'lightgreen', 'Negatywny': 'lightcoral', 'Neutralny': 'lightskyblue'}
                pie_colors = [colors.get(label, 'gray') for label in sentiment_counts.index]

                ax.pie(sentiment_counts, labels = sentiment_counts.index, autopct = '%1.1f%%', startangle = 90, colors = pie_colors)
                ax.axis('equal')
                st.pyplot(fig)
                st.dataframe(sentiment_counts)

                st.markdown("---")
                if analysis_results["positive_text"]:
                    st.subheader("Najczęściej używane słowa w recenzjach pozytywnych")
                    wordcloud_pos = WordCloud(width = 800, height = 400, background_color = 'white').generate(analysis_results["positive_text"])
                    fig_pos, ax_pos = plt.subplots()
                    ax_pos.imshow(wordcloud_pos, interpolation = 'bilinear')
                    ax_pos.axis('off')
                    st.pyplot(fig_pos)

                if analysis_results["negative_text"]:
                    st.subheader("Najczęściej używane słowa w recenzjach negatywnych")
                    wordcloud_neg = WordCloud(width = 800, height = 400, background_color = 'black', colormap = 'Reds').generate(analysis_results["negative_text"])
                    fig_neg, ax_neg = plt.subplots()
                    ax_neg.imshow(wordcloud_neg, interpolation = 'bilinear')
                    ax_neg.axis('off')
                    st.pyplot(fig_neg)

                st.markdown("---")
                st.subheader("Najbardziej skrajne recenzje")

                scored_reviews = analysis_results["reviews_with_scores"]
                scored_reviews.sort(key=lambda x: x["score"])

                st.markdown("#### Najbardziej negatywne opinie:")
                for i in range(min(3, len(scored_reviews))):
                    review = scored_reviews[i]
                    if review["score"] < -0.05:
                        icon = "👎" if not review["voted_up"] else "👍"
                        st.error(f"**Opinia autora: {icon} | Wynik sentymentu: {review['score']:.2f}**\n\n{review['text']}")

                st.markdown("#### Najbardziej pozytywne opinie:")
                for i in range(min(3, len(scored_reviews))):
                    review = scored_reviews[-(i+1)]
                    if review["score"] > 0.05:
                        icon = "👍" if review["voted_up"] else "👎"
                        st.success(f"**Opinia autora: {icon} | Wynik sentymentu: {review['score']:.2f}**\n\n{review['text']}")

            else:
                st.warning("Nie znaleziono wystarczającej liczby anglojęzycznych recenzji dla tej gry.")
    else:
        st.warning("Proszę wpisać fragment nazwy i wybrać grę z listy przed analizą.")