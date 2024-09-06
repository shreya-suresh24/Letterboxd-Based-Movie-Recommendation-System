import requests
from bs4 import BeautifulSoup
import pandas as pd
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

def fetch_liked_movies_from_pages(user):
    base_likes_url = f'https://letterboxd.com/{user}/likes/films/page/'
    page_number = 1
    liked_movies = set()

    while True:
        url = f'{base_likes_url}{page_number}/'
        print(f"Fetching likes page {page_number}...")
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Failed to retrieve likes page {page_number}. Status code: {response.status_code}")
            break
        
        soup = BeautifulSoup(response.text, 'html.parser')
        poster_list = soup.find('ul', class_='poster-list')
        
        if not poster_list:
            print(f"No poster list found on likes page {page_number}. Ending fetch.")
            break
        
        posters = poster_list.find_all('li', class_='poster-container')
        if not posters:
            print(f"No more movies found on likes page {page_number}. Ending fetch.")
            break
        
        for poster in posters:
            img_tag = poster.find('img')
            if img_tag and img_tag.has_attr('alt'):
                movie_title = img_tag['alt']
                liked_movies.add(movie_title)
        
        page_number += 1

    return liked_movies

def fetch_liked_high_rated_movies(user):
    base_url = f'https://letterboxd.com/{user}/films/diary/page/'
    page_number = 1
    liked_high_rated_movies = set()  # Use a set to avoid duplicates
    
    while True:
        url = f'{base_url}{page_number}/'
        print(f"Fetching diary page {page_number}...")
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Failed to retrieve diary page {page_number}. Status code: {response.status_code}")
            break
        
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all('tr', class_='diary-entry-row')

        if not rows:
            print(f"No diary entries found on page {page_number}. Ending fetch.")
            break

        for row in rows:
            # Check if the entry is liked
            liked = row.find('span', class_='icon-liked')
            if liked:
                # Extract rating
                rating_tag = row.find('input', class_='rateit-field')
                rating_value = int(rating_tag['value']) if rating_tag else 0
                
                if rating_value >= 4:
                    # Extract movie title
                    movie_title_tag = row.find('h3', class_='headline-3 prettify')
                    if movie_title_tag:
                        movie_title = movie_title_tag.get_text(strip=True)
                        liked_high_rated_movies.add(movie_title)  # Use set to avoid duplicates

        page_number += 1

    return liked_high_rated_movies

def fetch_liked_and_high_rated_movies(user):
    try:
        liked_movies_from_diary = fetch_liked_high_rated_movies(user)
        liked_movies_from_likes_page = fetch_liked_movies_from_pages(user)

        combined_liked_movies = liked_movies_from_diary.union(liked_movies_from_likes_page)
        print("Combined liked movies rated 4 stars and above:")
        for movie in combined_liked_movies:
            print(movie)
    except Exception as e:
        print(f"An error occurred: {e}")

def fetch_watchlist_movies(user):
    base_watchlist_url = f'https://letterboxd.com/{user}/watchlist/page/'
    page_number = 1
    watchlist_movies = set()

    while True:
        url = f'{base_watchlist_url}{page_number}/'
        print(f"Fetching watchlist page {page_number}...")
        response = requests.get(url)
        # print(response)
        if response.status_code != 200:
            print(f"Failed to retrieve watchlist page {page_number}. Status code: {response.status_code}")
            break
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all 'div' elements with the class 'film-poster' which contain the movie details
        posters = soup.find_all('div', class_='film-poster')
        # print("posters")
        # print(page_number)
        # print(posters)

        if not posters:
            print(f"No more movies found on watchlist page {page_number}. Ending fetch.")
            break

        # Extract movie titles from the nested img tag's 'alt' attribute
        for poster in posters:
            img_tag = poster.find('img')  # Find the img tag inside the div
            if img_tag and img_tag.has_attr('alt'):
                movie_title = img_tag['alt']
                watchlist_movies.add(movie_title)

        # Find the next page link; if it doesn't exist, break the loop
        next_page = soup.find('a', class_='next')
        if not next_page:
            print(f"No more pages. Ending at page {page_number}.")
            break

        page_number += 1

    return watchlist_movies

def display_watchlist_movies(user):
    try:
        watchlist_movies = fetch_watchlist_movies(user)
        print(f"\nMovies in {user}'s watchlist:")
        for movie in watchlist_movies:  # Sorting for neat display
            print(movie)
    except Exception as e:
        print(f"An error occurred: {e}")

user = 'ShreSuresh'  # Replace with the desired username
print(fetch_liked_and_high_rated_movies(user))
print(display_watchlist_movies(user))

# Load IMDb Basics dataset
title_basics = pd.read_csv(r"C:\Users\Shreya\Downloads\title.basics (1).tsv.gz", sep='\t', dtype=str, compression='gzip', low_memory=False)
movies_df = title_basics[title_basics['titleType'] == 'movie']

def get_imdb_info(movie_title, movies_df):
    # Find the closest match from the IMDb dataset
    best_match = process.extractOne(movie_title, movies_df['primaryTitle'])
    
    if best_match:
        matched_title = best_match[0]
        matched_row = movies_df[movies_df['primaryTitle'] == matched_title]
        
        if not matched_row.empty:
            imdb_id = matched_row['tconst'].values[0]
            year = matched_row['startYear'].values[0]
            genres = matched_row['genres'].values[0]
            
            return {
                'title': matched_title,
                'imdb_id': imdb_id,
                'year': year,
                'genres': genres
            }
    return None

watchlist_movies = fetch_watchlist_movies(user)
movies_info = []

for movie_title in watchlist_movies:
    imdb_info = get_imdb_info(movie_title, movies_df)
    
    if imdb_info:
        movie_data = {
            'title': imdb_info['title'],
            'year': imdb_info['year'],
            'genres': imdb_info['genres']
        }
        print(movie_data)
#         movies_info.append(movie_data)

# # Display or save the information
# for movie in movies_info:
#     print(f"Title: {movie['title']}, Year: {movie['year']}, Genres: {movie['genres']}")


