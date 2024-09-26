import sys
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service as ChromeService 
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
import time  # For page load delays
import requests

# Set encoding to UTF-8 to handle Unicode characters in the output
sys.stdout.reconfigure(encoding='utf-8')

TMDB_API_KEY = '702eb6ef09130c827f22f66330aef45e'

def get_tmdb_movie_details(movie_name, release_year):
    """
    Fetches movie details from TMDB, including genre, director, and top 3 cast members.
    """
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}&year={release_year}"
    response = requests.get(search_url)
    
    try:
        data = response.json()
        if data['results']:
            movie_id = data['results'][0]['id']
            movie_details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
            details_response = requests.get(movie_details_url)
            movie_data = details_response.json()

            # Extract genre(s)
            genres = [genre['name'] for genre in movie_data.get('genres', [])]

            # Extract director from credits
            director = None
            for crew_member in movie_data['credits']['crew']:
                if crew_member['job'] == 'Director':
                    director = crew_member['name']
                    break

            # Extract top 3 cast members
            cast = [cast_member['name'] for cast_member in movie_data['credits']['cast'][:3]]

            return {
                'genres': genres,
                'director': director,
                'top_cast': cast
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching TMDB details for {movie_name} ({release_year}): {e}")
        return None

def fetch_liked_movies(user):
    base_url = f'https://letterboxd.com/{user}/likes/films/page/'
    page_number = 1
    films = []

    driver = webdriver.Chrome(service=ChromeService( 
        ChromeDriverManager().install())) 

    while True:
        url = f'{base_url}{page_number}/'
        print(f"Fetching likes page {page_number}...")
        driver.get(url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        content_div = soup.find('div', id='content', class_='site-body')
        if not content_div:
            break
        
        content_wrap = content_div.find('div', class_="content-wrap")
        likes_page_content = content_wrap.find('div', class_="likes-page-content")
        movie_list = likes_page_content.find('ul', class_='poster-list -p70 -grid film-list clear')
        
        if not movie_list:
            break
        
        movie_items = movie_list.find_all('li', class_='poster-container')
        if not movie_items:
            break
        
        name_pattern = r'data-film-name="([^"]+)"'
        year_pattern = r'data-film-release-year="(\d{4})"'
        
        for html_item in movie_items:
            html_string = str(html_item)
            film_name_match = re.search(name_pattern, html_string)
            release_year_match = re.search(year_pattern, html_string)

            if film_name_match and release_year_match:
                film_name = film_name_match.group(1)
                release_year = release_year_match.group(1)
                films.append({"name": film_name, "year": release_year})
        
        page_number += 1

    driver.quit()
    return films

def fetch_high_rated_films(user):
    base_url = f'https://letterboxd.com/{user}/films/page/'
    page_number = 1
    films = []

    driver = webdriver.Chrome(service=ChromeService( 
        ChromeDriverManager().install())) 

    while True:
        url = f'{base_url}{page_number}/'
        print(f"Fetching films page {page_number}...")
        driver.get(url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        content_div = soup.find('div', id='content', class_='site-body')
        if not content_div:
            break
        
        content_wrap = content_div.find('div', class_="content-wrap")
        movie_list = content_wrap.find('ul', class_='poster-list -p70 -grid film-list clear')
        
        if not movie_list:
            break
        
        movie_items = movie_list.find_all('li', class_='poster-container')
        if not movie_items:
            break

        name_pattern = r'data-film-name="([^"]+)"'
        year_pattern = r'data-film-release-year="(\d{4})"'
        rating_pattern = r'(<span class="rating -micro -darker rated-\d+">)(.*?)</span>'

        for html_item in movie_items:
            html_string = str(html_item)
            film_name_match = re.search(name_pattern, html_string)
            release_year_match = re.search(year_pattern, html_string)
            rating_match = re.search(rating_pattern, html_string)

            if film_name_match and release_year_match and rating_match:
                film_name = film_name_match.group(1)
                release_year = release_year_match.group(1)
                rating_text = rating_match.group(2).strip()
                rating_value = rating_text.count('★')

                if rating_value >= 4:
                    films.append({"name": film_name, "year": release_year})

        page_number += 1

    driver.quit()
    return films

def fetch_watchlist_films(user):
    base_url = f'https://letterboxd.com/{user}/watchlist/page/'
    page_number = 1
    films = []

    driver = webdriver.Chrome(service=ChromeService( 
        ChromeDriverManager().install())) 

    while True:
        url = f'{base_url}{page_number}/'
        print(f"Fetching watchlist page {page_number}...")
        driver.get(url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        content_div = soup.find('div', id='content', class_='site-body')
        if not content_div:
            break

        content_wrap = content_div.find('div', class_="content-wrap")
        movie_list = content_wrap.find('ul', class_='poster-list -p125 -grid -scaled128')
        
        if not movie_list:
            break
        
        movie_items = movie_list.find_all('li', class_='poster-container')
        if not movie_items:
            break

        name_pattern = r'data-film-name="([^"]+)"'
        year_pattern = r'data-film-release-year="(\d{4})"'

        for html_item in movie_items:
            html_string = str(html_item)
            film_name_match = re.search(name_pattern, html_string)
            release_year_match = re.search(year_pattern, html_string)

            if film_name_match and release_year_match:
                film_name = film_name_match.group(1)
                release_year = release_year_match.group(1)
                films.append({"name": film_name, "year": release_year})
        
        page_number += 1

    driver.quit()
    return films

def fetch_all_films(user):
    base_url = f'https://letterboxd.com/{user}/films/page/'
    page_number = 1
    films = []

    driver = webdriver.Chrome(service=ChromeService( 
        ChromeDriverManager().install())) 

    while True:
        url = f'{base_url}{page_number}/'
        print(f"Fetching all films page {page_number}...")
        driver.get(url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        content_div = soup.find('div', id='content', class_='site-body')
        if not content_div:
            break
        
        content_wrap = content_div.find('div', class_="content-wrap")
        movie_list = content_wrap.find('ul', class_='poster-list -p70 -grid film-list clear')
        
        if not movie_list:
            break
        
        movie_items = movie_list.find_all('li', class_='poster-container')
        if not movie_items:
            break
        
        name_pattern = r'data-film-name="([^"]+)"'
        year_pattern = r'data-film-release-year="(\d{4})"'

        for html_item in movie_items:
            html_string = str(html_item)
            film_name_match = re.search(name_pattern, html_string)
            release_year_match = re.search(year_pattern, html_string)

            if film_name_match and release_year_match:
                film_name = film_name_match.group(1)
                release_year = release_year_match.group(1)
                films.append({"name": film_name, "year": release_year})
        
        page_number += 1

    driver.quit()
    return films

def fetch_and_map_movies(user):
    liked_movies = fetch_liked_movies(user)
    high_rated_movies = fetch_high_rated_films(user)
    watchlist_movies = fetch_watchlist_films(user)
    all_movies = fetch_all_films(user)

    # Combine all the movies into one list
    combined_movies = liked_movies + high_rated_movies + watchlist_movies
    unique_movies = {f"{movie['name']} ({movie['year']})": movie for movie in combined_movies}

    movie_details = {}
    for movie_key, movie_info in unique_movies.items():
        details = get_tmdb_movie_details(movie_info['name'], movie_info['year'])
        if details:
            movie_details[movie_key] = details

    return movie_details

# Usage
user = 'shresuresh'  # Replace with the desired username
movie_details = fetch_and_map_movies(user)
print(movie_details)
