import sys
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service as ChromeService 
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
import time
import requests
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers

# Set encoding to UTF-8
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
                'id': movie_id,
                'genres': genres,
                'director': director,
                'top_cast': cast
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching TMDB details for {movie_name} ({release_year}): {e}")
        return None

# [Keep your original fetch_liked_movies, fetch_high_rated_films, fetch_watchlist_films, and fetch_all_films functions]

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
    # Get movies using your original functions
    liked_movies = fetch_liked_movies(user)
    high_rated_movies = fetch_high_rated_films(user)
    watchlist_movies = fetch_watchlist_films(user)
    all_movies = fetch_all_films(user)

    # Create target values based on the movie source
    for movie in liked_movies:
        movie['target'] = 1.0  # Liked movies
    for movie in high_rated_movies:
        movie['target'] = 0.8  # Highly rated movies
    for movie in watchlist_movies:
        movie['target'] = 0.6  # Watchlist movies
    for movie in all_movies:
        movie['target'] = 0.4  # Other watched movies
    
    # Combine all movies
    combined_movies = []
    seen_movies = set()  # To prevent duplicates
    
    for movie in liked_movies + high_rated_movies + watchlist_movies + all_movies:
        movie_key = (movie['name'], movie['year'])
        if movie_key not in seen_movies:
            movie_details = get_tmdb_movie_details(movie['name'], movie['year'])
            if movie_details:
                combined_movies.append({
                    'user_id': user,
                    'movie_id': movie_details['id'],
                    'name': movie['name'],
                    'year': movie['year'],
                    'genres': movie_details['genres'],
                    'director': movie_details['director'],
                    'top_cast': movie_details['top_cast'],
                    'target': movie['target']
                })
                seen_movies.add(movie_key)
    
    return combined_movies

def prepare_features(movies_df, genre_dict=None, director_dict=None):
    """
    Prepare features with optional pre-existing dictionaries to ensure consistent dimensions
    """
    # Create feature encodings if not provided
    if genre_dict is None:
        all_genres = set()
        for genres in movies_df['genres']:
            all_genres.update(genres)
        genre_dict = {genre: idx for idx, genre in enumerate(sorted(all_genres))}
    
    if director_dict is None:
        all_directors = set(movies_df['director'].dropna().unique())
        director_dict = {director: idx for idx, director in enumerate(sorted(all_directors))}
    
    # Create feature matrices
    num_movies = len(movies_df)
    genre_features = np.zeros((num_movies, len(genre_dict)))
    director_features = np.zeros((num_movies, len(director_dict)))
    
    # Fill in genre features
    for i, genres in enumerate(movies_df['genres']):
        for genre in genres:
            if genre in genre_dict:
                genre_features[i, genre_dict[genre]] = 1
    
    # Fill in director features
    for i, director in enumerate(movies_df['director']):
        if director in director_dict:
            director_features[i, director_dict[director]] = 1
    
    # Combine features
    features = np.hstack([genre_features, director_features])
    return features, genre_dict, director_dict

def create_model(num_users, num_items, num_features):
    # User embedding
    user_input = layers.Input(shape=(1,))
    user_embedding = layers.Embedding(num_users, 32)(user_input)
    user_vec = layers.Flatten()(user_embedding)
    
    # Movie embedding
    movie_input = layers.Input(shape=(1,))
    movie_embedding = layers.Embedding(num_items, 32)(movie_input)
    movie_vec = layers.Flatten()(movie_embedding)
    
    # Content features
    features_input = layers.Input(shape=(num_features,))
    features_dense = layers.Dense(64, activation='relu')(features_input)
    
    # Combine all features
    concat = layers.Concatenate()([user_vec, movie_vec, features_dense])
    dense1 = layers.Dense(128, activation='relu')(concat)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    output = layers.Dense(1, activation='sigmoid')(dense2)
    
    model = keras.Model(
        inputs=[user_input, movie_input, features_input],
        outputs=output
    )
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_recommendation_model(movies_df):
    """Modified to return feature dictionaries"""
    # Create user and movie mappings
    user_ids = sorted(movies_df['user_id'].unique())
    movie_ids = sorted(movies_df['movie_id'].unique())
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    
    # Prepare features and get the dictionaries
    content_features, genre_dict, director_dict = prepare_features(movies_df)
    
    # Prepare training data
    user_indices = np.array([user_to_idx[user] for user in movies_df['user_id']])
    movie_indices = np.array([movie_to_idx[movie] for movie in movies_df['movie_id']])
    targets = movies_df['target'].values
    
    # Create and train model
    model = create_model(
        num_users=len(user_ids),
        num_items=len(movie_ids),
        num_features=content_features.shape[1]
    )
    
    model.fit(
        [user_indices, movie_indices, content_features],
        targets,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )
    
    return model, user_to_idx, movie_to_idx, content_features, genre_dict, director_dict

def normalize_title(title):
    """
    Normalizes movie titles to help catch variations
    e.g., "The Godfather: Part II" and "The Godfather Part II" should match
    """
    # Remove special characters and convert to lowercase
    normalized = re.sub(r'[^\w\s]', '', title.lower())
    # Remove common words that might vary between versions
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
    words = normalized.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def are_similar_movies(movie1, movie2):
    """
    Check if two movies are likely the same by comparing normalized titles and years
    """
    title1 = normalize_title(movie1['name'])
    title2 = normalize_title(movie2['name'])
    
    # Check for exact title match
    if title1 == title2:
        # If years are available, check if they're close (for different versions)
        if movie1['year'] and movie2['year']:
            return abs(int(movie1['year']) - int(movie2['year'])) <= 2
        return True
    
    # Check for substring match (for partial titles)
    if (title1 in title2 or title2 in title1) and len(title1) > 5 and len(title2) > 5:
        return True
    
    return False

def get_tmdb_discover_movies(page=1, include_older=True):
    """
    Fetches both popular and classic movies from TMDB's discover endpoint.
    """
    movies = []
    seen_movies = set()  # Track movies we've already added
    
    # Fetch current popular movies
    discover_url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&sort_by=popularity.desc&page={page}"
    
    # If including older movies, also fetch highly rated classics
    if include_older:
        decades = ['1950', '1960', '1970', '1980', '1990', '2000', '2010']
        for decade in decades:
            decade_url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&sort_by=vote_average.desc&vote_count.gte=1000&primary_release_year.gte={decade}&primary_release_year.lte={str(int(decade) + 9)}&page={page}"
            try:
                response = requests.get(decade_url)
                data = response.json()
                
                for movie in data.get('results', []):
                    movie_id = movie['id']
                    # Skip if we've already processed this movie ID
                    if movie_id in seen_movies:
                        continue
                        
                    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
                    details_response = requests.get(details_url)
                    movie_data = details_response.json()
                    
                    director = None
                    for crew_member in movie_data.get('credits', {}).get('crew', []):
                        if crew_member['job'] == 'Director':
                            director = crew_member['name']
                            break
                    
                    new_movie = {
                        'movie_id': movie_id,
                        'name': movie['title'],
                        'year': movie['release_date'][:4] if movie.get('release_date') else None,
                        'genres': [genre['name'] for genre in movie_data.get('genres', [])],
                        'director': director,
                        'top_cast': [cast['name'] for cast in movie_data.get('credits', {}).get('cast', [])[:3]],
                        'vote_average': movie.get('vote_average', 0),
                        'vote_count': movie.get('vote_count', 0)
                    }
                    
                    # Check if this is a duplicate before adding
                    is_duplicate = False
                    for existing_movie in movies:
                        if are_similar_movies(existing_movie, new_movie):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        movies.append(new_movie)
                        seen_movies.add(movie_id)
                    
                    time.sleep(0.25)
                    
            except Exception as e:
                print(f"Error fetching movies from {decade}s: {e}")
                continue
    
    # Fetch current popular movies
    try:
        response = requests.get(discover_url)
        data = response.json()
        
        for movie in data.get('results', []):
            movie_id = movie['id']
            if movie_id in seen_movies:
                continue
                
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
            details_response = requests.get(details_url)
            movie_data = details_response.json()
            
            director = None
            for crew_member in movie_data.get('credits', {}).get('crew', []):
                if crew_member['job'] == 'Director':
                    director = crew_member['name']
                    break
            
            new_movie = {
                'movie_id': movie_id,
                'name': movie['title'],
                'year': movie['release_date'][:4] if movie.get('release_date') else None,
                'genres': [genre['name'] for genre in movie_data.get('genres', [])],
                'director': director,
                'top_cast': [cast['name'] for cast in movie_data.get('credits', {}).get('cast', [])[:3]],
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0)
            }
            
            # Check if this is a duplicate before adding
            is_duplicate = False
            for existing_movie in movies:
                if are_similar_movies(existing_movie, new_movie):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                movies.append(new_movie)
                seen_movies.add(movie_id)
            
            time.sleep(0.25)
            
    except Exception as e:
        print(f"Error fetching current popular movies: {e}")
    
    return movies

def recommend_movies(user_id, model, user_to_idx, movie_to_idx, content_features, movies_df, genre_dict, director_dict, n=5):
    if user_id not in user_to_idx:
        print(f"User {user_id} not found in training data")
        return []
    
    user_idx = user_to_idx[user_id]
    
    # Get all watched movies with better title normalization
    print("Fetching user's complete watch history...")
    all_watched_films = fetch_all_films(user_id)
    watched_movies = []
    for movie in all_watched_films:
        watched_movies.append({
            'name': movie['name'],
            'year': movie['year'],
            'normalized_title': normalize_title(movie['name'])
        })
    
    print(f"User has watched {len(watched_movies)} movies in total")
    
    # Fetch potential movies from TMDB (including classics)
    print("Fetching movies from TMDB (including classics)...")
    new_movies = []
    for page in range(1, 4):
        new_movies.extend(get_tmdb_discover_movies(page, include_older=True))
    
    # Filter out watched movies and duplicates with better detection
    filtered_movies = []
    seen_normalized_titles = set()
    
    for movie in new_movies:
        # Skip if year or director is missing
        if not movie['year'] or not movie['director']:
            continue
            
        # Check if movie is already watched using normalized comparison
        is_watched = False
        for watched_movie in watched_movies:
            if are_similar_movies(movie, watched_movie):
                is_watched = True
                break
        
        if is_watched:
            continue
            
        # Check for duplicates in our filtered list
        normalized_title = normalize_title(movie['name'])
        if normalized_title in seen_normalized_titles:
            continue
            
        # Calculate quality score
        vote_count_weight = min(movie['vote_count'] / 1000, 1)
        quality_score = movie['vote_average'] * vote_count_weight
        movie['quality_score'] = quality_score
        
        filtered_movies.append(movie)
        seen_normalized_titles.add(normalized_title)
    
    print(f"Found {len(filtered_movies)} unique unwatched movies to evaluate")
    
    if not filtered_movies:
        print("No new movies found to recommend")
        return []
    
    # Prepare features and make predictions
    new_movies_df = pd.DataFrame(filtered_movies)
    new_content_features, _, _ = prepare_features(new_movies_df, genre_dict, director_dict)
    
    num_new_movies = len(filtered_movies)
    user_input = np.full(num_new_movies, user_idx)
    default_movie_idx = 0
    movie_indices = np.full(num_new_movies, default_movie_idx)
    
    predictions = model.predict([
        user_input,
        movie_indices,
        new_content_features
    ], verbose=0)
    
    # Combine model predictions with TMDB quality scores
    for i, movie in enumerate(filtered_movies):
        model_weight = 0.7
        quality_weight = 0.3
        combined_score = (model_weight * predictions[i][0] + 
                         quality_weight * (movie['quality_score'] / 10))
        movie['combined_score'] = combined_score
    
    # Sort by combined score and get top N unique recommendations
    filtered_movies.sort(key=lambda x: x['combined_score'], reverse=True)
    recommendations = []
    seen_titles = set()
    
    for movie in filtered_movies:
        normalized_title = normalize_title(movie['name'])
        if normalized_title not in seen_titles:
            recommendations.append({
                'name': movie['name'],
                'year': movie['year'],
                'genres': movie['genres'],
                'director': movie['director'],
                'top_cast': movie['top_cast'],
                'score': movie['combined_score'],
                'tmdb_rating': movie['vote_average'],
                'vote_count': movie['vote_count']
            })
            seen_titles.add(normalized_title)
            
            if len(recommendations) >= n:
                break
    
    return recommendations

def main():
    try:
        user = "shresuresh"  # Replace with actual username
        print(f"Fetching movies for user: {user}")
        
        # Get movies for training
        movies = fetch_and_map_movies(user)
        if not movies:
            print("No movies found for user")
            return
            
        # Convert to DataFrame
        movies_df = pd.DataFrame(movies)
        print(f"Found {len(movies_df)} movies for training")
        
        # Train model
        print("Training recommendation model...")
        model, user_to_idx, movie_to_idx, content_features, genre_dict, director_dict = train_recommendation_model(movies_df)
        
        # Generate recommendations
        print("Generating recommendations...")
        recommendations = recommend_movies(
            user,
            model,
            user_to_idx,
            movie_to_idx,
            content_features,
            movies_df,
            genre_dict,
            director_dict,
            n=10
        )
        
        # Display results
        if recommendations:
            print("\nRecommended Movies:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['name']} ({rec['year']})")
                print(f"   Director: {rec['director']}")
                print(f"   Cast: {', '.join(rec['top_cast'])}")
                print(f"   Genres: {', '.join(rec['genres'])}")
                print(f"   TMDB Rating: {rec['tmdb_rating']}/10 ({rec['vote_count']} votes)")
                print(f"   Recommendation Score: {rec['score']:.2f}")
                print()
        else:
            print("No recommendations could be generated.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()