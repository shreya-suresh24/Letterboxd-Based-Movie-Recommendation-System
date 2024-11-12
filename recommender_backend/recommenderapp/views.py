# views.py
from django.http import JsonResponse
from .utils import UserPreferences
import pandas as pd
from .modsetup import fetch_and_map_movies, train_recommendation_model, search_tmdb_movies, recommend_movies

from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def get_movie_recommendations(request):
    try:
        # Get Letterboxd username
        username = request.GET.get('username')
        if not username:
            return JsonResponse({"error": "Letterboxd username is required"}, status=400)
        
        # Get user preferences from request
        preferences = UserPreferences(request)
        
        # Fetch user watch history
        movies = fetch_and_map_movies(username)
        if not movies:
            return JsonResponse({"error": "No movies found for user"}, status=404)
        
        # Convert to DataFrame
        movies_df = pd.DataFrame(movies)
        
        # Train recommendation model
        model, user_to_idx, movie_to_idx, content_features, genre_dict, director_dict = train_recommendation_model(movies_df)
        
        # Search TMDB with preferences
        new_movies = []
        for page in range(1, 4):  # Get first 3 pages of results
            new_movies.extend(search_tmdb_movies(preferences, page))
        
        if not new_movies:
            return JsonResponse({"error": "No movies found matching preferences"}, status=404)
        
        # Generate recommendations
        recommendations = recommend_movies(
            username,
            model,
            user_to_idx,
            movie_to_idx,
            content_features,
            movies_df,
            genre_dict,
            director_dict,
            n=10,
            candidate_movies=new_movies
        )
        
        # Format the recommendations for JSON response
        if recommendations:
            response_data = [
                {
                    "name": rec['name'],
                    "year": rec['year'],
                    "director": rec['director'],
                    "top_cast": rec['top_cast'],
                    "genres": rec['genres'],
                    "runtime": rec.get('runtime'),
                    "tmdb_rating": rec['tmdb_rating'],
                    "vote_count": rec['vote_count'],
                    "score": rec['score']
                }
                for rec in recommendations
            ]
            return JsonResponse({"recommendations": response_data})
        else:
            return JsonResponse({"error": "No recommendations could be generated"}, status=404)
        
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)