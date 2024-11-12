# utils.py
import requests
from datetime import datetime

class UserPreferences:
    def __init__(self, request):
        # Initialize from request GET parameters
        self.preferred_genres = request.GET.getlist('preferred_genres', [])
        self.excluded_genres = request.GET.getlist('excluded_genres', [])
        self.preferred_decades = request.GET.getlist('preferred_decades', [])
        self.min_rating = float(request.GET.get('min_rating', 0))
        self.min_votes = int(request.GET.get('min_votes', 0))
        self.preferred_languages = request.GET.getlist('preferred_languages', [])
        self.max_runtime = int(request.GET.get('max_runtime', 0)) if request.GET.get('max_runtime') else None
        self.include_adult = request.GET.get('include_adult', 'false').lower() == 'true'

    def fetch_tmdb_genres(self):
        url = f"https://api.themoviedb.org/3/genre/movie/list?api_key=702eb6ef09130c827f22f66330aef45e"
        response = requests.get(url)
        genres = [genre['name'] for genre in response.json()['genres']]
        return genres
