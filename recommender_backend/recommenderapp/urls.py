# recommenderapp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Map root URL to the 'home' view
    path('get_movie_recommendations/', views.get_movie_recommendations, name='get_movie_recommendations'),
]
