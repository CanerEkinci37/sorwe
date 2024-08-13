from django.urls import path

from .views import classify_review, list_reviews

urlpatterns = [
    path("", list_reviews, name="reviews"),
    path("classify/", classify_review, name="classify"),
]
