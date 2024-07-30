from django.urls import path

from .views import classify_dataset, classify_review

urlpatterns = [
    path("classify_review/", classify_review, name="classify_review"),
    path("classify_dataset/", classify_dataset, name="classify_dataset"),
]
