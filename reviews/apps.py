from django.apps import AppConfig

from .constants import initialize


class ReviewsConfig(AppConfig):
    name = "reviews"

    def ready(self) -> None:
        initialize()
