from django.test import SimpleTestCase
from django.urls import resolve, reverse

from .views import home


class HomepageTests(SimpleTestCase):
    def setUp(self) -> None:
        url = reverse("home")
        self.response = self.client.get(url)

    def test_homepage_status_code(self):
        self.assertEqual(self.response.status_code, 200)

    def test_homepage_contains_correct_html(self):
        self.assertContains(self.response, "Sorwe App Home")

    def test_homepage_does_not_contain_incorrect_html(self):
        self.assertNotContains(self.response, "Random App Page")

    def test_homepage_view(self):
        view = resolve("/")
        self.assertEqual(view.func.__name__, home.__name__)
