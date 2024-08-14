from django.test import TestCase
from django.urls import reverse

from .models import Review


class ReviewTests(TestCase):
    def setUp(self) -> None:
        url = reverse("classify")
        self.response = self.client.post(
            url, data={"review": "Yemekler aşırı kötüydü fakat temizlik iyiydi."}
        )
        self.empty_response = self.client.post(url, data={"review": ""})
        self.invalid_response = self.client.get(url)

    def test_classify_review_post(self):
        self.assertEqual(self.response.status_code, 200)
        self.assertEqual(Review.objects.all().count(), 1)
        self.assertEqual(
            Review.objects.last().text, "Yemekler aşırı kötüydü fakat temizlik iyiydi."
        )

    def test_classify_review_empty_post(self):
        self.assertEqual(self.empty_response.status_code, 400)
        self.assertEqual(
            self.empty_response.content.decode(), "Plese enter the sentences."
        )

    def test_classify_review_invalid_method(self):
        self.assertEqual(self.invalid_response.status_code, 405)
        self.assertEqual(
            self.invalid_response.content.decode(), "Invalid request method."
        )
