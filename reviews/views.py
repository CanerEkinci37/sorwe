import time

import pandas as pd
from django.http.response import (
    HttpResponseBadRequest,
    HttpResponseNotAllowed,
    JsonResponse,
)
from django.views.decorators.csrf import csrf_exempt

from . import utils
from .models import Review


def list_reviews(request):
    reviews = Review.objects.all().order_by("-id")[:10]
    reviews_list = []

    for review in reviews:
        review_dict = {"text": review.text, "topics": []}
        for topic in review.topics:
            topic_dict = {"category": topic["category"], "emotions": {}}
            for sentiment, score in topic["emotions"].items():
                topic_dict["emotions"][sentiment] = score
            review_dict["topics"].append(topic_dict)
        reviews_list.append(review_dict)

    return JsonResponse(reviews_list, safe=False)


@csrf_exempt
def classify_review(request):
    start_at = time.time()
    if request.method == "POST":
        review = request.POST.get("review", "")
        if review == "":
            return HttpResponseBadRequest("Plese enter the sentences.")

        results = {"review": review}
        preprocessed_review = utils.preprocess(review)

        predicted_results = utils.predict(preprocessed_review)
        results.update(predicted_results)

        topics = []
        for topic_name, topic_info in results["topics"].items():
            emotions = topic_info["emotions"]

            formatted_emotions = {
                sentiment: score for sentiment, score in emotions.items()
            }
            topic = {"category": topic_name, "emotions": formatted_emotions}
            topics.append(topic)

        review = Review(text=review, topics=topics)
        review.save()
    else:
        return HttpResponseNotAllowed(["POST"], "Invalid request method.")

    results["elapsed_time"] = round(time.time() - start_at, 3)
    return JsonResponse(results)
