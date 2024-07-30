import time

import pandas as pd
from django.http.response import (
    HttpResponseBadRequest,
    HttpResponseNotAllowed,
    JsonResponse,
)
from django.views.decorators.csrf import csrf_exempt

from . import utils


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

        # db_results = []
        # for aspect, data in results["topics"].items():
        #     sentiment = data.get("sentiment")
        #     confidence = float(data.get("confidence"))

        #     topics = {}
        #     db_results.append({"sentiment": sentiment, "confidence": confidence})
    else:
        return HttpResponseNotAllowed(["POST"], "Invalid request method.")

    results["elapsed_time"] = round(time.time() - start_at, 3)

    return JsonResponse(results)


@csrf_exempt
def classify_dataset(request):
    """Handles POST requests to classify a dataset's sentiment."""

    start_at = time.time()
    results = {}

    if request.method == "POST":
        try:
            file = request.FILES.get("file")
            if not file:
                return HttpResponseBadRequest("No file uploaded.")

            df = pd.read_csv(file)
            if "review" not in df.columns:
                return HttpResponseBadRequest(
                    "The CSV file must contain a 'review' column."
                )

            # df["review"] = df["review"].apply(helpers.preprocess)
            reviews = df["review"].values

            predicts_results = utils.predict(reviews)
            results = {
                **predicts_results,
                "elapsed_time": round(time.time() - start_at, 3),
            }
        except Exception as e:
            return HttpResponseBadRequest(
                f"An error occurred while processing the dataset.\n{e}"
            )
    else:
        return HttpResponseNotAllowed(["POST"], "Invalid request method.")

    return JsonResponse(results)
