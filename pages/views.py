from django.http.response import HttpResponse


def home(request):
    return HttpResponse("Sorwe App Home")
