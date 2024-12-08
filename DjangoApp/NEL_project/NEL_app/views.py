from django.shortcuts import render

def index(request):
    return render(request, 'NEL_app/index.html', {'message': 'Hello, Django!'})