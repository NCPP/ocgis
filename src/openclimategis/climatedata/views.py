from django.http import HttpResponse

def index(request):
    return HttpResponse(
        "Hello, world. "
        "You're at the index page for OpenClimateGIS website. "
        "Move along, nothing to see here..."
    )
