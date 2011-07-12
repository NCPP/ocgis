from django.http import HttpResponse
from climatedata.models import Archive
from django.shortcuts import render_to_response


def display_archives(request):
    qs = Archive.objects.all()
    
    data = []
    for archive in qs:
        for cm in archive.climatemodel_set.all():
            for dataset in cm.datset_set.all():
                for variable in dataset.variable_set.all():
                    data.append(dict(archive=archive,
                                     cm=cm,
                                     dataset=dataset,
                                     variable=variable))
    order = (('archive','Archive'))
                    
    
    return render_to_response('archives.html',dict(archives=qs))

def index(request):
    return HttpResponse(
        "Hello, world. "
        "You're at the index page for OpenClimateGIS website. "
        "Move along, nothing to see here..."
    )