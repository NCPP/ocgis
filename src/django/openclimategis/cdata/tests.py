from django.test import TestCase
from django.test.client import Client
import itertools
from ocgis.api.operations import OcgOperations
from cdata.models import Address
import os.path
from ocgis.util.shp_cabinet import ShpCabinet
import subprocess


class Test(TestCase):
    fixtures = ['cdata.json']
    c = Client()
    
    def iter_datasets(self):
        for address in Address.objects.all():
            uri = address.uri
            variable = os.path.split(uri)[1].split('_')[0]
            yield({'uri':uri,'variable':variable})
            
    def open_in_chrome(self,url):
        subprocess.call(["google-chrome",'http://127.0.0.1:8000'+url])
            
    def test_subset(self):
        for dataset in self.iter_datasets():
            ops = OcgOperations(dataset,snippet=True,output_format='keyed')
            url = ops.as_url()
            resp = self.c.get(url)
            self.assertTrue(len(resp.content) > 100)

    def test_inspect(self):
        template = '/inspect?uri={0}{1}'
        for dataset,use_variable in itertools.product(self.iter_datasets(),
                                                      [True,False]):
            if use_variable:
                variable_string = '&variable=' + dataset['variable']
            else:
                variable_string = ''
            url = template.format(dataset['uri'],variable_string)
            resp = self.c.get(url)
            self.assertTrue(len(resp.content) > 100)

    def test_shp(self):
        sc = ShpCabinet()
        keys = sc.keys()
        for key in keys:
            url = '/shp/{0}'.format(key)
            resp = self.c.get(url)
            self.assertTrue(len(resp.content) > 100)
