from django.core.management.base import BaseCommand
from climatedata.tests import import_examples


class Command(BaseCommand):
    help = 'Load example NC files from disk.'

    def handle(self, *args, **options):
        try:
            import_examples()
        except:
            import sys
            self.stdout.write("Unexpected error:", sys.exc_info()[0])