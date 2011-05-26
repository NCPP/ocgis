from django.core.management.base import BaseCommand
from openclimategis.climatedata.models import Grid, GridCell

class Command(BaseCommand):
    help = 'Creates sample grid data'

    def handle(self, *args, **options):
        try:
            lon_min=10.0
            lon_step=1.0
            lon_cells=4
            lon_max=lon_min+lon_cells*lon_step
            lat_min=0.0
            lat_step=2.0
            lat_cells=3
            lat_max=lat_min+lat_cells*lat_step
            # create the Grid record
            boundary_wkt = ('POLYGON (({w} {s}, '
                                      '{w} {n}, '
                                      '{e} {n}, '
                                      '{e} {s}, '
                                      '{w} {s}))').format(
                            n=lat_max,
                            s=lat_min,
                            e=lon_max,
                            w=lon_min,
                        )
            grid_obj, created = Grid.objects.get_or_create(
                boundary_geom=boundary_wkt,
                native_srid=4326,
                description='a trivial grid intended for testing purposes',
            )
            if created:
            # create the associated GridCell records
                for col in range(0,lon_cells):
                    for row in range(0,lat_cells):
                        # create a grid record, using a convention that the 
                        # origin (row,col=0,0) is in the upper left
                        wkt = ('POLYGON (({w} {s}, '
                                         '{w} {n}, '
                                         '{e} {n}, '
                                         '{e} {s}, '
                                         '{w} {s}))').format(
                            n=lat_max-row*lat_step,
                            s=lat_max-row*(lat_step+1),
                            e=lon_min+col*(lon_step+1),
                            w=lon_min+col*lon_step,
                        )
                        z = GridCell(
                            grid=grid_obj,
                            row=row,
                            col=col,
                            geom=wkt,
                        )
                        z.save()
                self.stdout.write('Successfully created a sample grid\n')
            else:
                self.stdout.write('The sample grid already exists in the database!\n')
        except:
            import sys
            self.stdout.write("Unexpected error:", sys.exc_info()[0])