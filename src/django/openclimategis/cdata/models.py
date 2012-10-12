from django.db import models

# Create your models here.
class Address(models.Model):
    uid = models.IntegerField(primary_key=True)
    uri = models.TextField(blank=False,null=False)