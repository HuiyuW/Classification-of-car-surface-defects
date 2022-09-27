from django.contrib import admin

# Register your models here.
from . import models

admin.site.register(models.Post)
admin.site.register(models.UploadPhotos)
admin.site.register(models.LabelPhotos2)
admin.site.register(models.ProcessPhotos)

