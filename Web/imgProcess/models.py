from pydoc import describe
from django.db import models
import os

from numpy import delete

# Create your models here.
class Post(models.Model):
    title = models.CharField(max_length=155)
    content = models.TextField()
    slug = models.SlugField(max_length=255)
    image = models.ImageField(upload_to="images/", default="images/default.png")

    def __str__(self):
        return self.title


class LabelPhotos2(models.Model):
    image = models.ImageField(upload_to="label_images/", null=False, blank=False)
    description = models.TextField()
    labelstatus = models.BooleanField()

    def filename(self):
        return os.path.basename(self.image.name)

    def delete(self, *arg, **kwargs):
        self.image.delete()
        super().delete(*arg, **kwargs)


class UploadPhotos(models.Model):
    image = models.ImageField(upload_to="images/", null=False, blank=False)
    description = models.TextField()
    # label = models.CharField(max_length=155)
    def filename(self):
        return os.path.basename(self.image.name)
    
    def delete(self, *arg, **kwargs):
        self.image.delete()
        super().delete(*arg, **kwargs)


class ProcessPhotos(models.Model):
    image = models.ImageField(upload_to="images/", null=False, blank=False)
    description = models.TextField()

    def filename(self):
        return os.path.basename(self.image.name)
    '''  
    def delete(self, *arg, **kwargs):
        self.image.delete()
        super().delete(*arg, **kwargs)
    '''  
class ConfusePhotos(models.Model):
    image = models.ImageField(upload_to="images/", null=False, blank=False)
    description = models.TextField()

    def filename(self):
        return os.path.basename(self.image.name)
