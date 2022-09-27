# Generated by Django 2.2.5 on 2022-07-12 13:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imgProcess', '0008_labelphotos'),
    ]

    operations = [
        migrations.CreateModel(
            name='LabelPhotos2',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='label_images/')),
                ('description', models.TextField()),
                ('labelstatus', models.TextField()),
            ],
        ),
        migrations.DeleteModel(
            name='LabelPhotos',
        ),
    ]