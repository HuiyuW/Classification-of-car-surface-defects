# Generated by Django 4.0.5 on 2022-06-10 09:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imgProcess', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='image',
            field=models.ImageField(default='images/default.png', upload_to='images/'),
        ),
    ]