# Generated by Django 4.1.5 on 2023-01-24 08:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0002_alter_usermodel_user_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='usermodel',
            name='user_image',
            field=models.ImageField(upload_to='user/'),
        ),
    ]
