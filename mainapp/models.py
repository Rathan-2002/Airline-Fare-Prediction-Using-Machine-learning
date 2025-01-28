from django.db import models

# Create your models here.
class UserModel(models.Model):
    user_id = models.AutoField(primary_key=True)
    user_username = models.CharField(max_length=100)
    user_passportnumber=models.CharField(max_length=20)
    user_email = models.EmailField(max_length=100)
    user_password = models.CharField(max_length=100)
    user_contact = models.CharField(max_length=15)
    user_address=models.TextField()
    user_image = models.ImageField(upload_to='user/images')
 
    class Meta:
        db_table = 'User_Details'