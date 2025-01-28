from django.db import models

# Create your models here.
class Dataset(models.Model):
    data_id = models.AutoField(primary_key=True)
    data_set = models.FileField(upload_to='files/')
    lr_Accuracy = models.FloatField(null=True)
    lr_algo = models.CharField(max_length=50,null=True)
    knn_Accuracy = models.FloatField(null=True)
    knn_algo = models.CharField(max_length=50,null=True)
    svr_Accuracy = models.FloatField(null=True)
    svr_algo = models.CharField(max_length=50,null=True)
    rf_Accuracy = models.FloatField(null=True)
    rf_algo = models.CharField(max_length=50,null=True)
    dt_Accuracy = models.FloatField(null=True)
    dt_algo = models.CharField(max_length=50,null=True)
    class Meta:
        db_table = 'dataset'
