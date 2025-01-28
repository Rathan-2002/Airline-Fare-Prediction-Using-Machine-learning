from django.db import models

# Create your models here.

# class PredictionModel(models.Model):
#     id = models.AutoField(primary_key=True)
#     source = models.CharField(max_length=100)
#     to = models.CharField(max_length=100)
#     airline= models.CharField(max_length=100)
#     dept_time = models.DateTimeField(max_length=50)
#     stops=models.IntegerField()
#     arr_time=models.DateTimeField(max_length=50)

#     class Meta:
#         db_table = 'predictionmodel'

# class TestModel(models.Model):
#     id=models.AutoField(primary_key=True)
#     airline=models.CharField(max_length=50)
#     source=models.IntegerField()
#     to=models.IntegerField()
#     daysleft_travel=models.IntegerField()
#     dept_time = models.IntegerField()
#     stops=models.IntegerField()
#     class1=models.IntegerField()
#     arr_time=models.IntegerField()

#     class Meta:
#         db_table='Testdata Details'

# class TestDataModel(models.Model):
#     id=models.AutoField(primary_key=True)
#     airline=models.CharField(max_length=50)
#     source=models.IntegerField()
#     to=models.IntegerField()
#     arr_date=models.DateTimeField()
#     dept_date=models.DateTimeField()
#     stops=models.IntegerField()

#     class Meta:
#         db_table='TestData'

class PredModel(models.Model):
    id = models.AutoField(primary_key=True)
    source = models.CharField(max_length=100)
    to = models.CharField(max_length=100)
    airline= models.CharField(max_length=100)
    dept_time = models.DateTimeField(max_length=50)
    stops=models.IntegerField()
    arr_time=models.DateTimeField(max_length=50)

    class Meta:
        db_table = 'predmodel'

class TestingModel(models.Model):
    id = models.AutoField(primary_key=True)
    Total_Stops=models.IntegerField()
    Air_India=models.IntegerField()
    GoAir=models.IntegerField()
    IndiGo=models.IntegerField()
    Jet_Airways=models.IntegerField()
    Jet_Airways_Business=models.IntegerField()
    Multiple_carriers=models.IntegerField()
    Multiple_carriers_Premium_economy=models.IntegerField()
    SpiceJet=models.IntegerField()
    Trujet=models.IntegerField()
    Vistara=models.IntegerField()
    Vistara_Premium_economy=models.IntegerField()
    Chennai=models.IntegerField()
    Delhi=models.IntegerField()
    Kolkata=models.IntegerField()
    Mumbai=models.IntegerField()
    Cochin=models.IntegerField()
    Hyderabad=models.IntegerField()
    journey_day=models.IntegerField()
    journey_month=models.IntegerField()
    Dep_Time_hour=models.IntegerField()
    Dep_Time_min=models.IntegerField()
    Arrival_Time_hour=models.IntegerField()
    Arrival_Time_min=models.IntegerField()
    dur_hour=models.IntegerField()
    dur_min=models.IntegerField()


    class Meta:
        db_table='datatestingmodel'





















