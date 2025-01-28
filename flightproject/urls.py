"""flightproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from adminapp import views as adminapp_views
from userapp import views as userapp_views
from mainapp import views as mainapp_views
from django.contrib import messages
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score
import pandas as pd

urlpatterns = [
    path('admin/', admin.site.urls),

    path('admin-index',adminapp_views.admin_index,name='admin_index'),
    path('admin-uploaddata',adminapp_views.admin_uploaddata,name='admin_uploaddata'),
    path('admin-run-algorithms/',adminapp_views.admin_run_algorithms,name='admin_run_algorithms'),
    path('admin-score/<int:id>/',adminapp_views.score,name='score'),
    path('DecisionTree/<int:id>',adminapp_views.DecisionTree,name='DecisionTree'),
    path('LinearRegression/<int:id>/',adminapp_views.LinearRegressor,name='LinearRegressor'),
    path('KNeighborsRegressor/<int:id>',adminapp_views.KNeighborsRegressor,name='KNeighborsRegressor'),
    path('admin-sentiment-analysis.html',adminapp_views.admin_sentiment,name='admin_sentiment'),
    path('SVR/<int:id>',adminapp_views.SVR,name='SVR'),
    path('RandomForest/<int:id>',adminapp_views.RandomForest,name='RandomForest'),
    path('button/<int:id>',adminapp_views.button,name='button'),



    path('',mainapp_views.main_index,name='main_index'),
    path('main-about',mainapp_views.main_about,name='main_about'),
    path('main-admin-login',mainapp_views.main_admin_login,name='main_admin_login'),
    path('main-user-login',mainapp_views.main_user_login,name='main_user_login'),
    path('main-user-registration',mainapp_views.main_user_registration,name='main_user_registration'),
    path('main-contact',mainapp_views.main_contact,name='main_contact'),
         
    path('user-index',userapp_views.user_index,name='user_index'),
    path('user-myprofile',userapp_views.user_myprofile,name='user_myprofile'),
    path('Predict/<int:id>',userapp_views.Predict,name='Predict'),
]+ static (settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
