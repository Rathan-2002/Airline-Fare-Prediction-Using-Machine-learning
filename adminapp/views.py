from django.shortcuts import render,redirect
from adminapp.models import *
from django.contrib import messages
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score,r2_score
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
from adminapp.models import * 
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from userapp.models import *
from mainapp.models import *

from sklearn.svm import SVR
import numpy as np

# Create your views here.
def admin_index(request):
    dataset=Dataset.objects.all().count()
    user=UserModel.objects.all().count()
    test=TestingModel.objects.all().count()
    return render(request,'admin/admin-index.html',{'Dataset':dataset,'user':user,'test':test})

def admin_uploaddata(request):
    if request.method == 'POST' :
        dataset = request.FILES['dataset']
        data = Dataset.objects.create(data_set = dataset)
        data = data.data_id
        print(type(data),'type')


        return redirect('admin_run_algorithms')
    return render(request,'admin/admin-uploaddata.html')

def admin_run_algorithms(request):
    data = Dataset.objects.all().order_by('-data_id').first()
    
    print(data,type(data),'sssss')
    file = str(data.data_set)
    df = pd.read_csv('./media/'+ file)
    table = df.to_html(table_id='data_table')
    # print(df.iloc[0,[1]])
    # print(len(df))
    # for i in range(len(df)):
    #     print(df.iloc[i,[1]],'loop')
    # print(df[0:5],type(df),'database tablesssssssssss')

    return render(request,'admin/admin-run-algorithms.html',{'i':data,'t':table})



 


def score(request,id):
    data = Dataset.objects.get(data_id=id)

    return render(request,'admin/admin-score.html',{'i':data})

def admin_sentiment(request):
    try:    
        data = Dataset.objects.all().order_by('-data_id').first()
        dt_ac = data.dt_Accuracy*100
    
        sv_ac = data.svr_Accuracy*100
        
        nb_ac = data.knn_Accuracy*100

        lr_ac = data.lr_Accuracy*100

        rf_ac = data.rf_Accuracy*100

        print(rf_ac,lr_ac,nb_ac,sv_ac,dt_ac)


    
        context = {
            'lr_ac':lr_ac,
            
            'nb_ac':nb_ac,
            
            'dt_ac':dt_ac,

            'rf_ac':rf_ac,

            'sv_ac':sv_ac,
            
        }
        return render(request,'admin/admin-sentiment-analysis.html',context)
    except:
        messages.info(request,'Run all 4 algorithms')

        return redirect('admin_run_algorithms')


def RandomForest(request,id):
    Accuracy = None
    data = Dataset.objects.get(data_id=id)
    id = data.data_id
    file = str(data.data_set)
    df = pd.read_csv('./media/'+ file)
    X=df[['Total_Stops','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy','Chennai','Delhi','Kolkata','Mumbai',
          'Cochin','Hyderabad','journey_day','journey_month','Dep_Time_hour','Dep_Time_min','Arrival_Time_hour','Arrival_Time_min','dur_hour','dur_min']]
    y=df['Price']
    print(y.head(),'gggggggggggggggggggggggggggggggggggg')
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
    from sklearn.metrics import accuracy_score,confusion_matrix
    def prediction(ml_model):
        print('Model is: {}'.format(ml_model))
        model= ml_model.fit(X_train,y_train)
        print("Training score: {}".format(model.score(X_train,y_train)))
        predictions = model.predict(X_test)
        print("Predictions are: {}".format(predictions))
        print('\n')
        Accuracy=r2_score(y_test,predictions) 
        print(Accuracy,'ssssssssssssssssssssssssss')
        print("r2 score is: {}".format(Accuracy))
        print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
        print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
        print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
        from sklearn.model_selection import RandomizedSearchCV
        random_grid = {
        'n_estimators' : [100, 120, 150, 180, 200,220],
        'max_features':['auto','sqrt'],
       'max_depth':[5,10,15,20], }
        rf=RandomForestRegressor()
        rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1,)

        rf_random.fit(X_train,y_train)

# best parameter
        rf_random.best_params_
        prediction = rf_random.predict(X_test)
        Accuracy2=r2_score(y_test,prediction)
        data.rf_Accuracy=Accuracy2
        data.rf_algo = "Random Forest"
        data.save()
        import joblib
        file=open('airline_rf.pkl','wb')
        joblib.dump(rf_random,file)
    from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
    prediction(RandomForestRegressor())
    # import pickle
    # file=open('airline_rf.pkl','wb')
    # pickle.dump(model,file)
    return redirect('score',id=id)

def button(request,id):
    import pickle
   
    test=TestingModel.objects.get(pk=id)
    print(test,'jjjjjjjjjjjjjjjjjjjjjjjj')
    X_test= [[test.Total_Stops,test.Air_India,test.GoAir,test.IndiGo,test.Jet_Airways,test.Jet_Airways_Business
    ,test.Multiple_carriers,test.Multiple_carriers_Premium_economy,test.SpiceJet,test.Trujet,test.Vistara,test.Vistara_Premium_economy,
    test.Chennai,test.Delhi,test.Kolkata,test.Mumbai,test.Cochin,test.Hyderabad,test.journey_day,test.journey_month,
    test.Dep_Time_hour,test.Dep_Time_min,test.Arrival_Time_hour,test.Arrival_Time_min,test.dur_hour,test.dur_min]]
    
    print(X_test,'iiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
    import joblib
    import pickle
    model=open('airline_rf.pkl','rb')
    rf_random=joblib.load(model)
    # from sklearn.ensemble import RandomForestRegressor
    y_pred=rf_random.predict(X_test)
    # y_prediction=forest.predict(data1)
    # Accuracy=metrics.r2_score(y_test,y_prediction)
    print(y_pred,'uuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
    messages.info(request,y_pred[0])
    # messages.warning(request,Accuracy)
    messages.success(request,'Predicted Successfully')
    return redirect('user_index')

  


    # return redirect('score',id=33)

    # try:
       
    #      print(id,'iiiiiiiiiidddddddddddd')
    #     test=TestModel.objects.get(id=id)
            
    #     test=TestModel.objects.get(pk=id)
    #     print(test,'jjjjjjjjjjjjjjjjjjjjjjjj')
    #     data1= [[test.airline,test.source,test.to,test.daysleft_travel,test.dept_time,test.arr_time,test.class1,test.stops]]
    #     y_test=reg_rf.predict(data1)
    #     print(y_test,'yyyyyy')
    #     messages.info(request,y_test[0])

    #     messages.warning(request,Accuracy)
    #     messages.success(request,'Predicted Successfully') 

    #     return redirect('user_index')
    
    
    # except:
    #         pass
    
     
    # return redirect('score',id=33)

def DecisionTree(request,id):
    Accuracy = None
    data = Dataset.objects.get(data_id=id)
    id = data.data_id
    file = str(data.data_set)
    df = pd.read_csv('./media/'+ file)
    X=df[['Total_Stops','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy','Chennai','Delhi','Kolkata','Mumbai',
          'Cochin','Delhi','Hyderabad','Kolkata','journey_day','journey_month','Dep_Time_hour','Dep_Time_min','Arrival_Time_hour','Arrival_Time_min','dur_hour','dur_min']]
    y=df['Price']
    print(y.head(),'gggggggggggggggggggggggggggggggggggg')
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
    from sklearn.metrics import accuracy_score,confusion_matrix
    def prediction(ml_model):
        print('Model is: {}'.format(ml_model))
        model= ml_model.fit(X_train,y_train)
        print("Training score: {}".format(model.score(X_train,y_train)))
        predictions = model.predict(X_test)
        print("Predictions are: {}".format(predictions))
        print('\n')
        Accuracy=r2_score(y_test,predictions) 
        print(Accuracy,'ssssssssssssssssssssssssss')
        print("r2 score is: {}".format(Accuracy))
        print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
        print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
        print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
        data.dt_Accuracy=Accuracy
        data.dt_algo = "DecisionTree"
        data.save()
    from sklearn.tree import DecisionTreeRegressor
    prediction(DecisionTreeRegressor())
   
    return redirect('score',id=id)
   
    return redirect('score',id=id)

def KNeighborsRegressor(request,id):
    Accuracy = None
    data = Dataset.objects.get(data_id=id)
    id = data.data_id
    file = str(data.data_set)
    df = pd.read_csv('./media/'+ file)
    X=df[['Total_Stops','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy','Chennai','Delhi','Kolkata','Mumbai',
          'Cochin','Delhi','Hyderabad','Kolkata','journey_day','journey_month','Dep_Time_hour','Dep_Time_min','Arrival_Time_hour','Arrival_Time_min','dur_hour','dur_min'
]]
    y=df['Price']
    print(y.head(),'gggggggggggggggggggggggggggggggggggg')
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
    from sklearn.metrics import accuracy_score,confusion_matrix
    def prediction(ml_model):
        print('Model is: {}'.format(ml_model))
        model= ml_model.fit(X_train,y_train)
        print("Training score: {}".format(model.score(X_train,y_train)))
        predictions = model.predict(X_test)
        print("Predictions are: {}".format(predictions))
        print('\n')
        Accuracy=r2_score(y_test,predictions) 
        print(Accuracy,'ssssssssssssssssssssssssss')
        print("r2 score is: {}".format(Accuracy))
        print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
        print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
        print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
        data.knn_Accuracy=Accuracy
        data.knn_algo = "KNNeighbor"
        data.save()
    from sklearn.neighbors import KNeighborsRegressor
    prediction(KNeighborsRegressor())
   
    return redirect('score',id=id)

def LinearRegressor(request,id):
    Accuracy = None
    data = Dataset.objects.get(data_id=id)
    id = data.data_id
    file = str(data.data_set)
    df = pd.read_csv('./media/'+ file)
    X=df[['Total_Stops','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy','Chennai','Delhi','Kolkata','Mumbai',
          'Cochin','Delhi','Hyderabad','Kolkata','journey_day','journey_month','Dep_Time_hour','Dep_Time_min','Arrival_Time_hour','Arrival_Time_min','dur_hour','dur_min'
]]
    y=df['Price']
    print(y.head(),'gggggggggggggggggggggggggggggggggggg')
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
    from sklearn.metrics import accuracy_score,confusion_matrix
    def prediction(ml_model):
        print('Model is: {}'.format(ml_model))
        model= ml_model.fit(X_train,y_train)
        print("Training score: {}".format(model.score(X_train,y_train)))
        predictions = model.predict(X_test)
        print("Predictions are: {}".format(predictions))
        print('\n')
        Accuracy=r2_score(y_test,predictions) 
        print(Accuracy,'ssssssssssssssssssssssssss')
        print("r2 score is: {}".format(Accuracy))
        print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
        print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
        print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
        data.lr_Accuracy=Accuracy
        data.lr_algo = "Linear Regressor"
        data.save()
    from sklearn.linear_model import LogisticRegression
    prediction(LogisticRegression())
    return redirect('score',id=id)

def SVR(request,id):
    Accuracy = None
    data = Dataset.objects.get(data_id=id)
    id = data.data_id
    file = str(data.data_set)
    df = pd.read_csv('./media/'+ file)
    X=df[['Total_Stops','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy','Chennai','Delhi','Kolkata','Mumbai',
          'Cochin','Delhi','Hyderabad','Kolkata','journey_day','journey_month','Dep_Time_hour','Dep_Time_min','Arrival_Time_hour','Arrival_Time_min','dur_hour','dur_min'
]]
    y=df['Price']
    print(y.head(),'gggggggggggggggggggggggggggggggggggg')
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
    from sklearn.metrics import accuracy_score,confusion_matrix
    def prediction(ml_model):
        print('Model is: {}'.format(ml_model))
        model= ml_model.fit(X_train,y_train)
        print("Training score: {}".format(model.score(X_train,y_train)))
        predictions = model.predict(X_test)
        print("Predictions are: {}".format(predictions))
        print('\n')
        Accuracy=r2_score(y_test,predictions) 
        print(Accuracy,'ssssssssssssssssssssssssss')
        print("r2 score is: {}".format(Accuracy))
        print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
        print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
        print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
        data.svr_Accuracy=Accuracy
        data.svr_algo = "SVR"
        data.save()
    from sklearn.svm import SVR
    prediction(SVR())
    return redirect('score',id=id)


