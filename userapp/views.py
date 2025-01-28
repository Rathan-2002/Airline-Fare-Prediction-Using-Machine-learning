from django.shortcuts import render,redirect
from mainapp.models import *
from django.contrib import messages
from userapp.models import *
from adminapp.models import *
import pandas as pd

# Create your views here.
def user_index(request):
    user_id = request.session['user_id']
    user = UserModel.objects.get(user_id=user_id)
    if request.method == 'POST':

        source= request.POST.get("source")
        
        to=request.POST.get('to')
       
        airline= request.POST.get("airline")
        


        dept_time = request.POST.get("dept_time")
        

        stops=request.POST.get('stops')
        arr_time=request.POST.get('arr_time')
        
        print(source,to,airline,dept_time,stops,arr_time)
        obj = PredModel.objects.create(source=source,to=to,airline=airline,dept_time=dept_time,stops=stops,arr_time=arr_time)
        print(obj,'kkkkkkkkkkkkkkkkkk')
        return redirect("Predict",id=obj.id)
        

    return render(request,'user/user-index.html')

def user_myprofile(request):
    user_id = request.session['user_id']
    user = UserModel.objects.get(user_id=user_id)

    if request.method == 'POST':
            username = request.POST.get("user_username")
            userppnum=request.POST.get('user_passportnumber')
            email = request.POST.get("user_email")
            contact = request.POST.get("user_contact")
            password = request.POST.get("user_password")
            address=request.POST.get('user_address')
            print(username,userppnum,email,contact,password,address)
            
            if len(request.FILES) != 0:
                
                        image = request.FILES["user_image"]
                        
                        user.user_passportnumber=userppnum
                        user.user_username = username
                        user.user_contact = contact
                        user.user_email=email
                        user.user_password = password
                        user.user_image = image
                        user.user_address=address
                        user.save()
                        messages.success(request,'Updated Successfully')
            else:
                        user.user_username = username
                        user.user_passportnumber=userppnum
                        user.user_contact = contact
                        user.user_contact = contact
                        user.user_email=email
                        # user.user_image=image
                        user.user_password = password
                        user.user_address=address
                        user.save()
                        messages.success(request,'Updated Successfully')
            
                        
            return redirect('user_myprofile')
    
    return render(request,'user/user-myprofile.html',{'user':user})


def Predict(request,id):
    data = Dataset.objects.all().first()
    user_data = PredModel.objects.get(pk=id)
   
    if(user_data.source == 'Chennai'):
        Chennai=1
        Delhi=0
        Kolkata=0
        Mumbai=0
        Cochin=0
        Hyderabad=0

    elif(user_data.source == 'Delhi'):
        Chennai=0
        Delhi=1
        Kolkata=0
        Mumbai=0
        Cochin=0
        Hyderabad=0
    elif(user_data.source == 'Kolkata'):
        Chennai=0
        Delhi=0
        Kolkata=1
        Mumbai=0
        Cochin=0
        Hyderabad=0
    elif(user_data.source == 'Mumbai'):
        Chennai=0
        Delhi=0
        Kolkata=0
        Mumbai=1
        Cochin=0
        Hyderabad=0
    elif(user_data.source == 'Cochin'):
        Chennai=0
        Delhi=0
        Kolkata=0
        Mumbai=0
        Cochin=1
        Hyderabad=0
   
    elif(user_data.source == 'Hyderabad'):
        Chennai=0
        Delhi=0
        Kolkata=0
        Mumbai=0
        Cochin=0
        Hyderabad=1
    else:
        Chennai=0
        Delhi=0
        Kolkata=0
        Mumbai=0
        Cochin=0
        Hyderabad=0
    if(user_data.to == 'Chennai'):
        Chennai=1
        Delhi=0
        Kolkata=0
        Mumbai=0
        Cochin=0
        Hyderabad=0
    elif(user_data.to == 'Delhi'):
        Chennai=0
        Delhi=1
        Kolkata=0
        Mumbai=0
        Cochin=0
        Hyderabad=0
    elif(user_data.to == 'Kolkata'):
        Chennai=0
        Delhi=0
        Kolkata=1
        Mumbai=0
        Cochin=0
        Hyderabad=0
    elif(user_data.to == 'Mumbai'):
        Chennai=0
        Delhi=0
        Kolkata=0
        Mumbai=1
        Cochin=0
        Hyderabad=0
    elif(user_data.to == 'Cochin'):
        Chennai=0
        Delhi=0
        Kolkata=0
        Mumbai=0
        Cochin=1
        Hyderabad=0
    
    elif(user_data.to == 'Hyderabad'):
        Chennai=0
        Delhi=0
        Kolkata=0
        Mumbai=0
        Cochin=0
        Hyderabad=1
    else:
        Chennai=0
        Delhi=0
        Kolkata=0
        Mumbai=0
        Cochin=0
        Hyderabad=0

    if(user_data.airline == 'Air_India'):
        Air_India=1
        GoAir=0
        IndiGo=0
        Jet_Airways=0
        Jet_Airways_Business=0
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=0
        SpiceJet=0
        Trujet=0
        Vistara=0
        Vistara_Premium_economy=0
    elif(user_data.airline == 'GoAir'):
        Air_India=0
        GoAir=1
        IndiGo=0
        Jet_Airways=0
        Jet_Airways_Business=0
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=0
        SpiceJet=0
        Trujet=0
        Vistara=0
        Vistara_Premium_economy=0
    elif(user_data.airline =='IndiGo'):
        Air_India=0
        GoAir=0
        IndiGo=1
        Jet_Airways=0
        Jet_Airways_Business=0
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=0
        SpiceJet=0
        Trujet=0
        Vistara=0
        Vistara_Premium_economy=0
    elif(user_data.airline == 'Jet_Airways'):
        Air_India=0
        GoAir=0
        IndiGo=0
        Jet_Airways=1
        Jet_Airways_Business=0
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=0
        SpiceJet=0
        Trujet=0
        Vistara=0
        Vistara_Premium_economy=0
    elif(user_data.airline == 'Jet_Airways_Business'):
        Air_India=0
        GoAir=0
        IndiGo=0
        Jet_Airways=0
        Jet_Airways_Business=1
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=0
        SpiceJet=0
        Trujet=0
        Vistara=0
        Vistara_Premium_economy=0

    elif(user_data.airline == 'Multiple_carriers'):
        GoAir=0
        IndiGo=0
        Jet_Airways=0
        Jet_Airways_Business=0
        Multiple_carriers=1
        Multiple_carriers_Premium_economy=0
        SpiceJet=0
        Trujet=0
        Vistara=0
        Vistara_Premium_economy=0
    elif(user_data.airline == 'Multiple_carriers_Premium_economy'):
        Air_India=0
        GoAir=0
        IndiGo=0
        Jet_Airways_Business=0
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=1
        SpiceJet=0
        Trujet=0
        Vistara=0
        Vistara_Premium_economy=0

    elif(user_data.airline == 'SpiceJet'):
        Air_India=0
        GoAir=0
        IndiGo=0
        Jet_Airways=0
        Jet_Airways_Business=0
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=0
        SpiceJet=1
        Trujet=0
        Vistara=0
        Vistara_Premium_economy=0
    elif(user_data.airline == 'Trujet'):
        Air_India=0
        IndiGo=0
        Jet_Airways=0
        Jet_Airways_Business=0
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=0
        SpiceJet=0
        Trujet=1
        Vistara=0
        Vistara_Premium_economy=0
    elif(user_data.airline == 'Vistara'):
        Air_India=0
        GoAir=0
        IndiGo=0
        Jet_Airways=0
        Jet_Airways_Business=0
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=0
        SpiceJet=0
        Trujet=0
        Vistara=1
        Vistara_Premium_economy=0
    elif(user_data.airline == 'Vistara_Premium_economy'):
        Air_India=0
        GoAir=0
        IndiGo=0
        Jet_Airways=0
        Jet_Airways_Business=0
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=0
        SpiceJet=0
        Trujet=0
        Vistara=0
        Vistara_Premium_economy=1
    else:
        Air_India=0
        GoAir=0
        Jet_Airways=0
        Jet_Airways_Business=0
        Multiple_carriers=0
        Multiple_carriers_Premium_economy=0
        SpiceJet=0
        Trujet=0
        Vistara=0
        Vistara_Premium_economy=0

    journey_day=int(pd.to_datetime(user_data.dept_time,format="%Y-%m-%dT%H:%M").day)
    journey_month=int(pd.to_datetime(user_data.dept_time,format="%Y-%m-%dT%H:%M").month)
    Dep_Time_hour=int(pd.to_datetime(user_data.dept_time,format="%Y-%m-%dT%H:%M").hour)
    Dep_Time_min=int(pd.to_datetime(user_data.dept_time,format="%Y-%m-%dT%H:%M").minute)
    Arrival_Time_hour=int(pd.to_datetime(user_data.arr_time,format="%Y-%m-%dT%H:%M").hour)
    Arrival_Time_min=int(pd.to_datetime(user_data.arr_time,format="%Y-%m-%dT%H:%M").minute)

    dur_hour=abs(Arrival_Time_hour-Dep_Time_hour)
    dur_min=abs(Arrival_Time_min-Dep_Time_min)
               

               
   
   
   
   
    
    lp = [journey_day,Chennai,Hyderabad,Cochin,Mumbai,Air_India,Jet_Airways,Jet_Airways_Business,Multiple_carriers,Multiple_carriers_Premium_economy,IndiGo,Vistara_Premium_economy,Vistara,
          Trujet,SpiceJet,dur_hour,dur_min,int(user_data.stops),journey_month,Dep_Time_hour,Dep_Time_min,Arrival_Time_hour,Arrival_Time_min,Delhi,GoAir,Kolkata]


    # output=lp
    print(lp,'lllllllllllllllllllllll')
    # print(Predict,'llllllllllll')
    # from sklearn.ensemble import RandomForestRegressor
    # reg_rf=RandomForestRegressor()
    # data = Dataset.objects.get(data_id = data)

    # # y=reg_rf.fit(output)
    # y_pred=reg_rf.predict(lp)
    # print(y_pred)
    # output1=round(y_pred,2)
    #     # print(lp)
    # id = request.session['id']
    # user = PredictModel.objects.get(pk=id)
   
    test = TestingModel.objects.create(Total_Stops=lp[17],Air_India=lp[5] ,Jet_Airways=lp[6],journey_day =lp[0],
                                      Chennai=lp[1],Hyderabad=lp[2],Cochin=lp[3],
                                       Mumbai=lp[4],Jet_Airways_Business=lp[7],Multiple_carriers=lp[8],
                                       Multiple_carriers_Premium_economy=lp[9],
                                       IndiGo=lp[10],Vistara_Premium_economy=lp[11],Vistara=lp[12],Trujet=lp[13],
                                       SpiceJet=lp[14],dur_hour=lp[15],dur_min=lp[16],journey_month=lp[18],Dep_Time_hour=lp[19],Dep_Time_min=lp[20],
                                       Arrival_Time_hour=lp[21],Arrival_Time_min=lp[22],
                                       Delhi=lp[23],GoAir=lp[24],Kolkata=lp[25])
    print(test,'kkkkkkkkkkkkkkkkkk')

    print(test.id,'jjjjjjjj')
    
    


     





      

    return redirect('button',id=test.id)
    
           







    # file = str(data.data_set)
    # df = pd.read_csv('./media/'+ file)
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # for col in lp:
    #     if type(lp[col]) == 'str':
    #         lp[col] = le.transform(lp[col])
        

    # return render(request,'user/user-index.html')