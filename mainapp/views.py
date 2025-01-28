from django.shortcuts import render,redirect
from django.contrib import messages
from mainapp.models import *

# Create your views here.
def main_index(request):
    return render(request,'main/main-index.html')


def main_admin_login(request):
    if request.method=='POST':
        
        username=request.POST.get('username')
        userpassword=request.POST.get('password')
        print(username,userpassword)

        if username =="admin" and userpassword == "admin":
            print('suceeeee')
            messages.success(request,"admin successfully login")
            return redirect('admin_index')
        else:
            messages.error(request,"invalid credentials")
            return redirect('main_admin_login')
  

        messages.success(request,"invalid credentials")
    return render(request,'main/main-admin-login.html')

def main_about(request):
    return render(request,'main/main-about.html')

def main_contact(request):
    return render(request,'main/main-contact.html')

def main_user_login(request):
    # user_id = request.session['user_id']
    # user = UserModel.objects.get(user_id=user_id)
    if request.method == 'POST':
        useremail = request.POST.get('email')
        password = request.POST.get('password')
        print(useremail, password)

        try:
            user = UserModel.objects.get(
                user_email=useremail, user_password=password)
            request.session['user_id'] = user.pk
            messages.success(request,"successfully login")
            return redirect('user_index')
        except:
            messages.error(request,"invalid credentials")
            return redirect('main_user_login')
    return render(request,'main/main-user-login.html')

def main_user_registration(request):
    # user_id = request.session['user_id']
    # user = UserModel.objects.get(user_id=user_id)

    if request.method == 'POST' and request.FILES["image"]:
       username = request.POST.get("name")
       userppnumber=request.POST.get("number")
       email = request.POST.get("email")
       password = request.POST.get("password")
       contact = request.POST.get("contact")
       address = request.POST.get("address")
       image = request.FILES["image"]
       print(username,email,password,contact,image,address,userppnumber)
                    
       user=UserModel.objects.create(user_address=address,user_username=username ,user_passportnumber=userppnumber, user_email=email , user_password=password , user_contact=contact , user_image=image)
       if user:
            messages.success(request, 'successfully registered')
            return redirect('main_user_login')
       else:
            messages.error(request, 'Invalid registration')
            return redirect('main_user_registration')

    return render(request,'main/main-user-registration.html')



