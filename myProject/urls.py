from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
        path('index/', views.index, name="index"),
    path('about/', views.about, name="about"),
    path('services/', views.services, name="services"),
    path('project/', views.project, name='project'),
    path('contact/', views.contact, name='contact'),
    path('', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('camera/', views.camera, name='camera'),
    path('face/', views.face, name='face')



]
