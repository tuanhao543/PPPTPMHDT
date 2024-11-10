from django.urls import path
from . import views
urlpatterns = [
    # trang chu
    path('', views.predict_price, name='predict_price'),
    # path('trangchu/', views.trangchu, name='trangchu'),
    # path('thongke', views.lab6_thongke, name='lab6_thongke'),
    # path('predict/', views.predict_price, name='predict_price'),
    path('visualize/', views.visualize_data, name='visualize_data'),
]