from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),  # Camera UI
    path('display/', views.display_view, name='display'),  # Display UI
    path('api/start/', views.start_session, name='start_session'),
    path('api/end/', views.end_session, name='end_session'),
    path('api/upload/', views.upload_frame, name='upload_frame'),
    path('api/status/', views.session_status, name='session_status'),
    path('api/latest/', views.get_latest_result, name='get_latest_result'),
]