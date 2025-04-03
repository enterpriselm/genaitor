from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Criar router e registrar viewsets
router = DefaultRouter()
router.register(r'agents', views.AgentViewSet, basename='agent')
router.register(r'tasks', views.TaskViewSet, basename='task')
router.register(r'genaitor', views.GenaitorViewSet, basename='genaitor')

urlpatterns = [
    path('', include(router.urls)),
] 