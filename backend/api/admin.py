from django.contrib import admin
from .models import AgentModel, TaskModel


admin.site.register(AgentModel)

@admin.register(TaskModel)
class TaskModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'is_active', 'created_at', 'updated_at')
    list_filter = ('is_active', 'created_at', 'updated_at')
    search_fields = ('name', 'description', 'prompt_template')
    ordering = ('name',)
    date_hierarchy = 'created_at'
    readonly_fields = ('created_at', 'updated_at')