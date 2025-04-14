from rest_framework import serializers
from .models import AgentModel, TaskModel, Response

class AgentSerializer(serializers.ModelSerializer):
    """
    ModelSerializer para os agentes disponíveis
    """
    class Meta:
        model = AgentModel
        fields = [
            'id', 'name', 'role', 'tasks', 'config',
            'conversation_history', 'task_history', 
            'max_retries', 'timeout', 'validation_required',
            'cache_results', 'created_at', 'updated_at',
            'is_active',
        ]

class TaskSerializer(serializers.ModelSerializer):
    """
    ModelSerializer para as tarefas que podem ser atribuídas aos agentes
    """
    class Meta:
        model = TaskModel
        fields = [
            'id', 'name', 'description', 'goal', 'output_format', 
            'prompt_template', 'parameters', 'max_retries', 'timeout', 
            'validation_required', 'cache_results', 'is_active', 
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
        
class ResponseSerializer(serializers.ModelSerializer):
    """
    ModelSerializer para as respostas da API Genaitor
    """
    class Meta:
        model = Response
        fields = ['response']
        
    response = serializers.CharField(
        help_text="Resposta gerada pelo agente selecionado"
    ) 