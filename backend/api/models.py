from django.db import models

# Create your models here.

class AgentModel(models.Model):
    """
    Modelo que representa um agente disponível no sistema.
    Não será realmente armazenado no banco de dados, apenas usado para API.
    """
    name = models.CharField(max_length=100, unique=True)
    
    class Meta:
        managed = False  # Não será gerenciado pelo Django ORM
        
    def __str__(self):
        return self.name
        
class TaskModel(models.Model):
    """
    Modelo que representa uma tarefa que pode ser atribuída a agentes.
    Este modelo será armazenado no banco de dados.
    Corresponde à estrutura de um GeneralTask.
    """
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(help_text="Descrição da tarefa - o que ela faz")
    goal = models.TextField(help_text="Objetivo da tarefa - o que ela deve alcançar")
    output_format = models.TextField(help_text="Formato esperado da saída da tarefa")
    prompt_template = models.TextField(blank=True, help_text="Template de prompt para execução da tarefa (opcional)")
    parameters = models.JSONField(default=dict, blank=True, help_text="Parâmetros adicionais da tarefa")
    max_retries = models.IntegerField(default=3, help_text="Número máximo de tentativas")
    timeout = models.IntegerField(default=60, help_text="Tempo limite em segundos")
    validation_required = models.BooleanField(default=True, help_text="Se é necessário validar o resultado")
    cache_results = models.BooleanField(default=True, help_text="Se os resultados devem ser cacheados")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'genaitor_tasks'  # Nome personalizado da tabela no banco
        verbose_name = 'Task'
        verbose_name_plural = 'Tasks'
        ordering = ['name']
        
    def __str__(self):
        return self.name
        
    def to_general_task(self, llm_provider):
        """
        Converte o modelo para uma instância de GeneralTask
        que pode ser utilizada pelo framework Genaitor
        """
        # Assumindo que GeneralTask é importado de presets.tasks_objects
        from presets.tasks_objects import GeneralTask
        
        # Criando uma instância de GeneralTask com os dados do modelo
        task = GeneralTask(
            description=self.description,
            goal=self.goal,
            output_format=self.output_format,
            llm_provider=llm_provider
        )
        
        return task

class Response(models.Model):
    """
    Modelo que representa uma resposta de processamento.
    Não será realmente armazenado no banco de dados, apenas usado para API.
    """
    response = models.TextField()
    
    class Meta:
        managed = False  # Não será gerenciado pelo Django ORM
