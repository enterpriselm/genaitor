from django.apps import AppConfig
import os
import sys


class GenaitorApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    
    def ready(self):
        """
        Carrega todas as tarefas do banco de dados como GeneralTask quando a aplicação inicia.
        """
        try:
            # Adicionar o caminho do projeto ao sys.path para importar os módulos
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.append(project_root)
            
            # Importando aqui para evitar problemas de importação circular
            from .models import TaskModel
            
            # Importar o necessário para criar GeneralTask
            from presets.tasks_objects import GeneralTask
            from presets.providers import gemini_provider
            
            # Obter provider padrão
            provider = gemini_provider()
            
            # Carregar todas as tarefas salvas no banco
            tasks = TaskModel.objects.filter(is_active=True)
            
            for task in tasks:
                try:
                    # Criar instância de GeneralTask
                    general_task = GeneralTask(
                        description=task.description,
                        goal=task.goal,
                        output_format=task.output_format,
                        llm_provider=provider
                    )
                    
                    # Registrar no namespace global
                    task_var_name = f"{task.name}_task"
                    globals()[task_var_name] = general_task
                    
                    # Registrar no módulo de tarefas se disponível
                    if 'presets.tasks' in sys.modules:
                        import presets.tasks
                        setattr(presets.tasks, task_var_name, general_task)
                        
                    print(f"Tarefa '{task.name}' carregada com sucesso como GeneralTask.")
                except Exception as e:
                    print(f"Erro ao carregar tarefa '{task.name}': {str(e)}")
                
        except Exception as e:
            print(f"Erro ao inicializar tasks: {str(e)}")
