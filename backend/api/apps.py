import os
import sys
import logging
from django.apps import AppConfig
from django.db import connection

logger = logging.getLogger(__name__)

def table_exists(table_name):
    """
    Verifica se uma tabela existe no banco de dados.
    """
    return table_name in connection.introspection.table_names()

class GenaitorApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        """
        Carrega tarefas e agentes predefinidos ao iniciar a aplicação.
        """
        # Ignora carregamento durante comandos de migração
        if 'makemigrations' in sys.argv or 'migrate' in sys.argv or 'collectstatic' in sys.argv:
            logger.info("Ignorando carregamento de tarefas e agentes durante migração ou coleta de arquivos estáticos.")
            return

        try:
            # Adicionar root do projeto ao sys.path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.append(project_root)

            # Carregar tarefas e agentes, se tabelas existirem
            self._load_tasks()
            self._load_agents()

        except Exception as e:
            logger.error(f"Erro ao inicializar API: {str(e)}")

    def _load_tasks(self):
        try:
            if not table_exists('api_taskmodel'):
                logger.info("Tabela 'api_taskmodel' ainda não existe. Ignorando _load_tasks.")
                return

            from .models import TaskModel
            from genaitor.presets.tasks_objects import GeneralTask
            from genaitor.presets.providers import gemini_provider

            provider = gemini_provider()
            tasks = TaskModel.objects.filter(is_active=True)

            for task in tasks:
                try:
                    general_task = GeneralTask(
                        description=task.description,
                        goal=task.goal,
                        output_format=task.output_format,
                        llm_provider=provider
                    )

                    task_var_name = f"{task.name}_task"
                    globals()[task_var_name] = general_task

                    if 'genaitor.presets.tasks' in sys.modules:
                        import genaitor.presets.tasks
                        setattr(genaitor.presets.tasks, task_var_name, general_task)

                    logger.info(f"Tarefa '{task.name}' carregada com sucesso.")
                except Exception as e:
                    logger.warning(f"Erro ao carregar tarefa '{task.name}': {str(e)}")
        except Exception as e:
            logger.error(f"Erro ao carregar tarefas: {str(e)}")

    def _load_agents(self):
        try:
            if not table_exists('api_agentmodel'):
                logger.info("Tabela 'api_agentmodel' ainda não existe. Ignorando _load_agents.")
                return

            from .models import AgentModel
            from genaitor.presets.tasks_objects import GeneralTask
            import genaitor.presets.agents as agents_module
            from genaitor.presets.providers import gemini_provider

            agents = AgentModel.objects.all()
            provider = gemini_provider()

            for agent_instance in agents:
                try:
                    tasks = [
                        GeneralTask(
                            description=task.description,
                            goal=task.goal,
                            output_format=task.output_format,
                            llm_provider=provider
                        )
                        for task in agent_instance.tasks.all()
                        if task.is_active
                    ]

                    agent_name = f"{agent_instance.role.lower()}_agent"

                    agent_object = {
                        "role": agent_instance.role,
                        "tasks": tasks,
                        "provider": agent_instance.llm_provider,
                        "config": agent_instance.config,
                        "history": agent_instance.conversation_history,
                    }

                    setattr(agents_module, agent_name, agent_object)
                    globals()[agent_name] = agent_object

                    logger.info(f"Agente '{agent_instance.role}' carregado do banco com sucesso.")
                except Exception as e:
                    logger.warning(f"Erro ao carregar agente '{agent_instance}': {str(e)}")

        except Exception as e:
            logger.error(f"Erro ao carregar agentes do banco: {str(e)}")
