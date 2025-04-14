from django.shortcuts import render
import sys
import asyncio
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response as DRFResponse
from .serializers import ResponseSerializer, AgentSerializer, TaskSerializer
from .models import AgentModel, TaskModel, Response as ResponseModel
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# Importações atualizadas dos módulos src
from genaitor.core import Orchestrator, Flow, ExecutionMode
from genaitor.presets.agents import *
from genaitor.presets.tasks import *
from genaitor.presets.tasks_objects import *
from genaitor.presets.providers import *

class AgentViewSet(viewsets.ModelViewSet):
    """
    ViewSet para gerenciar agentes.
    Permite criar, ler, atualizar e excluir agentes no banco de dados.
    """
    queryset = AgentModel.objects.all()
    serializer_class = AgentSerializer
    
    def get_queryset(self):
        db_agents = list(AgentModel.objects.all())
        db_agent_names = [agent.name for agent in db_agents]
        agent_vars = [var for var in globals() if var.endswith("_agent")]

        for agent_var_name in agent_vars:
            if not AgentModel.objects.filter(name=agent_var_name).exists():
                agent_obj = globals()[agent_var_name]
                agent_model = AgentModel(name=agent_var_name)

                # Role (convert enum para string)
                if hasattr(agent_obj, "role"):
                    role = agent_obj.role
                    agent_model.role = role.name if hasattr(role, "name") else str(role)
                else:
                    agent_model.role = f"Agente: {agent_var_name}"

                # Configurações
                config = getattr(agent_obj, "config", None)
                agent_model.max_retries = getattr(config, "max_retries", 1)
                agent_model.timeout = getattr(config, "timeout", 30)
                agent_model.validation_required = getattr(config, "validation_required", False)
                agent_model.cache_results = getattr(config, "cache_results", False)

                # Salva primeiro para poder usar .set() nas tasks depois
                agent_model.save()

                if hasattr(agent_obj, "tasks"):
                    try:
                        
                        task_objs = []
                        for t in agent_obj.tasks:
                            
                            try:
                                task_from_db = TaskModel.objects.get(description=t.description)
                                task_objs.append(task_from_db)
                                print(f"   ✓ Encontrada no banco: {task_from_db}")
                            except TaskModel.DoesNotExist:
                                new_task = TaskModel.objects.create(
                                name=t.description,  # Nome simples baseado na descrição
                                description=t.description,
                                goal=t.goal,
                                output_format=t.output_format,
                                is_active=True,
                                )
                                task_objs.append(new_task)
                                print(f"   ➕ Criada nova Task no banco: {new_task}")
                        
                        print(f"Tasks válidas encontradas: {task_objs}")
                        agent_model.tasks.set(task_objs)
                        print(f"✓ Tasks associadas com sucesso ao agente '{agent_var_name}'.")

                    except Exception as e:
                        print(f"✗ Erro ao extrair tasks do agente '{agent_var_name}': {str(e)}")
                        agent_model.tasks.clear()
                else:
                    print(f"Agente '{agent_var_name}' NÃO possui atributo 'tasks'. Limpando tasks.")
                    agent_model.tasks.clear()
                    db_agents.append(agent_model)

        return sorted(db_agents, key=lambda x: x.name)

    @swagger_auto_schema(
        operation_description="Retorna a lista completa de agentes disponíveis",
        responses={200: openapi.Response("Lista de Agentes", AgentSerializer(many=True))}
    )
    def list(self, request):
        """Lista todos os agentes disponíveis."""
        agents = self.get_queryset()
        serializer = self.get_serializer(agents, many=True)
        return DRFResponse(serializer.data)
    
    @swagger_auto_schema(
        operation_description="Cria um novo agente no banco de dados como Agent",
        request_body=AgentSerializer,
        responses={
            201: openapi.Response("Agente criado", AgentSerializer),
            400: openapi.Response("Dados inválidos", 
                schema=openapi.Schema(type=openapi.TYPE_OBJECT)
            )
        }
    )
    def create(self, request):
        """Cria um novo agente no banco de dados e o registra como Agent."""
        serializer = self.get_serializer(data=request.data)
        
        if serializer.is_valid():
            # Salvar a tarefa no banco de dados
            agent = serializer.save()
            
            try:              
                provider = gemini_provider()
                                
                # Criar uma instância de Agent
                agent_obj = Agent(
                    role=agent.role,
                    tasks=agent.tasks,
                    llm_provider=provider
                )
                
                # Registrar o agente no módulo global
                agent_var_name = f"{agent.name}_agent"
                globals()[agent_var_name] = agent
                
                # Se o módulo de agents estiver acessível, também registrar lá
                if 'presets.agents' in sys.modules:
                    import genaitor.presets.agents
                    setattr(genaitor.presets.agents, agent_var_name, agent_obj)

                agent_mapping = {}
                for var in globals():
                    if var.endswith('_agent'):
                        agent_mapping[var] = globals()[var]
                        
                return DRFResponse(
                    self.get_serializer(agent).data,
                    status=status.HTTP_201_CREATED
                )
            except Exception as e:
                # Se falhar ao registrar como Agent, ainda retornamos sucesso
                # mas logamos o erro para debugging
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Erro ao registrar agent como Agent: {str(e)}")
                
                return DRFResponse(
                    {
                        **self.get_serializer(agent).data,
                        "warning": "O agente foi criada no banco de dados, mas não foi possível registrá-lo como Agent no sistema."
                    },
                    status=status.HTTP_201_CREATED
                )
        
        return DRFResponse(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )
    


class GenaitorViewSet(viewsets.ViewSet):
    """
    ViewSet para as operações principais da API Genaitor.
    """
    
    @swagger_auto_schema(
        operation_description="Verifica se a API está funcionando e mostra endpoints disponíveis",
        responses={200: openapi.Response("API Status", schema=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='Mensagem de status'),
                'endpoints': openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    description='Lista de endpoints disponíveis'
                )
            }
        ))}
    )
    @action(detail=False, methods=['get'])
    def status(self, request):
        """
        Endpoint para verificar o status da API
        """
        return DRFResponse({
            "message": "Genaitor API running!",
            "endpoints": {
                "status": "GET /api/genaitor/status/ - Informações sobre a API",
                "process": "POST /api/genaitor/process/ - Processa uma solicitação usando um agente",
                "agents": "GET /api/agents/ - Lista todos os agentes disponíveis",
                "tasks": "GET /api/tasks/ - Lista todas as tarefas disponíveis",
                "tasks_create": "POST /api/tasks/ - Cria uma nova tarefa",
                "docs": "GET /docs/ - Documentação Swagger da API"
            }
        })
    
    @swagger_auto_schema(
        operation_description="Processa uma solicitação utilizando um agente específico",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['agent_name', 'input_data'],
            properties={
                'agent_name': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Nome do agente a ser utilizado (ex: qa_agent, data_analysis_agent, etc.)"
                ),
                'input_data': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Dados de entrada a serem processados pelo agente selecionado"
                ),
            }
        ),
        responses={
            200: openapi.Response("Resposta bem-sucedida", ResponseSerializer),
            400: openapi.Response("Solicitação inválida", schema=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'error': openapi.Schema(type=openapi.TYPE_STRING, description='Mensagem de erro'),
                    'message': openapi.Schema(type=openapi.TYPE_STRING, description='Descrição detalhada do erro')
                }
            )),
            500: openapi.Response("Erro interno do servidor", schema=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'error': openapi.Schema(type=openapi.TYPE_STRING, description='Mensagem de erro')
                }
            ))
        }
    )
    @action(detail=False, methods=['post'])
    def process(self, request):
        """
        Processa uma solicitação utilizando um agente específico do Genaitor
        """
        data = request.data
        errors = {}

        if not data.get('agent_name'):
            errors['agent_name'] = ["Este campo é obrigatório."]

        if not data.get('input_data'):
            errors['input_data'] = ["Este campo é obrigatório."]

        if errors:
            return DRFResponse(errors, status=status.HTTP_400_BAD_REQUEST)

        agent_name = data.get('agent_name')
        input_data = data.get('input_data')

        agent_instance = None
        agent_callable = None

        # 1. Verifica se está no AGENT_MAPPING
        if agent_name in AGENT_MAPPING:
            agent_callable = AGENT_MAPPING[agent_name]
        else:
            # 2. Tenta buscar no banco
            try:
                agent_instance = AgentModel.objects.get(role=agent_name)
                agent_callable = agent_instance.to_agent()  # método do model que instancia um Agent real
            except AgentModel.DoesNotExist:
                return DRFResponse(
                    {
                        "error": "Agent not found!",
                        "message": f"O agente '{agent_name}' não está disponível. Use o endpoint /api/agents/ para ver a lista de agentes disponíveis."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )

        try:
            # Configurar o orquestrador com o agente solicitado
            orchestrator = Orchestrator(
                agents={agent_name: agent_callable},
                flows={"default_flow": Flow(agents=[agent_name], context_pass=[True])},
                mode=ExecutionMode.SEQUENTIAL
            )

            # Processar a requisição de forma assíncrona
            result = asyncio.run(orchestrator.process_request(input_data, flow_name='default_flow'))

            # Criar objeto Response e serializar
            response_obj = ResponseModel(response=result["content"].get(agent_name))
            response_serializer = ResponseSerializer(response_obj)

            return DRFResponse(response_serializer.data)

        except Exception as e:
            return DRFResponse(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TaskViewSet(viewsets.ModelViewSet):
    """
    ViewSet para gerenciar tarefas.
    Permite criar, ler, atualizar e excluir tarefas no banco de dados.
    """
    queryset = TaskModel.objects.all()
    serializer_class = TaskSerializer
    
    def get_queryset(self):
        """
        Retorna as tarefas do banco de dados e adiciona as tarefas do módulo tasks
        que ainda não foram salvas no banco.
        """
        # Obtém as tarefas já salvas no banco de dados
        db_tasks = list(TaskModel.objects.all())
        
        # Nomes das tarefas já salvas no banco
        db_task_names = [task.name for task in db_tasks]
        
        # Lista de variáveis que contêm "_task" em seus nomes
        task_vars = [var for var in globals() if "_task" in var]
        
        # Adiciona tarefas do módulo que não estão no banco
        for task_var in task_vars:
            if task_var not in db_task_names:
                task_obj = globals().get(task_var)
                if task_obj:
                    # Criar e salvar a tarefa no banco de dados
                    task = TaskModel()
                    task.name = task_var
                    
                    # Extrair informações da tarefa, se disponível
                    if hasattr(task_obj, 'description'):
                        task.description = task_obj.description
                    else:
                        task.description = f"Tarefa: {task_var}"
                        
                    if hasattr(task_obj, 'goal'):
                        task.goal = task_obj.goal
                    else:
                        task.goal = f"Executar a tarefa {task_var}"
                        
                    if hasattr(task_obj, 'output_format'):
                        task.output_format = task_obj.output_format
                    else:
                        task.output_format = "Texto em formato livre"
                        
                    if hasattr(task_obj, 'prompt'):
                        task.prompt_template = task_obj.prompt
                    else:
                        task.prompt_template = "Sem template disponível"
                    
                    # Configurações da tarefa
                    if hasattr(task_obj, 'config'):
                        if hasattr(task_obj.config, 'max_retries'):
                            task.max_retries = task_obj.config.max_retries
                        if hasattr(task_obj.config, 'timeout'):
                            task.timeout = task_obj.config.timeout
                        if hasattr(task_obj.config, 'validation_required'):
                            task.validation_required = task_obj.config.validation_required
                        if hasattr(task_obj.config, 'cache_results'):
                            task.cache_results = task_obj.config.cache_results
                    
                    # Salvar no banco
                    task.save()
                    db_tasks.append(task)
        
        return sorted(db_tasks, key=lambda x: x.name)
    
    @swagger_auto_schema(
        operation_description="Retorna a lista completa de tarefas disponíveis",
        responses={200: openapi.Response("Lista de Tarefas", TaskSerializer(many=True))}
    )
    def list(self, request):
        """Lista todas as tarefas disponíveis."""
        tasks = self.get_queryset()
        serializer = self.get_serializer(tasks, many=True)
        return DRFResponse(serializer.data)
    
    @swagger_auto_schema(
        operation_description="Cria uma nova tarefa no banco de dados como GeneralTask",
        request_body=TaskSerializer,
        responses={
            201: openapi.Response("Tarefa criada", TaskSerializer),
            400: openapi.Response("Dados inválidos", 
                schema=openapi.Schema(type=openapi.TYPE_OBJECT)
            )
        }
    )
    def create(self, request):
        """Cria uma nova tarefa no banco de dados e a registra como GeneralTask."""
        serializer = self.get_serializer(data=request.data)
        
        if serializer.is_valid():
            # Salvar a tarefa no banco de dados
            task = serializer.save()
            
            try:              
                provider = gemini_provider()
                
                # Criar uma instância de GeneralTask
                general_task = GeneralTask(
                    description=task.description,
                    goal=task.goal,
                    output_format=task.output_format,
                    llm_provider=provider
                )
                
                # Registrar a tarefa no módulo global
                task_var_name = f"{task.name}_task"
                globals()[task_var_name] = general_task
                
                # Se o módulo de tarefas estiver acessível, também registrar lá
                if 'presets.tasks' in sys.modules:
                    import genaitor.presets.tasks
                    setattr(presets.tasks, task_var_name, general_task)
                
                # Adicionar a nova tarefa ao mapeamento de tarefas existentes
                # Isso torna-a disponível imediatamente para uso
                task_mapping = {}
                for var in globals():
                    if var.endswith('_task'):
                        task_mapping[var] = globals()[var]
                        
                return DRFResponse(
                    self.get_serializer(task).data,
                    status=status.HTTP_201_CREATED
                )
            except Exception as e:
                # Se falhar ao registrar como GeneralTask, ainda retornamos sucesso
                # mas logamos o erro para debugging
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Erro ao registrar task como GeneralTask: {str(e)}")
                
                return DRFResponse(
                    {
                        **self.get_serializer(task).data,
                        "warning": "A tarefa foi criada no banco de dados, mas não foi possível registrá-la como GeneralTask no sistema."
                    },
                    status=status.HTTP_201_CREATED
                )
        
        return DRFResponse(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # @swagger_auto_schema(
    #     operation_description="Obtém uma tarefa específica por ID",
    #     responses={
    #         200: openapi.Response("Detalhes da tarefa", TaskSerializer),
    #         404: openapi.Response("Tarefa não encontrada", 
    #             schema=openapi.Schema(
    #                 type=openapi.TYPE_OBJECT,
    #                 properties={
    #                     'error': openapi.Schema(type=openapi.TYPE_STRING)
    #                 }
    #             )
    #         )
    #     }
    # )
    # def retrieve(self, request, pk=None):
    #     """Obtém uma tarefa específica."""
    #     try:
    #         task = TaskModel.objects.get(pk=pk)
    #         serializer = self.get_serializer(task)
    #         return DRFResponse(serializer.data)
    #     except TaskModel.DoesNotExist:
    #         return DRFResponse(
    #             {"error": f"Tarefa com ID {pk} não encontrada"},
    #             status=status.HTTP_404_NOT_FOUND
    #         )
    
    # @swagger_auto_schema(
    #     operation_description="Testa a execução de uma tarefa com dados de entrada",
    #     request_body=openapi.Schema(
    #         type=openapi.TYPE_OBJECT,
    #         properties={
    #             'input_data': openapi.Schema(
    #                 type=openapi.TYPE_STRING,
    #                 description="Dados de entrada para teste da tarefa"
    #             )
    #         },
    #         required=['input_data']
    #     ),
    #     responses={
    #         200: openapi.Response("Resultado da execução", ResponseSerializer),
    #         400: openapi.Response("Dados inválidos", 
    #             schema=openapi.Schema(type=openapi.TYPE_OBJECT)
    #         ),
    #         404: openapi.Response("Tarefa não encontrada", 
    #             schema=openapi.Schema(
    #                 type=openapi.TYPE_OBJECT,
    #                 properties={
    #                     'error': openapi.Schema(type=openapi.TYPE_STRING)
    #                 }
    #             )
    #         )
    #     }
    # )
    # @action(detail=True, methods=['post'])
    # def test(self, request, pk=None):
    #     """Testa a execução de uma tarefa com dados de entrada."""
    #     try:
    #         task = TaskModel.objects.get(pk=pk)
    #     except TaskModel.DoesNotExist:
    #         return DRFResponse(
    #             {"error": f"Tarefa com ID {pk} não encontrada"},
    #             status=status.HTTP_404_NOT_FOUND
    #         )
        
    #     # Validação dos dados de entrada
    #     input_data = request.data.get('input_data')
    #     if not input_data:
    #         return DRFResponse(
    #             {"error": "O campo 'input_data' é obrigatório"},
    #             status=status.HTTP_400_BAD_REQUEST
    #         )
            
    #     try:
    #         # Importar o provider padrão
    #         from presets.providers import gemini_provider
    #         provider = gemini_provider()
            
    #         # Converter para GeneralTask
    #         general_task = task.to_general_task(provider)
            
    #         # Executar a tarefa
    #         result = general_task.execute(input_data)
            
    #         # Retornar o resultado
    #         response_obj = ResponseModel(response=result.content)
    #         response_serializer = ResponseSerializer(response_obj)
            
    #         return DRFResponse(response_serializer.data)
                
    #     except Exception as e:
    #         return DRFResponse(
    #             {"error": str(e)},
    #             status=status.HTTP_500_INTERNAL_SERVER_ERROR
    #         )
    
    # @swagger_auto_schema(
    #     operation_description="Atualiza uma tarefa existente",
    #     request_body=TaskSerializer,
    #     responses={
    #         200: openapi.Response("Tarefa atualizada", TaskSerializer),
    #         400: openapi.Response("Dados inválidos", 
    #             schema=openapi.Schema(type=openapi.TYPE_OBJECT)
    #         ),
    #         404: openapi.Response("Tarefa não encontrada", 
    #             schema=openapi.Schema(
    #                 type=openapi.TYPE_OBJECT,
    #                 properties={
    #                     'error': openapi.Schema(type=openapi.TYPE_STRING)
    #                 }
    #             )
    #         )
    #     }
    # )
    # def update(self, request, pk=None):
    #     """Atualiza uma tarefa existente."""
    #     try:
    #         task = TaskModel.objects.get(pk=pk)
    #     except TaskModel.DoesNotExist:
    #         return DRFResponse(
    #             {"error": f"Tarefa com ID {pk} não encontrada"},
    #             status=status.HTTP_404_NOT_FOUND
    #         )
        
    #     serializer = self.get_serializer(task, data=request.data)
        
    #     if serializer.is_valid():
    #         # Salvar atualização no banco
    #         updated_task = serializer.save()
            
    #         try:
    #             # Importar o GeneralTask e provider
    #             from presets.tasks_objects import GeneralTask
    #             from presets.providers import gemini_provider
    #             provider = gemini_provider()
                
    #             # Criar ou atualizar a instância de GeneralTask
    #             general_task = GeneralTask(
    #                 description=updated_task.description,
    #                 goal=updated_task.goal,
    #                 output_format=updated_task.output_format,
    #                 llm_provider=provider
    #             )
                
    #             # Atualizar a tarefa no módulo global
    #             task_var_name = f"{updated_task.name}_task"
    #             globals()[task_var_name] = general_task
                
    #             # Se o módulo de tarefas estiver acessível, também atualizar lá
    #             import sys
    #             if 'presets.tasks' in sys.modules:
    #                 import presets.tasks
    #                 setattr(presets.tasks, task_var_name, general_task)
                
    #             return DRFResponse(serializer.data)
    #         except Exception as e:
    #             # Se falhar ao atualizar como GeneralTask, ainda retornamos sucesso
    #             # mas logamos o erro para debugging
    #             import logging
    #             logger = logging.getLogger(__name__)
    #             logger.error(f"Erro ao atualizar task como GeneralTask: {str(e)}")
                
    #             return DRFResponse(
    #                 {
    #                     **serializer.data,
    #                     "warning": "A tarefa foi atualizada no banco de dados, mas não foi possível atualizar sua instância GeneralTask no sistema."
    #                 }
    #             )
            
    #     return DRFResponse(
    #         serializer.errors,
    #         status=status.HTTP_400_BAD_REQUEST
    #     )
    
    # @swagger_auto_schema(
    #     operation_description="Exclui uma tarefa existente",
    #     responses={
    #         204: openapi.Response("Tarefa excluída com sucesso"),
    #         404: openapi.Response("Tarefa não encontrada", 
    #             schema=openapi.Schema(
    #                 type=openapi.TYPE_OBJECT,
    #                 properties={
    #                     'error': openapi.Schema(type=openapi.TYPE_STRING)
    #                 }
    #             )
    #         )
    #     }
    # )
    # def destroy(self, request, pk=None):
    #     """Exclui uma tarefa específica."""
    #     try:
    #         task = TaskModel.objects.get(pk=pk)
            
    #         # Remover a instância GeneralTask global, se existir
    #         task_var_name = f"{task.name}_task"
    #         if task_var_name in globals():
    #             del globals()[task_var_name]
            
    #         # Remover do módulo de tarefas, se existir
    #         try:
    #             import sys
    #             if 'presets.tasks' in sys.modules:
    #                 import presets.tasks
    #                 if hasattr(presets.tasks, task_var_name):
    #                     delattr(presets.tasks, task_var_name)
    #         except Exception as e:
    #             import logging
    #             logger = logging.getLogger(__name__)
    #             logger.error(f"Erro ao remover GeneralTask ao excluir: {str(e)}")
            
    #         # Excluir a tarefa do banco de dados
    #         task.delete()
    #         return DRFResponse(status=status.HTTP_204_NO_CONTENT)
    #     except TaskModel.DoesNotExist:
    #         return DRFResponse(
    #             {"error": f"Tarefa com ID {pk} não encontrada"},
    #             status=status.HTTP_404_NOT_FOUND
    #         )
