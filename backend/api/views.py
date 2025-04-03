from django.shortcuts import render
import sys
import os
import asyncio
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response as DRFResponse
from .serializers import ResponseSerializer, AgentSerializer, TaskSerializer
from .models import AgentModel, TaskModel, Response as ResponseModel
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# Adicionar o caminho do projeto ao sys.path para importar os módulos
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Importações atualizadas dos módulos src
from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import *
from presets.tasks import *
from presets.tasks_objects import *
from presets.providers import *

# Mapeamento dos agentes disponíveis
AGENT_MAPPING = {
    "qa_agent": qa_agent,
    "autism_agent": autism_agent,
    "agent_creation": agent_creation,
    "data_understanding_agent": data_understanding_agent,
    "statistics_agent": statistics_agent,
    "anomalies_detection_agent": anomalies_detection_agent,
    "data_analysis_agent": data_analysis_agent,
    "problem_analysis_agent": problem_analysis_agent,
    "numerical_analysis_agent": numerical_analysis_agent,
    "pinn_modeling_agent": pinn_modeling_agent,
    "preferences_agent": preferences_agent,
    "payment_agent": payment_agent,
    "proposal_agent": proposal_agent,
    "review_agent": review_agent,
    "extraction_agent": extraction_agent,
    "matching_agent": matching_agent,
    "scoring_agent": scoring_agent,
    "report_agent": report_agent,
    "optimization_agent": optimization_agent,
    "educational_agent": educational_agent,
    "research_agent": research_agent,
    "content_agent": content_agent,
    "personalization_agent": personalization_agent,
    "financial_agent": financial_agent,
    "summarization_agent": summarization_agent,
    "linkedin_agent": linkedin_agent,
    "pinn_tuning_agent": pinn_tuning_agent,
    "html_analysis_agent": html_analysis_agent,
    "scraper_generation_agent": scraper_generation_agent,
    "equation_solver_agent": equation_solver_agent,
    "pinn_generation_agent": pinn_generation_agent,
    "hyperparameter_optimization_agent": hyperparameter_optimization_agent,
    "orchestrator_agent": orchestrator_agent,
    "validator_agent": validator_agent,
    "requirements_agent": requirements_agent,
    "architecture_agent": architecture_agent,
    "code_generation_agent": code_generation_agent,
    "destination_selection_agent": destination_selection_agent,
    "budget_estimation_agent": budget_estimation_agent,
    "itinerary_planning_agent": itinerary_planning_agent,
    "feature_selection_agent": feature_selection_agent,
    "signal_analysis_agent": signal_analysis_agent,
    "residual_evaluation_agent": residual_evaluation_agent,
    "lstm_model_agent": lstm_model_agent,
    "lstm_residual_evaluation_agent": lstm_residual_evaluation_agent,
    "document_agent": document_agent,
    "question_agent": question_agent,
    "search_agent": search_agent,
    "response_agent": response_agent,
    "performance_agent": performance_agent,
    "fatigue_agent": fatigue_agent,
    "tactical_agent": tactical_agent,
    "scraping_agent": scraping_agent,
    "analysis_agent": analysis_agent,
    "disaster_analysis_agent": disaster_analysis_agent,
    "agro_analysis_agent": agro_analysis_agent,
    "ecological_analysis_agent": ecological_analysis_agent,
    "air_quality_analysis_agent": air_quality_analysis_agent,
    "vegetation_analysis_agent": vegetation_analysis_agent,
    "soil_moisture_analysis_agent": soil_moisture_analysis_agent
}

class AgentViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet para listar todos os agentes disponíveis.
    """
    queryset = []
    serializer_class = AgentSerializer
    
    def get_queryset(self):
        # Cria objetos AgentModel a partir do mapeamento
        agents = []
        for agent_name in sorted(AGENT_MAPPING.keys()):
            agent = AgentModel()
            agent.name = agent_name
            agents.append(agent)
        return agents
    
    @swagger_auto_schema(
        operation_description="Retorna a lista completa de agentes disponíveis",
        responses={200: openapi.Response("Lista de Agentes", AgentSerializer(many=True))}
    )
    def list(self, request):
        """Lista todos os agentes disponíveis."""
        agents = self.get_queryset()
        serializer = self.get_serializer(agents, many=True)
        return DRFResponse(serializer.data)


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
        # Validação manual dos dados
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
        
        if agent_name not in AGENT_MAPPING:
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
                agents={agent_name: AGENT_MAPPING[agent_name]},
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
                    import presets.tasks
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
