# Backend Genaitor

API desenvolvida com Django e Django REST Framework para o framework Genaitor de orquestração de agentes de IA.

## Visão Geral da Arquitetura

Este backend é construído utilizando o framework web [Django](https://www.djangoproject.com/) e o [Django REST Framework (DRF)](https://www.django-rest-framework.org/) para a criação de APIs RESTful.

*   **Framework:** Django 5.1
*   **API:** Django REST Framework
*   **Banco de Dados:** SQLite (configuração padrão de desenvolvimento)
*   **Documentação da API:** drf-yasg (Swagger/OpenAPI)
*   **CORS:** django-cors-headers (habilitado para todas as origens em desenvolvimento)

A estrutura segue o padrão Django, com uma app principal (`api`) contendo a lógica de negócio e os endpoints, e uma pasta de configuração do projeto (`backend`).

## Configuração do Ambiente de Desenvolvimento

Siga os passos abaixo para configurar e executar o backend localmente:

1.  **Clone o Repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd genaitor/backend
    ```

2.  **Crie e Ative um Ambiente Virtual:**
    ```bash
    python -m venv ../env  # Cria o ambiente na pasta raiz do projeto
    source ../env/bin/activate  # Linux/macOS
    # ou
    ..\env\Scripts\activate  # Windows
    ```

3.  **Instale as Dependências:**
    Certifique-se de que o ambiente virtual está ativado.
    ```bash
    pip install -r ../requirements.txt
    ```
    *Nota: O arquivo `requirements.txt` está na raiz do projeto.*

4.  **Configure Variáveis de Ambiente (se necessário):**
    Se o projeto utilizar um arquivo `.env` para variáveis de ambiente (ex: chaves de API, configurações de banco de dados em produção), crie um arquivo `.env` na raiz do projeto (onde está o `manage.py`) baseado em um possível `.env.example`.

5.  **Aplique as Migrações do Banco de Dados:**
    ```bash
    python manage.py migrate
    ```

6.  **Crie um Superusuário (opcional, para acesso ao Admin):**
    ```bash
    python manage.py createsuperuser
    ```
    Siga as instruções para criar um usuário administrador.

7.  **Execute o Servidor de Desenvolvimento:**
    ```bash
    python manage.py runserver
    ```
    O servidor estará disponível em `http://127.0.0.1:8000/`.

## Estrutura do Projeto (`backend/`)

```
backend/
├── api/
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── serializers.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── backend/      
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── db.sqlite3
├── manage.py
└── README.md
```

*   **`api/`**: Contém a lógica central da API, incluindo modelos de dados (`models.py`), serializadores para conversão de dados (`serializers.py`), lógica de visualização/endpoints (`views.py`) e roteamento específico da API (`urls.py`).
*   **`backend/`**: Diretório de configuração do projeto Django. Inclui configurações globais (`settings.py`), roteamento principal (`urls.py`), e configurações para ASGI/WSGI.
*   **`llm/`**: Provavelmente contém código relacionado à interação com modelos de linguagem grandes (LLMs) ou à lógica específica dos agentes Genaitor. (Necessita de análise mais aprofundada se necessário).
*   **`manage.py`**: Script utilitário do Django para tarefas administrativas como rodar o servidor, aplicar migrações, etc.

## Endpoints da API

A API expõe os seguintes grupos principais de endpoints:

*   **`/api/agents/`**: Endpoints para gerenciar Agentes (CRUD via DRF ViewSet).
*   **`/api/tasks/`**: Endpoints para gerenciar Tarefas (CRUD via DRF ViewSet).
*   **`/api/genaitor/`**: Endpoints relacionados à orquestração Genaitor (detalhes no `GenaitorViewSet`).

### Documentação Interativa (Swagger/OpenAPI)

A documentação completa e interativa da API está disponível quando o servidor está rodando:

*   **Swagger UI:** `http://127.0.0.1:8000/docs/`
*   **ReDoc UI:** `http://127.0.0.1:8000/redoc/`

Use essas interfaces para explorar os endpoints em detalhes, ver os modelos de dados esperados e testar as chamadas da API diretamente do navegador.
