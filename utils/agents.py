import google.generativeai as genai
from transformers import pipeline

from api.utils import Log


class Task:
    def __init__(self, description, goal, output_format):
        self.description = description
        self.goal = goal
        self.output_format = output_format


class Agent:
    def __init__(self, role, task):
        self.role = role
        self.task = task

    def generate_conversation(self, user_query, specialist_answer=None):
        task = self.task
        conversation = (
            f"You're genaitor. An AI Assistant specialized in {self.role}\n\n"
            f"{task.description}\n\n"
            f"{task.goal}\n\n"
            f"{task.output_format}\n\n"
        )
        if specialist_answer is not None:
            conversation += f"Specialist explanation: {specialist_answer}\n\n"
            temp_cv = conversation.partition("If you don't have the answer, just say it.")[0]
            temp_cv+="If you don't have the answer, just say it."
            temp_cv+='Specialist explanation:'
            temp_cv+=conversation.partition("Specialist explanation:")[2]
            conversation = temp_cv
        conversation+=f"User requests: {user_query}\n\n"
        return conversation


class Orchestrator:
    def __init__(self, api_key):
        self.api_key = api_key
        configure_genai(self.api_key)

    def summarize_history(self, history, max_length=1000, min_length=100):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(history, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]["summary_text"]

    def gemini_questions(self, query, temperature=0.1, max_tokens=1000):
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            query,
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )
        return response.candidates[0].content.parts[0].text

    def handle_specialist_request(self, agent, user_query):
        conversation = agent.generate_conversation(user_query)
        return self.gemini_questions(conversation)

    def validate_answer(self, user_query, main_answer):
        validation_task = Task(
            "Your work here is to analyze an user request and the answer provided by the user.",
            "\nYou then have to decide if the user request was solved or not.",
            "\nAfter that you will return an answer on this format:"
            "\n\nCustomer needs solved? {yes_or_no}"
            "\n\nExplanation: {here_the_explanation}"
        )
        agent = Agent("Customer Service Validator", validation_task)
        
        prompt = (
            f"\n\n{user_query}\n\n"
            f"Assistent answer: \n\n{main_answer}"
        )
        
        conversation = agent.generate_conversation(prompt)
        validation = self.gemini_questions(conversation)
        answer = validation.partition('solved? ')[2].partition('\n')[0]
        explanation = validation.partition('Explanation: ')[2]
        return {"answer": answer, "explanation": explanation}
    
    def main_pipeline(self, user_query, agents):
        conversation_history = ""
        main_answer = ""
        Log.info("Iniciando pipeline...")

        main_agent = agents['main']
        conversation = main_agent.generate_conversation(user_query)
        main_answer = self.gemini_questions(conversation)

        conversation_history += f"User query: {conversation}\n\nMain Agent Answer: {main_answer}\n\n"

        specialist_answer=''
        for key, agent in agents.items():
            if key != "main" and f"I need to talk to a {agent.role} specialist" in main_answer:
                specialist_answer+=f"Helper Agent ({agent.role}) "
                specialist_answer+=f"Answer: {self.handle_specialist_request(agent, user_query)}\n\n"

        if specialist_answer!='':
            conversation_history += specialist_answer
        
            conversation = main_agent.generate_conversation(user_query, specialist_answer)  
            conversation_history += conversation
            main_answer=self.gemini_questions(conversation)

            conversation_history += main_answer
        
        Log.info("Pipeline concluído com sucesso!")
        return main_answer, conversation_history

def configure_genai(api_key):
    genai.configure(api_key=api_key)


# if __name__ == "__main__":
#     API_KEY = 'AIzaSyDA3r3LpI8cIGm4AVoaDQ65mDMD10GNTVM'
#     USER_QUERY = "I need to build a PINN for solving a simple linear ODE"

#     main_task = Task(
#         "Your work here is to create solid, robust, and good Physics Informed Neural Networks models using PyTorch",
#         "You will receive a user request with all information needed to solve the problem, including boundary conditions, initial conditions, equations, and specific architectures.",
#         "If you know how to solve the request, the format of your answer should be: Python Code: {the_python_code}."
#         "\n\nIf you don't have the answer, just say it."
#         "\n\nIf you need more information for solve the problem you can request help of a physics specialist" 
#         " saying 'I need to talk to a Physics specialist' and explaining the problem you need help"
#         "\n\nYou can also request help of a Computer Science specialist saying 'I need to talk to a Computer Science specialist' and explaining the problem you need help"
#     )

#     physics_task = Task(
#         "Your work here is to clarify and answer any Physics concepts about any topics and complexity",
#         "\n\nYou need to clarify all the user doubts about it.",
#         "\n\nPhysics Explanation: {ai_explanation}"
#     )

#     cs_task = Task(
#         "\n\nYour work here is to clarify and answer any Software architecture and coding concepts about any topics and complexity",
#         "\n\nYou need to clarify all the user doubts about it.",
#         "\n\nComputer Science Explanation: {ai_explanation}"
#     )

#     agents = {
#         "main": Agent("Physics Informed Neural Networks", main_task),
#         "physics": Agent("Physics", physics_task),
#         "computer_science": Agent("Computer Science", cs_task)
#     }

#     manager = Orchestrator(API_KEY)
#     answer, history = manager.main_pipeline(USER_QUERY, agents)
#     print(answer)