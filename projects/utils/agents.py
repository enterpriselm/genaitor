from utils.request_helper import make_llama_request

class Agent:
    def __init__(self, role, goal="", system_message=""):
        self.role = role
        self.goal = goal
        self.system_message = system_message

    def perform_task(self, input_text):
        response = make_llama_request(input_text, system_message=self.system_message)
        if response.get("error"):
            return {"error": f"{self.role} encountered an error: {response['error']}"}
        return response["content"]