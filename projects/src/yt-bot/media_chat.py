from flask import Flask, request, jsonify
from langchain_community.document_loaders import YoutubeLoader
from genaitor.config import config
from genaitor.utils.agents import Agent, Orchestrator, Task
from flask_cors import CORS
from genaitor.utils.utils import extract_text_from_doc, extract_text_from_json, extract_text_from_pdf, extract_text_from_ppt, read_csv, read_excel, transcribe_audio_file, image_to_text

agents = {
    'extractor': Agent(
        role='Text Extractor Agent',
        system_message=(
            "You are an expert in analyzing and extracting relevant information from text. "
            "When given a user's query and a text, identify and return the part of the text that justifies the answer. "
        ),
        temperature=0.1,
        max_tokens=2000,
        max_iterations=1
    ),
    'validator': Agent(
        role='Validation Agent',
        system_message=(
            "You are an expert in logical reasoning and validation. "
            "Given a user's question and the answer extracted from the text, evaluate if the answer makes sense in the context of the question. "
            "Provide a response indicating whether the answer is coherent and why."
        ),
        temperature=0.1,
        max_tokens=2000,
        max_iterations=1
    ),
    'refiner': Agent(
        role='Refinement Agent',
        system_message=(
            "You are an expert in refining and improving text. "
            "Given a question and its initial response, refine the response to make it more precise, coherent, and helpful."
        ),
        temperature=0.1,
        max_tokens=2000,
        max_iterations=1
    )
}

class QueryProcessingTasks():

    def extraction_task(self, agent):
        return Task(
            description=f"""
            This task involves extracting relevant information from a provided text based on the user's query. The extracted content must directly address the user's question and justify the response with specific portions of the text.
            """,
            expected_output=f"""
            In plain text: A concise excerpt from the input text that directly justifies the answer to the user's query.
            """,
            agent=agent,
            output_file='extracted_text.txt',
            goal="""Extract the most relevant information from the provided text that directly addresses the user's query. Include specific portions of the text to justify the answer.

                    ### Input:
                    Text: {text}
                    User Query: {user_query}

                    ### Output Format:
                    A concise excerpt from the input text that directly justifies the answer to the user's query, in plain text.

            """
        )

    def validation_task(self, agent):
        return Task(
            description=f"""
            This task evaluates whether the answer extracted from the text aligns logically and contextually with the user's query. 
            The agent will provide a rationale for its assessment.
            """,
            expected_output=f"""
            A plain text response stating whether the answer makes sense, followed by a brief explanation.
            """,
            agent=agent,
            output_file='validation_report.txt',
            goal="""Assess whether the extracted answer aligns with the user's query in terms of logic and context. Provide a rationale for your assessment.

                    ### Input:
                    User Query: {user_query}
                    Extracted Answer: {extracted_answer}

                    ### Output Format:
                    - Validation result: [Yes/No]
                    - Explanation: [A brief rationale for your assessment]
"""
        )

    def refinement_task(self, agent):
        return Task(
            description=f"""
            This task refines the answer to ensure it is precise, coherent, and directly addresses the user's query.
            """,
            expected_output=f"""
            A plain text version of the refined answer.
            """,
            agent=agent,
            output_file='refined_answer.txt',
            goal="""Refine the extracted answer to ensure it is precise, coherent, and directly addresses the user's query.

                    ### Input:
                    Initial Answer: {extracted_answer}
                    User Query: {user_query}

                    ### Output Format:
                    The refined answer in plain text.
            """
        )

def run_genaitor(text, user_query):
    query_processing_tasks = QueryProcessingTasks()

    tasks = [
        query_processing_tasks.extraction_task(
            agent=agents['extractor']
        )
    ]

    orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)
    result = orchestrator.kickoff(user_query=user_query, text=text)

    tasks = [
        query_processing_tasks.validation_task(
            agent=agents['validator']
        ),
        query_processing_tasks.refinement_task(
            agent=agents['refiner']
        )
    ]

    for key_answer in result['output']['output'][0].keys():
        extracted_answer = result['output']['output'][0][key_answer]

    print(extracted_answer)

    orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)
    extracted_answer = str(extracted_answer) if not isinstance(extracted_answer, str) else extracted_answer
    return orchestrator.kickoff(user_query=user_query, extracted_answer=extracted_answer)


app = Flask(__name__)
CORS(app)

@app.route('/text_analyzer', methods=['POST'])
def get_answer():
    data = request.json
    media_data = data.get('media_data')
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    text = ''
    for media in media_data:
        if media.startswith('https://'):
            try:
                video_id = media.partition('watch?v=')[2]
                loader = YoutubeLoader(video_id)
                text+=loader.load()[0].page_content
                text+='\n'
            except:
                pass
        elif media.endswith('.mp3') or media.endswith('.mp4'):
            try:
                text+=transcribe_audio_file(media)
                text+='\n'
            except:
                pass
        elif media.endswith('.doc'):
            try:
                text+=extract_text_from_doc(media)
                text+='\n'
            except:
                pass
        elif media.endswith('.json'):
            try:
                text+=extract_text_from_json(media)
                text+='\n'
            except:
                pass
        elif media.endswith('.pdf'):
            try:
                text+=extract_text_from_pdf(media)
                text+='\n'
            except:
                pass
        elif media.endswith('.ppt') or media.endswith('pptx'):
            try:
                text+=extract_text_from_ppt(media)
                text+='\n'
            except:
                pass
        elif media.endswith('.xlsx') or media.endswith('.xls'):
            try:
                text+=read_excel(media)
                text+='\n'
            except:
                pass
        elif media.endswith('.csv'):
            try:
                text+=read_csv(media)
                text+='\n'
            except:
                pass
        elif media.endswith('.jpg') or media.endswith('.jpeg') or media.endswith('.png'):
            text+=image_to_text(media)
            text+='\n'
        else:
            pass
    
    #text=text.replace('Estimate on the Generalization Error for TPBVP arising from the PMP application to OCP (cont’d)', '')
    answer = run_genaitor(text=text, user_query=user_query)
    for answer_key in answer['output']['output'][0].keys():
        answer = answer['output']['output'][-1][answer_key]
    return jsonify({"answer":answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
