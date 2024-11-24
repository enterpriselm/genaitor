from flask import Flask, request, jsonify
from genaitor.video_utils import transcribe_audio
import requests
from langchain_community.document_loaders import YoutubeLoader
from genaitor.config import config
from genaitor.utils.agents import Agent, Orchestrator, Task
import os
from flask_cors import CORS

agents = {
    'extractor': Agent(
        role='Text Extractor Agent',
        system_message=(
            "You are an expert in analyzing and extracting relevant information from text. "
            "When given a user's query and a text, identify and return the part of the text that justifies the answer. "
        ),
        temperature=0.7,
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
        temperature=0.6,
        max_tokens=1500,
        max_iterations=1
    ),
    'refiner': Agent(
        role='Refinement Agent',
        system_message=(
            "You are an expert in refining and improving text. "
            "Given a question and its initial response, refine the response to make it more precise, coherent, and helpful."
        ),
        temperature=0.8,
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
            goal="""Identify the most relevant part of the provided text that justifies the answer to the user's query.
            User's Query: {user_query}
            Input Text: {input_text}"""
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
            goal="""Evaluate the coherence of the extracted answer in the context of the user's query.
            User's Query: {user_query}
            Extracted Answer: {extracted_answer}"""
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
            goal="""Refine the extracted answer to make it more precise and helpful.
            User's Query: {user_query}
            Initial Answer: {extracted_answer}"""
        )


app = Flask(__name__)
CORS(app)

def run_genaitor(video_transcription, user_query):
    query_processing_tasks = QueryProcessingTasks()

    tasks = [
        query_processing_tasks.extraction_task(
            agent=agents['extractor']
        )
    ]

    orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)
    result = orchestrator.kickoff(user_query=user_query, input_text=video_transcription)

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

    orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)
    return orchestrator.kickoff(user_query=user_query, extracted_answer=extracted_answer)


@app.route('/youtube', methods=['POST'])
def get_answer():
    data = request.json
    youtube_url = data.get('youtube_url')
    user_query = data.get('user_query')
    #file_path = data.get('file_path')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    video_transcript = None

    if youtube_url:
        video_id = youtube_url.partition('watch?v=')[2]
        loader = YoutubeLoader(video_id)
        video_transcript = loader.load()[0].page_content

    #elif file_path != '':
    #    video_transcript = transcribe_audio(file_path)
    else:
        return jsonify({"error": "You must provide a YouTube URL"}), 400

    answer = run_genaitor(video_transcription=video_transcript, user_query=user_query)
    return jsonify({"answer":answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
