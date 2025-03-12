import asyncio
import uvicorn
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import *

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI(title="Genaitor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestModel(BaseModel):
    agent_name: str
    input_data: str

@app.post("/process/")
async def process_request(request: RequestModel):
    if request.agent_name not in AGENT_MAPPING:
        raise HTTPException(status_code=400, detail="Agente not found!")

    orchestrator = Orchestrator(
        agents={request.agent_name: AGENT_MAPPING[request.agent_name]},
        flows={"default_flow": Flow(agents=[request.agent_name], context_pass=[True])},
        mode=ExecutionMode.SEQUENTIAL
    )

    result = await orchestrator.process_request(request.input_data, flow_name='default_flow')

    return {"response": result["content"].get(request.agent_name)}

@app.get("/")
async def root():
    return {"message": "Genaitor API running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
