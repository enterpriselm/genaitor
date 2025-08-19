from genaitor.core import Agent, AgentRole
from genaitor.llm.base import LLMProvider
from genaitor.presets.tasks import create_preset_tasks

def create_preset_agents(llm_provider: LLMProvider, preset_tasks):
    if not isinstance(llm_provider, LLMProvider):
        raise TypeError("llm_provider must be an instance of LLMProvider.")

    dev_requirements_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["get_requirements_task"]],
        llm_provider=provider
    )

    backend_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["backend_dev_task"]],
        llm_provider=provider
    )

    frontend_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["frontend_dev_task"]],
        llm_provider=provider
    )

    cicd_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["cicd_task"]],
        llm_provider=provider
    )

    product_manager_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["product_manager_task"]],
        llm_provider=provider
    )

    dev_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["dev_agent_task"]],
        llm_provider=provider
    )

    ui_designer_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["ui_designer_task"]],
        llm_provider=provider
    )

    growth_hacker_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["growth_hacker_task"]],
        llm_provider=provider
    )

    qa_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["qa_task"]],
        llm_provider=provider
    )

    debugging_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["debugging_task"]],
        llm_provider=provider
    )

    autism_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["autism_task"]],
        llm_provider=provider
    )

    agent_creation = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["agent_creation_task"]],
        llm_provider=provider
    )

    data_understanding_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["data_understanding_task"]],
        llm_provider=provider
    )

    statistics_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["statistics_task"]],
        llm_provider=provider
    )

    anomalies_detection_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["anomalies_detection_task"]],
        llm_provider=provider
    )

    data_analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["data_analysis_task"]],
        llm_provider=provider
    )

    problem_analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["problem_analysis_task"]],
        llm_provider=provider
    )

    numerical_analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["numerical_modeling_task"]],
        llm_provider=provider
    )

    pinn_modeling_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["pinn_modeling_task"]],
        llm_provider=provider
    )

    preferences_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["preferences_task"]],
        llm_provider=provider
    )

    payment_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["payment_task"]],
        llm_provider=provider
    )

    proposal_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["proposal_task"]],
        llm_provider=provider
    )

    review_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["review_task"]],
        llm_provider=provider
    )

    extraction_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[preset_tasks["data_extraction"]],
        llm_provider=provider
    )

    matching_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[preset_tasks["skill_matching"]],
        llm_provider=provider
    )

    scoring_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[preset_tasks["cv_scoring"]],
        llm_provider=provider
    )

    report_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[preset_tasks["report_generation"]],
        llm_provider=provider2
    )

    optimization_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[
            preset_tasks["model_selection_task"],
            preset_tasks["hyperparameter_tuning_task"],
            preset_tasks["model_evaluation_task"],
            preset_tasks["regularization_task"]
        ],
        llm_provider=provider
    )

    educational_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[
            preset_tasks["performance_task"],
            preset_tasks["prediction_task"],
            preset_tasks["material_task"],
            preset_tasks["language_task"]
        ],
        llm_provider=provider
    )

    research_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["audience_research"]],
        llm_provider=provider
    )

    content_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["email_generation"]],
        llm_provider=provider
    )

    personalization_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["email_personalization"]],
        llm_provider=provider
    )

    financial_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[
            preset_tasks["investment_task"],
            preset_tasks["credit_risk_task"],
            preset_tasks["portfolio_task"],
            preset_tasks["fraud_detection_task"]
        ],
        llm_provider=provider
    )

    summarization_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["paper_summarization"]],
        llm_provider=provider
    )

    linkedin_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["linkedin_post_generation"]],
        llm_provider=provider
    )

    pinn_tuning_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["pinn_tuning_task"]],
        llm_provider=provider
    )

    html_analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["html_analysis_task"]],
        llm_provider=provider
    )

    scraper_generation_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["scraper_generation"]],
        llm_provider=provider
    )

    equation_solver_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["equation_solver_task"]],
        llm_provider=provider
    )

    pinn_generation_agent = Agent(
        role=AgentRole.ENGINEER,
        tasks=[preset_tasks["pinn_generation_task"]],
        llm_provider=provider
    )

    hyperparameter_optimization_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[preset_tasks["hyperparameter_optimization_task"]],
        llm_provider=provider
    )

    orchestrator_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["orchestrator_task"]],
        llm_provider=provider
    )

    validator_agent = Agent(
        role=AgentRole.ENGINEER,
        tasks=[preset_tasks["validator_task"]],
        llm_provider=provider
    )

    requirements_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["requirements_analysis"]],
        llm_provider=provider
    )

    architecture_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["architecture_planning"]],
        llm_provider=provider
    )

    code_generation_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["code_generation"]],
        llm_provider=provider
    )

    destination_selection_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["destination_selection_task"]],
        llm_provider=provider
    )

    budget_estimation_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["budget_estimation_task"]],
        llm_provider=provider
    )

    itinerary_planning_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["itinerary_planning_task"]],
        llm_provider=provider
    )

    feature_selection_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["feature_selection_task"]],
        llm_provider=provider
    )

    signal_analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["signal_analysis_task"]],
        llm_provider=provider
    )

    residual_evaluation_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["residual_evaluation_task"]],
        llm_provider=provider
    )

    lstm_model_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["lstm_model_task"]],
        llm_provider=provider
    )

    lstm_residual_evaluation_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["lstm_residual_evaluation_task"]],
        llm_provider=provider
    )

    document_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["document_analysis"]],
        llm_provider=provider
    )

    question_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["question_analysis"]],
        llm_provider=provider
    )

    search_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["answer_search"]],
        llm_provider=provider
    )

    response_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["response_generation"]],
        llm_provider=provider
    )

    performance_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["performance_analysis"]],
        llm_provider=provider
    )

    fatigue_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["fatigue_detection"]],
        llm_provider=provider
    )

    tactical_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["tactical_adjustment"]],
        llm_provider=provider
    )

    scraping_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["security_scraping"]],
        llm_provider=provider2
    )

    analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["vulnerability_analysis"]],
        llm_provider=provider2
    )

    report_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preset_tasks["security_report"]],
        llm_provider=provider
    )

    disaster_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[DisasterAnalysisTask(provider)],
        llm_provider=provider
    )

    agro_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[AgroAnalysisTask(provider)],
        llm_provider=provider
    )

    ecological_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[EcologicalAnalysisTask(provider)],
        llm_provider=provider
    )

    air_quality_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[AirQualityAnalysisTask(provider)],
        llm_provider=provider
    )

    vegetation_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[VegetationAnalysisTask(provider)],
        llm_provider=provider
    )

    soil_moisture_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[SoilMoistureAnalysisTask(provider)],
        llm_provider=provider
    )

    structure_data_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[preset_tasks["structuring_task"]],
        llm_provider=provider
    )

    ocr_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[preset_tasks["ocr_task"]],
        llm_provider=provider
    )

    return {
        "dev_requirements_agent": dev_requirements_agent,
        "backend_agent": backend_agent,
        "frontend_agent": frontend_agent,
        "ci_cd_agent": cicd_agent,
        "product_manager_agent": product_manager_agent,
        "dev_agent": dev_agent,
        "ui_designer_agent": ui_designer_agent,
        "growth_hacker_agent": growth_hacker_agent,
        "qa_agent": qa_agent,
        "debugging_agent": debugging_agent,
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
        "soil_moisture_analysis_agent": soil_moisture_analysis_agent,
        "structure_data_agent": structure_data_agent,
        "ocr_agent": ocr_agent
    }

