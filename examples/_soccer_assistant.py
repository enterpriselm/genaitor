import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import SoccerAIProvider, SoccerAIConfig

llm_provider = SoccerAIProvider(SoccerAIConfig(api_key="your_api_key"))

class PerformanceAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Performance Analysis", "Analyze player performance based on real-time stats", "JSON format with performance insights and improvement suggestions")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Match Statistics: {input_data}

        Identify players who need more defensive or offensive support based on performance stats.
        Identify which player is contributing the most to the game's build-up?
        Which players are most exposed to counterattacks due to defensive positioning?
        Considering the statistics of high pressing and the average distance between players, how effective is our pressing strategy? 
       
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "performance_analysis"})

class FatigueDetectionTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Fatigue Detection", "Detect player fatigue and suggest adjustments", "JSON format with player fatigue levels and recommended actions")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Movement and Positioning Data: {input_data}

        Identify tired players and suggest replacements or role adjustments.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "fatigue_detection"})

class TacticalAdjustmentTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Tactical Adjustment", "Optimize team tactics based on match data", "JSON format with suggested tactical changes")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Match Context: {input_data}

        Suggest tactical adjustments to improve game performance, considering positioning and team dynamics.
        Suggest a strategy to improve goal-scoring opportunities, considering the weaknesses of the opposing defense.
        Suggest a tactical adjustment to enhance the connection between midfielders and forwards.
        Suggest adjustments to the defensive line or coverage to prevent counterattacks.        
        Analyzing ball possession time and the number of runs, suggest an adjustment in playing style to optimize pressure on opponents without compromising the team's energy.
        Suggest a tactical change to force an error in the opponent’s half.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "tactical_adjustment"})

performance_analysis_agent = Agent(
    role=AgentRole.ENGINEER,
    tasks=[PerformanceAnalysisTask(llm_provider)],
    llm_provider=llm_provider
)

fatigue_detection_agent = Agent(
    role=AgentRole.SPECIALIST,
    tasks=[FatigueDetectionTask(llm_provider)],
    llm_provider=llm_provider
)

tactical_adjustment_agent = Agent(
    role=AgentRole.CUSTOM,
    tasks=[TacticalAdjustmentTask(llm_provider)],
    llm_provider=llm_provider
)

orchestrator = Orchestrator(
    agents={
        "performance_analysis_agent": performance_analysis_agent,
        "fatigue_detection_agent": fatigue_detection_agent,
        "tactical_adjustment_agent": tactical_adjustment_agent
    },
    flows={
        "match_analysis_flow": Flow(
            agents=["performance_analysis_agent", "fatigue_detection_agent", "tactical_adjustment_agent"],
            context_pass=[True, True, True]
        )
    },
    mode=ExecutionMode.SEQUENTIAL
)

result_process = orchestrator.process_request(
    "Live match statistics and positional data",
    flow_name="match_analysis_flow"
)
result = asyncio.run(result_process)

print(result)
