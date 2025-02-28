from typing import Dict, List, Any, Optional
from .base import TaskResult, AgentRole
from .agent import Agent
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging  # Added logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"

class Flow:
    def __init__(self, agents: List[str], context_pass: List[bool], orchestrator_agent: str = '', validator_agent: str = ''):
        self.agents = agents  # List of agent names
        self.context_pass = context_pass  # List of booleans indicating if context should be passed
        self.orchestrator_agent = orchestrator_agent # Name of the agent to run the orchestrator between agents
        self.validator_agent = validator_agent # Name of the agent to check all the inputs and outputs of agents

class Orchestrator:
    def __init__(
        self,
        agents: Dict[str, Agent],
        flows: Dict[str, Flow],
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        max_workers: int = 5
    ):
        self.agents = agents
        self.flows = flows
        self.mode = mode
        self.max_workers = max_workers
        self.execution_history: List[Dict[str, Any]] = []

    async def _execute_agent(
        self,
        agent: Agent,
        request: Any,
        context: Optional[Dict[str, Any]]
    ) -> TaskResult:
        """Execute a single agent with context"""
        if agent.role == AgentRole.MAIN:
            request = f"[ROLE: MAIN] You are the primary agent overseeing this task. Coordinate the execution effectively, ensuring clarity, coherence, and completeness. Provide a structured response with a well-defined approach.\nRequest: {request}"

        elif agent.role == AgentRole.SPECIALIST:
            request = f"[ROLE: SPECIALIST] You are an expert in a specific domain. Provide deep technical insights, precise details, and advanced explanations relevant to the subject matter. Use domain-specific terminology where applicable.\nRequest: {request}"

        elif agent.role == AgentRole.VALIDATOR:
            request = f"[ROLE: VALIDATOR] Your task is to verify and validate the accuracy, consistency, and reliability of the given information. Identify inconsistencies, logical flaws, or areas needing further confirmation.\nRequest: {request}"

        elif agent.role == AgentRole.REFINER:
            request = f"[ROLE: REFINER] Improve the clarity, structure, and coherence of the given information. Optimize phrasing, remove redundancy, and enhance readability while preserving the original intent.\nRequest: {request}"

        elif agent.role == AgentRole.SUMMARIZER:
            request = f"[ROLE: SUMMARIZER] Extract the most important points from the given information and present a concise yet comprehensive summary. Ensure clarity, coherence, and factual accuracy.\nRequest: {request}"

        elif agent.role == AgentRole.SCIENTIST:
            request = f"[ROLE: SCIENTIST] Approach the task with a research-oriented mindset. Provide rigorous analysis, theoretical insights, and evidence-based reasoning. If applicable, cite relevant methodologies or scientific principles.\nRequest: {request}"

        elif agent.role == AgentRole.ENGINEER:
            request = f"[ROLE: ENGINEER] Focus on practical implementation and problem-solving. Provide clear steps, technical specifications, and actionable recommendations for real-world applications.\nRequest: {request}"

        elif agent.role == AgentRole.ARCHITECT:
            request = f"[ROLE: ARCHITECT] Design and structure the solution effectively. Focus on system-level thinking, high-level planning, and optimal organization of components and dependencies.\nRequest: {request}"

        elif agent.role == AgentRole.CUSTOM:
            request = f"[ROLE: CUSTOM] You have a unique role defined by the user. Adapt your response to the given task, ensuring it aligns with the user's specific requirements and objectives.\nRequest: {request}"        
        
        try:
            return await agent.process_request(request, context)
        except Exception as e:
            logging.error(f"Error executing agent '{agent.role}': {str(e)}")  # Log the error
            return TaskResult(success=False, content=None, error=f"Error executing agent '{agent.role}': {str(e)}")

    async def process_request(self, request: str, flow_name: str) -> Dict[str, Any]:
        """Process a request using the specified flow"""
        try:
            flow = self.flows[flow_name]
            results = {}
            context = {}
            print(f"Agents: {flow.agents}, Context Pass: {flow.context_pass}")

            for i, agent_name in enumerate(flow.agents):
                agent = self.agents[agent_name]
                should_pass_context = flow.context_pass[i]
                result = await self._execute_agent(agent, request, context if should_pass_context else None)
                results[agent_name] = result
                if result.success:
                    context[agent_name] = result.content  # Update context with the agent's response

            return {
                "success": True,
                "content": results
            }
        except KeyError as e:
            logging.error(f"Flow '{flow_name}' not found: {str(e)}")  # Log specific error
            return {
                "success": False,
                "content": None,
                "error": f"Flow '{flow_name}' not found."
            }
        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")  # Log the error
            return {
                "success": False,
                "content": None,
                "error": f"An error occurred while processing the request: {str(e)}"
            }

    async def _process_sequential(
        self,
        request: Any,
        flow_name: str
    ) -> Dict[str, Any]:
        """Process request sequentially through agents allowing memory"""
        try:
            flow = self.flows[flow_name]
            results = {}
            context = {}
            history = ""

            for i, agent_name in enumerate(flow.agents):
                agent = self.agents[agent_name]
                should_pass_context = flow.context_pass[i]
                request_with_history = f"{history}\n{request}"
                
                result = await self._execute_agent(agent, request_with_history, context if should_pass_context else None)
                results[agent_name] = result
                if result.success:
                    context[agent_name] = result.content
                    history += f"\n{result.content}"

            return {"success": True, "content": results}
        except Exception as e:
            logging.error(f"Error processing sequential request: {str(e)}")
            return {"success": False, "content": None, "error": f"An error occurred while processing the sequential request: {str(e)}"}

    async def _process_parallel(
        self,
        request: Any,
        flow_name: str
    ) -> Dict[str, Any]:
        """Process request in parallel through agents"""
        try:
            flow = self.flows[flow_name]
            tasks = []
            results = {}
            context = {}

            for i, agent_name in enumerate(flow.agents):
                agent = self.agents[agent_name]
                should_pass_context = flow.context_pass[i]
                tasks.append(self._execute_agent(agent, request, context if should_pass_context else None))
            
            responses = await asyncio.gather(*tasks)
            for agent_name, response in zip(flow.agents, responses):
                results[agent_name] = response
                if response.success:
                    context[agent_name] = response.content

            return {"success": True, "content": results}
        except Exception as e:
            logging.error(f"Error processing parallel request: {str(e)}")
            return {"success": False, "content": None, "error": f"An error occurred while processing the parallel request: {str(e)}"}

    async def _process_adaptative(
        self,
        request: Any,
        flow_name: str
    ) -> Dict[str, Any]:
        """Process request adaptively, selecting agents dynamically based on need"""
        try:
            flow = self.flows[flow_name]
            orchestrator = self.agents[flow.orchestrator_agent]
            validator = self.agents[flow.validator_agent]
            available_agents = {name: self.agents[name] for name in flow.agents}
            agents_context = [
                ', '.join(task.description for task in agent.tasks)
                for agent in (self.agents[name] for name in flow.agents)]
            
            context = {}
            results = {}
            current_input = request

            while True:
                orchestrator_prompt = (
                    f"User Input: {current_input}\n"
                    f"Available Agents: {'; '.join(available_agents.keys())}\n"
                    f"Context of Agents: {'; '.join(agents_context)}\n"
                    f"Context So Far: {context}\n"
                    f"What is the best agent to handle this next?"
                )
                
                orchestrator_response = await self._execute_agent(orchestrator, orchestrator_prompt, context)
                chosen_agent_name = orchestrator_response.content.strip().split('\n')[-1]
                
                if chosen_agent_name not in available_agents:
                    return {"success": False, "content": None, "error": f"Invalid agent selection: {chosen_agent_name}"}
                
                chosen_agent = available_agents[chosen_agent_name]
                print(f"User: {request}\n")
                print(f"Orchestrator decision: Passing input to {chosen_agent_name}")
                
                agent_response = await self._execute_agent(chosen_agent, current_input, context)
                results[chosen_agent_name] = agent_response
                context[chosen_agent_name] = agent_response.content
                
                print(f"{chosen_agent_name} response:\n {agent_response.content}")
                
                # Validator checks if the response is sufficient
                validation_prompt = (
                    f"User Input: {request}\n"
                    f"Agent Response: {agent_response.content}\n"
                    f"Available Agents: {'; '.join(available_agents.keys())}\n"
                    f"Context of Agents: {'; '.join(agents_context)}\n"
                    f"Does this response fully answer the user's question? If not, which agent should be consulted next?"
                    "If your decision is to give the user the aswer, at the end of your explanaton and answer, say: My Final decision is that is complete."
                    "In other hand, if you want to check other agent, say, at the end of your explanation and answer: My Final decision is to consult the {agent_name}."
                    "Where, the agent_name should be the name of the agent on the available agents list."
                )
                validator_response = await self._execute_agent(validator, validation_prompt, context)
                validation_decision = validator_response.content.split('ecision: ')[-1]
                
                print(f"Validator decision: {validation_decision}")
                
                if "complete" in validation_decision.lower() or "is complete" in validator_response.content.lower():
                    break  # Exit loop if the validator confirms completion
                elif "consult the" in validation_decision.lower() or "assing input to " in validation_decision.lower():
                    current_input = agent_response.content  # Pass last response as new input
                else:
                    return {"success": False, "content": None, "error": f"Invalid validation decision: {validation_decision}"}
                
            return {"success": True, "content": results}
        
        except Exception as e:
            logging.error(f"Error processing adaptive request: {str(e)}")
            return {"success": False, "content": None, "error": f"An error occurred while processing the adaptive request: {str(e)}"}

    def _should_continue_processing(self, content):
        return "requires further processing" in content.lower()

    def _combine_results(self, results: List[TaskResult]) -> TaskResult:
        """Combine multiple results into a single result"""
        successful_results = [r for r in results if r.success]
        if not successful_results:
            return TaskResult(success=False, content=None, error="All agents failed")

        combined_content = {
            f"agent_{i}": result.content 
            for i, result in enumerate(successful_results)
        }
        return TaskResult(success=True, content=combined_content)

    def _format_response(
        self,
        result: TaskResult,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format the final response"""
        if result is None:
            return {
                "success": False,
                "content": {},
                "error": "No result produced",
                "context": context
            }

        content = result.content
        if isinstance(content, str):
            content = {"main": content}
        
        return {
            "success": result.success,
            "content": content,
            "error": result.error,
            "context": context
        }

    def _update_execution_history(
        self,
        request: Any,
        result: TaskResult,
        sequence: List[str]
    ) -> None:
        """Update execution history"""
        self.execution_history.append({
            "request": request,
            "result": result,
            "agent_sequence": sequence,
            "mode": self.mode.value
        }) 
