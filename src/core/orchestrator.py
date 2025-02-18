from typing import Dict, List, Any, Optional
from .base import TaskResult
from .agent import Agent
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"

class Orchestrator:
    def __init__(
        self,
        agents: Dict[str, Agent],
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        max_workers: int = 5
    ):
        self.agents = agents
        self.mode = mode
        self.max_workers = max_workers
        self.execution_history: List[Dict[str, Any]] = []

    async def process_request(self, request: str) -> Dict[str, Any]:
        """Process a request using the configured agents"""
        try:
            if self.mode == ExecutionMode.SEQUENTIAL:
                results = {}
                for name, agent in self.agents.items():
                    result = agent.process_request(request)
                    results[name] = result
                return {
                    "success": True,
                    "content": results
                }
            elif self.mode == ExecutionMode.PARALLEL:
                return await self._process_parallel(request)
            elif self.mode == ExecutionMode.ADAPTIVE:
                return await self._process_adaptive(request)
            else:
                return await self._process_sequential(request)
        except Exception as e:
            return {
                "success": False,
                "content": None,
                "error": str(e)
            }

    async def _process_sequential(
        self,
        request: Any,
        agent_sequence: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Process request sequentially through agents"""
        result = None
        context = {}
        sequence = agent_sequence or list(self.agents.keys())

        for agent_name in sequence:
            agent = self.agents[agent_name]
            result = await self._execute_agent(agent, request, contexty)
            if not result.success:
                break
            context.update({agent_name: result.content})

        self._update_execution_history(request, result, sequence)
        return self._format_response(result, context)

    async def _process_parallel(
        self,
        request: Any
    ) -> Dict[str, Any]:
        """Process request in parallel through agents"""
        sequence = list(self.agents.keys())
        tasks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loop = asyncio.get_event_loop()
            for agent_name in sequence:
                agent = self.agents[agent_name]
                task = loop.run_in_executor(
                    executor,
                    agent.process_request,
                    request
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks)
        context = {
            name: result.content 
            for name, result in zip(sequence, results)
            if result.success
        }

        final_result = self._combine_results(results)
        self._update_execution_history(request, final_result, sequence)
        return self._format_response(final_result, context)

    async def _process_adaptive(
        self,
        request: Any
    ) -> Dict[str, Any]:
        """Adaptively choose between sequential and parallel processing"""
        # Implement adaptive logic based on request complexity,
        # agent dependencies, and system load
        pass

    async def _execute_agent(
        self,
        agent: Agent,
        request: Any,
        context: Dict[str, Any]
    ) -> TaskResult:
        """Execute a single agent with context"""
        try:
            return agent.process_request(request)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

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

        # Se o conteúdo for string, converte para dict
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