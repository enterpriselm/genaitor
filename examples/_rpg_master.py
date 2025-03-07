import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
from dotenv import load_dotenv
load_dotenv('.env')

# Define custom tasks for RPG strategy, combat, and world-building

class CharacterActionStrategyTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the following RPG character and their abilities, provide the best strategy for action based on their strengths and weaknesses:
        
        Input: {input_data}
        
        Please consider the character's stats, abilities, weaknesses, and the current scenario.
        Suggest actions that maximize strengths and minimize weaknesses.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "character_action_strategy"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

class CombatTacticsTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the combat scenario, provide creative ways to use the character's abilities to gain an advantage over the enemy:
        
        Input: {input_data}
        
        Please suggest ways to manipulate the environment, exploit enemy weaknesses, or use abilities creatively to gain the upper hand in combat.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "combat_tactics"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

class DynamicWorldCreationTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the following desired RPG game style, design a dynamic and immersive game world that reacts to player actions and changes the narrative.
        
        Input: {input_data}
        
        Consider NPCs, story arcs, environmental interactions, music, challenges, monsters, and dynamic quests that influence the environment and the story.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "dynamic_world_creation"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing RPG Strategy and World-Building Demo...")
    test_keys = [os.getenv('API_KEY_1'), os.getenv('API_KEY_2')]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)
    
    # Initialize tasks
    character_action_strategy_task = CharacterActionStrategyTask(
        description="Analyze RPG character's strengths and weaknesses to suggest an action strategy",
        goal="Provide best strategy for action based on character's abilities and weaknesses",
        output_format="Suggested strategy for action",
        llm_provider=provider
    )

    combat_tactics_task = CombatTacticsTask(
        description="Analyze combat scenario and suggest tactical advantages",
        goal="Suggest creative combat tactics using character's abilities",
        output_format="Suggested tactics for combat",
        llm_provider=provider
    )

    dynamic_world_creation_task = DynamicWorldCreationTask(
        description="Design a dynamic and interactive RPG world",
        goal="Create a dynamic RPG world based on player style and actions",
        output_format="Detailed game world design",
        llm_provider=provider
    )

    # Initialize agents
    strategy_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[character_action_strategy_task],
        llm_provider=provider
    )
    
    combat_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[combat_tactics_task],
        llm_provider=provider
    )

    world_creation_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[dynamic_world_creation_task],
        llm_provider=provider
    )

    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={
            "strategy": strategy_agent,
            "combat": combat_agent,
            "world_creation": world_creation_agent
        },
        flows={
            "default_flow": Flow(agents=["strategy", "combat", "world_creation"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    # Example inputs
    user_inputs = [
        "Character: A warrior with high strength and moderate intelligence, weak against magic. The enemy is a spellcaster. What is the best action strategy?",
        "Combat scenario: A warrior fights against a powerful wizard in an open field. The wizard uses fire spells. How can the warrior use their abilities to gain an advantage?",
        "Desired game world: A dark fantasy world with an ever-changing environment, quests that influence the plot, and NPCs with their own agendas. How should I set this up?"
    ]
    
    # Process each input
    for user_input in user_inputs:
        print(f"\nUser Input: {user_input}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(user_input, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    for agent_name, content in result["content"].items():
                        if content and content.success:
                            print(f"\nResponse from {agent_name}:")
                            print("-" * 80)
                            formatted_text = content.content.strip()
                            for line in formatted_text.split('\n'):
                                if line.strip():
                                    print(line)
                                else:
                                    print()
                        else:
                            print(f"Empty response from {agent_name}")
                else:
                    print(result["content"] or "Empty response")
            else:
                print(f"\nError: {result['error']}")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            break

if __name__ == "__main__":
    asyncio.run(main())
