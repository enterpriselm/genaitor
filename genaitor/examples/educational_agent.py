import asyncio

from core import Orchestrator, Flow, ExecutionMode
from presets.agents import educational_agent

async def main():
    print("\nInitializing Educational Improvement System...")

    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={"education_agent": educational_agent},
        flows={
            "default_flow": Flow(agents=["education_agent"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    user_inputs = [
        "Student performance data: Grades in Math: B+, Science: A, English: C+. Suggest topics that need more focus.",
        "Student performance data: Struggled with physics and calculus in past semesters. Predict difficulties in upcoming chemistry and suggest teaching adjustments.",
        "Student has basic knowledge in Algebra. Suggest materials for improving calculus understanding.",
        "Student is learning Spanish. Based on assessments, suggest activities to improve vocabulary and grammar."
    ]
    
    # Process each input
    for user_input in user_inputs:
        print(f"\nUser Input: {user_input}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(user_input, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("education_agent")
                    if content and content.success:
                        print("\nSuggestions for Improvement:")
                        print("-" * 80)
                        
                        formatted_text = content.content.strip()
                        
                        formatted_text = formatted_text.replace("**", "")
                        
                        for line in formatted_text.split('\n'):
                            if line.strip():
                                print(line)
                            else:
                                print()
                    else:
                        print("Empty response received")
                else:
                    print(result["content"] or "Empty response")
            else:
                print(f"\nError: {result['error']}")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            break

if __name__ == "__main__":
    asyncio.run(main())
