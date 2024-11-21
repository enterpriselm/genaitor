from genaitor.config import config
from genaitor.utils.agents import Agent, Orchestrator, Task

agents = {
    'local_expert': Agent(
        role='Local Expert at this city',
        system_message=(
            "You're a knowledgeable local guide with extensive information about the city, it's attractions and customs"
            "Provide the BEST insights about the inputed city"
        ),
        temperature=0.8,
        max_tokens=2000,
        max_iterations=2
    ),
    'city_selection_agent': Agent(
        role='City Selection Expert',
        system_message=(
            "You are an AI agent expert in analyzing travel data to pick ideal destinations"
            "Given the user's query, you will select the best city based on weather, season and prices."
        ),
        temperature=0.8,
        max_tokens=2000,
        max_iterations=2
    ),
    'travel_concierge': Agent(
        role='Travel Concierge',
        system_message=(
            "You are an AI agent Specialist in travel planning and logistics with decades of experience"
            "You will create the most amazing travel itineraries with budget and packing suggestions for the city."
        ),
        temperature=0.8,
        max_tokens=2000,
        max_iterations=2
    )
}

class TravelTasks():

    def location_task(self, agent, from_city, destination_city, date_from, date_to, budget, number_of_people):
        return Task(
            description=f"""
            This task involves a comprehensive data collection process to provide the traveler with essential information about their destination. It includes researching and compiling details on various accommodations, ranging from budget-friendly hostels to luxury hotels, as well as estimating the cost of living in the area. The task also covers transportation options, visa requirements, and any travel advisories that may be relevant.
            consider also the weather conditions forcast on the travel dates. and all the events that may be relevant to the traveler during the trip period.
            
            Traveling from : {from_city}
            Destination city : {destination_city}
            Arrival Date : {date_from}
            Departure Date : {date_to}
            Budget: {budget} 
            Number of People: {number_of_people}
            """,
            expected_output=f"""
            In markdown format : A detailed markdown report that includes a curated list of recommended places to stay, a breakdown of daily living expenses, and practical travel tips to ensure a smooth journey.
            """,
            agent=agent,
            output_file='city_report.md',
        )
    
    def guide_task(self, agent, destination_city, interests, date_from, date_to, budget, number_of_people):    
        return Task(
            description=f"""
            Tailored to the traveler's personal {interests}, this task focuses on creating an engaging and informative guide to the city's attractions. It involves identifying cultural landmarks, historical spots, entertainment venues, dining experiences, and outdoor activities that align with the user's preferences such {interests}. The guide also highlights seasonal events and festivals that might be of interest during the traveler's visit.
            Destination city : {destination_city}
            interests : {interests}
            Arrival Date : {date_from}
            Departure Date : {date_to}
            Budget: {budget} 
            Number of People: {number_of_people}
            
            """,
            expected_output=f"""
            An interactive markdown report that presents a personalized itinerary of activities and attractions, complete with descriptions, locations, and any necessary reservations or tickets. Also suggests bands of music for them.
            """,
    
            agent=agent,
            output_file='guide_report.md',
        )

    
    def planner_task(self, context, agent, destination_city, interests, date_from, date_to, budget, number_of_people):
        return Task(
            description=f"""
            This task synthesizes all collected information into a detaileds introduction to the city (description of city and presentation, in 3 pragraphes) cohesive and practical travel plan. and takes into account the traveler's schedule, preferences, and budget to draft a day-by-day itinerary. The planner also provides insights into the city's layout and transportation system to facilitate easy navigation.
            Destination city : {destination_city}
            interests : {interests}
            Arrival Date : {date_from}
            Departure Date : {date_to}
            Budget: {budget} 
            Number of People: {number_of_people}
            
            """,
            expected_output="""
            A rich markdown document with emojis on each title and subtitle, that :
            In markdown format : 
            # Welcome to {destination_city} :
            A 4 paragraphes markdown formated including :
            - a curated articles of presentation of the city, 
            - a breakdown of daily living expenses, and spots to visit.
            # Here's your Travel Plan to {destination_city} :
            Outlines a daily detailed travel plan list with time allocations and details for each activity, along with an overview of the city's highlights based on the guide's recommendations, and also bands of music.
            """,
            context=context,
            agent=agent,
            output_file='travel_plan.md',
            )


origin = "New York"
destination = "Paris"
start_date = "21-11-2024"
end_date = "12-12-2024"
interests = "Soccer, wine and beaches"
budget = "10k euros"
number_of_people = 3
travel_tasks = TravelTasks()

tasks = [
    travel_tasks.location_task(
        agent=agents['local_expert'],
        from_city=origin,
        destination_city=destination,
        date_from=start_date,
        date_to=end_date,
        budget=budget,
        number_of_people=number_of_people
    ),
    travel_tasks.guide_task(
        agent=agents['city_selection_agent'],
        destination_city=destination,
        interests=interests,
        date_from=start_date,
        date_to=end_date,
        budget=budget,
        number_of_people=number_of_people
    ),
    travel_tasks.planner_task(
        context=[],
        agent=agents['travel_concierge'],
        destination_city=destination,
        interests=interests,
        date_from=start_date,
        date_to=end_date,
        budget=budget,
        number_of_people=number_of_people
    )
]

pm_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

user_query = 'Plan my trip from New York to Paris with details on attractions, costs, and logistics.'
result = pm_orchestrator.kickoff(user_query)

print('Genaitor Results: ')
for task, result_item in zip(tasks, result["output"]):
    print(f"Result: {result_item}\n")  