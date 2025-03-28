import json
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from google.oauth2 import service_account
from google.cloud import aiplatform

from langgraph.graph import StateGraph, MessageGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List, Dict, TypedDict
from langchain_core.runnables import chain
import functools

# Load zone data (Make sure unified_data.json exists in the same directory)
with open("unified_data.json", "r") as f:
    zone_data = json.load(f)

# Initialize Vertex AI
credentials = service_account.Credentials.from_service_account_file(
    "vertical-setup-450217-n2-8904fd8695bd.json"  # Replace with your credentials file
)
aiplatform.init(project="strategy-agent", location="us-central1", credentials=credentials)
print("Vertex AI initialized successfully!")

# Initialize Gemini LLM
llm = VertexAI(
    model_name="gemini-1.5-flash", credentials=credentials, max_output_tokens=3000, temperature=0.7
)

# Initialize Vertex AI Embeddings
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003", credentials=credentials)

# Define base prompt template with a placeholder for dynamic location context
template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.

Zone Data: {context}
Idea: {idea}
Location Context: {location_context}

Objective:
 creating innovative, scalable, and inclusive solutions to mitigate and adapt to climate change. The solutions should focus on reducing GHG emissions, enhancing resilience, and leveraging technology to bridge the gap between innovation and implementation.

Context:
The given data provides a comprehensive overview of the land cover, environmental factors, risk metrics, trends, socioeconomic status, industries, and research insights of a specific zone. Your strategy should align with real-world constraints, regulations, and feasibility while maximizing sustainability, economic growth, and social well-being.

Instructions for Strategy Development:
1. Overview of the Zone
Provide a brief analysis of the zone’s characteristics based on the given data.
Identify key opportunities and challenges based on land cover, risk metrics, and environmental factors.
2. Strategy Plan & Implementation
Develop a well-structured, phased strategy addressing the identified challenges.
Ensure the plan is realistic, feasible, and backed  with dynamic timeline based on strategy by real-world constraints.

Include specific projects, initiatives, or interventions necessary for improving the zone.
Provide numerical estimations where applicable (e.g., expected impact in GDP growth, employment generation).
Consider a dynamic timeline for implementing the strategy, breaking it down into short-term, mid-term, and long-term goals.
3. Budget & Resource Allocation
Provide a detailed financial breakdown, including:

Infrastructure Costs (e.g., roads, housing, public utilities)
Environmental Restoration (e.g., afforestation, water conservation)
Public Services (e.g., healthcare, education, disaster preparedness)
Technology & Innovation Investments
Other Necessary Expenditures
Additionally, provide:

Projected ROI (Return on Investment) or cost-benefit analysis for major investments.
Manpower Requirements including:
Government bodies involved.
Engineers, urban planners, environmentalists, etc.
Local workforce and skilled labor.
4. Legal & Policy Considerations
Identify laws, regulations, and policies applicable to the location.
Ensure compliance with environmental, urban planning, and labor laws specific to the region.
Highlight any permits or approvals required for implementation.
5. Risk Mitigation & Contingency Plan
Identify potential risks (e.g., climate risks, financial risks, social resistance).
Propose mitigation strategies with numerical justifications where possible (e.g., estimated financial losses from a past disaster and the cost of preventive measures).
If applicable, reference historical cases where similar risks impacted projects and how those issues were addressed.
Include a disaster management plan based on risk metrics.
6. Monitoring & Evaluation
Define KPIs (Key Performance Indicators) to measure success.
Provide a timeline for periodic assessments to track progress.
Suggest a feedback mechanism for continuous improvement.

Output Format:
Generate the response in a structured format with:
✅ Headings & bullet points for clarity.
✅ Financial tables where applicable.
✅ Numerical estimates to support key recommendations.
✅ Relevant case studies or historical references where applicable.

Constraints:
❌ Avoid unrealistic assumptions—all strategies must be based on real-world feasibility.
✅ Ensure practicality and effectiveness with data-backed recommendations.
✅ Maintain alignment with the region’s economic, environmental, and social goals.
"""
prompt = PromptTemplate(input_variables=["context", "idea", "location_context"], template=template)

# Populate vector store
documents = [
    Document(page_content=f"Zone {zone['zone']}: {json.dumps(zone)}", metadata={"zone_id": zone["zone"]})
    for zone in zone_data
]
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


# Function to generate dynamic location context based on zone data
def generate_location_context(zone_id):
    # Find the zone data for the given zone_id
    zone = next((z for z in zone_data if z["zone"] == zone_id), None)
    if not zone:
        return f"No location data available for Zone {zone_id}."

    # Extract key details
    lat, lon = zone["location"]["latitude"], zone["location"]["longitude"]
    land_cover = zone["land_cover"]
    socio_economic = zone["socioeconomic_status"]
    industries = zone["industries_present"]
    env_factors = zone["environmental_factors"]

    # Infer characteristics
    is_rural = land_cover["built"] < 0.3 and (socio_economic["hospitals"] + socio_economic["banks"]) < 10
    area_type = "rural" if is_rural else "semi-urban"
    climate = "sunny climate with over 300 sunny days annually" if lat < 20 else "moderate climate"
    precipitation = (
        "low precipitation (~600-800 mm/year)" if env_factors["precipitation"] < 0.03 else "moderate precipitation"
    )
    key_industry = industries[0]["name"] if industries else "unknown locality"

    # Construct location context
    context = (
        f"The zone is located near {key_industry}, Andhra Pradesh, India (latitude {lat}, longitude {lon}), "
        f"a {area_type} area with {climate} and {precipitation}. "
        f"It features {land_cover['crops']*100:.1f}% crop coverage and {land_cover['built']*100:.1f}% built areas, "
        f"with {socio_economic['schools']} schools, {socio_economic['hospitals']} hospitals, and {socio_economic['banks']} banks."
    )
    return context


# Define the state
class GraphState(TypedDict):
    messages: List[BaseMessage]
    zone_id: int
    idea: str


# Define nodes
def generate_strategy(state: GraphState) -> List[BaseMessage]:
    """Generates the strategy plan based on the current state."""
    zone_id = state["zone_id"]
    idea_text = state["idea"]
    messages = state["messages"]

    # Extract the latest user query
    latest_query = messages[-1].content if messages else ""

    query = f"Zone {zone_id}"
    retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 1}).invoke(query)
    context = retrieved_docs[0].page_content if retrieved_docs else f"Zone {zone_id}: No data available."

    # Generate dynamic location context
    location_context = generate_location_context(zone_id)

    # Format the prompt with context, idea, and location_context
    filled_prompt = prompt.format(context=context, idea=idea_text, location_context=location_context)

    # Include the latest query in the prompt to guide the strategy generation
    final_prompt = filled_prompt + f"\nUser Query: {latest_query}\nStrategy:"

    # Directly invoke the LLM
    strategy = llm.invoke(final_prompt)
    return [AIMessage(content=strategy)]


def user_node(state: GraphState, user_message: str) -> List[BaseMessage]:
    """Adds the user's message to the message list."""
    return [HumanMessage(content=user_message)]


# Define edges
def should_continue(state: GraphState):
    """Determines whether to continue generating the strategy or respond to the user."""
    messages = state["messages"]
    # Check if the latest message is from the user
    if messages and isinstance(messages[-1], HumanMessage):
        return "user_query"
    else:
        return "generate"


# def decide_to_generate(messages: List[BaseMessage]):
#     """
#     Determines whether to generate a response or respond to the user.
#     """
#     most_recent_message = messages[-1]
#     if "continue" in most_recent_message.content.lower():
#         return "generate"
#     elif "exit" in most_recent_message.content.lower():
#         return "end"
#     else:
#         return "user_query"  # Default to sending back to the user


# Build the graph
builder = StateGraph(GraphState)

builder.add_node("user_query", user_node)
builder.add_node("generate", generate_strategy)
builder.set_entry_point("user_query")  # Start with the user query

builder.add_conditional_edges(
    "user_query",
    should_continue,
    {
        "generate": "generate",
        "user_query": "user_query",
    },
)
builder.add_conditional_edges(
    "generate",
    should_continue,
    {
        "generate": "generate",
        "user_query": "user_query",
    },
)

graph = builder.compile()


# Function to run the graph
def run_langgraph(zone_id: int, idea: str, first_query: str):
    """Runs the LangGraph for initial strategy generation and handles follow-up queries."""
    initial_state = {
        "messages": [HumanMessage(content=first_query)],
        "zone_id": zone_id,
        "idea": idea,
    }
    results = graph.stream(initial_state)
    # Print the stream
    all_messages: List[BaseMessage] = []
    response = ""
    for output in results:
        for key, value in output.items():
            if key == "generate":
                response += str(value[0].content)
            if key == "user_query":
                continue

    # Print the final response
    return response


# Example usage:
zone_id = 14  # Example: Zone 14
idea = "it park establishment"  # Example: Build an it park

first_query = "Develop a high-level strategy for this zone."
strategy = run_langgraph(zone_id, idea, first_query)
print("Initial Strategy:\n", strategy)

# Simulate a follow-up question
follow_up_query = "Give me some legal consideration on this zone."
strategy = run_langgraph(zone_id, idea, follow_up_query)
print("\nResponse to Follow-up Query:\n", strategy)