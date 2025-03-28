import json
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains import SequentialChain, LLMChain
from google.oauth2 import service_account
from google.cloud import aiplatform

# Load zone data
with open("unified_data.json", "r") as f:
    zone_data = json.load(f)

# Initialize Vertex AI
credentials = service_account.Credentials.from_service_account_file("inspired-rock-450806-r5-d18f93b7f5ba.json")
aiplatform.init(project="inspired-rock-450806-r5", location="us-central1", credentials=credentials)
print("Vertex AI initialized successfully!")

# Initialize Gemini LLM
llm = VertexAI(model_name="gemini-1.5-flash", credentials=credentials, max_output_tokens=3000, temperature=0.7)

# Initialize Vertex AI Embeddings
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003", credentials=credentials)

# Define refined prompt templates for multi-stage chain
overview_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Task: Develop a detailed, actionable, and phased strategy plan based on the zone overview and idea, aligning with real-time initiatives in Andhra Pradesh, India.
Zone Data: {context}
Location Context: {location_context}

Instructions:
- Analyze land cover, environmental factors, risk metrics, socioeconomic status, and industries.
- Identify key opportunities (e.g., renewable energy potential) and challenges (e.g., flood risk).
- Highlight climate vulnerabilities (e.g., GHG emission hotspots, extreme weather risks).
- Keep it brief and data-driven.
Output Format:
- Headings: "Overview", "Opportunities", "Challenges", "Climate Vulnerabilities"
- Bullet points for clarity
"""
overview_prompt = PromptTemplate(input_variables=["context", "location_context"], template=overview_template)

strategy_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Zone Data: {context}
Idea: {idea}
Location Context: {location_context}
Zone Overview: {overview}

Objective:
- Create scalable, practical solutions to mitigate and adapt to climate change.The solutions should focus on reducing GHG emissions, enhancing resilience, and leveraging technology to bridge the gap between innovation and implementation.

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
Consider a dynamic timeline for implementing the strategy, breaking it down into different termed goals.
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
- Provide numerical estimates.
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
strategy_prompt = PromptTemplate(input_variables=["context", "idea", "location_context", "overview"], template=strategy_template)

# Q&A prompt for interactive feature
qna_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Given Strategy: {strategy}
User Question: {question}

Instructions:
- Provide a clear, concise, data-driven answer based on the strategy and zone data.
- If needed, infer additional context or fetch real-world examples (e.g., solar costs).
Output Format:
- Plain text response
"""
qna_prompt = PromptTemplate(input_variables=["strategy", "question"], template=qna_template)

# Modification prompt for updating the strategy
modify_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Zone Data: {context}

Location Context: {location_context}
Zone Overview: {overview}
Modification Request: {modification_request}
Instructions:
- Update the original strategy to incorporate the user's modification request while maintaining alignment with climate goals.
- Adjust relevant sections (e.g., implementation, budget, risk mitigation) to reflect the change.
- Ensure feasibility and consistency with the original data and constraints.
- Provide numerical estimates where applicable (e.g., revised costs, emission impacts).
Output Format:
- Same as the original strategy: headings, bullet points, financial tables, numerical estimates.
- Highlight modified sections with a note (e.g., "Modified: Added wind energy").
"""
modify_prompt = PromptTemplate(input_variables=["strategy", "modification_request", "context", "location_context", "overview"], template=modify_template)

# Populate vector store
documents = [Document(page_content=f"Zone {zone['zone']}: {json.dumps(zone)}", metadata={"zone_id": zone["zone"]}) for zone in zone_data]
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Function to generate dynamic location context
def generate_location_context(zone_id):
    zone = next((z for z in zone_data if z["zone"] == zone_id), None)
    if not zone:
        return f"No location data available for Zone {zone_id}."
    
    lat, lon = zone["location"]["latitude"], zone["location"]["longitude"]
    land_cover = zone["land_cover"]
    socio_economic = zone["socioeconomic_status"]
    industries = zone["industries_present"]
    env_factors = zone["environmental_factors"]
    
    is_rural = land_cover["built"] < 0.3 and (socio_economic["hospitals"] + socio_economic["banks"]) < 10
    area_type = "rural" if is_rural else "semi-urban"
    climate = "sunny climate with over 300 sunny days annually" if lat < 20 else "moderate climate"
    precipitation = "low precipitation (~600-800 mm/year)" if env_factors["precipitation"] < 0.03 else "moderate precipitation"
    key_industry = industries[0]["name"] if industries else "unknown locality"
    
    context = (
        f"The zone is located near {key_industry}, Andhra Pradesh, India (latitude {lat}, longitude {lon}), "
        f"a {area_type} area with {climate} and {precipitation}. "
        f"It features {land_cover['crops']*100:.1f}% crop coverage and {land_cover['built']*100:.1f}% built areas, "
        f"with {socio_economic['schools']} schools, {socio_economic['hospitals']} hospitals, and {socio_economic['banks']} banks."
    )
    return context

# Multi-stage chain for strategy generation
overview_chain = LLMChain(llm=llm, prompt=overview_prompt, output_key="overview")
strategy_chain = LLMChain(llm=llm, prompt=strategy_prompt, output_key="strategy")
full_chain = SequentialChain(
    chains=[overview_chain, strategy_chain],
    input_variables=["context", "idea", "location_context"],
    output_variables=["overview", "strategy"]
)

# Q&A chain for interactive queries
qna_chain = LLMChain(llm=llm, prompt=qna_prompt)

# Modification chain for updating strategies
modify_chain = LLMChain(llm=llm, prompt=modify_prompt, output_key="modified_strategy")

# Function to run the agent
def run_agent(zone_id, idea_text):
    query = f"Zone {zone_id}"
    retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 1}).invoke(query)
    context = retrieved_docs[0].page_content if retrieved_docs else f"Zone {zone_id}: No data available."
    
    location_context = generate_location_context(zone_id)
    
    # Run the multi-stage chain
    result = full_chain.invoke({
        "context": context,
        "idea": idea_text,
        "location_context": location_context
    })
    return result["strategy"]

# Function to modify the strategy
def modify_strategy(zone_id, original_strategy, modification_request):
    query = f"Zone {zone_id}"
    retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 1}).invoke(query)
    context = retrieved_docs[0].page_content if retrieved_docs else f"Zone {zone_id}: No data available."
    
    location_context = generate_location_context(zone_id)
    
    # Run the modification chain
    result = modify_chain.invoke({
        "strategy": original_strategy,
        "modification_request": modification_request,
        "context": context,
        "location_context": location_context
    })
    return result["modified_strategy"]

# Function to handle interactive session (Q&A and modifications)
def interactive_session(zone_id, strategy):
    print("\nInteractive Session: You can ask questions or request modifications to the strategy.")
    print("Commands:")
    print("- Type a question (e.g., 'Why solar?') to get an answer.")
    print("- Type 'modify: [request]' (e.g., 'modify: Reduce the budget by 20%') to update the strategy.")
    print("- Type 'exit' to stop.")
    
    current_strategy = strategy
    while True:
        user_input = input("\nYour input: ")
        if user_input.lower() == "exit":
            print("Exiting interactive session.")
            break
        elif user_input.lower().startswith("modify:"):
            modification_request = user_input[7:].strip()  # Extract request after "modify:"
            try:
                current_strategy = modify_strategy(zone_id, current_strategy, modification_request)
                print("\nModified Strategy for Zone 5:\n", current_strategy)
            except Exception as e:
                print(f"Error modifying strategy: {str(e)}. Please try again.")
        else:
            # Treat as a question
            try:
                answer = qna_chain.invoke({"strategy": current_strategy, "question": user_input})
                print("Answer:", answer["text"] if isinstance(answer, dict) else answer)
            except Exception as e:
                print(f"Error answering question: {str(e)}. Please try again.")
    return current_strategy

# Test the agent and start interactive session
zone_id = 11  # Testing with Zone 5
idea = "Give Pollution Mitigation strategy with respect to geographical and scio-economic conditions"
strategy = run_agent(zone_id, idea)
print("Original Strategy for Zone 5:\n", strategy)

# Start interactive session
final_strategy = interactive_session(zone_id, strategy)

# Validation
print("\nValidation Check for Original Strategy:")
if "GHG" in strategy and "solar" in strategy.lower() and "budget" in strategy.lower():
    print("- Climate focus (GHG reduction) and tech integration (solar) present.")
else:
    print("- Missing climate focus or tech integration.")
if "20%" in strategy or "tons CO2e" in strategy:
    print("- Numerical estimates present.")
else:
    print("- Missing numerical estimates.")

print("\nValidation Check for Final Strategy:")
if "GHG" in final_strategy and "solar" in final_strategy.lower() and "budget" in final_strategy.lower():
    print("- Climate focus (GHG reduction) and tech integration (solar) present.")
else:
    print("- Missing climate focus or tech integration.")
if "20%" in final_strategy or "tons CO2e" in final_strategy:
    print("- Numerical estimates present.")
else:
    print("- Missing numerical estimates.")