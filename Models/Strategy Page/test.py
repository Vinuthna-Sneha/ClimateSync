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

# Initialize Gemini LLM with higher temperature for flexibility
llm = VertexAI(model_name="gemini-1.5-flash", credentials=credentials, max_output_tokens=3000, temperature=1.0)

# Initialize Vertex AI Embeddings
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003", credentials=credentials)

# Define refined prompt templates
overview_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Task: Provide a brief overview based on the zone data and location context, focusing on factors relevant to the user's idea.
Zone Data: {context}
Location Context: {location_context}

Instructions:
- Analyze land cover, environmental factors, risk metrics, socioeconomic status, and industries.
- Identify key opportunities and challenges relevant to the user's intent.
- Highlight vulnerabilities or strengths tied to the zone’s characteristics.
- Keep it brief and data-driven.
Output Format:
- Headings: "Overview", "Opportunities", "Challenges", "Vulnerabilities"
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
- Develop a strategy plan that directly addresses the user-provided idea: '{idea}'.
- Incorporate scalable, practical solutions tailored to the idea, leveraging zone-specific data.
- If applicable, align with broader goals like climate resilience, economic growth, or social well-being, but only as secondary considerations unless explicitly mentioned in the idea.

Instructions for Strategy Development:
1. Overview of the Zone
   - Provide a brief analysis of the zone’s characteristics relevant to the idea: '{idea}'.
   - Identify key opportunities and challenges tied to the idea based on land cover, risk metrics, and environmental factors.
2. Strategy Plan & Implementation
   - Develop a phased strategy directly addressing the idea.
   - Include specific projects or interventions tailored to the idea (e.g., pollution control measures if the idea is pollution mitigation).
   - Provide a dynamic timeline with short-, medium-, and long-term goals.
   - Include numerical estimations (e.g., pollution reduction in tons, cost savings).
3. Budget & Resource Allocation
   - Provide a financial breakdown specific to the idea (e.g., costs for air quality monitoring).
   - Include projected ROI or cost-benefit analysis.
   - Estimate manpower needs tied to the idea.
4. Legal & Policy Considerations
   - Identify regulations relevant to the idea (e.g., pollution standards in Andhra Pradesh).
5. Risk Mitigation & Contingency Plan
   - Address risks specific to the idea (e.g., resistance to pollution regulations).
   - Propose mitigation strategies with numerical justifications.
6. Monitoring & Evaluation
   - Define KPIs tied to the idea (e.g., reduction in PM2.5 levels).
   - Suggest a timeline and feedback mechanism.

Output Format:
- Use headings and bullet points.
- Include financial tables and numerical estimates where applicable.

Constraints:
- Base the strategy on the idea '{idea}' and zone data, avoiding unrelated assumptions.
- Ensure practicality and alignment with the region’s context.
"""
strategy_prompt = PromptTemplate(input_variables=["context", "idea", "location_context", "overview"], template=strategy_template)

qna_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Given Strategy: {strategy}
User Question: {question}

Instructions:
- Provide a clear, concise, data-driven answer based on the strategy and zone data.
- If needed, infer additional context or fetch real-world examples.
Output Format:
- Plain text response
"""
qna_prompt = PromptTemplate(input_variables=["strategy", "question"], template=qna_template)

modify_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Zone Data: {context}
Location Context: {location_context}
Original Strategy: {strategy}
Modification Request: {modification_request}
Zone Overview: {overview}

Instructions:
- Update the original strategy to incorporate the user's modification request while maintaining alignment with the idea.
- Adjust relevant sections (e.g., implementation, budget, risk mitigation) to reflect the change.
- Ensure feasibility and consistency with the original data and constraints.
- Provide numerical estimates where applicable (e.g., revised costs, impacts).
Output Format:
- Same as the original strategy: headings, bullet points, financial tables, numerical estimates.
- Highlight modified sections with a note (e.g., "Modified: Adjusted budget").
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

# Preprocess the idea to make it more actionable
def preprocess_idea(idea):
    idea_lower = idea.lower()
    if "pollution mitigation" in idea_lower:
        return (
            "Develop a strategy to mitigate pollution, focusing on air, water, and soil quality improvements "
            "based on geographical and socioeconomic conditions."
        )
    elif "economic development" in idea_lower:
        return (
            "Develop a strategy to boost economic development, leveraging local industries and resources "
            "based on geographical and socioeconomic conditions."
        )
    elif "agricultural productivity" in idea_lower:
        return (
            "Develop a strategy to enhance agricultural productivity, focusing on crop yield and sustainability "
            "based on geographical and socioeconomic conditions."
        )
    return idea  # Default to original if no specific preprocessing

# Function to run the agent
def run_agent(zone_id, idea_text):
    query = f"Zone {zone_id}"
    retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 1}).invoke(query)
    context = retrieved_docs[0].page_content if retrieved_docs else f"Zone {zone_id}: No data available."
    
    location_context = generate_location_context(zone_id)
    processed_idea = preprocess_idea(idea_text)
    
    result = full_chain.invoke({
        "context": context,
        "idea": processed_idea,
        "location_context": location_context
    })
    return result["strategy"]

# Function to modify the strategy
def modify_strategy(zone_id, original_strategy, modification_request):
    query = f"Zone {zone_id}"
    retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 1}).invoke(query)
    context = retrieved_docs[0].page_content if retrieved_docs else f"Zone {zone_id}: No data available."
    
    location_context = generate_location_context(zone_id)
    overview = full_chain.invoke({
        "context": context,
        "idea": "Overview generation",
        "location_context": location_context
    })["overview"]
    
    result = modify_chain.invoke({
        "strategy": original_strategy,
        "modification_request": modification_request,
        "context": context,
        "location_context": location_context,
        "overview": overview
    })
    return result["modified_strategy"]

# Function to validate strategy alignment with idea
def validate_strategy(strategy, idea):
    idea_keywords = set(idea.lower().split())
    strategy_lower = strategy.lower()
    matched_keywords = [kw for kw in idea_keywords if kw in strategy_lower]
    match_ratio = len(matched_keywords) / len(idea_keywords)
    if match_ratio < 0.5:
        print(f"Warning: Strategy may not fully address the idea '{idea}'. Matched keywords: {matched_keywords}")
    else:
        print(f"Strategy aligns with idea '{idea}'. Matched keywords: {matched_keywords}")
    return match_ratio >= 0.5

# Function to handle interactive session (Q&A and modifications)
def interactive_session(zone_id, strategy):
    print("\nInteractive Session: You can ask questions or request modifications to the strategy.")
    print("Commands:")
    print("- Type a question (e.g., 'Why focus on air pollution?') to get an answer.")
    print("- Type 'modify: [request]' (e.g., 'modify: Reduce budget by 20%') to update the strategy.")
    print("- Type 'exit' to stop.")
    
    current_strategy = strategy
    while True:
        user_input = input("\nYour input: ")
        if user_input.lower() == "exit":
            print("Exiting interactive session.")
            break
        elif user_input.lower().startswith("modify:"):
            modification_request = user_input[7:].strip()
            try:
                current_strategy = modify_strategy(zone_id, current_strategy, modification_request)
                print("\nModified Strategy for Zone 11:\n", current_strategy)
            except Exception as e:
                print(f"Error modifying strategy: {str(e)}. Please try again.")
        else:
            try:
                answer = qna_chain.invoke({"strategy": current_strategy, "question": user_input})
                print("Answer:", answer["text"] if isinstance(answer, dict) else answer)
            except Exception as e:
                print(f"Error answering question: {str(e)}. Please try again.")
    return current_strategy

# Test the agent and start interactive session
zone_id = 11
idea = "improve health infrastructure "
strategy = run_agent(zone_id, idea)
print("Original Strategy for Zone 11:\n", strategy)
validate_strategy(strategy, idea)

# Start interactive session
final_strategy = interactive_session(zone_id, strategy)

# Validation
print("\nValidation Check for Original Strategy:")
if validate_strategy(strategy, idea):
    print("- Strategy aligns with the provided idea.")
else:
    print("- Strategy may not fully address the idea.")
if "budget" in strategy.lower() and any(char.isdigit() for char in strategy):
    print("- Financial and numerical estimates present.")
else:
    print("- Missing financial or numerical estimates.")

print("\nValidation Check for Final Strategy:")
if validate_strategy(final_strategy, idea):
    print("- Strategy aligns with the provided idea.")
else:
    print("- Strategy may not fully address the idea.")
if "budget" in final_strategy.lower() and any(char.isdigit() for char in final_strategy):
    print("- Financial and numerical estimates present.")
else:
    print("- Missing financial or numerical estimates.")