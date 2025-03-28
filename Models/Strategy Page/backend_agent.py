import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains import SequentialChain, LLMChain
from google.oauth2 import service_account
from google.cloud import aiplatform
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="Government Strategy Planner API")

# Load zone data
with open("unified_data.json", "r") as f:
    zone_data = json.load(f)

# Initialize Vertex AI
credentials = service_account.Credentials.from_service_account_file("inspired-rock-450806-r5-d18f93b7f5ba.json")
aiplatform.init(project="inspired-rock-450806-r5", location="us-central1", credentials=credentials)

# Initialize Gemini LLM and Embeddings
llm = VertexAI(model_name="gemini-1.5-flash", credentials=credentials, max_output_tokens=3000, temperature=0.7)
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003", credentials=credentials)

# Define prompt templates
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
- Create scalable, practical solutions to mitigate and adapt to climate change. The solutions should focus on reducing GHG emissions, enhancing resilience, and leveraging technology to bridge the gap between innovation and implementation.

Context:
The given data provides a comprehensive overview of the land cover, environmental factors, risk metrics, trends, socioeconomic status, industries, and research insights of a specific zone. Your strategy should align with real-world constraints, regulations, and feasibility while maximizing sustainability, economic growth, and social well-being.

Instructions for Strategy Development:
1. Overview of the Zone
Provide a brief analysis of the zone’s characteristics based on the given data.
Identify key opportunities and challenges based on land cover, risk metrics, and environmental factors.
2. Strategy Plan & Implementation
Develop a well-structured, phased strategy addressing the identified challenges.
Ensure the plan is realistic, feasible, and backed with dynamic timeline based on strategy by real-world constraints.

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

modify_template = """
Role Assignment:
You are Government Strategy Planner AI, a highly capable AI assistant specializing in creating detailed, practical, and data-driven strategy plans for urban and rural development. Your goal is to analyze the provided unified zone data and develop a realistic, implementable strategy plan tailored to the specified location.
Zone Data: {context}
Strategy: {strategy}
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

# Pydantic models for request bodies
class StrategyRequest(BaseModel):
    zone_id: int
    idea: str

class QnARequest(BaseModel):
    strategy: str
    question: str

class ModifyRequest(BaseModel):
    zone_id: int
    strategy: str
    modification_request: str

# Helper function to generate location context
def generate_location_context(zone_id: int) -> str:
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

# Multi-stage chain setup
overview_chain = LLMChain(llm=llm, prompt=overview_prompt, output_key="overview")
strategy_chain = LLMChain(llm=llm, prompt=strategy_prompt, output_key="strategy")
full_chain = SequentialChain(
    chains=[overview_chain, strategy_chain],
    input_variables=["context", "idea", "location_context"],
    output_variables=["overview", "strategy"]
)

# Q&A and Modification chains
qna_chain = LLMChain(llm=llm, prompt=qna_prompt)
modify_chain = LLMChain(llm=llm, prompt=modify_prompt, output_key="modified_strategy")

# API Endpoints
@app.post("/generate-strategy/")
async def generate_strategy(request: StrategyRequest):
    try:
        query = f"Zone {request.zone_id}"
        retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 1}).invoke(query)
        context = retrieved_docs[0].page_content if retrieved_docs else f"Zone {request.zone_id}: No data available."
        
        location_context = generate_location_context(request.zone_id)
        
        result = full_chain.invoke({
            "context": context,
            "idea": request.idea,
            "location_context": location_context
        })
        return {"strategy": result["strategy"], "overview": result["overview"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating strategy: {str(e)}")

@app.post("/ask-question/")
async def ask_question(request: QnARequest):
    try:
        answer = qna_chain.invoke({"strategy": request.strategy, "question": request.question})
        return {"answer": answer["text"] if isinstance(answer, dict) else answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.post("/modify-strategy/")
async def modify_strategy(request: ModifyRequest):
    try:
        query = f"Zone {request.zone_id}"
        retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 1}).invoke(query)
        context = retrieved_docs[0].page_content if retrieved_docs else f"Zone {request.zone_id}: No data available."
        
        location_context = generate_location_context(request.zone_id)
        
        result = modify_chain.invoke({
            "strategy": request.strategy,
            "modification_request": request.modification_request,
            "context": context,
            "location_context": location_context,
            "overview": "Overview placeholder"  # You might want to fetch the original overview if needed
        })
        return {"modified_strategy": result["modified_strategy"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error modifying strategy: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Government Strategy Planner API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)