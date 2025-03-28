import google.generativeai as genai
import os
import pandas as pd
import numpy as np
import json

genai.configure(api_key="AIzaSyCUS9uEV_iQHG_I3BofUVN4bPtOA5L6P0E")

# Define file paths
UPDATED_DATA_FILE = "updated_data_with_cri_sklm.csv"
ZONED_DATA_FILE = "sklm_zone_wise_analysis.csv"
OUTPUT_DATA_FILE = "output.json"
SHAP_DATA_FILE = "shap_insights_sklm.csv"

# Load data
updated_data = pd.read_csv(UPDATED_DATA_FILE)
zoned_data = pd.read_csv(ZONED_DATA_FILE)
shap_data = pd.read_csv(SHAP_DATA_FILE)

with open(OUTPUT_DATA_FILE, "r", encoding="utf-8") as f:
    output_data = json.load(f)

# Initialize Generative AI Model
model_bot = genai.GenerativeModel(model_name="gemini-2.0-pro-exp-02-05")

def generate_zone_report(zone_number, updated_data, zoned_data, output_data, shap_data):
    """Generates a structured report for a specific zone."""
    try:
        # Filter data for the given zone
        zone_updated_data = updated_data[updated_data['zone'] == zone_number].drop(columns=['zone']).to_csv(index=False)
        zone_zoned_data = zoned_data[zoned_data['zone'] == zone_number].drop(columns=['zone']).to_csv(index=False)
        zone_shap_data = shap_data[shap_data['Zone'] == zone_number].drop(columns=['Zone']).to_csv(index=False)
        
        # Extract socioeconomic data from output.json
        zone_output_data = next(
            (item for item in output_data if 'location' in item and
             any(np.isclose(updated_data[updated_data['zone'] == zone_number]['latitude'], item['location']['latitude'])) and
             any(np.isclose(updated_data[updated_data['zone'] == zone_number]['longitude'], item['location']['longitude']))),
            None
        )
        
        zone_output_data_string = json.dumps(zone_output_data, indent=4) if zone_output_data else "No data found in output.json for this zone."
        
        # Define report generation prompt
        prompt_with_data = f"""
        Objective:
Create a clear and detailed report for each zone in Krishna district to help a District Collector understand its sustainability, ability to handle climate challenges, and ways to reduce greenhouse gas (GHG) emissions. Use all available data, fill gaps with online research if needed, and provide practical, step-by-step strategies tailored to each zone’s unique situation.
  Web search: If critical data is unavailable in the provided links, supplement it through external searches.
    See If any of the data is missing to give the report please browser it online and get information you want. donot mention does thing in report . If donot find any of the data online just mention in data gaps and research required donot say it in any of the sections

    Data:
        Updated Data:
        csv
        {zone_updated_data}
        

        Zone-wise Analysis Data:
        csv
        {zone_zoned_data}
        

        Socioeconomic and Research Data:
        json
        {zone_output_data_string}
        

        SHAP ANALYSIS:
        csv
        {zone_shap_data}
        
- If any information is missing, quietly fetch it from links in output.json or search trustworthy online sources (e.g., government reports, research papers). Do not mention the search process in the report—just use the data. If data still can’t be found, note it in the “Data Gaps” section with ideas to get it later.

Task:
For each zone, write a separate, easy-to-read report that analyzes its current state and suggests practical improvements. Use the following structure with simple headings and explanations. For every fact or number, explain why it’s that way (e.g., what’s causing it) and how it affects the zone. Base your analysis on the data provided, trends from SHAP insights, and extra research if needed.

---

### Report Structure for Each Zone

#### 1. Zone Basics & Land Details
- Zone Name: Use the zone number (e.g., “Zone 1”).
- Location: Share latitude and longitude from the CSV data.
- Land Breakdown: List percentages of bare land, built-up areas, crops, trees, and water from the CSV.
- Why It’s Like This: Explain what shapes the land use—like city growth, farming habits, or tree cutting—and how it impacts the zone (e.g., more buildings might mean less nature).

#### 2. Environment & Health Snapshot
- Health Risk Level: Take the Health Risk Index (HRI) from the CSV and say if it’s high, medium, or low.
  - Why: Point out what’s driving it—like dirty air, few trees, or nearby factories—and how it affects people’s health.
- Green Cover Quality: Describe how good or bad the zone’s trees and plants are (Green Score from CSV).
  - Why: Is it shrinking because of building projects, poor soil, or drought? How does this change the air or temperature?
- Key Trends (SHAP Insights): Use SHAP data to highlight the biggest factors affecting health and green cover (e.g., pollution, weather, land changes).
  - Why: Explain how these factors make things better or worse for the zone.

#### 3. People & Community Resources
- Population Info: If available, include basics like number of people or age groups from output.json. If missing, look online (e.g., census data) or note it’s unavailable.
- Community Facilities: List schools, hospitals, banks, and grocery stores from output.json.
  - Why: Show how these services help or hurt daily life, health, and the local economy (e.g., no hospital nearby means tougher health access).

#### 4. Jobs & Businesses
- Industries: Name the main types (e.g., factories, shops) from output.json.
- Farming: Describe crops or farming activities from output.json or online sources if missing.
- Services: Mention any service jobs (e.g., shops, offices) in the zone.
- Environmental Effects:
  - Why: Explain how these activities add to pollution, create jobs, or boost money—e.g., a factory might pollute but employ many people.

#### 5. Climate & Nature Check
- Water & Drainage: Use output.json for water quality or drainage details. If missing, suggest how to find out (e.g., test local rivers).
- Weather Patterns: Share temperature, rainfall, and wind speed from the CSV.
  - Why: Highlight how these affect farming, floods, or heat risks in the zone.
- Air Quality: Report levels of NO₂, SO₂, CH₄, and aerosols from the CSV.
  - Why: Identify what’s causing bad air (e.g., traffic, industries, few trees) and how it harms people or nature.

#### 6. Missing Pieces & Next Steps
- Pull extra details from output.json research links to fill gaps.
- If anything’s still missing (e.g., water quality, population), list it here and suggest simple ways to get it—like surveys, water tests, or checking government records.

#### 7. Step-by-Step Improvement Plan
Based on the zone’s challenges (e.g., pollution, climate risks, limited resources), suggest clear, doable actions that:
Reduce GHG emissions by cutting down on fossil fuels and boosting natural solutions like tree planting.
Enhance resilience to climate threats like floods, heat, or drought with smart planning and infrastructure.
Leverage technology to connect new ideas with real-world action—like using apps, sensors, or clean energy tools.
Make each step specific to the zone’s needs, practical for the Collector to implement, and sensitive to local people, jobs, and resources. Number the steps (e.g., Step 1, Step 2) and explain how they tackle the zone’s issues while bridging innovation and action.
---

### Output Rules
- Write a unique report for each zone, labeled clearly (e.g., “Report for Zone 1”).
- Use headings, bullet points, and short sentences so it’s easy to read.
- For every number or fact, explain why it’s high or low and what’s behind it.
- Keep strategies detailed, step-by-step, and tailored—no vague or repeat ideas.
- Today’s date is March 1, 2025—use it if needed (e.g., for timelines).

---

### Example Report (Hypothetical Zone 1)

#### Report for Zone {zone_number}

#### 1. Zone Basics & Land Details
- Zone Name: Zone {zone_number}
- Location: Latitude 16.5°N, Longitude 80.6°E
- Land Breakdown: Bare land 10%, Built-up 40%, Crops 30%, Trees 15%, Water 5%
- Why It’s Like This: Lots of buildings show city growth, pushing out trees and farms. Crops are still big because of old farming traditions, but water’s low due to poor rivers nearby. This means less nature to cool the area or clean the air.

#### 2. Environment & Health Snapshot
- Health Risk Level: HRI = 0.7 (High)
  - Why: Too many cars and a nearby factory pump out smoke, making people sick. Few trees don’t help either.
- Green Cover Quality: Green Score = 0.4 (Low)
  - Why: Trees got cut for houses, and dry soil stops new ones from growing. Less green means hotter days and dirtier air.
- Key Trends (SHAP Insights): Pollution (NO₂) and low tree cover are the biggest risks to health here.
  - Why: These make breathing harder and heat worse, hitting kids and elders the most.

#### 3. People & Community Resources
- Population Info: About 50,000 people (guessed from online district averages).
- Community Facilities: 2 schools, 1 hospital, 3 banks, 5 grocery stores
  - Why: Decent services help, but one hospital isn’t enough for so many, raising health worries. Banks and stores keep money moving, though.

#### 4. Jobs & Businesses
- Industries: Small textile factory
- Farming: Rice and sugarcane
- Services: Local shops and a bus depot
- Environmental Effects:
  - Why: The factory spits out smoke but gives jobs to 200 people. Farming needs water, which is scarce, and shops add plastic waste.

#### 5. Climate & Nature Check
- Water & Drainage: No data—could be muddy rivers from nearby fields.
- Weather Patterns: Temp 32°C avg, Rainfall 900mm/year, Wind 10km/h
  - Why: Hot weather dries crops, and low rain stresses water supply. Calm winds trap pollution.
- Air Quality: NO₂ high, SO₂ moderate, CH₄ low, Aerosols high
  - Why: Traffic and factory smoke clog the air, worsened by few trees to filter it.

#### 6. Missing Pieces & Next Steps
- No water quality info—test rivers and drains near the factory.
- Exact population missing—check local census records.

#### 7. Step-by-Step Improvement Plan
- Step 1: Plant 1,000 trees near homes and the factory to cut CO₂ and clean air. Use hardy local types that grow fast.
- Step 2: Push the factory to switch to solar power—offer tax breaks to lower coal use and emissions.
- Step 3: Fix drainage with new channels to stop flooding in rains—start with a pilot near crops.
- Step 4: Add a mobile health clinic to ease hospital pressure—run it twice a week.
- Step 5: Teach farmers drip irrigation to save water and keep crops alive—train 50 farmers first.

Note:
NOTE :Do not mention the source or format of the input data (e.g., from web search, JSON, CSV) in the report.
Simply use the data as if it’s readily available and present the findings naturally under each section.
If data is missing, note it in the "Missing Pieces & Next Steps" section without referencing where it might be sourced from. """
        
        response = model_bot.generate_content(
            prompt_with_data, 
            generation_config={
                "temperature": 0.7, 
                "max_output_tokens": 4096, 
                "top_p": 0.95, 
                "top_k": 40
            }
        )
        
        return response.text if response else "Error: No response from AI model."
    except Exception as e:
        return f"Error generating report for Zone {zone_number}: {e}"

# Generate reports and save them
zone_reports = {}

with open("report.txt", "w", encoding="utf-8") as txt_file:
    for zone_number in sorted(updated_data['zone'].unique()):
        report = generate_zone_report(zone_number, updated_data, zoned_data, output_data, shap_data)
        
        # Store in dictionary for JSON output
        zone_reports[f"Zone {zone_number}"] = report
        
        # Write to TXT file with separator
        txt_file.write(report)
        txt_file.write("\n" + "-" * 50 + "\n\n")

# Save reports to JSON
with open("report.json", "w", encoding="utf-8") as json_file:
    json.dump(zone_reports, json_file, ensure_ascii=False, indent=4)

print("Reports saved as 'report.txt' and 'report.json'!")
