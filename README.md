# ClimateSync

**ClimateSync** is a platform to fight climate change by equipping governments, industries, and individuals with data-driven tools for sustainable action. This project is my submission for the [Google Solution Challenge 2025](https://developers.google.com/community/gdsc-solution-challenge), driven by a passion to address one of the world’s most pressing challenges—climate change—through technology.

## Why ClimateSync?

Climate change impacts every corner of the globe, yet actionable solutions often remain out of reach for decision-makers. I’m building ClimateSync to bridge this gap, starting with governments. By integrating real-time environmental data and AI-powered insights, it empowers policymakers to act decisively—whether it’s mitigating floods, reducing emissions, or planning resilient cities. My goal is to make climate action accessible, scalable, and impactful, one stakeholder at a time.

The Government MVP is the first step, providing tools to analyze climate risks and devise strategies. Future iterations will extend this to industries and individuals, creating a unified ecosystem for sustainability.

## Features (Government MVP)
- **Zone Mapping**: Visualize climate zones using Google Earth Engine.
- **Risk Reports**: AI-generated summaries of weather, air quality, and more via Gemini AI.
- **Action Plans**: Predictive strategies from Vertex AI to guide government policies.

## Tech Stack
- **Frontend**: React, TypeScript, Vite.
- **Backend**: Node.js, Python (Google Earth Engine, Vertex AI, Gemini AI).
- **Data**: Google Earth Engine, external APIs (e.g., weather, AQICN).

## Setup Instructions

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)
- Git

### Steps
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/Vinuthna-Sneha/ClimateSync.git
   cd ClimateSync
2. **Frontend Setup**:
   ```bash
   cd frontend/frontend-1/frontend
   npm install
   npm run dev
3. **Backend Setup**:
   ```bash
   cd frontend/frontend-1/backend
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   node server.js

## Demo
Watch the Government MVP in action:
https://youtu.be/9zhwala_Vv0

## Team Members
- Challapalli Srinivasu
- Tanmayi Kona
- Maheshwari Doddipatla
- Athota Vinuthna Sneha
