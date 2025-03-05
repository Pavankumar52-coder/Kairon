# Importing necessary libraries
import os
import streamlit as st
import logging
from dataclasses import dataclass
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langgraph.graph import StateGraph, END
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Set up logging to track events
logging.basicConfig(level=logging.INFO)

# Set up environment variables API keys of Tavily and Gemini
os.environ["TAVILY_API_KEY"] = "place your tavily api key"
GENAI_API_KEY = "place your gemini api key"  # Replace with your actual API key
genai.configure(api_key=GENAI_API_KEY)

# Initialize API tools and using LLM model
tavily_tool = TavilySearchResults()
gemini_model = GenerativeModel("gemini-2.0-flash")

# Define the state for LangGraph
@dataclass
class ResearchState:
    query: str
    research_data: str = ""
    drafted_answer: str = ""

# Agent - 1 : Research Agent
def research_agent(state: ResearchState) -> ResearchState:
    try:
        logging.info(f"Searching for research througn websites: {state.query}")
        search_results = tavily_tool.run(state.query)
        return ResearchState(query=state.query, research_data=search_results)
    except Exception as e:
        logging.error(f"Error in research_agent: {e}", exc_info=True)
        return ResearchState(query=state.query, research_data="Error retrieving research data.")

# Agent -2 : Answer Drafting Agent
def draft_agent(state: ResearchState) -> ResearchState:
    if not state.research_data or state.research_data == "Error retrieving research data.":
        return ResearchState(query=state.query, research_data=state.research_data, drafted_answer="No valid research data available.")

    prompt = f"""
    Based on the following research data:
    {state.research_data}
    Provide a well-structured answer:
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return ResearchState(query=state.query, research_data=state.research_data, drafted_answer=response.text)
    except Exception as e:
        logging.error(f"Error in draft_agent: {e}", exc_info=True)
        return ResearchState(query=state.query, research_data=state.research_data, drafted_answer="Error generating response.")

# Build Graph Workflow for agents functioning
graph = StateGraph(ResearchState)
graph.add_node("research", research_agent)
graph.add_node("draft", draft_agent)

graph.add_edge("research", "draft")
graph.add_edge("draft", END)
graph.set_entry_point("research")

# Function to run the research system
def run_research_system(query: str) -> str:
    logging.info(f"Running research system for to get desired research answers: {query}")
    executable = graph.compile()
    final_state = executable.invoke(ResearchState(query))
    if isinstance(final_state, dict):  
        final_state = ResearchState(**final_state)

    return final_state.drafted_answer

# Streamlit UI for user interaction
st.title("AI Powered Deep Research Agent")
query = st.text_input("Please enter your research topic to proceed further:")

if st.button("Start Research"):
    if query.strip():
        with st.spinner("Researching..."):
            response = run_research_system(query)
            st.subheader("Generated Research Answer:")
            st.write(response)
    else:
        st.warning("Please enter a research topic.")