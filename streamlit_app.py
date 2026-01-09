import streamlit as st
import os
from typing import TypedDict, List, Annotated
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="Research Synthesizer", page_icon="🕵️‍♂️", layout="wide")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    
    st.divider()
    st.info("This tool performs web research, extracts themes, and writes a report based on your selection.")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# Check for keys
if not openai_api_key or not tavily_api_key:
    st.warning("Please enter your API keys in the sidebar to proceed.")
    st.stop()

# Set Environment Variables for LangChain
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# --- LANGGRAPH SETUP ---

# 1. Define State
class ResearchState(TypedDict):
    topic: str
    initial_notes: str
    suggested_themes: List[str] # Structured list for UI
    selected_themes: List[str]
    deep_research_notes: str
    report_draft: str

# 2. Define Nodes
llm = ChatOpenAI(model="gpt-4o", temperature=0)
search_tool = TavilySearchResults(max_results=3)

def preliminary_search(state: ResearchState):
    """Step 1: Broad Search"""
    query = f"Overview and key current issues regarding {state['topic']}"
    results = search_tool.invoke(query)
    content = "\n".join([r["content"] for r in results])
    return {"initial_notes": content}

def extract_themes(state: ResearchState):
    """Step 2: Extract Themes"""
    # We ask for a strict list format to make parsing easier for Streamlit
    prompt = f"""
    Analyze the following notes and extract 5 broad distinct themes or sub-topics.
    Return ONLY a Python-parseable list of strings. Example: ["Theme A", "Theme B"]
    
    NOTES:
    {state['initial_notes']}
    """
    response = llm.invoke([SystemMessage(content="You are a data extractor."), HumanMessage(content=prompt)])
    
    # Simple cleanup to ensure it evaluates to a list
    try:
        themes = eval(response.content)
        if not isinstance(themes, list):
            themes = ["General Overview", "Market Trends", "Challenges"]
    except:
        themes = ["General Overview", "Key Technologies", "Future Outlook"]
        
    return {"suggested_themes": themes}

def deep_dive(state: ResearchState):
    """Step 3: Deep Research on Selected Themes"""
    combined_notes = ""
    # Progress bar logic would go here in a CLI, but in Streamlit we use st.spinner in the UI layer
    for theme in state['selected_themes']:
        query = f"Detailed analysis, data, and statistics for '{theme}' related to {state['topic']}"
        results = search_tool.invoke(query)
        notes = "\n".join([r["content"] for r in results])
        combined_notes += f"\n\n### THEME: {theme}\n{notes}"
    
    return {"deep_research_notes": combined_notes}

def write_report(state: ResearchState):
    """Step 4: Write Report"""
    prompt = f"""
    Write a comprehensive executive briefing on '{state['topic']}'.
    Focus strictly on these selected themes: {state['selected_themes']}.
    
    Use Markdown formatting. Include a Title, Executive Summary, and a section for each theme.
    
    RESEARCH DATA:
    {state['deep_research_notes']}
    """
    response = llm.invoke([SystemMessage(content="You are an expert analyst."), HumanMessage(content=prompt)])
    return {"report_draft": response.content}

# 3. Compile Graph (Cached Resource)
@st.cache_resource
def get_graph():
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("search", preliminary_search)
    workflow.add_node("extract", extract_themes)
    workflow.add_node("deep_research", deep_dive)
    workflow.add_node("write", write_report)
    
    workflow.set_entry_point("search")
    workflow.add_edge("search", "extract")
    
    # Interrupt after extraction to let user select themes
    workflow.add_edge("extract", "deep_research") 
    
    workflow.add_edge("deep_research", "write")
    workflow.add_edge("write", END)
    
    memory = MemorySaver()
    # We interrupt BEFORE deep_research to allow human input
    return workflow.compile(checkpointer=memory, interrupt_before=["deep_research"])

app_graph = get_graph()

# --- STREAMLIT UI LOGIC ---

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"
if "current_step" not in st.session_state:
    st.session_state.current_step = "input" # input -> select -> result

st.title("🕵️‍♂️ Research Synthesizer")

# --- SCREEN 1: INPUT ---
if st.session_state.current_step == "input":
    st.markdown("#### Step 1: Define Your Topic")
    topic = st.text_input("What do you want to research?", placeholder="e.g. The impact of AI on Biotech")
    
    if st.button("Start Discovery"):
        if not topic:
            st.error("Please enter a topic.")
        else:
            with st.status("🔍 Performing preliminary research...", expanded=True) as status:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Start the graph, it will run until the interrupt
                for event in app_graph.stream({"topic": topic}, config):
                    pass # Just run to the interrupt
                
                status.update(label="Discovery Complete!", state="complete", expanded=False)
            
            # Update state to move to next screen
            st.session_state.current_step = "select"
            st.rerun()

# --- SCREEN 2: THEME SELECTION ---
elif st.session_state.current_step == "select":
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    # Fetch current state from memory
    current_state = app_graph.get_state(config).values
    topic = current_state.get("topic")
    suggested = current_state.get("suggested_themes", [])
    
    st.markdown(f"#### Step 2: Select Themes for '{topic}'")
    st.write("I found the following themes. Select the ones you want to explore in depth:")
    
    # Multiselect for themes
    selected = st.multiselect("Themes", options=suggested, default=suggested[:2])
    
    # Option to add custom theme
    custom_theme = st.text_input("Add a specific angle (optional):")
    if custom_theme:
        selected.append(custom_theme)
        
    if st.button("Generate Report"):
        if not selected:
            st.error("Please select at least one theme.")
        else:
            # Update the state with user choice
            app_graph.update_state(config, {"selected_themes": selected})
            
            with st.status("🚀 Deep diving & Drafting...", expanded=True) as status:
                # Resume execution (None input resumes from current state)
                for event in app_graph.stream(None, config):
                    pass
                status.update(label="Report Generated!", state="complete", expanded=False)
            
            st.session_state.current_step = "result"
            st.rerun()

# --- SCREEN 3: RESULT ---
elif st.session_state.current_step == "result":
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    final_state = app_graph.get_state(config).values
    report = final_state.get("report_draft")
    
    st.markdown("#### Step 3: Research Report")
    st.markdown(report)
    
    st.divider()
    st.download_button("Download Report", data=report, file_name="research_report.md")
    
    if st.button("Start New Research"):
        st.session_state.clear()
        st.rerun()