'''
'''
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

# --- 0. CONFIGURATION ---
# Ensure OPENAI_API_KEY and TAVILY_API_KEY are set in your environment
llm = ChatOpenAI(model="gpt-4o", temperature=0)
search_tool = TavilySearchResults(max_results=3)

# --- 1. DEFINE STATE ---
class ResearchState(TypedDict):
    topic: str
    initial_notes: str
    suggested_themes: str
    selected_themes: List[str] # Human provides this
    deep_research_notes: str
    report_draft: str
    user_feedback: str # Human provides this

# --- 2. DEFINE NODES ---

def preliminary_search(state: ResearchState):
    print(f"--- STEP 1: Broad Search on '{state['topic']}' ---")
    # In a real app, you might run multiple queries here
    results = search_tool.invoke(f"Overview and key issues regarding {state['topic']}")
    # Naive summarization of search results
    content = "\n".join([r["content"] for r in results])
    return {"initial_notes": content}

def extract_themes(state: ResearchState):
    print("--- STEP 2: Extracting Themes ---")
    prompt = f"""
    Analyze the following notes and extract 5 broad themes. 
    Highlight the top 2-3 most important ones and explain why.
    
    NOTES:
    {state['initial_notes']}
    """
    response = llm.invoke([SystemMessage(content="You are a Research Assistant."), HumanMessage(content=prompt)])
    return {"suggested_themes": response.content}

def deep_dive(state: ResearchState):
    print(f"--- STEP 5: Deep Research on {state['selected_themes']} ---")
    combined_notes = ""
    for theme in state['selected_themes']:
        print(f"   -> Searching for: {theme}")
        results = search_tool.invoke(f"Detailed analysis, data, and case studies for {theme} in context of {state['topic']}")
        notes = "\n".join([r["content"] for r in results])
        combined_notes += f"\n\nTHEME: {theme}\nNOTES: {notes}"
    
    return {"deep_research_notes": combined_notes}

def write_report(state: ResearchState):
    print("--- STEP 6: Drafting Report ---")
    prompt = f"""
    Write a comprehensive report on '{state['topic']}' based on the deep research below.
    Focus ONLY on these themes: {state['selected_themes']}
    
    RESEARCH MATERIALS:
    {state['deep_research_notes']}
    
    PREVIOUS FEEDBACK (If any):
    {state.get('user_feedback', 'None')}
    """
    response = llm.invoke([SystemMessage(content="You are an expert report writer."), HumanMessage(content=prompt)])
    return {"report_draft": response.content}

# --- 3. BUILD GRAPH ---
workflow = StateGraph(ResearchState)

# Add Nodes
workflow.add_node("search", preliminary_search)
workflow.add_node("extract", extract_themes)
workflow.add_node("deep_research", deep_dive)
workflow.add_node("write", write_report)

# Define Edges
workflow.set_entry_point("search")
workflow.add_edge("search", "extract")

# INTERRUPT HERE: We stop after 'extract' to let the human choose themes
workflow.add_edge("extract", "deep_research") 

workflow.add_edge("deep_research", "write")
workflow.add_edge("write", END) 

# Compile with a checkpointer (memory) to allow pausing
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["deep_research", "write"])

# --- 4. EXECUTION SIMULATION ---

# START: Run until the first interruption (Theme Selection)
thread_id = {"configurable": {"thread_id": "1"}}
initial_input = {"topic": "The Future of Solid State Batteries"}

print("\n=== STARTING WORKFLOW ===")
for event in app.stream(initial_input, thread_id):
    pass 

# RETRIEVE STATE: The app has paused before 'deep_research'. 
# We inspect what the agent found.
current_state = app.get_state(thread_id).values
print("\n=== SUGGESTED THEMES ===")
print(current_state["suggested_themes"])

# HUMAN INPUT: Simulate the user selecting themes
user_selection = ["Manufacturing Scalability", "Energy Density vs Safety"] 
print(f"\n=== USER SELECTED: {user_selection} ===")

# RESUME: Update state with selection and continue
app.update_state(thread_id, {"selected_themes": user_selection})

# CONTINUE: Run until the next interruption (Draft Review) or End
# Note: In a real app, you would handle the loop for feedback here.
for event in app.stream(None, thread_id):
    pass

final_state = app.get_state(thread_id).values
print("\n=== FINAL DRAFT REPORT ===")
print(final_state["report_draft"])