import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ============================================================
# PAGE TITLE
# ============================================================

# Display the main application title
st.title("Market Research Assistant")

# Short description of the app purpose
st.write("Generate an industry report based on Wikipedia.")


# ============================================================
# SIDEBAR (Always Visible)
# ============================================================

# Dropdown for selecting the LLM model (assignment requires only one in final version)
selected_model = st.sidebar.selectbox(
    "Select LLM",
    ["gpt-5.2"]
)

# Text input for user to paste OpenAI API key (client-side input)
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

# Initialize storage for retrieved Wikipedia documents
if "docs" not in st.session_state:
    st.session_state.docs = []

# Initialize storage for the validated industry term
if "industry" not in st.session_state:
    st.session_state.industry = ""

# Initialize storage for the generated industry report
if "report" not in st.session_state:
    st.session_state.report = ""


# ============================================================
# STEP 1 – INDUSTRY INPUT + SEMANTIC VALIDATION
# ============================================================

# Section header for Step 1
st.header("Step 1: Enter an Industry")

# Input box for user to type industry keyword
industry_input = st.text_input("Industry Name", key="industry_input")

# Button to trigger Step 1 processing
step1_clicked = st.button("Generate (Step 1)")

if step1_clicked:

    # Validate empty input
    if not industry_input.strip():
        st.error("Please enter an industry keyword.")
        st.stop()

    # Require API key before using LLM
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    # Create LLM instance for query rewriting (keyword → industry term)
    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0,
        openai_api_key=api_key
    )

    # Prompt instructs LLM to convert keyword into a real economic industry
    rewrite_prompt = f"""
    Convert the following user keyword into a real economic industry term.

    Rules:
    - Must represent a real industry, market, or sector
    - Keep it concise
    - If the keyword is too abstract, return "INVALID"

    Keyword: {industry_input}
    """

    # Call LLM for rewriting
    rewrite_response = llm.invoke([
        SystemMessage(content="You are an economic classification assistant."),
        HumanMessage(content=rewrite_prompt)
    ])

    # Extract normalized industry term
    industry_term = rewrite_response.content.strip().lower()

    # ==============================
    # SEMANTIC VALIDATION
    # ==============================

    # If LLM determines keyword is abstract, reject input
    if industry_term == "invalid":
        st.error("The keyword is too abstract. Please re-enter an industry-related term.")
        st.stop()

    # Ensure rewritten term resembles an economic domain
    industry_keywords = ["industry", "market", "sector", "services"]

    if not any(word in industry_term for word in industry_keywords):
        st.error("The keyword is too far from a real economic industry. Please refine your input.")
        st.stop()

    # ==============================
    # WIKIPEDIA VALIDATION
    # ==============================

    # Retrieve top 5 relevant Wikipedia documents
    retriever = WikipediaRetriever(top_k_results=5, doc_content_chars_max=2000)
    docs = retriever.invoke(industry_term)

    # If no documents found, reject
    if not docs:
        st.error("No relevant industry found. Please refine your keyword.")
        st.stop()

    # Save validated results into session state
    st.session_state.docs = docs
    st.session_state.industry = industry_term

    # Notify successful validation
    st.success(f"Industry validated: {industry_term}")


# ============================================================
# STEP 2 – DISPLAY 5 WIKIPEDIA URLS
# ============================================================

# Section header for Step 2
st.header("Step 2: Top 5 Relevant Wikipedia Pages")

# If documents exist, display source URLs
if st.session_state.get("docs"):
    for doc in st.session_state.docs:
        if "source" in doc.metadata:
            st.write(doc.metadata["source"])
else:
    # Placeholder text before Step 1 is completed
    st.info("Wikipedia results will appear here after Step 1.")

# ============================================================
# STEP 3 – INDUSTRY REPORT
# ============================================================

st.header("Step 3: Industry Report")

# Initialization
if "report" not in st.session_state:
    st.session_state.report = None

step3_clicked = st.button("Generate Report (Step 3)")

if step3_clicked:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not st.session_state.get("docs"):
        st.error("No context found. Please complete Step 1 & 2 (Validation & Retrieval) first.")
    else:
        try:
            # Add a visual loader so the user knows the app is working
            with st.spinner(f"Using {selected_model} to synthesize your report..."):
                llm = ChatOpenAI(
                    model=selected_model, 
                    temperature=0.1, 
                    openai_api_key=api_key
                )

                # Fetch documents from session state
                context_text = "\n\n".join([doc.page_content for doc in st.session_state.docs])

                # Improved prompt structure
                messages = [
                    SystemMessage(content="You are a professional market research analyst specializing in industry classification."),
                    HumanMessage(content=f"""
                        Based strictly on the following Wikipedia context, write a professional industry report.
                        
                        Context: {context_text}
                        
                        Required Headers:
                        ## Industry Overview
                        ## Key Industry Insights
                        ## Industry Summary
                        
                        Constraint: Use exactly these headers. Professional tone. Under 480 words.
                        The report should be written in clear, professional English and maintain the tone of a market research or consulting-style analysis. The content must consist of full sentences and well-structured paragraphs, avoiding fragmented expressions, bullet-style writing, or shorthand. The body text must remain in plain text format, ensuring readability and consistency in presentation. Section titles must appear on their own lines and should follow a consistent structural format to support logical flow and clarity of interpretation.

The report must strictly follow a three-part structure:
Industry Overview: 20%
This section introduces the industry by providing background information, definition, and contextual understanding. It should remain concise and serve as an entry point for readers unfamiliar with the sector.
Key Industry Insights: 60%
This section represents the core analytical component of the report and must contain the most substantial content. It should highlight key market dynamics, structural characteristics, growth drivers, competitive forces, and major challenges. The emphasis should be on interpretation and business relevance rather than simple description.
Industry Summary: 20%
This section synthesizes the main findings into strategic implications and a forward-looking perspective. It should conclude the report naturally without repeating earlier content.

The analysis must be based strictly on the retrieved Wikipedia material, ensuring that no unsupported assumptions or fabricated information are introduced. The system is designed to operate within a constrained information environment, reinforcing the reliability and traceability of insights.
The report must remain concise and must not exceed 480 words. At the same time, it should maintain coherence, completeness, and logical continuity. The system should avoid abrupt termination of sentences when approaching the word limit and ensure a natural ending.
Overall, Step 3 aims to transform raw encyclopedic information into structured industry intelligence, bridging the gap between general knowledge retrieval and business-oriented insight generation.
                    """)
                ]

                # Invoke the LLM
                response = llm.invoke(messages)
                
                if response and response.content:
                    # Content cleaning
                    final_report = response.content.replace("###", "##")
                    
                    # Hard word limit enforcement 
                    word_list = final_report.split()
                    if len(word_list) > 480:
                        final_report = " ".join(word_list[:480]) + "..."
                    
                    st.session_state.report = final_report
                    st.success("Report successfully generated!")
                else:
                    st.error("The model returned an empty response. Please try again.")

        except Exception as e:
            # This will show exactly WHY it's failing (e.g., Wrong API key or Invalid Model)
            st.error(f"Error during generation: {str(e)}")

# ============================================================
# DISPLAY MODULE
# ============================================================

if st.session_state.report:
    st.markdown("---")
    # Rendering directly as Markdown ensures headers appear correctly
    st.markdown(st.session_state.report)
    
    # Validation info for the user/instructor
    current_count = len(st.session_state.report.split())
    st.caption(f"Word Count: {current_count} words | Model: {selected_model}")