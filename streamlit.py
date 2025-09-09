import streamlit as st
import requests
import json
import random
from typing import Dict, Any, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Employee Attrition Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration
EXTERNAL_IP = "YOUR_EXTERNAL_IP_HERE"  # Replace with your GCP VM external IP
API_BASE_URL = f"http://{EXTERNAL_IP}:8000"

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    border-bottom: 2px solid #f0f2f6;
    margin-bottom: 2rem;
}

.option-card {
    border: 2px solid #e6e9ef;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    background-color: #f8f9fa;
}

.option-card:hover {
    border-color: #ff4b4b;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.risk-high { color: #ff4444; font-weight: bold; }
.risk-medium { color: #ff8800; font-weight: bold; }
.risk-low { color: #00aa44; font-weight: bold; }

.chat-container {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    max-height: 400px;
    overflow-y: auto;
}

.user-message {
    background-color: #e3f2fd;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
    text-align: right;
}

.ai-message {
    background-color: #f1f8e9;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint: str, data: Dict = None, method: str = "GET") -> Optional[Dict]:
    """Make API request to backend server"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to server at {API_BASE_URL}. Please ensure the API server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timeout. Please try again.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def display_prediction_results(prediction: Dict[str, Any], analysis: str):
    """Display prediction results"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Prediction Results")
        
        prob = prediction['attrition_probability']
        st.metric("Employee", prediction['employee_name'])
        st.metric("Attrition Probability", f"{prob:.1%}")
        st.metric("Prediction", "Will likely leave" if prediction['will_leave'] else "Will likely stay")
        
        # Risk level
        if prob >= 0.7:
            risk_level = "🔴 HIGH RISK"
            st.markdown(f'<p class="risk-high">{risk_level}</p>', unsafe_allow_html=True)
        elif prob >= 0.4:
            risk_level = "🟡 MEDIUM RISK"
            st.markdown(f'<p class="risk-medium">{risk_level}</p>', unsafe_allow_html=True)
        else:
            risk_level = "🟢 LOW RISK"
            st.markdown(f'<p class="risk-low">{risk_level}</p>', unsafe_allow_html=True)
        
        if 'actual_attrition' in prediction and prediction['actual_attrition'] is not None:
            actual = "Left" if prediction['actual_attrition'] == 1 else "Stayed"
            st.metric("Actual Result", actual)
    
    with col2:
        st.subheader("🧠 AI Analysis")
        with st.container():
            st.markdown(analysis)

def render_test_data_option():
    """Render test data option interface"""
    st.markdown("""
    <div class="option-card">
        <h3>📋 Option 1: Analyze Test Employee</h3>
        <p>Select an employee from our test database for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get test data info
    test_info = make_api_request("test-info")
    if not test_info:
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info(f"Available employees: 0 to {test_info['total_employees'] - 1}")
        
        # Option to select specific or random
        selection_type = st.radio(
            "Selection method:",
            ["Random selection", "Specific employee"],
            key="selection_type"
        )
        
        if selection_type == "Specific employee":
            employee_index = st.number_input(
                "Employee Index:",
                min_value=0,
                max_value=test_info['total_employees'] - 1,
                value=0,
                key="employee_index"
            )
        else:
            employee_index = None
    
    with col2:
        if st.button("🔍 Analyze Employee", key="analyze_test", type="primary"):
            with st.spinner("Analyzing employee data..."):
                request_data = {
                    "choice": 1,
                    "employee_index": employee_index
                }
                
                result = make_api_request("analyze", request_data, "POST")
                
                if result:
                    st.session_state.current_analysis = result
                    st.session_state.analysis_complete = True
                    st.rerun()

def render_custom_data_option():
    """Render custom data input interface"""
    st.markdown("""
    <div class="option-card">
        <h3>👤 Option 2: Analyze Custom Employee</h3>
        <p>Enter employee details for personalized analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("employee_form"):
        st.subheader("Employee Information")
        
        # Basic Info
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Employee Name*", key="name")
            age = st.number_input("Age*", min_value=18, max_value=65, value=30, key="age")
            monthly_income = st.number_input("Monthly Income*", min_value=1000, max_value=50000, value=5000, key="income")
            years_company = st.number_input("Years at Company*", min_value=0, max_value=40, value=3, key="years_company")
            total_experience = st.number_input("Total Experience*", min_value=0, max_value=45, value=5, key="experience")
            
        with col2:
            distance = st.number_input("Distance from Home (km)*", min_value=1, max_value=50, value=10, key="distance")
            overtime = st.selectbox("Works Overtime*", ["Yes", "No"], key="overtime")
            job_satisfaction = st.selectbox("Job Satisfaction*", ["Low", "Medium", "High", "Very High"], index=2, key="job_sat")
            work_life_balance = st.selectbox("Work-Life Balance*", ["Bad", "Good", "Better", "Best"], index=2, key="wlb")
            env_satisfaction = st.selectbox("Environment Satisfaction*", ["Low", "Medium", "High", "Very High"], index=2, key="env_sat")
        
        st.subheader("Professional Details")
        col3, col4 = st.columns(2)
        
        with col3:
            job_level = st.selectbox("Job Level*", ["Entry Level", "Junior Level", "Mid Level", "Senior Level", "Executive Level"], index=2, key="job_level")
            department = st.selectbox("Department*", ["Sales", "Research & Development", "Human Resources"], key="department")
            marital_status = st.selectbox("Marital Status*", ["Single", "Married", "Divorced"], key="marital")
            business_travel = st.selectbox("Business Travel*", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], key="travel")
            education = st.selectbox("Education*", ["Below College", "College", "Bachelor", "Master", "Doctor"], index=2, key="education")
        
        with col4:
            education_field = st.selectbox("Education Field*", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"], key="edu_field")
            gender = st.selectbox("Gender*", ["Male", "Female"], key="gender")
            job_involvement = st.selectbox("Job Involvement*", ["Low", "Medium", "High", "Very High"], index=2, key="job_inv")
            job_role = st.selectbox("Job Role*", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"], key="job_role")
            relationship_satisfaction = st.selectbox("Relationship Satisfaction*", ["Low", "Medium", "High", "Very High"], index=2, key="rel_sat")
            performance_rating = st.selectbox("Performance Rating*", ["Excellent", "Outstanding"], key="performance")
        
        submit_button = st.form_submit_button("🔍 Analyze Employee", type="primary")
        
        if submit_button:
            if not name:
                st.error("Please enter employee name")
                return
            
            with st.spinner("Analyzing employee data..."):
                employee_data = {
                    "name": name,
                    "Age": age,
                    "MonthlyIncome": monthly_income,
                    "YearsAtCompany": years_company,
                    "TotalWorkingYears": total_experience,
                    "DistanceFromHome": distance,
                    "OverTime": overtime,
                    "JobSatisfaction": job_satisfaction,
                    "WorkLifeBalance": work_life_balance,
                    "EnvironmentSatisfaction": env_satisfaction,
                    "JobLevel": job_level,
                    "Department": department,
                    "MaritalStatus": marital_status,
                    "BusinessTravel": business_travel,
                    "Education": education,
                    "EducationField": education_field,
                    "Gender": gender,
                    "JobInvolvement": job_involvement,
                    "JobRole": job_role,
                    "RelationshipSatisfaction": relationship_satisfaction,
                    "PerformanceRating": performance_rating
                }
                
                request_data = {
                    "choice": 2,
                    "employee_data": employee_data
                }
                
                result = make_api_request("analyze", request_data, "POST")
                
                if result:
                    st.session_state.current_analysis = result
                    st.session_state.analysis_complete = True
                    st.rerun()

def render_chat_interface():
    """Render follow-up chat interface"""
    if 'current_analysis' not in st.session_state:
        return
    
    analysis = st.session_state.current_analysis
    prediction = analysis['prediction']
    
    st.markdown("---")
    st.subheader("💬 Follow-up Consultation")
    
    # Display suggested questions
    st.markdown("**💡 Suggested Questions:**")
    suggested_questions = analysis.get('suggested_questions', [])
    
    for i, question in enumerate(suggested_questions[:5], 1):
        if st.button(f"{i}. {question}", key=f"suggest_{i}"):
            st.session_state.current_question = question
    
    # Chat interface
    st.markdown("**Ask your own question:**")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    question = st.text_input("Your question:", key="chat_input", placeholder="Ask about retention strategies, risk factors, etc.")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("Ask", type="primary")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Handle question from button or input
    current_question = st.session_state.get('current_question', question)
    if ask_button and current_question or st.session_state.get('current_question'):
        if st.session_state.get('current_question'):
            current_question = st.session_state.current_question
            st.session_state.current_question = None
        
        if current_question:
            with st.spinner("Getting AI response..."):
                chat_data = {
                    "question": current_question,
                    "employee_name": prediction['employee_name'],
                    "attrition_probability": prediction['attrition_probability']
                }
                
                response = make_api_request("chat", chat_data, "POST")
                
                if response:
                    st.session_state.chat_history.append({
                        "question": current_question,
                        "response": response['response']
                    })
                    st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("**Chat History:**")
        chat_container = st.container()
        
        with chat_container:
            for i, chat in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {chat['question']}
                </div>
                <div class="ai-message">
                    <strong>AI Consultant:</strong><br>
                    {chat['response']}
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Employee Attrition Analysis System</h1>
        <p>Predict employee attrition risk and get AI-powered engagement strategies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Show results if analysis is complete
    if st.session_state.analysis_complete and 'current_analysis' in st.session_state:
        analysis = st.session_state.current_analysis
        display_prediction_results(analysis['prediction'], analysis['analysis'])
        render_chat_interface()
        
        # Option to start new analysis
        st.markdown("---")
        if st.button("🔄 New Analysis", type="secondary"):
            for key in ['current_analysis', 'analysis_complete', 'chat_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    else:
        # Main interface - show options
        st.markdown("### Choose Analysis Method:")
        
        # Create tabs for the two options
        tab1, tab2 = st.tabs(["📋 Test Data Analysis", "👤 Custom Employee Analysis"])
        
        with tab1:
            render_test_data_option()
        
        with tab2:
            render_custom_data_option()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Employee Attrition Analysis System | Powered by AI</p>
        <p><small>Server: {}</small></p>
    </div>
    """.format(API_BASE_URL), unsafe_allow_html=True)

if __name__ == "__main__":
    main()