# =========================================================================
# src/llm_engagement.py - Standalone LLM-based Employee Engagement Analysis
# =========================================================================

import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import json
import warnings
from typing import Dict, Any, List

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables from .env file
load_dotenv()

class EmployeeEngagementAnalyzer:
    def __init__(self):
        """Initialize the LLM-based engagement analyzer"""
        # Check if API key is available
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file or environment.")
        
        # Create outputs directory if it doesn't exist
        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Initialize Gemini LLM (FIXED - removed streaming parameter)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        
        # Create system prompt
        self.system_prompt = SystemMessagePromptTemplate.from_template(
            """You are an expert HR Analytics consultant specializing in employee retention and engagement psychology. 
            
            Your task is to analyze raw employee data and attrition probability to provide actionable insights.
            
            **Analysis Framework:**
            - Assess attrition risk based on probability score
            - Identify key psychological and workplace factors
            - Provide specific, actionable recommendations
            - Focus on human psychology behind the data
            
            **Output Format Requirements:**
            - Use clear headings with ##
            - Use bullet points for lists
            - Keep sections concise and focused
            - Provide specific action items
            - Write in professional, empathetic tone
            
            **Key Areas to Address:**
            1. Risk Assessment
            2. Key Contributing Factors  
            3. Recommended Actions
            4. Timeline for Implementation
            5. Success Metrics"""
        )
        
        # Create human prompt template
        self.human_prompt = HumanMessagePromptTemplate.from_template(
            """## Employee Attrition Analysis Request

            **Employee:** {employee_name}
            **Attrition Probability:** {attrition_probability:.2%}

            **Raw Employee Data:**
            {raw_data}
            
            **Key Features:**
            {key_features}
            
            Please provide a comprehensive analysis and engagement strategy."""
        )
        
        # Combine prompts
        self.chat_prompt = ChatPromptTemplate.from_messages([
            self.system_prompt,
            self.human_prompt
        ])
        
        # Initialize session variables
        self.current_session_file = None
        self.chat_log = []
        
    def analyze_attrition_risk(self, prediction_result: Dict[str, Any]) -> str:
        """Analyze employee attrition risk and provide engagement strategies"""
        try:
            # Extract employee info
            employee_name = prediction_result.get('employee_name', 'Unknown Employee')
            attrition_prob = prediction_result.get('attrition_probability', 0)
            
            # Format employee data
            raw_data, key_features = self._format_employee_data(prediction_result)
            
            # Create the prompt
            messages = self.chat_prompt.format_messages(
                employee_name=employee_name,
                attrition_probability=attrition_prob,
                raw_data=raw_data,
                key_features=key_features
            )
            
            print("🤖 AI Analysis:")
            print("-" * 50)
            
            # Get LLM response
            response = self.llm.invoke(messages)
            analysis_content = response.content
            
            # Simple streaming effect (optional)
            self._simulate_streaming(analysis_content)
            
            print("\n" + "-" * 50)
            
            # Save to file
            self._save_analysis_to_file(employee_name, attrition_prob, analysis_content)
            
            return analysis_content
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(error_msg)
            return error_msg
    
    def chat_with_llm(self, question: str, context: Dict[str, Any] = None) -> str:
        """Interactive chat with LLM for follow-up questions (ENHANCED VERSION)"""
        try:
            if context:
                context_str = f"""
                **Context:**
                - Employee: {context.get('employee_name', 'N/A')}
                - Attrition Probability: {context.get('attrition_probability', 0):.2%}
                
                **Previous Analysis Summary:** See previous conversation for detailed analysis.
                """
                
                # Enhanced system message for more detailed responses
                detailed_system_message = """You are an expert HR Analytics consultant specializing in employee retention and engagement psychology. 

                **Response Guidelines:**
                - Provide comprehensive, actionable responses (3-5 sentences minimum)
                - Include 2-3 specific bullet points or recommendations when relevant
                - Use professional but conversational tone
                - Draw insights from HR best practices and psychology
                - Structure responses with clear reasoning and practical steps
                - Provide context and rationale for your recommendations
                
                **Response Format:**
                - Start with a brief analysis or context
                - Provide 2-3 specific actionable points
                - End with implementation guidance or next steps
                
                Continue the conversation about employee engagement with detailed, helpful responses."""
                
                messages = [
                    SystemMessage(content=detailed_system_message),
                    HumanMessage(content=f"{context_str}\n\n**Question:** {question}")
                ]
            else:
                # Enhanced system message for general HR questions
                detailed_system_message = """You are an expert HR Analytics consultant specializing in employee retention and engagement psychology.
                
                **Response Guidelines:**
                - Provide comprehensive, detailed responses (3-5 sentences minimum)
                - Include specific recommendations with 2-3 bullet points when applicable
                - Use professional yet accessible language
                - Draw from HR best practices and organizational psychology
                - Provide practical implementation steps
                - Give context and reasoning behind recommendations
                
                **Response Structure:**
                - Brief context or analysis of the question
                - 2-3 specific actionable recommendations
                - Implementation guidance or next steps
                
                Provide detailed, actionable guidance on employee engagement and retention strategies."""
                
                messages = [
                    SystemMessage(content=detailed_system_message),
                    HumanMessage(content=question)
                ]
            
            print(f"\n💬 AI Response:")
            print("-" * 30)
            
            # Get response
            response = self.llm.invoke(messages)
            chat_response = response.content
            
            # Simple streaming effect
            self._simulate_streaming(chat_response)
            
            print("\n" + "-" * 30)
            
            # Log the chat
            self._log_chat_interaction(question, chat_response)
            
            return chat_response
            
        except Exception as e:
            error_msg = f"Chat failed: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _simulate_streaming(self, content: str):
        """Simple streaming simulation for better UX"""
        import time
        import sys
        
        words = content.split(' ')
        for i, word in enumerate(words):
            print(word, end=' ', flush=True)
            if i % 15 == 0 and i > 0:  # Add small delay every 15 words
                time.sleep(0.08)
        print()  # New line at the end
    
    def _format_employee_data(self, prediction_result: Dict[str, Any]) -> tuple:
        """Format employee data for LLM analysis"""
        features = prediction_result.get('features', {})
        
        # Format raw data
        raw_data_parts = []
        
        # Basic demographics
        if 'Age' in features:
            raw_data_parts.append(f"Age (scaled): {features['Age']:.2f}")
        if 'MonthlyIncome' in features:
            raw_data_parts.append(f"Monthly Income (scaled): {features['MonthlyIncome']:.2f}")
        if 'YearsAtCompany' in features:
            raw_data_parts.append(f"Years at Company: {features['YearsAtCompany']:.2f}")
        if 'TotalWorkingYears' in features:
            raw_data_parts.append(f"Total Experience: {features['TotalWorkingYears']:.2f}")
        if 'DistanceFromHome' in features:
            raw_data_parts.append(f"Distance from Home (scaled): {features['DistanceFromHome']:.2f}")
        
        raw_data = "\n".join(raw_data_parts) if raw_data_parts else "Limited demographic data available"
        
        # Extract key workplace factors
        key_features = self._extract_key_features(features, prediction_result)
        
        return raw_data, key_features
    
    def _extract_key_features(self, features: Dict[str, Any], prediction_result: Dict[str, Any] = None) -> str:
        """Extract key workplace factors for analysis"""
        key_factors = []
        
        # Use simplified input if available (for custom entries)
        if prediction_result and 'simplified_input' in prediction_result:
            simple_data = prediction_result['simplified_input']
            
            for key, value in simple_data.items():
                if key != 'name':  # Skip name field
                    key_factors.append(f"{key.replace('_', ' ').title()}: {value}")
            
            # Add LLM context if available
            if prediction_result and 'llm_context' in prediction_result:
                llm_data = prediction_result['llm_context']
                for key, value in llm_data.items():
                    key_factors.append(f"{key.replace('_', ' ').title()} (Custom): {value}")
                    
        else:
            # Extract from encoded features (for test data)
            # Department
            dept_features = [k for k in features.keys() if k.startswith('Department_') and features[k] == 1]
            if dept_features:
                dept = dept_features[0].replace('Department_', '')
                key_factors.append(f"Department: {dept}")
            
            # Job Role
            role_features = [k for k in features.keys() if k.startswith('JobRole_') and features[k] == 1]
            if role_features:
                role = role_features[0].replace('JobRole_', '')
                key_factors.append(f"Job Role: {role}")
                
            # Work-Life Balance
            wlb_features = [k for k in features.keys() if k.startswith('WorkLifeBalance_') and features[k] == 1]
            if wlb_features:
                wlb = wlb_features[0].replace('WorkLifeBalance_', '')
                key_factors.append(f"Work-Life Balance: {wlb}")
            
            # Job Satisfaction
            js_features = [k for k in features.keys() if k.startswith('JobSatisfaction_') and features[k] == 1]
            if js_features:
                js = js_features[0].replace('JobSatisfaction_', '')
                key_factors.append(f"Job Satisfaction: {js}")
            
            # Environment Satisfaction
            env_features = [k for k in features.keys() if k.startswith('EnvironmentSatisfaction_') and features[k] == 1]
            if env_features:
                env = env_features[0].replace('EnvironmentSatisfaction_', '')
                key_factors.append(f"Environment Satisfaction: {env}")
            
            # Overtime
            if 'OverTime_Yes' in features:
                overtime_status = "Yes" if features['OverTime_Yes'] == 1 else "No"
                key_factors.append(f"Overtime: {overtime_status}")
            
            # Marital Status
            marital_features = [k for k in features.keys() if k.startswith('MaritalStatus_') and features[k] == 1]
            if marital_features:
                marital = marital_features[0].replace('MaritalStatus_', '')
                key_factors.append(f"Marital Status: {marital}")
            
            # Business Travel
            travel_features = [k for k in features.keys() if k.startswith('BusinessTravel_') and features[k] == 1]
            if travel_features:
                travel = travel_features[0].replace('BusinessTravel_', '')
                key_factors.append(f"Business Travel: {travel}")
        
        return "\n".join(key_factors) if key_factors else "Limited feature information available"
    
    def _save_analysis_to_file(self, employee_name: str, attrition_prob: float, analysis_content: str):
        """Save analysis to markdown file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        employee_clean = employee_name.replace(" ", "_").replace("/", "_")
        filename = f"employee_analysis_{employee_clean}_{timestamp}.md"
        filepath = self.outputs_dir / filename
        
        # Create markdown content
        markdown_content = f"""# Employee Attrition Analysis Report

**Employee:** {employee_name}  
**Analysis Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Attrition Probability:** {attrition_prob:.2%}  

---

{analysis_content}

---

*Generated by Employee Attrition Analysis System*
"""
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"\n📄 Analysis saved to: {filepath}")
        
        # Set current session file for chat logging
        self.current_session_file = filepath
        self.chat_log = []
    
    def _log_chat_interaction(self, question: str, response: str):
        """Log chat interactions to the current session file"""
        if not self.current_session_file:
            return
        
        # Add to chat log
        chat_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'question': question,
            'response': response
        }
        self.chat_log.append(chat_entry)
        
        # Append to file
        chat_section = f"""

## Follow-up Chat

"""
        
        for entry in self.chat_log:
            chat_section += f"""
### Question ({entry['timestamp']})
{entry['question']}

### Response
{entry['response']}

---
"""
        
        # Read existing content and append chat
        with open(self.current_session_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        # Remove previous chat section if exists
        if "## Follow-up Chat" in existing_content:
            existing_content = existing_content.split("## Follow-up Chat")[0]
        
        # Write updated content
        with open(self.current_session_file, 'w', encoding='utf-8') as f:
            f.write(existing_content + chat_section)
        
        print(f"💾 Chat logged to: {self.current_session_file}")
    
    def get_conversation_starter(self, attrition_probability: float) -> List[str]:
        """Get suggested follow-up questions based on attrition probability"""
        if attrition_probability >= 0.7:  # High risk (70%+)
            return [
                "What immediate retention actions should we take?",
                "How can we address their top concerns quickly?",
                "What compensation adjustments might help?",
                "Should we involve senior management?",
                "What's the timeline for intervention?"
            ]
        elif attrition_probability >= 0.4:  # Medium risk (40-70%)
            return [
                "What preventive measures should we implement?",
                "How can we enhance their job satisfaction?",
                "What development opportunities can we offer?",
                "How can we improve their work environment?",
                "What recognition strategies would be effective?"
            ]
        else:  # Low risk (<40%)
            return [
                "How can we maintain their current engagement?",
                "What growth opportunities should we provide?",
                "How can they mentor others?",
                "What new challenges might motivate them?",
                "How can we leverage their strengths?"
            ]

    def run_standalone_demo(self):
        """Run a standalone demo of the LLM engagement analyzer"""
        print("🤖 Employee Engagement Analyzer (Standalone Demo)")
        print("=" * 60)
        
        # Create sample prediction result for demo
        sample_result = {
            "employee_name": "Demo Employee",
            "attrition_probability": 0.75,
            "will_leave": True,
            "simplified_input": {
                "name": "Demo Employee",
                "Age": 28,
                "MonthlyIncome": 4500,
                "YearsAtCompany": 1,
                "DistanceFromHome": 25,
                "OverTime": "Yes",
                "JobSatisfaction": "Low",
                "WorkLifeBalance": "Bad",
                "EnvironmentSatisfaction": "Low",
                "JobLevel": "Junior Level",
                "Department": "Sales",
                "MaritalStatus": "Single",
                "BusinessTravel": "Travel_Frequently"
            },
            "features": {}
        }
        
        print("📊 Running analysis on demo employee data...")
        print("=" * 60)
        
        # Analyze the sample data
        analysis = self.analyze_attrition_risk(sample_result)
        
        # Interactive chat demo
        print("\n" + "=" * 60)
        print("💬 INTERACTIVE CHAT DEMO")
        print("=" * 60)
        
        # Show suggested questions
        questions = self.get_conversation_starter(sample_result['attrition_probability'])
        print("\n💡 Suggested Questions:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
        
        # Chat context
        chat_context = {
            'employee_name': sample_result['employee_name'],
            'attrition_probability': sample_result['attrition_probability']
        }
        
        print("\n💡 Type your questions below (or 'quit' to exit):")
        
        # Chat loop
        while True:
            try:
                user_question = input("\n➤ Your question: ").strip()
                
                if user_question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Demo session ended. Check the outputs folder for saved analysis!")
                    break
                
                if user_question:
                    self.chat_with_llm(user_question, chat_context)
                else:
                    print("Please enter a question or type 'quit' to exit")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Check the outputs folder for saved analysis!")
                break

    def run_custom_analysis(self):
        """Run analysis with custom employee data input"""
        print("🤖 Employee Engagement Analyzer (Custom Analysis)")
        print("=" * 60)
        
        # Get custom employee data
        print("\n=== CUSTOM EMPLOYEE DATA ENTRY ===")
        
        # Simple data collection for demo
        employee_name = input("Enter employee name: ")
        
        try:
            attrition_prob = float(input("Enter attrition probability (0.0 to 1.0): "))
            if not 0 <= attrition_prob <= 1:
                attrition_prob = 0.5  # Default
                print("Invalid probability, using 0.5")
        except ValueError:
            attrition_prob = 0.5
            print("Invalid input, using default probability 0.5")
        
        # Create custom result structure
        custom_result = {
            "employee_name": employee_name,
            "attrition_probability": attrition_prob,
            "will_leave": attrition_prob >= 0.5,
            "simplified_input": {
                "name": employee_name,
                # You can expand this to collect more data
            },
            "features": {}
        }
        
        print(f"\n📊 Running analysis for {employee_name}...")
        print("=" * 60)
        
        # Analyze the custom data
        analysis = self.analyze_attrition_risk(custom_result)
        
        # Interactive chat
        print("\n" + "=" * 60)
        print("💬 FOLLOW-UP CONSULTATION")
        print("=" * 60)
        
        # Show suggested questions
        questions = self.get_conversation_starter(custom_result['attrition_probability'])
        print("\n💡 Suggested Questions:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
        
        # Chat context
        chat_context = {
            'employee_name': custom_result['employee_name'],
            'attrition_probability': custom_result['attrition_probability']
        }
        
        print("\n💡 Type your questions below (or 'quit' to exit):")
        
        # Chat loop
        while True:
            try:
                user_question = input("\n➤ Your question: ").strip()
                
                if user_question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Session ended. Check the outputs folder for saved analysis!")
                    break
                
                if user_question:
                    self.chat_with_llm(user_question, chat_context)
                else:
                    print("Please enter a question or type 'quit' to exit")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Check the outputs folder for saved analysis!")
                break

    def run_standalone(self):
        """Main standalone runner with menu options"""
        print("🤖 Employee Engagement Analyzer - Standalone Mode")
        print("=" * 60)
        print("1. Run Demo Analysis (with sample data)")
        print("2. Run Custom Analysis (enter your own data)")
        print("3. Chat Only Mode (general HR questions)")
        
        while True:
            try:
                choice = int(input("\nSelect option (1, 2, or 3): "))
                if choice in [1, 2, 3]:
                    break
                else:
                    print("Please select 1, 2, or 3")
            except ValueError:
                print("Please enter a valid number")
        
        if choice == 1:
            self.run_standalone_demo()
        elif choice == 2:
            self.run_custom_analysis()
        elif choice == 3:
            self.run_chat_only_mode()

    def run_chat_only_mode(self):
        """Run in chat-only mode for general HR questions"""
        print("💬 HR Consultant Chat Mode")
        print("=" * 40)
        print("Ask me anything about employee engagement, retention, or HR strategies!")
        print("Type 'quit' to exit")
        
        while True:
            try:
                user_question = input("\n➤ Your HR question: ").strip()
                
                if user_question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Chat session ended!")
                    break
                
                if user_question:
                    self.chat_with_llm(user_question)
                else:
                    print("Please enter a question or type 'quit' to exit")
            except KeyboardInterrupt:
                print("\n\n👋 Chat session interrupted!")
                break