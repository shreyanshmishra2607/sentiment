# =========================================================================
# src/llm_engagement.py - LLM-based Employee Engagement Analysis
# =========================================================================

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import json
from typing import Dict, Any, List

# Load environment variables from .env file
load_dotenv()

class EmployeeEngagementAnalyzer:
    def __init__(self):
        """Initialize the LLM-based engagement analyzer"""
        # Check if API key is available
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file or environment.")
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        
        # Create system prompt
        self.system_prompt = SystemMessagePromptTemplate.from_template(
            """You are an expert HR Analytics consultant specializing in employee retention and engagement psychology. 
            Your role is to analyze employee data and predict attrition risks, then provide actionable, personalized engagement strategies.
            
            Focus on these key psychological and workplace factors:
            - Work-life balance and stress management
            - Career development and growth opportunities  
            - Job satisfaction and role fulfillment
            - Workplace relationships and team dynamics
            - Compensation and recognition
            - Management and leadership effectiveness
            - Work environment and company culture
            - Mental health and wellbeing factors
            - Performance pressure and workload management
            
            Always provide:
            1. Clear risk assessment explanation
            2. Root cause analysis based on data patterns
            3. Specific, actionable recommendations
            4. Psychological insights into employee motivation
            5. Preventive measures for retention
            
            Keep responses professional, empathetic, and focused on human psychology behind the numbers."""
        )
        
        # Create human prompt template
        self.human_prompt = HumanMessagePromptTemplate.from_template(
            """Please analyze this employee's attrition risk and provide engagement recommendations:

            EMPLOYEE PROFILE:
            {employee_summary}
            
            ATTRITION PREDICTION:
            - Probability of Leaving: {attrition_probability:.2%}
            - Risk Level: {risk_level}
            - Model Prediction: {prediction_text}
            
            KEY DATA POINTS:
            {key_features}
            
            Please provide a comprehensive analysis including:
            1. Risk Assessment & Key Concerns
            2. Psychological Factors Analysis  
            3. Specific Engagement Strategies
            4. Preventive Action Plan
            5. Follow-up Recommendations"""
        )
        
        # Combine prompts
        self.chat_prompt = ChatPromptTemplate.from_messages([
            self.system_prompt,
            self.human_prompt
        ])
        
    def analyze_attrition_risk(self, prediction_result: Dict[str, Any]) -> str:
        """Analyze employee attrition risk and provide engagement strategies"""
        try:
            # Format employee data for LLM
            employee_summary, key_features = self._format_employee_data(prediction_result)
            
            # Create prediction text
            prediction_text = "Will likely leave" if prediction_result.get('will_leave', False) else "Will likely stay"
            
            # Create the prompt
            messages = self.chat_prompt.format_messages(
                employee_summary=employee_summary,
                attrition_probability=prediction_result.get('attrition_probability', 0),
                risk_level=prediction_result.get('risk_level', 'Unknown'),
                prediction_text=prediction_text,
                key_features=key_features
            )
            
            # Get LLM response using invoke instead of deprecated __call__
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def chat_with_llm(self, question: str, context: Dict[str, Any] = None) -> str:
        """Interactive chat with LLM for follow-up questions"""
        try:
            if context:
                # Include context in the conversation
                context_str = f"""
                Previous Analysis Context:
                - Risk Level: {context.get('risk_level', 'N/A')}
                - Attrition Probability: {context.get('attrition_probability', 0):.2%}
                - Key Concerns: {context.get('key_concerns', 'See previous analysis')}
                """
                
                messages = [
                    SystemMessage(content="You are an HR expert continuing a conversation about employee engagement and retention strategies."),
                    HumanMessage(content=f"{context_str}\n\nQuestion: {question}")
                ]
            else:
                messages = [
                    SystemMessage(content="You are an HR expert specializing in employee engagement and retention strategies."),
                    HumanMessage(content=question)
                ]
            
            # Use invoke instead of deprecated __call__
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Chat failed: {str(e)}"
    
    def _format_employee_data(self, prediction_result: Dict[str, Any]) -> tuple:
        """Format employee data for LLM analysis"""
        features = prediction_result.get('features', {})
        
        # Extract key demographic info
        employee_summary = self._extract_employee_summary(features)
        
        # Extract key workplace factors
        key_features = self._extract_key_features(features, prediction_result)
        
        return employee_summary, key_features
    
    def _extract_employee_summary(self, features: Dict[str, Any]) -> str:
        """Extract basic employee demographic information"""
        summary_parts = []
        
        # Age
        if 'Age' in features:
            age_scaled = features['Age']
            # Approximate age (assuming scaling, adjust if needed)
            summary_parts.append(f"Age: {age_scaled:.2f} (scaled)")
        
        # Income
        if 'MonthlyIncome' in features:
            income_scaled = features['MonthlyIncome']
            summary_parts.append(f"Monthly Income: {income_scaled:.2f} (scaled)")
        
        # Experience
        if 'TotalWorkingYears' in features:
            exp_scaled = features['TotalWorkingYears']
            summary_parts.append(f"Total Experience: {exp_scaled:.2f} (scaled)")
        
        if 'YearsAtCompany' in features:
            company_years = features['YearsAtCompany']
            summary_parts.append(f"Years at Company: {company_years:.2f} (scaled)")
        
        # Department
        dept_features = [k for k in features.keys() if k.startswith('Department_') and features[k] == 1]
        if dept_features:
            dept = dept_features[0].replace('Department_', '')
            summary_parts.append(f"Department: {dept}")
        
        # Job Role
        role_features = [k for k in features.keys() if k.startswith('JobRole_') and features[k] == 1]
        if role_features:
            role = role_features[0].replace('JobRole_', '')
            summary_parts.append(f"Job Role: {role}")
        
        return " | ".join(summary_parts)
    
    def _extract_key_features(self, features: Dict[str, Any], prediction_result: Dict[str, Any] = None) -> str:
        """Extract key workplace factors for analysis"""
        key_factors = []
        
        # Use simplified input if available
        if prediction_result and 'simplified_input' in prediction_result:
            simple_data = prediction_result['simplified_input']
            
            if 'WorkLifeBalance' in simple_data:
                key_factors.append(f"Work-Life Balance: {simple_data['WorkLifeBalance']}")
            if 'JobSatisfaction' in simple_data:
                key_factors.append(f"Job Satisfaction: {simple_data['JobSatisfaction']}")
            if 'EnvironmentSatisfaction' in simple_data:
                key_factors.append(f"Environment Satisfaction: {simple_data['EnvironmentSatisfaction']}")
            if 'OverTime' in simple_data:
                key_factors.append(f"Overtime: {simple_data['OverTime']}")
            if 'DistanceFromHome' in simple_data:
                key_factors.append(f"Distance from Home: {simple_data['DistanceFromHome']} km")
            if 'MaritalStatus' in simple_data:
                key_factors.append(f"Marital Status: {simple_data['MaritalStatus']}")
            if 'BusinessTravel' in simple_data:
                key_factors.append(f"Business Travel: {simple_data['BusinessTravel']}")
        else:
            # Fallback to encoded features
            wlb_features = [k for k in features.keys() if k.startswith('WorkLifeBalance_') and features[k] == 1]
            if wlb_features:
                wlb = wlb_features[0].replace('WorkLifeBalance_', '')
                key_factors.append(f"Work-Life Balance: {wlb}")
            
            js_features = [k for k in features.keys() if k.startswith('JobSatisfaction_') and features[k] == 1]
            if js_features:
                js = js_features[0].replace('JobSatisfaction_', '')
                key_factors.append(f"Job Satisfaction: {js}")
            
            env_features = [k for k in features.keys() if k.startswith('EnvironmentSatisfaction_') and features[k] == 1]
            if env_features:
                env = env_features[0].replace('EnvironmentSatisfaction_', '')
                key_factors.append(f"Environment Satisfaction: {env}")
            
            if 'OverTime_Yes' in features and features['OverTime_Yes'] == 1:
                key_factors.append("Overtime: Yes")
            elif 'OverTime_Yes' in features:
                key_factors.append("Overtime: No")
            
            if 'DistanceFromHome' in features:
                distance = features['DistanceFromHome']
                key_factors.append(f"Distance from Home: {distance:.2f} (scaled)")
        
        return " | ".join(key_factors) if key_factors else "Limited feature information available"
    
    def get_conversation_starter(self, risk_level: str) -> List[str]:
        """Get suggested follow-up questions based on risk level"""
        if risk_level.lower() in ['high', 'very high']:
            return [
                "What immediate actions should we take?",
                "How can we improve their work-life balance?",
                "What career development opportunities might help?",
                "Should we consider a salary review?",
                "How can their manager better support them?"
            ]
        elif risk_level.lower() == 'medium':
            return [
                "What preventive measures should we implement?",
                "How can we enhance their job satisfaction?",
                "What training programs might benefit them?",
                "How can we improve team dynamics?",
                "What recognition strategies would work?"
            ]
        else:  # Low risk
            return [
                "How can we maintain their engagement?",
                "What growth opportunities can we provide?",
                "How can we leverage them as mentors?",
                "What new challenges might motivate them?",
                "How can we recognize their contributions?"
            ]