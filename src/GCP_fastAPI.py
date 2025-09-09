from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import random

# Import your existing modules
from predict_attrition import AttritionPredictor
from llm_engagement import EmployeeEngagementAnalyzer

app = FastAPI(title="Employee Attrition Analysis API")

# Initialize components
predictor = AttritionPredictor()
analyzer = EmployeeEngagementAnalyzer()

class EmployeeData(BaseModel):
    # Required fields from simplified_features.json
    name: str
    Age: int
    MonthlyIncome: float
    YearsAtCompany: float
    TotalWorkingYears: float
    DistanceFromHome: float
    OverTime: str
    JobSatisfaction: str
    WorkLifeBalance: str
    EnvironmentSatisfaction: str
    JobLevel: str
    Department: str
    MaritalStatus: str
    BusinessTravel: str
    Education: str
    EducationField: str
    Gender: str
    JobInvolvement: str
    JobRole: str
    RelationshipSatisfaction: str
    PerformanceRating: str
    # Optional fields
    YearsInCurrentRole: Optional[float] = None
    YearsSinceLastPromotion: Optional[float] = None
    YearsWithCurrManager: Optional[float] = None
    TrainingTimesLastYear: Optional[int] = None

class AnalysisRequest(BaseModel):
    choice: int  # 1 for test data, 2 for custom
    employee_data: Optional[EmployeeData] = None
    employee_index: Optional[int] = None

class ChatRequest(BaseModel):
    question: str
    employee_name: str
    attrition_probability: float

@app.post("/analyze")
def analyze_employee(request: AnalysisRequest):
    """Main analysis pipeline - replicates main.py logic exactly"""
    
    # Step 1: Get prediction based on choice
    if request.choice == 1:
        # Test data option
        if request.employee_index is None:
            # Random selection like main.py
            test_info = predictor.get_test_data_info()
            max_index = test_info['total_employees'] - 1
            employee_index = random.randint(0, max_index)
        else:
            employee_index = request.employee_index
            
        prediction_result = predictor.predict_from_test_data(employee_index)
        
        if "error" in prediction_result:
            raise HTTPException(status_code=404, detail=prediction_result["error"])
            
    elif request.choice == 2:
        # Custom data option
        if not request.employee_data:
            raise HTTPException(status_code=400, detail="Employee data required for choice 2")
        
        employee_dict = request.employee_data.dict()
        prediction_result = predictor.predict_simplified_input(employee_dict)
    else:
        raise HTTPException(status_code=400, detail="Choice must be 1 or 2")
    
    # Step 2: Get LLM analysis
    try:
        llm_analysis = analyzer.analyze_attrition_risk(prediction_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {str(e)}")
    
    # Step 3: Get suggested questions
    suggested_questions = analyzer.get_conversation_starter(prediction_result['attrition_probability'])
    
    return {
        "prediction": {
            "employee_name": prediction_result.get('employee_name'),
            "attrition_probability": prediction_result['attrition_probability'],
            "will_leave": prediction_result['will_leave'],
            "actual_attrition": prediction_result.get('actual_attrition')
        },
        "analysis": llm_analysis,
        "suggested_questions": suggested_questions
    }

@app.post("/chat")
def chat_followup(request: ChatRequest):
    """Chat with AI consultant - replicates chat loop from main.py"""
    
    context = {
        'employee_name': request.employee_name,
        'attrition_probability': request.attrition_probability
    }
    
    try:
        response = analyzer.chat_with_llm(request.question, context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/test-info")
def get_test_info():
    """Get available test data info"""
    return predictor.get_test_data_info()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)