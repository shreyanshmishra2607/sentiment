# =========================================================================
# src/vertexai.py - Full Attrition + LLM API including main pipeline
# =========================================================================

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import random
import warnings

from predict_attrition import AttritionPredictor
from llm_engagement import EmployeeEngagementAnalyzer

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------
# Initialize components
# --------------------------
app = FastAPI(title="Employee Attrition + Engagement API")
predictor = AttritionPredictor()
analyzer = EmployeeEngagementAnalyzer()

# ===========================
# Request models
# ===========================
class EmployeeData(BaseModel):
    name: Optional[str] = None
    Age: Optional[float] = None
    MonthlyIncome: Optional[float] = None
    YearsAtCompany: Optional[float] = None
    TotalWorkingYears: Optional[float] = None
    DistanceFromHome: Optional[float] = None
    OverTime: Optional[str] = None
    Department: Optional[str] = None
    JobLevel: Optional[str] = None
    JobSatisfaction: Optional[str] = None
    WorkLifeBalance: Optional[str] = None
    EnvironmentSatisfaction: Optional[str] = None
    MaritalStatus: Optional[str] = None
    BusinessTravel: Optional[str] = None
    JobRole: Optional[str] = None


class LLMAnalysisRequest(BaseModel):
    prediction_result: Dict[str, Any]


class LLMChatRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None


class PipelineRequest(BaseModel):
    test_index: Optional[int] = None
    employee_data: Optional[EmployeeData] = None


# ===========================
# Basic root
# ===========================
@app.get("/")
def root():
    return {"message": "Attrition Prediction API is live. Use /get_attrition or /pipeline."}


# ===========================
# Existing attrition prediction
# ===========================
@app.post("/get_attrition")
def get_attrition(employee: EmployeeData = None, test_index: int = None):
    try:
        if test_index is not None:
            result = predictor.predict_from_test_data(test_index)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            return result

        if employee is not None:
            emp_dict = employee.dict()
            emp_dict = {k: v for k, v in emp_dict.items() if v is not None}
            if not emp_dict:
                raise HTTPException(status_code=400, detail="No employee data provided")
            
            full_features, llm_context = predictor._create_full_feature_vector(emp_dict)
            scaled_data = predictor.scaler.transform([list(full_features.values())])
            prob = predictor.model.predict_proba(scaled_data)[0, 1]
            prediction = 1 if prob >= predictor.threshold else 0

            response = {
                "employee_name": emp_dict.get("name", "Unknown"),
                "attrition_probability": float(prob),
                "will_leave": bool(prediction),
                "simplified_input": emp_dict,
                "features": full_features,
                "llm_context": llm_context
            }
            return response

        raise HTTPException(status_code=400, detail="Provide either test_index or employee data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# Existing LLM routes
# ===========================
@app.post("/llm/analyze")
def llm_analyze(request: LLMAnalysisRequest):
    try:
        prediction_result = request.prediction_result
        if not prediction_result:
            raise HTTPException(status_code=400, detail="prediction_result is required")
        analysis_content = analyzer.analyze_attrition_risk(prediction_result)
        return {"analysis": analysis_content, "prediction_result": prediction_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/chat")
def llm_chat(request: LLMChatRequest):
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="question is required")
        chat_response = analyzer.chat_with_llm(request.question, request.context)
        return {"response": chat_response, "context": request.context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# NEW: Main pipeline route
# ===========================
@app.post("/pipeline")
def pipeline(request: PipelineRequest):
    """
    Full pipeline: predict attrition -> run LLM engagement analysis -> return context for chat
    """
    try:
        # Step 1: Prediction
        if request.test_index is not None:
            prediction_result = predictor.predict_from_test_data(request.test_index)
            if "error" in prediction_result:
                raise HTTPException(status_code=404, detail=prediction_result["error"])
        elif request.employee_data is not None:
            emp_dict = request.employee_data.dict()
            emp_dict = {k: v for k, v in emp_dict.items() if v is not None}
            if not emp_dict:
                raise HTTPException(status_code=400, detail="No employee data provided")
            
            full_features, llm_context = predictor._create_full_feature_vector(emp_dict)
            scaled_data = predictor.scaler.transform([list(full_features.values())])
            prob = predictor.model.predict_proba(scaled_data)[0, 1]
            prediction = 1 if prob >= predictor.threshold else 0
            
            prediction_result = {
                "employee_name": emp_dict.get("name", "Unknown"),
                "attrition_probability": float(prob),
                "will_leave": bool(prediction),
                "simplified_input": emp_dict,
                "features": full_features,
                "llm_context": llm_context
            }
        else:
            raise HTTPException(status_code=400, detail="Provide either test_index or employee_data")

        # Step 2: LLM Analysis
        analysis_content = analyzer.analyze_attrition_risk(prediction_result)

        # Step 3: Prepare chat context
        chat_context = {
            "employee_name": prediction_result.get("employee_name", "Employee"),
            "attrition_probability": prediction_result["attrition_probability"]
        }

        return {
            "prediction_result": prediction_result,
            "analysis": analysis_content,
            "chat_context": chat_context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/chat")
def pipeline_chat(request: LLMChatRequest):
    """
    Follow-up chat based on previous pipeline context
    """
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="question is required")
        chat_response = analyzer.chat_with_llm(request.question, request.context)
        return {"response": chat_response, "context": request.context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# Run locally
# ===========================
if __name__ == "__main__":
    uvicorn.run("vertexai:app", host="0.0.0.0", port=8080, reload=True)
