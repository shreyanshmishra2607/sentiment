# =========================================================================
# src/predict_attrition.py - Pure Prediction Logic with Simplified Input
# =========================================================================

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

class AttritionPredictor:
    def __init__(self):
        """Initialize the predictor by loading saved model components"""
        self.project_root = Path(__file__).parent.parent
        
        # Load model components
        self.model = joblib.load(self.project_root / 'models' / 'attrition_model.pkl')
        self.scaler = joblib.load(self.project_root / 'models' / 'scaler.pkl')
        self.feature_columns = joblib.load(self.project_root / 'models' / 'feature_columns.pkl')
        
        # Load test data
        self.test_data = pd.read_csv(self.project_root / 'data' / 'test_data' / 'test_data_with_predictions.csv')
        
        # Load simplified features config
        with open(self.project_root / 'config' / 'simplified_features.json', 'r') as f:
            self.config = json.load(f)
        
        self.threshold = 0.68
        
    def predict_from_test_data(self, employee_index):
        """Predict attrition for a specific employee from test data"""
        if employee_index >= len(self.test_data):
            return {"error": f"Employee index {employee_index} not found. Max index: {len(self.test_data)-1}"}
        
        employee_row = self.test_data.iloc[employee_index]
        feature_data = employee_row[self.feature_columns].values.reshape(1, -1)
        
        prob = self.model.predict_proba(feature_data)[0, 1]
        prediction = 1 if prob >= self.threshold else 0
        
        result = {
            "employee_name": f"Test Employee {employee_index}",
            "employee_index": employee_index,
            "attrition_probability": float(prob),
            "will_leave": bool(prediction),
            "risk_level": self._get_risk_level(prob),
            "actual_attrition": int(employee_row['Actual_Attrition']),
            "features": employee_row[self.feature_columns].to_dict()
        }
        
        return result
    
    def predict_simplified_input(self):
        """Collect simplified input and predict attrition"""
        print("\n=== SIMPLIFIED EMPLOYEE DATA COLLECTION ===")
        
        # Collect employee info
        employee_data = {}
        for key, config in self.config["employee_info"].items():
            value = input(f"{config['question']}: ")
            employee_data[key] = value
        
        # Collect feature data
        for key, config in self.config["features"].items():
            if config["type"] == "number":
                while True:
                    try:
                        value = float(input(f"{config['question']}: "))
                        if config.get("min") and value < config["min"]:
                            print(f"Value must be >= {config['min']}")
                            continue
                        if config.get("max") and value > config["max"]:
                            print(f"Value must be <= {config['max']}")
                            continue
                        employee_data[key] = value
                        break
                    except ValueError:
                        print("Please enter a valid number")
            
            elif config["type"] == "choice":
                print(f"{config['question']}")
                for i, option in enumerate(config["options"], 1):
                    print(f"{i}. {option}")
                
                while True:
                    try:
                        choice = int(input("Select option (number): "))
                        if 1 <= choice <= len(config["options"]):
                            employee_data[key] = config["options"][choice - 1]
                            break
                        else:
                            print("Invalid choice")
                    except ValueError:
                        print("Please enter a valid number")
        
        # Convert to full feature vector
        full_features = self._create_full_feature_vector(employee_data)
        
        # Scale and predict
        scaled_data = self.scaler.transform([list(full_features.values())])
        prob = self.model.predict_proba(scaled_data)[0, 1]
        prediction = 1 if prob >= self.threshold else 0
        
        result = {
            "employee_name": employee_data["name"],
            "attrition_probability": float(prob),
            "will_leave": bool(prediction),
            "risk_level": self._get_risk_level(prob),
            "simplified_input": employee_data,
            "features": full_features
        }
        
        return result
    
    def _create_full_feature_vector(self, simplified_data):
        """Convert simplified input to full 135-feature vector"""
        full_features = {col: 0.0 for col in self.feature_columns}
        
        # Map numerical features (normalize using test data stats)
        if "Age" in simplified_data:
            # Simple normalization - you can improve this
            full_features["Age"] = (simplified_data["Age"] - 30) / 15  # rough scaling
        
        if "MonthlyIncome" in simplified_data:
            full_features["MonthlyIncome"] = (simplified_data["MonthlyIncome"] - 6000) / 4000
            
        if "YearsAtCompany" in simplified_data:
            full_features["YearsAtCompany"] = (simplified_data["YearsAtCompany"] - 5) / 5
            
        if "DistanceFromHome" in simplified_data:
            full_features["DistanceFromHome"] = (simplified_data["DistanceFromHome"] - 10) / 10
        
        # Map categorical features (one-hot encoding)
        if simplified_data.get("OverTime") == "Yes":
            full_features["OverTime_Yes"] = 1
            
        if simplified_data.get("Department"):
            dept_col = f"Department_{simplified_data['Department']}"
            if dept_col in full_features:
                full_features[dept_col] = 1
        
        if simplified_data.get("JobLevel"):
            level_col = f"JobLevel_{simplified_data['JobLevel']}"
            if level_col in full_features:
                full_features[level_col] = 1
                
        if simplified_data.get("JobSatisfaction"):
            sat_col = f"JobSatisfaction_{simplified_data['JobSatisfaction']}"
            if sat_col in full_features:
                full_features[sat_col] = 1
                
        if simplified_data.get("WorkLifeBalance"):
            wlb_col = f"WorkLifeBalance_{simplified_data['WorkLifeBalance']}"
            if wlb_col in full_features:
                full_features[wlb_col] = 1
                
        if simplified_data.get("EnvironmentSatisfaction"):
            env_col = f"EnvironmentSatisfaction_{simplified_data['EnvironmentSatisfaction']}"
            if env_col in full_features:
                full_features[env_col] = 1
                
        if simplified_data.get("MaritalStatus"):
            marital_col = f"MaritalStatus_{simplified_data['MaritalStatus']}"
            if marital_col in full_features:
                full_features[marital_col] = 1
                
        if simplified_data.get("BusinessTravel"):
            travel_col = f"BusinessTravel_{simplified_data['BusinessTravel']}"
            if travel_col in full_features:
                full_features[travel_col] = 1
        
        return full_features
    
    def get_user_choice(self):
        """Ask user for data source preference"""
        print("\n=== EMPLOYEE ATTRITION PREDICTION ===")
        print("1. Use test data (demo)")
        print("2. Enter custom employee data")
        
        while True:
            try:
                choice = int(input("Select option (1 or 2): "))
                if choice in [1, 2]:
                    return choice
                else:
                    print("Please select 1 or 2")
            except ValueError:
                print("Please enter a valid number")
    
    def get_test_data_info(self):
        """Get information about available test data"""
        total_employees = len(self.test_data)
        return {
            "total_employees": total_employees,
            "available_indices": f"0 to {total_employees - 1}"
        }
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability >= 0.8:
            return 'Very High'
        elif probability >= 0.6:
            return 'High' 
        elif probability >= 0.4:
            return 'Medium'
        else:
            return 'Low'