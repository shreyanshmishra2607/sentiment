# =========================================================================
# src/predict_attrition.py - Standalone Prediction Logic with Enhanced Features
# =========================================================================

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*sklearn.*")
warnings.filterwarnings("ignore", message=".*version.*sklearn.*")

class AttritionPredictor:
    def __init__(self):
        """Initialize the predictor by loading saved model components"""
        self.project_root = Path(__file__).parent.parent
        
        # Load model components with warning suppression
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = joblib.load(self.project_root / 'models' / 'attrition_model.pkl')
                self.scaler = joblib.load(self.project_root / 'models' / 'scaler.pkl')
                self.feature_columns = joblib.load(self.project_root / 'models' / 'feature_columns.pkl')
        except Exception as e:
            print(f"Warning: Model loading issue - {e}")
            print("This is likely due to scikit-learn version differences.")
            print("The models will still work, but consider retraining with current sklearn version.")
        
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
                        if config.get("required", True):
                            value = float(input(f"{config['question']}: "))
                        else:
                            user_input = input(f"{config['question']} (optional, press Enter to skip): ")
                            if user_input.strip() == "":
                                # Set default values for optional fields
                                default_values = {
                                    "YearsInCurrentRole": 2,
                                    "YearsSinceLastPromotion": 1,
                                    "YearsWithCurrManager": 2,
                                    "TrainingTimesLastYear": 2
                                }
                                value = default_values.get(key, 0)
                                print(f"Using default value: {value}")
                            else:
                                value = float(user_input)
                        
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
            
            elif config["type"] == "choice_with_other":
                print(f"{config['question']}")
                for i, option in enumerate(config["options"], 1):
                    print(f"{i}. {option}")
                print(f"{len(config['options']) + 1}. Other (specify)")
                
                while True:
                    try:
                        choice = int(input("Select option (number): "))
                        if 1 <= choice <= len(config["options"]):
                            employee_data[key] = config["options"][choice - 1]
                            break
                        elif choice == len(config["options"]) + 1:
                            custom_value = input("Please specify: ")
                            employee_data[key] = f"OTHER: {custom_value}"
                            break
                        else:
                            print("Invalid choice")
                    except ValueError:
                        print("Please enter a valid number")
        
        # Convert to full feature vector
        full_features, llm_context = self._create_full_feature_vector(employee_data)
        
        # Scale and predict
        scaled_data = self.scaler.transform([list(full_features.values())])
        prob = self.model.predict_proba(scaled_data)[0, 1]
        prediction = 1 if prob >= self.threshold else 0

        result = {
            "employee_name": employee_data["name"],
            "attrition_probability": float(prob),
            "will_leave": bool(prediction),
            "simplified_input": employee_data,
            "features": full_features,
            "llm_context": llm_context
        }
        
        return result
    
    def _create_full_feature_vector(self, simplified_data):
        """Convert simplified input to full feature vector, separating model features from LLM context"""
        full_features = {col: 0.0 for col in self.feature_columns}
        llm_context = {}
        
        # Map numerical features (normalize using reasonable estimates)
        if "Age" in simplified_data:
            # Normalize age (assuming mean ~35, std ~10)
            full_features["Age"] = (simplified_data["Age"] - 35) / 10
        
        if "MonthlyIncome" in simplified_data:
            # Normalize income (assuming mean ~6000, std ~4000)
            full_features["MonthlyIncome"] = (simplified_data["MonthlyIncome"] - 6000) / 4000
            
        if "YearsAtCompany" in simplified_data:
            # Normalize years at company
            full_features["YearsAtCompany"] = (simplified_data["YearsAtCompany"] - 5) / 5
        
        if "TotalWorkingYears" in simplified_data:
            # Normalize total working years
            full_features["TotalWorkingYears"] = (simplified_data["TotalWorkingYears"] - 10) / 8
            
        if "DistanceFromHome" in simplified_data:
            # Normalize distance
            full_features["DistanceFromHome"] = (simplified_data["DistanceFromHome"] - 10) / 10
        
        # Handle optional numerical features
        optional_numeric_mappings = {
            "YearsInCurrentRole": ("YearsInCurrentRole", 2, 3),
            "YearsSinceLastPromotion": ("YearsSinceLastPromotion", 2, 3),
            "YearsWithCurrManager": ("YearsWithCurrManager", 2, 3),
            "TrainingTimesLastYear": ("TrainingTimesLastYear", 2, 2)
        }
        
        for key, (feature_name, mean, std) in optional_numeric_mappings.items():
            if key in simplified_data and feature_name in full_features:
                full_features[feature_name] = (simplified_data[key] - mean) / std
        
        # Map categorical features (one-hot encoding)
        for key, value in simplified_data.items():
            if key == "name":  # Skip name field
                continue
                
            if isinstance(value, str) and value.startswith("OTHER: "):
                # Store custom values for LLM context only
                llm_context[key] = value.replace("OTHER: ", "")
                continue
            
            # Handle known categorical values
            categorical_mappings = {
                "OverTime": {"Yes": "OverTime_Yes"},
                "Department": {
                    "Sales": "Department_Sales",
                    "Research & Development": "Department_Research & Development", 
                    "Human Resources": "Department_Human Resources"
                },
                "JobLevel": {
                    "Entry Level": "JobLevel_1",
                    "Junior Level": "JobLevel_2", 
                    "Mid Level": "JobLevel_3",
                    "Senior Level": "JobLevel_4",
                    "Executive Level": "JobLevel_5"
                },
                "JobSatisfaction": {
                    "Low": "JobSatisfaction_1",
                    "Medium": "JobSatisfaction_2",
                    "High": "JobSatisfaction_3", 
                    "Very High": "JobSatisfaction_4"
                },
                "WorkLifeBalance": {
                    "Bad": "WorkLifeBalance_1",
                    "Good": "WorkLifeBalance_2",
                    "Better": "WorkLifeBalance_3",
                    "Best": "WorkLifeBalance_4"
                },
                "EnvironmentSatisfaction": {
                    "Low": "EnvironmentSatisfaction_1",
                    "Medium": "EnvironmentSatisfaction_2", 
                    "High": "EnvironmentSatisfaction_3",
                    "Very High": "EnvironmentSatisfaction_4"
                },
                "MaritalStatus": {
                    "Single": "MaritalStatus_Single",
                    "Married": "MaritalStatus_Married",
                    "Divorced": "MaritalStatus_Divorced"
                },
                "BusinessTravel": {
                    "Travel_Rarely": "BusinessTravel_Travel_Rarely",
                    "Travel_Frequently": "BusinessTravel_Travel_Frequently",
                    "Non-Travel": "BusinessTravel_Non-Travel"
                },
                "Education": {
                    "Below College": "Education_1",
                    "College": "Education_2",
                    "Bachelor": "Education_3",
                    "Master": "Education_4", 
                    "Doctor": "Education_5"
                },
                "EducationField": {
                    "Life Sciences": "EducationField_Life Sciences",
                    "Medical": "EducationField_Medical",
                    "Marketing": "EducationField_Marketing",
                    "Technical Degree": "EducationField_Technical Degree",
                    "Human Resources": "EducationField_Human Resources",
                    "Other": "EducationField_Other"
                },
                "Gender": {
                    "Male": "Gender_Male",
                    "Female": "Gender_Female"
                },
                "JobInvolvement": {
                    "Low": "JobInvolvement_1",
                    "Medium": "JobInvolvement_2",
                    "High": "JobInvolvement_3",
                    "Very High": "JobInvolvement_4"
                },
                "JobRole": {
                    "Sales Executive": "JobRole_Sales Executive",
                    "Research Scientist": "JobRole_Research Scientist",
                    "Laboratory Technician": "JobRole_Laboratory Technician",
                    "Manufacturing Director": "JobRole_Manufacturing Director",
                    "Healthcare Representative": "JobRole_Healthcare Representative",
                    "Manager": "JobRole_Manager",
                    "Sales Representative": "JobRole_Sales Representative", 
                    "Research Director": "JobRole_Research Director",
                    "Human Resources": "JobRole_Human Resources"
                },
                "RelationshipSatisfaction": {
                    "Low": "RelationshipSatisfaction_1",
                    "Medium": "RelationshipSatisfaction_2",
                    "High": "RelationshipSatisfaction_3",
                    "Very High": "RelationshipSatisfaction_4"
                },
                "PerformanceRating": {
                    "Excellent": "PerformanceRating_3",
                    "Outstanding": "PerformanceRating_4"
                }
            }
            
            if key in categorical_mappings and value in categorical_mappings[key]:
                feature_col = categorical_mappings[key][value]
                if feature_col in full_features:
                    full_features[feature_col] = 1
        
        return full_features, llm_context
    
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

    def run_standalone(self):
        """Run the prediction system standalone"""
        print("🔍 Employee Attrition Prediction System (Standalone)")
        print("=" * 60)
        
        try:
            print("✅ System initialized successfully")
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return
        
        # Get user choice
        choice = self.get_user_choice()
        
        # Get prediction
        if choice == 1:
            # Test data option
            test_info = self.get_test_data_info()
            print(f"\nAvailable test employees: {test_info['available_indices']}")
            
            while True:
                try:
                    index = int(input("Enter employee index: "))
                    prediction_result = self.predict_from_test_data(index)
                    if "error" not in prediction_result:
                        break
                    else:
                        print(prediction_result["error"])
                except ValueError:
                    print("Please enter a valid number")
        
        else:
            # Custom data option
            prediction_result = self.predict_simplified_input()
        
        # Display results
        self.display_results(prediction_result)
    
    def display_results(self, prediction_result):
        """Display prediction results in a clean format"""
        print("\n" + "=" * 60)
        print("🎯 PREDICTION RESULTS")
        print("=" * 60)
        print(f"Employee: {prediction_result.get('employee_name', 'Unknown')}")
        print(f"Attrition Probability: {prediction_result['attrition_probability']:.2%}")
        print(f"Model Prediction: {'Will likely leave' if prediction_result['will_leave'] else 'Will likely stay'}")
        
        if 'actual_attrition' in prediction_result:
            actual_status = "Left" if prediction_result['actual_attrition'] == 1 else "Stayed"
            print(f"Actual Result: {actual_status}")
        
        # Risk level interpretation
        prob = prediction_result['attrition_probability']
        if prob >= 0.7:
            risk_level = "🔴 HIGH RISK"
            action = "Immediate intervention required"
        elif prob >= 0.4:
            risk_level = "🟡 MEDIUM RISK"
            action = "Monitor closely and consider preventive measures"
        else:
            risk_level = "🟢 LOW RISK"
            action = "Continue current engagement strategies"
        
        print(f"Risk Level: {risk_level}")
        print(f"Recommended Action: {action}")
        
        print("\n" + "=" * 60)
        print("📊 Raw prediction data saved to outputs folder (if running full system)")


# Standalone execution
if __name__ == "__main__":
    try:
        predictor = AttritionPredictor()
        predictor.run_standalone()
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check if all required files and models are in place.")