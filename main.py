# =========================================================================
# main.py - Employee Attrition Analysis Pipeline
# =========================================================================

from src.predict_attrition import AttritionPredictor
from src.llm_engagement import EmployeeEngagementAnalyzer

def main():
    """Main pipeline for employee attrition analysis"""
    print("🔍 Employee Attrition Analysis System")
    print("=" * 50)
    
    # Initialize components
    try:
        predictor = AttritionPredictor()
        analyzer = EmployeeEngagementAnalyzer()
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Get user choice
    choice = predictor.get_user_choice()
    
    # Get prediction
    if choice == 1:
        # Test data option
        test_info = predictor.get_test_data_info()
        print(f"\nAvailable test employees: {test_info['available_indices']}")
        
        while True:
            try:
                index = int(input("Enter employee index: "))
                prediction_result = predictor.predict_from_test_data(index)
                if "error" not in prediction_result:
                    break
                else:
                    print(prediction_result["error"])
            except ValueError:
                print("Please enter a valid number")
    
    else:
        # Custom data option
        prediction_result = predictor.predict_simplified_input()
    
    # Display prediction results
    print("\n" + "=" * 50)
    print("🎯 PREDICTION RESULTS")
    print("=" * 50)
    print(f"Employee: {prediction_result.get('employee_name', 'Unknown')}")
    print(f"Attrition Probability: {prediction_result['attrition_probability']:.2%}")
    print(f"Risk Level: {prediction_result['risk_level']}")
    print(f"Prediction: {'Will likely leave' if prediction_result['will_leave'] else 'Will likely stay'}")
    
    # Get LLM analysis
    print("\n🤖 Analyzing with AI...")
    analysis = analyzer.analyze_attrition_risk(prediction_result)
    
    print("\n" + "=" * 50)
    print("📋 AI ENGAGEMENT ANALYSIS")
    print("=" * 50)
    print(analysis)
    
    # Interactive chat
    print("\n" + "=" * 50)
    print("💬 FOLLOW-UP CHAT")
    print("=" * 50)
    
    # Show suggested questions
    questions = analyzer.get_conversation_starter(prediction_result['risk_level'])
    print("Suggested questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    # Chat loop
    chat_context = {
        'risk_level': prediction_result['risk_level'],
        'attrition_probability': prediction_result['attrition_probability'],
        'employee_name': prediction_result.get('employee_name', 'Employee')
    }
    
    print("\nType your questions (or 'quit' to exit):")
    
    while True:
        user_question = input("\nYou: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'bye']:
            print("👋 Thank you for using Employee Attrition Analysis System!")
            break
        
        if user_question:
            response = analyzer.chat_with_llm(user_question, chat_context)
            print(f"\nAI: {response}")
        else:
            print("Please enter a question or 'quit' to exit")

if __name__ == "__main__":
    main()