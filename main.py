
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
        # Test data option with random selection
        import random
        
        test_info = predictor.get_test_data_info()
        max_index = int(test_info['available_indices'].split(' to ')[1])
        
        # Randomly select an employee index
        random_index = random.randint(0, max_index)
        print(f"\nRandomly selected employee index: {random_index}")
        
        prediction_result = predictor.predict_from_test_data(random_index)
        
        # Display which employee was chosen
        print(f"Analyzing Employee #{random_index} from test data")

    else:
        # Custom data option
        employee_data = {}
        for key, config in predictor.config["employee_info"].items():
            value = input(f"{config['question']}: ")
            employee_data[key] = value
        prediction_result = predictor.predict_simplified_input(employee_data)

    
    # Display clean prediction results
    print("\n" + "=" * 50)
    print("🎯 PREDICTION RESULTS")
    print("=" * 50)
    print(f"Employee: {prediction_result.get('employee_name', 'Unknown')}")
    print(f"Attrition Probability: {prediction_result['attrition_probability']:.2%}")
    print(f"Model Prediction: {'Will likely leave' if prediction_result['will_leave'] else 'Will likely stay'}")
    
    if 'actual_attrition' in prediction_result:
        actual_status = "Left" if prediction_result['actual_attrition'] == 1 else "Stayed"
        print(f"Actual Result: {actual_status}")
    
    # Get LLM analysis with streaming
    print("\n" + "=" * 50)
    print("📋 AI ENGAGEMENT ANALYSIS")
    print("=" * 50)
    
    try:
        analysis = analyzer.analyze_attrition_risk(prediction_result)
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        analysis = None
    
    # Interactive chat
    print("\n" + "=" * 50)
    print("💬 FOLLOW-UP CONSULTATION")
    print("=" * 50)
    
    try:
        # Show suggested questions based on probability
        questions = analyzer.get_conversation_starter(prediction_result['attrition_probability'])
        print("\n💡 Suggested Questions:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
        
        # Chat context
        chat_context = {
            'employee_name': prediction_result.get('employee_name', 'Employee'),
            'attrition_probability': prediction_result['attrition_probability']
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
                    analyzer.chat_with_llm(user_question, chat_context)
                else:
                    print("Please enter a question or type 'quit' to exit")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Check the outputs folder for saved analysis!")
                break
                
    except Exception as e:
        print(f"Chat system error: {str(e)}")
        print("Please check if GOOGLE_API_KEY is set in your .env file.")

if __name__ == "__main__":
    main()