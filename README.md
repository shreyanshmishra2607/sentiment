# Employee Attrition Analysis System

> 🔍 **AI-Powered Employee Retention Prediction & Engagement Strategy System**

An intelligent HR analytics platform that predicts employee attrition risk and provides AI-generated engagement strategies using machine learning models and Google Gemini LLM.

## 🌟 Features

### Core Capabilities
- **📊 Attrition Prediction**: ML-based probability scoring for employee turnover risk
- **🧠 AI Analysis**: Google Gemini LLM provides detailed engagement strategies
- **💬 Interactive Consultation**: Chat with AI HR consultant for follow-up questions
- **📋 Dual Input Modes**: Test data analysis or custom employee data entry
- **📄 Report Generation**: Automated markdown reports with actionable insights

### Interface Options
- **🖥️ Command Line**: Full-featured CLI interface (`main.py`)
- **🌐 Web API**: FastAPI backend for integration (`GCP_fastAPI.py`)
- **📱 Web App**: Streamlit dashboard with intuitive UI (`streamlit.py`)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Cloud API key (for Gemini LLM)
- Required Python packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/shreyanshmishra2607/sentiment.git
cd sentiment

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your GOOGLE_API_KEY to .env file
```

### Environment Setup

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## 🔧 Usage

### Option 1: Command Line Interface

```bash
# Run the main application
python main.py
```

**Features:**
- Interactive employee data collection
- Real-time attrition prediction
- AI-powered engagement analysis
- Follow-up consultation chat
- Automatic report saving

### Option 2: Web Application (Streamlit)

```bash
# Launch the web interface
streamlit run streamlit.py
```

**Access:** Open `http://localhost:8501` in your browser

**Features:**
- User-friendly form interface
- Visual prediction results
- Interactive AI chat
- Risk level indicators
- Suggested questions

### Option 3: API Server (FastAPI)

```bash
# Start the API server
python src/GCP_fastAPI.py
```

**Access:** Open `http://localhost:8000/docs` for API documentation

**Endpoints:**
- `POST /analyze` - Predict attrition and get AI analysis
- `POST /chat` - Chat with AI consultant
- `GET /test-info` - Get available test data information

## 📊 Model Details

### Machine Learning Pipeline

**Algorithm:** Logistic Regression
- **Input Features:** 24+ employee attributes (demographics, job satisfaction, work-life balance, etc.)
- **Output:** Attrition probability (0-1) and binary prediction
- **Threshold:** 0.68 (optimized for balanced precision-recall)

### Feature Categories

**Demographic Features:**
- Age, Gender, Marital Status, Education Level

**Job-Related Features:**
- Department, Job Role, Job Level, Years at Company
- Monthly Income, Distance from Home, Overtime

**Satisfaction Metrics:**
- Job Satisfaction, Work-Life Balance
- Environment Satisfaction, Relationship Satisfaction

**Performance Indicators:**
- Performance Rating, Job Involvement
- Training Sessions, Years with Current Manager

### Model Performance
- **Training Accuracy:** Optimized through cross-validation
- **Feature Engineering:** One-hot encoding for categorical variables
- **Scaling:** StandardScaler for numerical features
- **Validation:** Test set evaluation with actual attrition outcomes

## 🤖 AI Analysis Engine

### LLM Integration
- **Model:** Google Gemini 1.5 Flash
- **Framework:** LangChain for prompt engineering
- **Temperature:** 0.7 (balanced creativity and consistency)

### Analysis Framework
1. **Risk Assessment:** Probability-based risk categorization
2. **Factor Analysis:** Key contributing factors identification
3. **Action Recommendations:** Specific, actionable retention strategies
4. **Timeline Planning:** Implementation schedules and success metrics
5. **Interactive Consultation:** Follow-up Q&A with context awareness

### Risk Categories
- 🔴 **High Risk (≥70%):** Immediate intervention required
- 🟡 **Medium Risk (40-69%):** Preventive measures recommended
- 🟢 **Low Risk (<40%):** Maintain current engagement strategies

## 📁 Project Structure

```
sentiment/
├── 📂 .vscode/              # VS Code configuration
├── 📂 config/
│   └── simplified_features.json    # Input schema configuration
├── 📂 data/
│   ├── 📂 attrition_employee/     # Original datasets
│   ├── 📂 cleaned_data/           # Processed datasets
│   ├── 📂 dataset/               # Raw data files
│   └── 📂 test_data/             # Test predictions
├── 📂 models/
│   ├── attrition_model.pkl       # Trained logistic regression model
│   ├── feature_columns.pkl       # Feature names and order
│   └── scaler.pkl                # StandardScaler for preprocessing
├── 📂 notebooks/
│   ├── File-1-Data-Preparation.ipynb    # Data preprocessing
│   └── File-2-ML-Training.ipynb         # Model training
├── 📂 outputs/                   # Generated analysis reports
├── 📂 src/
│   ├── GCP_fastAPI.py           # FastAPI backend server
│   ├── llm_engagement.py        # LLM analysis engine
│   └── predict_attrition.py     # Core prediction logic
├── main.py                      # CLI application
├── streamlit.py                 # Web dashboard
└── requirements.txt             # Python dependencies
```

## 🔧 Configuration

### Input Schema (`config/simplified_features.json`)

The system uses a structured configuration for employee data collection:

**Required Fields:**
- Personal: Name, Age, Gender, Marital Status
- Professional: Job Role, Department, Job Level, Income
- Experience: Years at Company, Total Experience
- Satisfaction: Job, Environment, Work-Life Balance ratings

**Optional Fields:**
- Years in Current Role, Years Since Promotion
- Training Sessions, Years with Current Manager

## 🌐 Deployment

### Local Development
```bash
# Run all components
python main.py              # CLI interface
streamlit run streamlit.py   # Web dashboard
python src/GCP_fastAPI.py   # API server
```

### Production Deployment

**Google Cloud Platform:**
1. Deploy FastAPI backend to Compute Engine
2. Configure external IP and firewall rules
3. Update Streamlit app with production API URL
4. Set up SSL certificates for HTTPS

**Docker Deployment:**
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "src/GCP_fastAPI.py"]
```

## 📈 API Usage Examples

### Analyze Test Employee
```python
import requests

response = requests.post("http://localhost:8000/analyze", json={
    "choice": 1,
    "employee_index": 42
})

result = response.json()
print(f"Attrition Probability: {result['prediction']['attrition_probability']:.2%}")
```

### Analyze Custom Employee
```python
employee_data = {
    "name": "John Doe",
    "Age": 28,
    "MonthlyIncome": 4500,
    "Department": "Sales",
    "JobSatisfaction": "Low",
    "WorkLifeBalance": "Bad",
    # ... other fields
}

response = requests.post("http://localhost:8000/analyze", json={
    "choice": 2,
    "employee_data": employee_data
})
```

### Chat with AI Consultant
```python
response = requests.post("http://localhost:8000/chat", json={
    "question": "What retention strategies work best for high-risk employees?",
    "employee_name": "John Doe",
    "attrition_probability": 0.75
})
```

## 🔍 Output Examples

### Prediction Results
```
🎯 PREDICTION RESULTS
Employee: John Doe
Attrition Probability: 75.32%
Model Prediction: Will likely leave
Risk Level: 🔴 HIGH RISK
Recommended Action: Immediate intervention required
```

### AI Analysis Sample
```markdown
# Attrition Risk Assessment

**Test Employee 157 presents an extremely high attrition probability (95.34%).**  
This necessitates immediate and focused intervention. The scaled data reveals several concerning factors contributing to this high risk. Negative values across age, monthly income, years at company, total experience, and distance from home suggest the employee may be falling below average in these areas compared to their peers, potentially leading to feelings of inadequacy, under-compensation, or lack of career progression.

---

## 🔍 Key Contributing Factors

Several psychological and workplace factors likely contribute to Test Employee 157's high attrition probability:

- **Compensation and Perceived Value**  
  The negative score for monthly income suggests potential dissatisfaction with pay relative to their experience and market value. This can lead to feelings of being undervalued and underappreciated, impacting motivation and engagement.

- **Career Stagnation**  
  Negative scores for years at company and total experience may indicate a lack of career progression opportunities. Employees need to feel they are growing and developing, and a lack of this can lead to boredom, frustration, and a desire to seek opportunities elsewhere.

- **Work-Life Integration**  
  While distance from home is relatively low, a holistic review of work-life balance (including work hours, stress levels, commute time, and overall well-being) is crucial. Even a seemingly manageable distance can become stressful when combined with other factors.

- **Single Marital Status**  
  Research suggests that single employees may experience different levels of support and work-life balance compared to married employees. This is a factor to consider in understanding the employee's overall work experience.

---

## ✅ Recommended Actions

To mitigate the high attrition risk, a multi-pronged approach focusing on addressing the employee's psychological needs and improving their workplace experience is recommended:

1. **Compensation Review**  
   Conduct a thorough review of Test Employee 157's salary, comparing it to market rates for similar roles and experience. A salary adjustment or bonus may be necessary to demonstrate appreciation and improve their sense of value.

2. **Career Development Plan**  
   Create a structured career development plan with clear goals, timelines, and opportunities for skill enhancement and advancement. This should include regular check-ins to monitor progress and address any concerns.

3. **Work-Life Balance Assessment**  
   Conduct a confidential meeting to discuss work-life balance. Explore flexible work arrangements, stress management resources, or other support systems to improve well-being.

4. **Enhanced Communication and Engagement**  
   Regular check-ins (weekly or bi-weekly) to discuss workload, progress, and any concerns. Active listening and open communication are crucial to building trust and fostering a sense of belonging.

5. **Mentorship Program**  
   Pair the employee with a senior colleague who can provide guidance, support, and mentorship, fostering a sense of connection and belonging.

---

## 📅 Timeline for Implementation

- **Week 1:** Initiate a compensation review and schedule a meeting to discuss career development.  
- **Week 2:** Conduct a confidential work-life balance assessment and discuss potential solutions.  
- **Week 3:** Implement the career development plan and introduce the mentorship program.  
- **Ongoing:** Regular check-ins (at least bi-weekly) to monitor progress and address concerns.

---

## 📊 Success Metrics

The effectiveness of the intervention will be measured by:

- **Attrition Probability Reduction:**  
  Track the employee's attrition probability score over the next few months. A significant decrease would indicate success.

- **Employee Engagement Surveys:**  
  Conduct regular employee engagement surveys to gauge the employee's satisfaction, motivation, and overall well-being.

- **Qualitative Feedback:**  
  Regular feedback from the employee, mentor, and manager to understand the impact of the interventions.

- **Performance Improvement:**  
  Monitor the employee’s performance to see if improvements are observed.

---

> This proactive approach, focused on addressing both the psychological and practical aspects of Test Employee 157's situation, will significantly increase the chances of retaining this valuable employee. The key is to demonstrate genuine care and commitment to their well-being and professional growth.

```

## 🛠️ Technical Requirements

### Dependencies
```txt
streamlit>=1.28.0
fastapi>=0.104.1
uvicorn>=0.24.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
langchain>=0.0.350
langchain-google-genai>=0.0.5
python-dotenv>=1.0.0
requests>=2.31.0
```

### System Requirements
- **RAM:** 4GB+ recommended
- **Storage:** 500MB for models and data
- **Network:** Internet connection for LLM API calls
- **OS:** Windows, macOS, Linux supported

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/ main.py streamlit.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Shreyansnh Mishra**
- AI/ML Developer
- GitHub: [@shreyanshmishra2607](https://github.com/shreyanshmishra2607)
- Email: shreyanshmishra2607@gmail.com

## 🙏 Acknowledgments

- **Google Gemini** for advanced LLM capabilities
- **Streamlit** for rapid web app development
- **FastAPI** for high-performance API framework
- **scikit-learn** for robust ML algorithms

## 📞 Support

For support, email shreyanshmishra2607@gmail.com or create an issue on GitHub.

---

<div align="center">

**⭐ Star this repository if it helped you!**

</div>