# Cold Email Generator for Software Services

## Overview
An AI-powered cold email generator built with LLaMA 3.1, specifically designed for software service companies. The system leverages job postings to create personalized outreach emails, integrating ChromaDB for data management and Streamlit for the user interface.

## Features
- Automated job posting analysis
- Personalized email generation
- Skills-based matching system
- Portfolio integration
- User-friendly Streamlit interface
- Cloud-based processing
- Persistent data storage

## Architecture
```
Cold Email Generator
├── Data Processing
│   ├── Job Scraper
│   ├── Skills Extractor
│   └── Portfolio Matcher
├── Database
│   └── ChromaDB Vector Store
├── LLM Integration
│   ├── LLaMA 3.1 Model
│   └── LangChain Pipelines
└── User Interface
    └── Streamlit Application
```

## Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB RAM (minimum)
- Internet connection for API access

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cold-email-generator.git
cd cold-email-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Environment Setup
Create a `.env` file:
```env
LLAMA_MODEL_PATH=path/to/llama/model
CHROMA_DB_PATH=path/to/chromadb
API_KEY=your_api_key_if_needed
```

### Model Configuration
```python
# config/model_config.yaml
llama:
  model_size: "3.1"
  temperature: 0.7
  max_tokens: 500
  
email_generation:
  tone: "professional"
  max_length: 300
  include_portfolio: true
```

## Implementation

### 1. Job Data Extraction
```python
from langchain import JobScraper
from utils.parser import SkillsExtractor

class JobDataProcessor:
    def __init__(self):
        self.scraper = JobScraper()
        self.extractor = SkillsExtractor()
        
    def process_job_posting(self, url):
        # Scrape job posting
        job_data = self.scraper.scrape(url)
        
        # Extract skills
        skills = self.extractor.extract_skills(job_data)
        
        return {
            'job_description': job_data,
            'required_skills': skills
        }
```

### 2. Database Integration
```python
import chromadb
from chromadb.config import Settings

class DatabaseManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./db",
            settings=Settings(
                allow_reset=True,
                is_persistent=True
            )
        )
        
    def store_portfolio(self, skills, portfolio_links):
        collection = self.client.get_or_create_collection("portfolio")
        collection.add(
            documents=portfolio_links,
            metadatas=[{"skill": skill} for skill in skills]
        )
```

### 3. Email Generation
```python
from llama import LlamaModel
from utils.templates import EmailTemplate

class EmailGenerator:
    def __init__(self):
        self.model = LlamaModel()
        self.template = EmailTemplate()
        
    def generate_email(self, job_data, company_portfolio):
        prompt = self.template.create_prompt(
            job_description=job_data['job_description'],
            skills=job_data['required_skills'],
            portfolio=company_portfolio
        )
        
        response = self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=500
        )
        
        return response
```

### 4. Streamlit Interface
```python
import streamlit as st
from email_generator import EmailGenerator

def main():
    st.title("Cold Email Generator")
    
    # Job URL input
    job_url = st.text_input("Enter Job Posting URL")
    
    if st.button("Generate Email"):
        # Process job posting
        processor = JobDataProcessor()
        job_data = processor.process_job_posting(job_url)
        
        # Generate email
        generator = EmailGenerator()
        email = generator.generate_email(job_data)
        
        # Display result
        st.text_area("Generated Email", email, height=300)

if __name__ == "__main__":
    main()
```

## Usage

### Running the Application
```bash
# Start the Streamlit application
streamlit run app.py
```

### API Usage
```python
from cold_email_generator import EmailGenerator

# Initialize generator
generator = EmailGenerator()

# Generate email
email = generator.generate_email(
    job_url="https://example.com/job-posting",
    company_profile={
        "name": "Tech Solutions Inc",
        "portfolio": ["project1", "project2"],
        "expertise": ["python", "machine_learning"]
    }
)
```

## Customization

### Email Templates
```yaml
# templates/email_templates.yaml
introduction:
  - "I noticed your job posting for {position}"
  - "I came across your requirement for {position}"

value_proposition:
  - "Our team has successfully delivered {project_count} similar projects"
  - "We specialize in {skills} with proven success"

closing:
  - "Would you be available for a brief discussion?"
  - "When would be a good time to discuss this further?"
```

## Performance Optimization

### ChromaDB Optimization
- Use batch processing for multiple entries
- Implement caching for frequently accessed data
- Optimize vector dimensions for storage efficiency

### LLaMA Model Optimization
- Use quantization for reduced memory usage
- Implement response caching
- Optimize batch processing for multiple requests

## Troubleshooting

### Common Issues
1. Model Loading Issues
```python
def verify_model():
    try:
        model = LlamaModel()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
```

2. Database Connection Issues
```python
def test_db_connection():
    try:
        client = chromadb.PersistentClient()
        client.heartbeat()
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection failed: {e}")
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments
- LLaMA team for the base model
- ChromaDB developers
- LangChain community
- Streamlit team
