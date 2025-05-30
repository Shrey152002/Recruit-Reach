from flask import Flask, request, jsonify, render_template
import os
import uuid
import pandas as pd
import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
import json
import logging
import datetime
import re
from urllib.parse import urlparse

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    PORTFOLIO_CSV_PATH = os.getenv("PORTFOLIO_CSV_PATH", "my_portfolio (1).csv")
    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore")

# Validate required environment variables
if not Config.GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")
# Initialize the ChatGroq LLM
try:
    llm = ChatGroq(
        temperature=0.1,
        groq_api_key=Config.GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        max_tokens=2048
    )
    logger.info("ChatGroq LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChatGroq LLM: {str(e)}")
    raise

# Global variables for database
client = None
collection = None

def validate_url(url):
    """Validate URL format and accessibility"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\.,\-\+\#]', ' ', text)
    return text.strip()

def initialize_database():
    """Initialize ChromaDB and load portfolio data"""
    global client, collection
    
    try:
        if not os.path.exists(Config.PORTFOLIO_CSV_PATH):
            logger.error(f"Portfolio CSV file not found: {Config.PORTFOLIO_CSV_PATH}")
            return False
            
        df = pd.read_csv(Config.PORTFOLIO_CSV_PATH)
        logger.info(f"Loaded portfolio data with {len(df)} entries")
        
        required_columns = ['Project_Title', 'Techstack', 'Description', 'Links', 'Client_Type', 'Project_Duration']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"CSV must contain columns: {required_columns}")
            return False
        
        client = chromadb.PersistentClient(Config.VECTORSTORE_PATH)
        collection = client.get_or_create_collection(name="portfolio")
        
        if collection.count() == 0:
            logger.info("Populating ChromaDB collection...")
            for idx, row in df.iterrows():
                try:
                    # Create rich document content for better matching
                    techstack = clean_text(str(row["Techstack"]) if pd.notna(row["Techstack"]) else "")
                    description = clean_text(str(row["Description"]) if pd.notna(row["Description"]) else "")
                    project_title = clean_text(str(row["Project_Title"]) if pd.notna(row["Project_Title"]) else "")
                    
                    # Combine all searchable content
                    document_content = f"{project_title} {techstack} {description}".strip()
                    
                    if document_content:
                        metadata = {
                            "project_title": project_title,
                            "techstack": techstack,
                            "description": description,
                            "links": str(row["Links"]) if pd.notna(row["Links"]) else "",
                            "client_type": str(row["Client_Type"]) if pd.notna(row["Client_Type"]) else "",
                            "project_duration": str(row["Project_Duration"]) if pd.notna(row["Project_Duration"]) else ""
                        }
                        
                        collection.add(
                            documents=document_content,
                            metadatas=metadata,
                            ids=[str(uuid.uuid4())]
                        )
                except Exception as e:
                    logger.warning(f"Failed to add row {idx} to collection: {str(e)}")
                    continue
            
            logger.info(f"ChromaDB collection populated with {collection.count()} entries")
        else:
            logger.info(f"ChromaDB collection already contains {collection.count()} entries")
            
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

def extract_job_data(page_content):
    """Extract job data from webpage content with improved prompting"""
    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        
        ### INSTRUCTION:
        You are analyzing a job posting webpage. Extract ALL available job information and return it as a JSON array.
        Each job should include these fields (use "Not specified" if information is missing):
        - company: Company name
        - role: Job title/position
        - experience: Required experience level
        - skills: Array of required technical skills, frameworks, and technologies
        - description: Brief job summary (2-3 sentences)
        - location: Job location if mentioned
        - employment_type: Full-time, Part-time, Contract, etc.
        
        Focus on extracting comprehensive skill requirements including programming languages, frameworks, databases, tools, and methodologies.
        
        Return ONLY valid JSON array format with no additional text or formatting.
        
        ### VALID JSON RESPONSE:
        """
    )
    
    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={'page_data': page_content})
    
    try:
        json_parser = JsonOutputParser()
        json_res = json_parser.parse(res.content)
        
        if not isinstance(json_res, list):
            if isinstance(json_res, dict):
                json_res = [json_res]
            else:
                raise ValueError("Invalid JSON format returned")
        
        # Clean and validate extracted data
        for job in json_res:
            if isinstance(job.get('skills'), str):
                # Convert string skills to array
                skills_str = job['skills']
                job['skills'] = [skill.strip() for skill in re.split(r'[,;|]', skills_str) if skill.strip()]
            elif not isinstance(job.get('skills'), list):
                job['skills'] = []
                
        return json_res
        
    except Exception as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")
        logger.error(f"Raw response: {res.content}")
        raise

def query_portfolio(skills, job_role="", company_type=""):
    """Enhanced portfolio query with better matching"""
    if not collection:
        raise Exception("Database not initialized")
        
    try:
        # Prepare query components
        skill_query = ""
        if isinstance(skills, list):
            valid_skills = [str(skill).strip() for skill in skills if skill and str(skill).strip()]
            skill_query = " ".join(valid_skills)
        elif isinstance(skills, str):
            skill_query = skills.strip()
        
        # Enhanced query combining skills, role, and context
        query_components = [skill_query]
        if job_role:
            query_components.append(clean_text(str(job_role)))
        
        combined_query = " ".join(filter(None, query_components))
        
        if not combined_query.strip():
            logger.warning("No valid query components for portfolio search")
            return []
        
        logger.info(f"Querying portfolio with: {combined_query}")
        
        # Query with higher result count for better selection
        results = collection.query(
            query_texts=[combined_query], 
            n_results=8,
            include=['metadatas', 'documents', 'distances']
        )
        
        portfolio_items = []
        if results and 'metadatas' in results and results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                if isinstance(metadata, dict):
                    distance = results['distances'][0][i] if 'distances' in results else 1.0
                    
                    # Filter results with reasonable similarity threshold
                    if distance < 1.2:  # Adjust threshold as needed
                        item = {
                            "project_title": metadata.get("project_title", ""),
                            "techstack": metadata.get("techstack", ""),
                            "description": metadata.get("description", ""),
                            "link": metadata.get("links", ""),
                            "client_type": metadata.get("client_type", ""),
                            "duration": metadata.get("project_duration", ""),
                            "relevance_score": round(1 - distance, 3)
                        }
                        
                        if item["link"] and item["link"].strip():
                            portfolio_items.append(item)
        
        # Sort by relevance and return top results
        portfolio_items.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.info(f"Found {len(portfolio_items)} relevant portfolio items")
        
        return portfolio_items[:5]  # Return top 5 most relevant
        
    except Exception as e:
        logger.error(f"Failed to query portfolio database: {str(e)}")
        raise

def generate_cold_email(job_data, company_name, portfolio_items):
    """Generate enhanced cold email with better structure"""
    # Prepare portfolio showcase
    portfolio_showcase = ""
    if portfolio_items:
        portfolio_showcase = "\n\nHere are some relevant projects from our portfolio that demonstrate our capabilities:\n"
        for item in portfolio_items:
            portfolio_showcase += f"\n• {item['project_title']} ({item['client_type']})\n"
            portfolio_showcase += f"  Technologies: {item['techstack']}\n"
            portfolio_showcase += f"  {item['description']}\n"
            portfolio_showcase += f"  View Project: {item['link']}\n"
    
    prompt_email = PromptTemplate.from_template(
        """
        ### JOB POSTING DETAILS:
        {job_description}

        ### PORTFOLIO EXAMPLES:
        {portfolio_showcase}

        ### INSTRUCTION:
        You are Shreyash Verma , Senior Business Development Executive at  Macrosoft. 

        Macrosoft is a premier AI & Software Consulting company with 8+ years of experience, specializing in:
        • Custom Software Development & AI Integration
        • Digital Transformation & Process Automation  
        • Cloud Solutions & DevOps Implementation
        • Mobile & Web Application Development
        • Data Analytics & Machine Learning Solutions

        We have successfully delivered 200+ projects across various industries including Healthcare, Finance, E-commerce, and Manufacturing, helping businesses achieve 40% cost reduction and 3x faster time-to-market.

        Write a compelling cold email to {company_name} for the mentioned job opportunity. The email should:
        1. Have an engaging subject line suggestion
        2. Show genuine interest in their specific requirements
        3. Highlight AtliQ's relevant expertise and proven track record
        4. Include 2-3 most relevant portfolio examples that match their tech stack
        5. Present a clear value proposition
        6. Include a professional call-to-action
        7. Maintain a consultative, professional tone

        Format the response as:
        **Subject:** [Subject line]
        
        **Email Body:**
        [Email content]

        Keep it concise (under 250 words) but impactful. Do NOT include generic statements or excessive self-promotion.

        ### PROFESSIONAL EMAIL:
        """
    )
    
    chain_email = prompt_email | llm
    email_res = chain_email.invoke({
        "job_description": str(job_data),
        "portfolio_showcase": portfolio_showcase,
        "company_name": company_name
    })
    
    return email_res.content

@app.route('/')
def index():
    """Render the enhanced main page"""
    return render_template('index.html')

@app.route('/api/validate-url', methods=['POST'])
def validate_url_endpoint():
    """Validate URL before processing"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({"valid": False, "message": "URL is required"})
        
        if not validate_url(url):
            return jsonify({"valid": False, "message": "Invalid URL format"})
        
        return jsonify({"valid": True, "message": "URL is valid"})
        
    except Exception as e:
        return jsonify({"valid": False, "message": f"Validation error: {str(e)}"})

@app.route('/api/portfolio-stats')
def portfolio_stats():
    """Get portfolio statistics"""
    try:
        if not collection:
            return jsonify({"error": "Database not initialized"}), 500
        
        stats = {
            "total_projects": collection.count(),
            "database_status": "healthy",
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process():
    """Enhanced process endpoint with better error handling and responses"""
    try:
        # Validate input
        data = request.get_json() if request.is_json else request.form
        
        if not data or 'link' not in data:
            return jsonify({
                "success": False, 
                "error": "Missing 'link' parameter",
                "error_type": "validation"
            }), 400
            
        input_link = data['link'].strip()
        if not input_link:
            return jsonify({
                "success": False,
                "error": "Link cannot be empty",
                "error_type": "validation"
            }), 400
            
        if not validate_url(input_link):
            return jsonify({
                "success": False,
                "error": "Invalid URL format. Please include http:// or https://",
                "error_type": "validation"
            }), 400
            
        logger.info(f"Processing link: {input_link}")
        
        # Check database status
        if not collection:
            return jsonify({
                "success": False,
                "error": "Portfolio database not initialized",
                "error_type": "system"
            }), 500
        
        # Load webpage content
        try:
            loader = WebBaseLoader(
                input_link,
                header_template={"User-Agent": Config.USER_AGENT}
            )
            pages = loader.load()
            
            if not pages or not pages[0].page_content.strip():
                return jsonify({
                    "success": False,
                    "error": "No content found at the provided URL. The page might be empty or require authentication.",
                    "error_type": "content"
                }), 400
                
            page_content = clean_text(pages[0].page_content)
            
        except Exception as e:
            logger.error(f"Failed to load webpage: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to load the webpage. Please check if the URL is accessible and try again.",
                "error_type": "network"
            }), 400
        
        # Extract job data
        try:
            job_data = extract_job_data(page_content)
            if not job_data:
                return jsonify({
                    "success": False,
                    "error": "No job postings found on this webpage. Please ensure the URL contains job listings.",
                    "error_type": "extraction"
                }), 400
                
        except Exception as e:
            logger.error(f"Failed to extract job data: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to extract job information. The webpage content might not be in a recognizable format.",
                "error_type": "extraction"
            }), 500
        
        # Process first job posting
        first_job = job_data[0]
        skills = first_job.get('skills', [])
        company = first_job.get('company', 'the client')
        role = first_job.get('role', 'the position')
        
        logger.info(f"Extracted job - Company: {company}, Role: {role}, Skills: {skills}")
        
        # Query portfolio
        try:
            portfolio_items = query_portfolio(skills, role, company)
            logger.info(f"Found {len(portfolio_items)} relevant portfolio items")
            
        except Exception as e:
            logger.error(f"Failed to query portfolio: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to find relevant portfolio items",
                "error_type": "portfolio"
            }), 500
        
        # Generate cold email
        try:
            email_content = generate_cold_email(job_data, company, portfolio_items)
            logger.info("Cold email generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate email: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to generate cold email",
                "error_type": "generation"
            }), 500
        
        # Return comprehensive response
        return jsonify({
            "success": True,
            "email_content": email_content,
            "job_details": {
                "company": company,
                "role": role,
                "skills": skills,
                "location": first_job.get('location', 'Not specified'),
                "experience": first_job.get('experience', 'Not specified'),
                "employment_type": first_job.get('employment_type', 'Not specified')
            },
            "portfolio_matches": len(portfolio_items),
            "portfolio_items": portfolio_items,
            "processing_time": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in process endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred. Please try again later.",
            "error_type": "system"
        }), 500

@app.route('/health')
def health_check():
    """Enhanced health check"""
    try:
        db_status = "healthy" if collection else "not initialized"
        collection_count = collection.count() if collection else 0
        
        # Test LLM connectivity
        llm_status = "healthy"
        try:
            test_response = llm.invoke("Test")
            if not test_response:
                llm_status = "unhealthy"
        except:
            llm_status = "unhealthy"
        
        return jsonify({
            "status": "healthy" if db_status == "healthy" and llm_status == "healthy" else "degraded",
            "database": db_status,
            "collection_count": collection_count,
            "llm_status": llm_status,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    if initialize_database():
        logger.info("Application starting with database initialized")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize database. Exiting.")
        exit(1)