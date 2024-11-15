from flask import Flask, request, jsonify, render_template
import os
import uuid
import pandas as pd
import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

app = Flask(__name__)

# Set environment variables
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "**"

# Set USER_AGENT environment variable
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Initialize the ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile",
    max_tokens=1024
)

# Load portfolio data and initialize ChromaDB
try:
    df = pd.read_csv("C:/Users/Shreyash Verma/Desktop/aryan/my_portfolio (1).csv")
    client = chromadb.PersistentClient('vectorstore')
    collection = client.get_or_create_collection(name="portfolio")

    if not collection.count():
        for _, row in df.iterrows():
            collection.add(
                documents=row["Techstack"],
                metadatas={"links": row["Links"]},
                ids=[str(uuid.uuid4())]
            )
except Exception as e:
    print(f"Error initializing database: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        input_link = request.form['link']

        # Configure WebBaseLoader with custom user agent
        loader = WebBaseLoader(
            input_link,
            header_template={"User-Agent": os.getenv("USER_AGENT")}
        )
        page_data = loader.load().pop().page_content

        # Define the prompt to extract job-related data
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the
            following keys: `company`, `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        # Use the chain to extract job data
        chain_extract = prompt_extract | llm
        res = chain_extract.invoke(input={'page_data': page_data})

        # Parse the JSON response
        try:
            json_parser = JsonOutputParser()
            json_res = json_parser.parse(res.content)
        except Exception as e:
            return jsonify({"error": f"Failed to parse JSON response: {str(e)}"})

        # Extract the first job description, skills, and company name
        if not json_res or not isinstance(json_res, list):
            return jsonify({"error": "No valid job postings found"})

        first_job = json_res[0]
        skills = first_job.get('skills', [])
        company = first_job.get('company', 'the client')

        # Query ChromaDB for relevant portfolio links
        try:
            results = collection.query(query_texts=[", ".join(skills)], n_results=5)
            links = [result_metadata.get("links") for result_metadata in results['metadatas'][0]]
        except Exception as e:
            return jsonify({"error": f"Failed to query portfolio database: {str(e)}"})

        # Generate the cold email
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools.
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability,
            process optimization, cost reduction, and heightened overall efficiency.
            Your job is to write a cold email to {company_name} regarding the job mentioned above describing the capability of AtliQ
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
            Remember you are Mohan, BDE at AtliQ.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | llm
        email_res = chain_email.invoke({
            "job_description": str(json_res),
            "link_list": links,
            "company_name": company
        })

        return jsonify({"content": email_res.content})

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
