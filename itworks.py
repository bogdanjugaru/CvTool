import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from dateutil.relativedelta import relativedelta
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader

# Download NLTK datasets required for tokenization, stopwords, and lemmatization
nltk.download('punkt')      # Tokenization models
nltk.download('stopwords')  # Stopwords list
nltk.download('wordnet')    # WordNet dictionary for lemmatization
nltk.download('omw-1.4')    # WordNet's morphological data for lemmatization

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime
from dateutil.relativedelta import relativedelta
from PyPDF2 import PdfReader

def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        with open(file_path, 'r') as file:
            return file.read()
    elif ext == '.pdf':
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
            return text
    else:
        return None  # Return None if the file type is unsupported

# Function to process the CV file
def process_cv(file_path):
    content = read_file(file_path)  # Read the file content

    if not content:
        # If the content is None or empty, return an error
        return {"error": "File content could not be read or the format is unsupported."}

    # Clean up the content - replace tabs and special characters
    content = content.replace('\t', ' ').replace('·', ' ')

    # Dictionary of words to replace
    replacements = {
        "Studii": "Education",
        "Studies": "Education",
        "Experiența Profesională": "Experience",
        "Licenta": "Bachelor",
        "Masterat": "Masters",
        "Doctorat": "Doctorate",
        "Ianuarie": "January",
        "Februarie": "February",
        "Martie": "March",
        "Aprilie": "April",
        "Mai": "May",
        "Iunie": "June",
        "Iulie": "July",
        "Septembrie": "September",
        "Octombrie": "October",
        "Noiembrie": "November",
        "Decembrie": "December",
        "Prezent": "Present",
        "Elev": "High School",
        "Pagina 1 din 2": "",
        "Pagina 2 din 2": ""
    }

    # Function to perform the replacements
    def replacement_titles(content, replacements):
        for old_word, new_word in replacements.items():
            content = re.sub(re.escape(old_word), new_word, content, flags=re.IGNORECASE)
        return content

    # Call the function
    content = replacement_titles(content, replacements)

    # Define dictionaries with skill sets
    ds_skills = {
        "Data_Science_Reporting": ["python", "r", "sql", "jupyter", "rstudio", "knime", "tableau", "machine learning", "deep learning", "data mining", "nlp", "nltk", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "ggplot2", "tensorflow", "keras", "pytorch", "scikit-learn", "xgboost", "lightgbm", "random forest", "decision trees", "linear regression", "logistic regression", "time series", "clustering", "dimensionality reduction", "feature engineering", "data visualization", "power bi", "excel", "shiny", "dplyr", "tidyverse", "databricks", "apache spark", "bigquery", "sas", "spss", "h2o", "mlflow", "model deployment", "azure ml", "aws sagemaker", "etl", "data wrangling"]
    }

    fe_skills = {
        "Front_End": ["javascript", "html", "css", "typescript", "visual studio code", "webstorm", "react", "angular", "vue", "jquery", "redux", "webpack", "babel", "bootstrap", "sass", "less", "tailwind css", "material ui", "npm", "yarn", "gulp", "vite", "storybook", "rxjs", "d3.js", "three.js", "next.js", "nuxt.js", "svelte", "ember.js", "gridsome", "pwa", "graphql", "apollo", "enzyme", "jest", "cypress", "protractor", "eslint", "prettier", "responsive design", "cross-browser compatibility", "ui/ux design", "web accessibility", "web components", "service workers", "webassembly", "dom manipulation", "ajax", "http", "rest api", "graphql", "fetch api", "axios"]
    }

    be_skills = {
        "Back_End": ["python", "java", "node.js", "php", "ruby", "c#", "intellij idea", "pycharm", "eclipse", "visual studio", "golang", "scala", "django", "flask", "spring", "laravel", "dotnet", "graphql", "sql", "nosql", "mongodb", "mysql", "postgresql", "redis", "cassandra", "elasticsearch", "neo4j", "firebase", "aws", "azure", "google cloud", "docker", "kubernetes", "microservices", "grpc", "rest api", "soap", "jwt", "oauth", "authentication", "authorization", "message queues", "rabbitmq", "kafka", "unit testing", "integration testing", "tdd", "bdd", "ci/cd pipelines", "log management", "caching", "session management"]
    }

    test_skills = {
        "Testing": ["python", "java", "selenium", "cypress", "jest", "mocha", "chai", "pytest", "junit", "testng", "xunit", "visual studio", "eclipse", "appium", "soapui", "postman", "rest-assured", "loadrunner", "jmeter", "gatling", "cucumber", "bdd", "tdd", "unit testing", "integration testing", "system testing", "uat", "manual testing", "automation testing", "performance testing", "load testing", "stress testing", "security testing", "api testing", "mobile testing", "cross-browser testing", "a/b testing", "test management tools", "defect tracking", "quality assurance", "continuous testing", "test case design", "test data management", "test environment setup", "test strategy", "test automation frameworks", "ui testing", "regression testing", "smoke testing"]
    }

    de_skills = {
        "Data_Engineering": ["python", "java", "scala", "sql", "r", "intellij idea", "pycharm", "datagrip", "kafka", "hadoop", "spark", "apache beam", "bigquery", "aws glue", "azure data factory", "databricks", "docker", "kubernetes", "airflow", "luigi", "etl", "data warehousing", "data lakes", "nosql", "mongodb", "cassandra", "redis", "apache storm", "apache flink", "presto", "hive", "pig", "snowflake", "redshift", "azure synapse", "gcp bigquery", "cloud dataflow", "data pipelines", "data governance", "data modeling", "data partitioning", "data migration", "orchestration tools", "stream processing", "batch processing"]
    }

    soft_skills = {
        "Soft_skills": [
            "communication", "teamwork", "adaptability", "problem solving", "emotional intelligence", "leadership",
            "time management", "conflict resolution", "critical thinking", "creativity", "active listening", "collaboration",
            "dependability", "flexibility", "interpersonal skills", "positive attitude", "initiative", "stress management",
            "work ethic", "decision making", "persuasion", "self motivation", "patience", "networking", "goal setting",
            "relationship building", "public speaking", "self awareness", "accountability", "open mindedness", "resourcefulness",
            "negotiation", "mentoring", "feedback delivery", "cultural awareness", "visionary thinking", "trustworthiness",
            "influence"
        ]
    }

    # Proceed with tokenization and cleaning
    content_tokens = word_tokenize(content)

    # Lowercase data
    content_tokens = [x.lower() for x in content_tokens]

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    content_tokens_v2 = [w for w in content_tokens if w not in stop_words]

    # Remove punctuations and duplicates
    content_tokens_v2 = [x for x in content_tokens_v2 if x.isalpha()]
    content_tokens_v2 = list(set(content_tokens_v2))

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    content_tokens_v3 = [lemmatizer.lemmatize(x, pos="n") for x in content_tokens_v2]

    # Classify each skill according to dictionary
    matched_skills = {
        "Data_Science_Reporting": [],
        "Front_End": [],
        "Back_End": [],
        "Testing": [],
        "Data_Engineering": [],
        "Soft_skills": []
    }

    for w in content_tokens_v3:
        # Check each word against the original dictionaries and add to the matched_skills dictionary
        if w in ds_skills["Data_Science_Reporting"]:
            matched_skills["Data_Science_Reporting"].append(w)
        if w in fe_skills["Front_End"]:
            matched_skills["Front_End"].append(w)
        if w in be_skills["Back_End"]:
            matched_skills["Back_End"].append(w)
        if w in test_skills["Testing"]:
            matched_skills["Testing"].append(w)
        if w in de_skills["Data_Engineering"]:
            matched_skills["Data_Engineering"].append(w)
        if w in soft_skills["Soft_skills"]:
            matched_skills["Soft_skills"].append(w)

    # Sort the matched_skills dictionary by the length of its lists in descending order
    sorted_matched_skills = {k: v for k, v in sorted(matched_skills.items(), key=lambda item: len(item[1]), reverse=True) if len(v) > 0}

    # Extract the labels (skills) and the sizes (length of each list)
    labels = list(sorted_matched_skills.keys())
    sizes = [len(skill_list) for skill_list in sorted_matched_skills.values()]

    # Reverse labels and sizes to display bars in descending order (top to bottom)
    labels.reverse()
    sizes.reverse()
    labels = [label.replace("_", " ") for label in sorted_matched_skills.keys()]

    # Create the horizontal bar chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size
    bars = plt.barh(labels, sizes, color='skyblue')

    # Add labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, str(width), ha='left', va='center')

    # Remove x-axis label
    plt.gca().axes.xaxis.set_visible(False)

    # Adjust font sizes for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Number of Matches per Skill', fontsize=18)

    # Ensure everything fits without overlapping
    plt.tight_layout()

    # Save the bar chart as an image in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    bar_chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    # Email extraction
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9_.+-]+\.[a-zA-Z]+"
    matched_email = re.search(pattern, content, re.IGNORECASE)
    email = matched_email.group() if matched_email else "No email found"

    # Education extraction
    education_ranking = {
        "Doctorate": 4,
        "Masters": 3,
        "Bachelor": 2,
        "College": 1,
        "High School": 0
    }

    # Extract the education segment
    education_pattern = re.compile(r'Education(.*?)(?=Skills|Experience|$)', re.DOTALL | re.IGNORECASE)
    matches = education_pattern.findall(content)

    education_entries = []
    degree_pattern = re.compile(r'\b(Doctorate|Masters|Bachelor|College|High School)\b', re.IGNORECASE)

    for match in matches:
        education_content = match.strip()

        # Split the content by new lines
        entries = [entry.strip() for entry in education_content.split('\n') if entry.strip()]
        
        # Filter entries that match degrees
        for entry in entries:
            if degree_pattern.search(entry):
                # Extract only the degree title
                degree_title = degree_pattern.search(entry).group(0)
                # Append the entry containing only the degree and the relevant part
                simplified_entry = entry.split(":")[1].strip() if ":" in entry else entry.strip()
                education_entries.append(f"{degree_title}: {simplified_entry}")

    # Sort the education entries based on the ranking
    def get_degree_rank(education_entry):
        highest_rank = -1
        for degree, rank in education_ranking.items():
            if degree.lower() in education_entry.lower():
                highest_rank = max(highest_rank, rank)
        return highest_rank  # Return the highest rank found

    sorted_education = sorted(education_entries, key=get_degree_rank, reverse=True)

    # Output the highest-ranked degree content
    highest_degree_info = ""
    if sorted_education:
        highest_degree_entry = sorted_education[0]
        degree_title, degree_info = highest_degree_entry.split(":")
        highest_degree_info = degree_info.strip()

    # Experience extraction
    months_dict = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }

    # Join month names into a regex pattern
    months = '|'.join(months_dict.values())

    # Compile the regex pattern for job titles and periods
    job_title_period_pattern = re.compile(
        rf'(?P<job_title>[\w\s-]+?)\s*(?:-|,)?\s*({months})\s+(?P<start_year>\d{{4}})\s*(?:to|-\s*)\s*(?:(?P<end_month>{months})\s+(?P<end_year>\d{{4}})|(?P<ongoing>current|present))',
        re.IGNORECASE
    )

    # Find matches
    matches = job_title_period_pattern.findall(content)

    job_data = []  # To store data for the DataFrame

    # Loop through matches and process job titles and dates
    for match in matches:
        job_title = match[0]  # Ensure this corresponds to your regex captures
        start_month = match[1]
        start_year = match[2]
        end_month = match[3]
        end_year = match[4]
        ongoing_status = match[5]

        # Handle ongoing jobs
        if ongoing_status:  # If "current" or "present"
            end_date = datetime.now()  # Ongoing job, so use current date
        else:
            end_date = datetime.strptime(f'{end_month} {end_year}', '%B %Y') if end_year else datetime.now()

        # Start date using English month
        start_date = datetime.strptime(f'{start_month} {start_year}', '%B %Y')

        # Calculate job duration in years
        job_duration_years = (end_date - start_date).days / 365.25  # Convert to years

        # Store job info in a dictionary for pandas DataFrame
        job_data.append({
            'Job Title': job_title.strip(),
            'Start Date': start_date.strftime('%B %Y'),
            'End Date': end_date.strftime('%B %Y') if end_date else 'N/A',
            'Duration (Years)': round(job_duration_years, 2)
        })

    # Calculate total job experience
    total_experience = sum(job['Duration (Years)'] for job in job_data)  # Sum all the job durations
    jobs_number = len(job_data)  # Get the number of jobs
    average_duration = total_experience / jobs_number if jobs_number > 0 else 0  # Avoid division by zero

    # Create a pie chart of job durations
    plt.figure(figsize=(8, 8))
    plt.title(f"Average time spent at a job: {average_duration:.2f} years", fontsize=20)
    _, texts, autotexts = plt.pie(
        [job['Duration (Years)'] for job in job_data],
        labels=[job['Job Title'] for job in job_data],
        autopct='%1.1f%%',
        startangle=140
    )
    for text in texts:
        text.set_fontsize(12)  # Adjust label font size
    for autotext in autotexts:
        autotext.set_fontsize(10)

    # Save the pie chart as an image in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    pie_chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    # Prepare the insights dictionary to return
    insights = {
        "message": "CV processed successfully",
        "email": email,
        "highest_degree": highest_degree_info,
        "skills": sorted_matched_skills,
        "total_experience_years": round(total_experience, 2),
        "average_job_duration_years": round(average_duration, 2),
        "skills_bar_chart": f"data:image/png;base64,{bar_chart_base64}",
        "experience_pie_chart": f"data:image/png;base64,{pie_chart_base64}"
    }

    return insights