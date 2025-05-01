import os
import re
import google.generativeai as genai
from typing import List
import json
import asyncio
import phonenumbers


def append_to_json(filename, data):
    """Safely appends each dictionary from a list to a JSON file."""
    if not isinstance(data, list):
        data = [data]  # Ensure data is always a list of dictionaries

    if os.path.exists(filename):
        with open(filename, 'r+') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []  # Handle cases where the file isn't a list
            except json.JSONDecodeError:
                existing_data = []  # Handle empty file case

            existing_data.extend(data)  # Append individual dicts

            f.seek(0)
            json.dump(existing_data, f, indent=4)
            f.truncate()  # Remove any trailing data if file was longer
    else:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)  # Write the list directly


def preprocess_resume_text(text):
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)  # remove emails
    text = re.sub(r'http\S+|www\S+', ' ', text)  # remove URLs
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text.strip()


def extract_skills(skill_extractor, text, threshold=0.7):
    clean_text = preprocess_resume_text(text)

    """Extract skills and categorize them into full and high-confidence partial matches."""
    annotations = skill_extractor.annotate(clean_text)

    full_matches = [skill['doc_node_value']
                    for skill in annotations["results"].get("full_matches", [])]

    partial_matches = [
        skill['doc_node_value']
        for skill in annotations["results"].get("ngram_scored", [])
        # Only include high-confidence matches
        if skill.get('score', 0) >= threshold
    ]

    # print(full_matches,partial_matches)

    BLACKLIST = {"com", "www", "e"}

    # Combine both lists and convert to lowercase for uniformity
    all_skills = set(full_matches + partial_matches)  # Remove duplicates
    # print(all_skills)
    # Filtered skills list
    final_skills = [skill.title() for skill in all_skills if skill.lower()
                    not in BLACKLIST]

    return sorted(final_skills)  # Return sorted list for consistency


def extract_phone_number(text):
    """
    Extract phone number using multiple strategies

    Args:
        text (str): Cleaned resume text

    Returns:
        str or None: Extracted phone number
    """
    # Phone number patterns
    phone_patterns = [
        # US phone formats
        r'\b(?:\+?1[-.\s]?)?(\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b',
        r'\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b',  # Alternative format
    ]

    # Try regex patterns first
    for pattern in phone_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    # Fallback to phonenumbers library
    try:
        for match in phonenumbers.PhoneNumberMatcher(text, "US"):
            if match:
                return phonenumbers.format_number(
                    match.number,
                    phonenumbers.PhoneNumberFormat.NATIONAL
                )
    except:
        pass

    return None


def extract_email(text):
    """
    Extract email address from text

    Args:
        text (str): Cleaned resume text

    Returns:
        str or None: Extracted email
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text, re.IGNORECASE)
    # print(match.group(0))
    return match.group(0) if match else None


#We do this by gemini ai to imrove accuracy
# def extract_name(nlp, resume_text):
#     """
#     Extract name using spaCy Matcher with NER validation.

#     Args:
#         resume_text (str): Text of the resume

#     Returns:
#         str or None: Extracted name
#     """
#     # Create a Matcher object
#     matcher = Matcher(nlp.vocab)

#     # Define name patterns
#     EXCLUDE_WORDS = {"Email", "Address", "Contact", "Phone", "Location"}

#     patterns = [
#         [{'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First Last
#         [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {
#             'POS': 'PROPN'}],  # First Middle Last
#         [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'},
#             {'POS': 'PROPN'}]  # First Middle Middle Last
#     ]

#     # Add patterns to matcher
#     matcher.add('NAME', patterns)

#     # Process the text
#     doc = nlp(resume_text)

#     # Find matches
#     matches = matcher(doc)

#     # Validate matches with NER
#     for match_id, start, end in matches:
#         span = doc[start:end]  # Get matched name

#         # Filter out unwanted words
#         if any(word.text in EXCLUDE_WORDS for word in span):
#             continue  # Skip this match

#         ner_doc = nlp(span.text)  # Apply NER on matched text

#         # Check if it contains a PERSON entity
#         if any(ent.label_ == "PERSON" for ent in ner_doc.ents):
#             # print(span.text)
#             return span.text  # Return first valid name

#     return None  # No valid name found


def parse_resume(resume_data_list, skill_extractor, gem_data_list):
    """Parses multiple resumes' other details using Gemini data."""

    all_combined_data = []
    # print(gem_data_list)
    # Process each resume's "other" data
    for data, gem_data in zip(resume_data_list, gem_data_list):
        # print(data)
        other_data = {
            # 'name': name,
            'email': extract_email(data),
            'phone': extract_phone_number(data),
            'skills': extract_skills(skill_extractor, data, 0.7),
            # 'references': []
        }

        # print(other_data)
        # Combine extracted data
        combined_data = {**other_data, **gem_data}
        # print(combined_data)

        # Save extracted data
        # append_to_json('gemini_data.json', gem_data)
        # append_to_json('other_data.json', other_data)
        # append_to_json('combined_data.json', combined_data)

        all_combined_data.append(combined_data)

    return all_combined_data


def get_resume_text(paths, layout):
    resume_list = []

    for doc in layout.pipe(paths):
        # edu, exp, ref, other = extract_section(
        #     doc)  # Extract structured sections
        resume_list.append(doc.text)
    # print(edu, exp, ref, other)
    return resume_list


async def parse_resume_edu_exp(resume_text: str) -> dict:
    """
    Parses a single resume and extracts structured information using Gemini AI.
    """
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    prompt = f"""
    Extract structured information from the following resume.
    Return the data in **valid JSON format** with these fields:

    - **name** (string): Full name of the candidate. If not available, return "N/A".

    - **education** (list of dictionaries):  
    - **degree** (string): Degree name.  
    - **location** (string): Name of the university/college/school.  
    - **result** (string): Percentage, GPA, CGPA, SGPA, or SPI (if available), otherwise "N/A".  

    - **experience** (list of dictionaries):  
    - **job_title** (string): Job role/title.  
    - **company** (string): Company name.  
    - **duration** (string): Start and end dates (or "current" if still employed).  

    - **references** (list of dictionaries)  If unavailable, return "N/A":  
    - **name** (string): Reference person full name.  
    - **designation** (string): Reference person job title or position.  
    - **contact** (string): Contact details (email or phone number). If unavailable, return "N/A".  

    Ensure **consistent formatting** for all responses. If any field is missing in the resume, return "N/A". 

    ### Resume Text:
    {resume_text}
    """

    try:
        # 🔥 Run Gemini API call in a separate thread (to avoid blocking)
        response = await asyncio.to_thread(model.generate_content, prompt)

        # Extract response text
        response_text = response.text if response else ""

        # 🔹 Clean JSON output using regex (handles cases like ```json ... ```)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        clean_json = json_match.group(0) if json_match else "{}"

        # Convert JSON response
        structured_data = json.loads(clean_json)

    except json.JSONDecodeError:
        structured_data = {"error": "Invalid JSON response from Gemini."}
    except Exception as e:
        structured_data = {"error": str(e)}

    # ✅ Ensure all fields exist (default values)
    default_keys = ["education", "experience"]
    for key in default_keys:
        structured_data.setdefault(key, [])

    return structured_data


async def parse_multiple_resumes_gemini(resumes: List[str]) -> List[dict]:
    """
    Parses multiple resumes asynchronously.
    """
    tasks = [parse_resume_edu_exp(resume) for resume in resumes]
    return await asyncio.gather(*tasks)


async def process_resumes_gemini(resume_texts):
    parsed_resumes = await parse_multiple_resumes_gemini(resume_texts)
    return parsed_resumes
