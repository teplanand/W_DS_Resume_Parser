import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import fuzz
import re
from datetime import datetime


def get_search_results(search_query):
    """Search Wikipedia and fetch the top article summary."""
    endpoint = f"https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1&redirects=1&srprop=size&origin=*&srsearch={search_query}"
    response = requests.get(endpoint)
    data = response.json()
    results = data.get("query", {}).get("search", [])
    # print(results)
    if results:
        title = results[0].get("title", "")
        return get_summary(title)
    return None


def get_summary(title):
    """Fetch the first 5 sentences from a Wikipedia article."""
    endpoint = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exsentences=20&explaintext=&origin=*&titles={title}"
    response = requests.get(endpoint)
    data = response.json()
    results = data.get("query", {}).get("pages", {})
    for result in results.values():
        return result.get("extract", "")
    return None


def extract_skills_from_text(text, skill_ex_obj):
    """Extract skills using SkillNER."""
    annotations = skill_ex_obj.annotate(text)
    return [skill['doc_node_value'] for skill in annotations["results"]["full_matches"]]


def expand_jd_skills(jd_skills, skill_ex_obj):
    """Expand JD skills using Wikipedia summaries and map them to their parents."""
    expanded_skills = {}  # Dictionary to store {expanded_skill: parent_skill}

    for skill in jd_skills:
        wiki_summary = get_search_results(f"{skill}")

        if wiki_summary:
            extracted_skills = extract_skills_from_text(
                wiki_summary, skill_ex_obj)

            for expanded_skill in extracted_skills:
                # Ensure proper case formatting and avoid duplicates
                expanded_skill = expanded_skill.lower()
                if expanded_skill not in jd_skills:
                    # Directly map to its parent
                    expanded_skills[expanded_skill] = skill

    return expanded_skills


def compute_tfidf_match_with_breakdown(resumes, jd_skills, skill_ex_obj):
    """
    Compute TF-IDF cosine similarity and determine skill contributions grouped by parent skills.
    Normalize contributions using the total JD match score.
    """
    jd_skills = [skill.title() for skill in jd_skills]

    # Get expanded skills mapped to parents {expanded_skill: parent_skill}
    expanded_skill_map = expand_jd_skills(jd_skills, skill_ex_obj)
    expanded_jd_skills = list(expanded_skill_map.keys())

    print("Original JD Skills:", jd_skills)
    print("Expanded JD Skills:", expanded_jd_skills)

    final_jd_skills = " ".join(jd_skills + expanded_jd_skills)

    texts = [final_jd_skills] + [
        " ".join([skill.lower() for skill in resume.get("skills", [])]) for resume in resumes
    ]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words=None,
                                 max_df=1.0, min_df=1, sublinear_tf=True)

    tfidf_matrix = vectorizer.fit_transform(texts)

    feature_names = vectorizer.get_feature_names_out()
    print("TF-IDF Features:", feature_names)

    jd_vector = tfidf_matrix[0].toarray().flatten()  # JD skill vector
    resume_vectors = tfidf_matrix[1:].toarray()  # Resume skill vectors
    scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()

    skill_contributions = []

    for idx, resume_vector in enumerate(resume_vectors):
        raw_skill_weights = {}

        # Compute raw skill weights
        for i, skill in enumerate(feature_names):
            if jd_vector[i] > 0 and resume_vector[i] > 0:
                raw_skill_weights[skill] = resume_vector[i]

        parent_skill_weights = {}

        for skill, weight in raw_skill_weights.items():
            # Check direct matches with JD skills
            if skill in jd_skills:
                parent_skill_weights[skill] = parent_skill_weights.get(
                    skill, 0) + weight
                continue

            # Check direct matches with expanded skills
            if skill in expanded_skill_map:
                parent = expanded_skill_map[skill]
                parent_skill_weights[parent] = parent_skill_weights.get(
                    parent, 0) + weight
                continue

            # Attempt substring matching for additional mapping
            matched_parent = None
            for parent in jd_skills:
                parent_lower = parent.lower()
                skill_lower = skill.lower()

                # Bigram matching (e.g., "machine learn" → "Machine Learning")
                if ' ' in skill and parent_lower.startswith(skill_lower.split()[0]):
                    matched_parent = parent
                    break

                # Partial containment (e.g., "Python" in "Python Development")
                elif skill_lower in parent_lower or parent_lower in skill_lower:
                    matched_parent = parent
                    break

            if matched_parent:
                parent_skill_weights[matched_parent] = parent_skill_weights.get(
                    matched_parent, 0) + weight

        # print(f"Resume {idx} Parent Skills:", parent_skill_weights)

        # Normalize skill weights using the total JD match score
        total_match_score = scores[idx] * 100  # Convert to percentage
        total_skill_weight = sum(parent_skill_weights.values())

        if total_skill_weight > 0:
            parent_skill_weights = {k: round(
                (v / total_skill_weight) * 100, 2) for k, v in parent_skill_weights.items()}

        # Ensure all JD skills appear in the output
        for skill in jd_skills:
            if skill not in parent_skill_weights:
                parent_skill_weights[skill] = 0.0

        skill_contributions.append({
            "resume_index": idx,
            "resume_name": resumes[idx]["name"],
            "score": round(total_match_score, 2),
            "skill_breakdown": parent_skill_weights
        })

    return skill_contributions


def normalize_scores(scores):
    """Normalize scores to a 0-100 scale using Min-Max Scaling."""
    if len(scores) == 0:
        return scores  # Avoid division by zero

    scaler = MinMaxScaler(feature_range=(0, 100))
    scores = np.array(scores).reshape(-1, 1)  # Convert to 2D array
    return scaler.fit_transform(scores).flatten()  # Return normalized 1D array


def normalize_education_result(education_list):
    """
    Extracts and normalizes the result from education entries.
    Handles various GPA scales with different delimiters.

    Args:
        education_list: List of education entries from resume

    Returns:
        float: Normalized score between 0 and 1
    """
    # print(education_list)
    if not education_list:
        return 0.0

    # Assume the first entry is the most recent/highest education
    latest_education = education_list[0]
    # print(latest_education)
    result_str = latest_education.get("result", "")

    # If no result is provided
    if not result_str or result_str == 'N/A':
        return 0.0

    # Extract numeric values and identify the scale

    # Extract numbers from the result string
    numbers = re.findall(r"(\d+\.?\d*)", result_str)
    if not numbers:
        return 0.0

    raw_score = float(numbers[0])

    # Normalize based on identified scale
    if "cgpa" in result_str.lower() or "gpa" in result_str.lower():
        # Check for explicitly mentioned scale with various delimiters

        # Pattern 1: "X / Y" format
        scale_match = re.search(r"\/\s*(\d+\.?\d*)", result_str)

        # Pattern 2: "X out of Y" format
        if not scale_match:
            scale_match = re.search(
                r"out\s+of\s+(\d+\.?\d*)", result_str, re.IGNORECASE)

        # Pattern 3: "X on a Y scale" or "X on Y scale" format
        if not scale_match:
            scale_match = re.search(
                r"on\s+(?:a\s+)?(\d+\.?\d*)(?:\s*\-?point)?\s+scale", result_str, re.IGNORECASE)

        # Pattern 4: "X:Y" or "X - Y" format (where Y is the scale)
        if not scale_match:
            scale_match = re.search(r"[:;-]\s*(\d+\.?\d*)", result_str)

        # Pattern 5: "score of X/Y" format
        if not scale_match:
            scale_match = re.search(
                r"score\s+of\s+\d+\.?\d*\s*\/\s*(\d+\.?\d*)", result_str, re.IGNORECASE)

        # If scale is explicitly mentioned in any format
        if scale_match:
            scale = float(scale_match.group(1))
            return min(raw_score / scale, 1.0)

        # Check for common scale indicators
        elif "4.0" in result_str or "4-point" in result_str.lower() or "4 point" in result_str.lower():
            return min(raw_score / 4.0, 1.0)

        elif "10.0" in result_str or "10-point" in result_str.lower() or "10 point" in result_str.lower():
            return min(raw_score / 10.0, 1.0)

        # Many Indian universities use 7.0 or 7-point scale
        elif "7.0" in result_str or "7-point" in result_str.lower() or "7 point" in result_str.lower():
            return min(raw_score / 7.0, 1.0)

        # If a second number exists, it might be the scale (like "8.5/10")
        elif len(numbers) >= 2:
            potential_scale = float(numbers[1])
            # Common scale values are 4, 5, 7, 10, 100
            if potential_scale in [4.0, 5.0, 7.0, 10.0, 100.0]:
                return min(raw_score / potential_scale, 1.0)

        # Infer scale based on the value
        else:
            # Likely on 10 scale (Indian/European)
            if raw_score > 4.5 and raw_score <= 10.0:
                return min(raw_score / 10.0, 1.0)
            # Likely on 5 scale (some European)
            elif raw_score > 4.0 and raw_score <= 5.0:
                return min(raw_score / 5.0, 1.0)
            # Likely on 4 scale (US)
            elif raw_score > 0.0 and raw_score <= 4.0:
                return min(raw_score / 4.0, 1.0)
            elif raw_score > 10.0 and raw_score <= 20.0:  # French 20-point scale
                return min(raw_score / 20.0, 1.0)

    # Percentage scores
    elif "%" in result_str or "percent" in result_str.lower() or (raw_score > 20 and raw_score <= 100):
        return min(raw_score / 100.0, 1.0)

    # Grade-based results with extended international grading
    elif any(grade in result_str.upper() for grade in ["A", "B", "C", "D", "E", "F", "O", "S", "P"]):
        # International grade mapping
        grade_map = {
            # US/UK style grades
            "A+": 1.0, "A": 0.95, "A-": 0.9,
            "B+": 0.85, "B": 0.8, "B-": 0.75,
            "C+": 0.7, "C": 0.65, "C-": 0.6,
            "D+": 0.55, "D": 0.5, "D-": 0.45,
            "F": 0.0,

            # Indian grading (O-Outstanding, A-Excellent, B-Good, etc.)
            "O": 1.0,
            "S": 0.9,  # Superior
            "E": 0.85,  # Excellent (also used in some European systems)
            "P": 0.4   # Pass
        }

        for grade, score in grade_map.items():
            if grade in result_str.upper():
                return score

        # If only found generic grade (like just "A" without + or -)
        if "A" in result_str.upper():
            return 0.95
        if "B" in result_str.upper():
            return 0.8
        if "C" in result_str.upper():
            return 0.65
        if "D" in result_str.upper():
            return 0.5
        if "E" in result_str.upper():
            return 0.4
        if "F" in result_str.upper():
            return 0.0

    # Handle other specialized scoring systems
    elif "distinction" in result_str.lower() or "honors" in result_str.lower() or "honours" in result_str.lower():
        return 0.9
    elif "merit" in result_str.lower() or "credit" in result_str.lower():
        return 0.75
    elif "pass" in result_str.lower() or "satisfactory" in result_str.lower():
        return 0.6
    elif "fail" in result_str.lower() or "unsatisfactory" in result_str.lower():
        return 0.0

    # Default normalization when system is unclear
    if raw_score > 20 and raw_score <= 100:
        return min(raw_score / 100.0, 1.0)
    elif raw_score > 10 and raw_score <= 20:
        return min(raw_score / 20.0, 1.0)  # French 20-point scale
    elif raw_score > 4.5 and raw_score <= 10:
        return min(raw_score / 10.0, 1.0)
    elif raw_score > 4.0 and raw_score <= 4.5:
        # This is a tricky range - could be a high US GPA or low 10-scale
        # Default to 5.0 scale as a middle ground
        return min(raw_score / 5.0, 1.0)
    else:
        return min(raw_score / 4.0, 1.0)


def get_candidate_degree_scores(education, required_degrees):
    """
    Score candidates based on their best degree match with required degrees.

    Args:
        resume_data: List of parsed resume dictionaries
        required_degrees: List of required degree strings

    Returns:
        List of dictionaries with candidate info and their best match score
    """

    if not education or len(education) == 0:
        return 0

    best_match_score = 0

    for edu in education:
        candidate_degree = edu.get("degree", "")
        # print(candidate_degree)

        for req_degree in required_degrees:
            # Calculate match score
            match_score = fuzz.token_sort_ratio(
                candidate_degree.lower(), req_degree.lower())

            # Update if this is the best match so far
            if match_score > best_match_score:
                best_match_score = match_score
                # best_match_degree = candidate_degree
                # best_match_requirement = req_degree

    return best_match_score


def calculate_total_experience(experience_list):
    """
    Calculate total years of experience from resume experience section.
    Handles various date formats including numeric MM/YYYY and 'present/now/current' references.

    Args:
        experience_list: List of dictionaries containing job experiences

    Returns:
        float: Total years of experience
    """
    if not experience_list:
        return 0

    total_months = 0
    current_date = datetime.now()
    current_indicators = ['present', 'now', 'current',
                          'today', 'ongoing', 'to date', 'currently']

    for job in experience_list:
        duration = job.get('duration', 'N/A')
        if duration == 'N/A':
            continue

        # Handle MM/YYYY - MM/YYYY format
        numeric_month_pattern = r'(\d{1,2})/(\d{4})\s*-\s*(\d{1,2})/(\d{4})'
        numeric_match = re.search(numeric_month_pattern, duration)
        if numeric_match:
            start_m, start_y, end_m, end_y = map(int, numeric_match.groups())
            try:
                start_date = datetime(start_y, start_m, 1)
                end_date = datetime(end_y, end_m, 1)
                months = (end_date.year - start_date.year) * \
                    12 + (end_date.month - start_date.month)
                total_months += max(0, months)
            except ValueError:
                continue
            continue

        # Handle formats like "May 2023 - June 2024"
        month_year_pattern = r'(\w+)\s+(\d{4})\s*-\s*(\w+)(?:\s+(\d{4})|(?:\s+|$))'
        month_match = re.search(month_year_pattern, duration, re.IGNORECASE)
        if month_match:
            start_month, start_year, end_month, end_year = month_match.groups()
            try:
                start_date = datetime.strptime(
                    f"1 {start_month} {start_year}", "%d %B %Y")
            except ValueError:
                try:
                    start_date = datetime.strptime(
                        f"1 {start_month} {start_year}", "%d %b %Y")
                except ValueError:
                    continue

            if end_year:
                try:
                    end_date = datetime.strptime(
                        f"1 {end_month} {end_year}", "%d %B %Y")
                except ValueError:
                    try:
                        end_date = datetime.strptime(
                            f"1 {end_month} {end_year}", "%d %b %Y")
                    except ValueError:
                        end_date = current_date
            else:
                if end_month.lower() in [ci.lower() for ci in current_indicators]:
                    end_date = current_date
                else:
                    try:
                        # just to check
                        datetime.strptime(f"1 {end_month} 2000", "%d %B %Y")
                        end_date = current_date
                    except ValueError:
                        end_date = current_date

            months = (end_date.year - start_date.year) * \
                12 + (end_date.month - start_date.month)
            total_months += max(0, months)
            continue

        # Fallback pattern: handles "2020 - Present", "2021 - 2022", "Jan. 2022 - now", etc.
        date_pattern = r'(\w+\.?\s*\d{4}|\d{4})\s*-\s*(\w+\.?\s*\d{4}|\d{4}|' + \
            '|'.join(current_indicators) + r')'
        match = re.search(date_pattern, duration, re.IGNORECASE)

        if not match:
            continue

        start_date_str, end_date_str = match.groups()

        # Parse start date
        try:
            if re.match(r'^\d{4}$', start_date_str):
                start_date = datetime.strptime(start_date_str, "%Y")
            else:
                try:
                    start_date = datetime.strptime(start_date_str, "%B %Y")
                except ValueError:
                    try:
                        start_date = datetime.strptime(
                            start_date_str, "%b. %Y")
                    except ValueError:
                        try:
                            start_date = datetime.strptime(
                                start_date_str, "%b %Y")
                        except ValueError:
                            year_match = re.search(r'\d{4}', start_date_str)
                            if year_match:
                                year = year_match.group(0)
                                start_date = datetime.strptime(
                                    f"January {year}", "%B %Y")
                            else:
                                start_date = datetime.strptime(
                                    "January 2000", "%B %Y")
        except ValueError:
            year_match = re.search(r'\d{4}', start_date_str)
            if year_match:
                year = year_match.group(0)
                start_date = datetime.strptime(f"January {year}", "%B %Y")
            else:
                start_date = datetime.strptime("January 2000", "%B %Y")

        # Parse end date
        end_date_lower = end_date_str.lower()
        is_current = any(indicator.lower()
                         in end_date_lower for indicator in current_indicators)

        if is_current:
            end_date = current_date
        else:
            try:
                if re.match(r'^\d{4}$', end_date_str):
                    end_date = datetime.strptime(end_date_str, "%Y")
                else:
                    try:
                        end_date = datetime.strptime(end_date_str, "%B %Y")
                    except ValueError:
                        try:
                            end_date = datetime.strptime(
                                end_date_str, "%b. %Y")
                        except ValueError:
                            try:
                                end_date = datetime.strptime(
                                    end_date_str, "%b %Y")
                            except ValueError:
                                year_match = re.search(r'\d{4}', end_date_str)
                                if year_match:
                                    year = year_match.group(0)
                                    end_date = datetime.strptime(
                                        f"December {year}", "%B %Y")
                                else:
                                    end_date = current_date
            except ValueError:
                year_match = re.search(r'\d{4}', end_date_str)
                if year_match:
                    year = year_match.group(0)
                    end_date = datetime.strptime(f"December {year}", "%B %Y")
                else:
                    end_date = current_date

        # Calculate duration in months
        months = (end_date.year - start_date.year) * \
            12 + (end_date.month - start_date.month)
        total_months += max(0, months)

    return round(total_months / 12, 1)


def rank_resumes(resumes, jd_skills, exp_weight, edu_weight, skill_weight, min_edu_score, required_degrees, skill_ex_obj):
    """
    Rank resumes based on experience, education, and detailed TF-IDF skill match breakdown.
    All components are normalized to 0-100 scale and final score is normalized to 0-100.

    Args:
        resumes: List of resume dictionaries
        jd_skills: Skills from job description
        exp_weight: Weight for experience (0-100)
        edu_weight: Weight for education (0-100)
        skill_weight: Weight for skills (0-100)
        min_edu_score: Minimum education score required
        required_degrees: List of required degrees
        skill_ex_obj: Skill extraction object

    Returns:
        list: Sorted list of candidates with scores
    """
    # Calculate skill TF-IDF scores with breakdown
    skill_contributions = compute_tfidf_match_with_breakdown(
        resumes, jd_skills, skill_ex_obj)

    # Create preliminary scores list with raw data
    prelim_scores = []

    for skill_data in skill_contributions:
        idx = skill_data["resume_index"]
        name = skill_data["resume_name"]
        tfidf_score = skill_data["score"]
        # print(tfidf_score)
        skill_breakdown = skill_data["skill_breakdown"]
        # print(skill_breakdown)
        resume = resumes[idx]
        experience = resume.get("experience", [])
        education = resume.get("education", [])

        # Calculate total experience in years
        exp_years = calculate_total_experience(experience)

        # Calculate degree match score
# Calculate degree match score
        raw_degree_score = 0
        edu_score = None  # Explicitly set default

        # If min_edu_score exists, normalize education result
        if min_edu_score is not None:
            edu_score = normalize_education_result(education)

        # If required_degrees exists, check degree matching
        if required_degrees is not None:
            raw_degree_score = get_candidate_degree_scores(
                education, required_degrees)

            # Apply minimum education filter if min_edu_score is also provided
            if min_edu_score is not None and edu_score is not None and edu_score < min_edu_score:
                raw_degree_score = 0

        # 🚀 Ensure degree_score is always present
        if min_edu_score is None and required_degrees is None:
            raw_degree_score = 0  # If both are None, set to 0
        print(tfidf_score)
        # Add to preliminary scores
        prelim_scores.append({
            "index": idx,
            "name": name,
            "raw_tfidf": tfidf_score,
            "exp_years": exp_years,
            "raw_degree_score": raw_degree_score,
            "skill_breakdown": skill_breakdown
        })

    # If no candidates passed the filters, return empty list
    if not prelim_scores:
        return []

    # Normalize all scores to 0-100 scale
    tfidf_values = np.array([score["raw_tfidf"]
                            for score in prelim_scores]).reshape(-1, 1)
    exp_values = np.array([score["exp_years"]
                          for score in prelim_scores]).reshape(-1, 1)
    degree_values = np.array([score["raw_degree_score"]
                             for score in prelim_scores]).reshape(-1, 1)

    # Create scalers for each component
    tfidf_scaler = MinMaxScaler(feature_range=(1, 100))  # Avoids exact zero
    exp_scaler = MinMaxScaler(feature_range=(0, 100))
    degree_scaler = MinMaxScaler(feature_range=(0, 100))

    normalized_tfidf = tfidf_scaler.fit_transform(tfidf_values).flatten() if len(
        set(tfidf_values.flatten())) > 1 else np.full_like(tfidf_values.flatten(), 0)
    normalized_exp = exp_scaler.fit_transform(exp_values).flatten() if len(
        set(exp_values.flatten())) > 1 else np.full_like(exp_values.flatten(), 0)
    normalized_degree = degree_scaler.fit_transform(degree_values).flatten() if len(
        set(degree_values.flatten())) > 1 else np.full_like(degree_values.flatten(), 0)

    final_scores = []

    for i, score in enumerate(prelim_scores):
        # Apply weights to normalized scores
        weighted_tfidf = normalized_tfidf[i] * skill_weight / 100
        weighted_exp = normalized_exp[i] * exp_weight / 100
        weighted_edu = normalized_degree[i] * edu_weight / 100

        # Calculate raw weighted sum (could exceed 100)
        raw_total = weighted_tfidf + weighted_exp + weighted_edu

        # Normalize the total score to 0-100 scale
        # We scale based on the theoretical maximum (sum of weights)
        max_possible = (skill_weight + exp_weight + edu_weight) / 100 * 100
        normalized_total = (raw_total / max_possible) * \
            100 if max_possible > 0 else 0

        # Adjust skill breakdown to reflect normalized TF-IDF
        normalized_skill_breakdown = {
            key: value * (normalized_tfidf[i] /
                          100) if score["raw_tfidf"] > 0 else 0
            for key, value in score["skill_breakdown"].items()
        }

        final_scores.append({
            "index": score["index"],
            "name": score["name"],
            "skill_score": round(normalized_tfidf[i], 2),
            "exp_score": round(normalized_exp[i], 2),
            "exp_years": round(score["exp_years"], 2),
            "degree_score": round(normalized_degree[i], 2),
            "weighted_skill": round(weighted_tfidf, 2),
            "weighted_exp": round(weighted_exp, 2),
            "weighted_degree": round(weighted_edu, 2),
            "raw_total": round(raw_total, 2),
            "total_score": round(normalized_total, 2),
            "skill_breakdown": normalized_skill_breakdown
        })

    # Sort by total score by default
    return sorted(final_scores, key=lambda x: x["total_score"], reverse=True)


def sort_resumes_by_criteria(ranked_resumes, sort_by="total"):
    """
    Sort resumes based on user preference.

    Parameters:
    - ranked_resumes: List of resume data with scores
    - sort_by: String indicating sort criteria ('total', 'experience', 'education', or 'skills')

    Returns:
    - Sorted list of resumes
    """
    if sort_by == "experience":
        return sorted(ranked_resumes, key=lambda x: x["exp_score"], reverse=True)
    elif sort_by == "education":
        return sorted(ranked_resumes, key=lambda x: x["edu_score"], reverse=True)
    elif sort_by == "skills":
        return sorted(ranked_resumes, key=lambda x: x["tfidf_score"], reverse=True)
    else:  # Default to total score
        return sorted(ranked_resumes, key=lambda x: x["total_score"], reverse=True)
