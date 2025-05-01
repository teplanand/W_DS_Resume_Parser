# Standard Library
import os
import json
import asyncio

# Third-Party Libraries
import numpy as np
import google.generativeai as genai
import spacy
from spacy.matcher import PhraseMatcher
from spacy_layout import spaCyLayout
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, g, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    current_user,
    UserMixin
)
from sqlalchemy.exc import IntegrityError

# Your Own Modules
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from rank import rank_resumes
from parser import get_resume_text, process_resumes_gemini, append_to_json, parse_resume


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'your_secret_key'  # Required for session storage
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


login_manager = LoginManager()
login_manager.init_app(app)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    # Added Gemini API key column
    gemini_api_key = db.Column(db.String(200), nullable=True)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        raw_password = request.form.get('password', '')
        gemini_api_key = request.form.get(
            'gemini_api_key', '').strip()  # Get Gemini API key

        if not name or not email or not raw_password:
            session['message'] = 'All fields are required.'
            session['message_type'] = 'warning'
            return redirect(url_for('index'))

        try:
            hashed_password = bcrypt.generate_password_hash(
                raw_password).decode('utf-8')
            user = User(name=name, email=email, password=hashed_password,
                        gemini_api_key=gemini_api_key)  # Save Gemini API key
            db.session.add(user)
            db.session.commit()
            session['message'] = 'Account created successfully! Please login.'
            session['message_type'] = 'success'
            return redirect(url_for('index'))

        except IntegrityError:
            db.session.rollback()
            session['message'] = 'Email is already registered. Try logging in or use a different email.'
            session['message_type'] = 'danger'
            return redirect(url_for('index'))

        except Exception as e:
            db.session.rollback()
            session['message'] = 'An unexpected error occurred. Please try again later.'
            session['message_type'] = 'danger'
            print(f"Error during signup: {e}")
            return redirect(url_for('index'))

    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        raw_password = request.form.get('password', '')

        if not email or not raw_password:
            session['message'] = 'Both email and password are required.'
            session['message_type'] = 'warning'
            return redirect(url_for('index'))

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, raw_password):
            login_user(user)
            session['name'] = user.name
            # session['message'] = 'Logged in successfully.'
            # session['message_type'] = 'success'
            return redirect(url_for('dashboard'))
        else:
            session['message'] = 'Login failed. Check email and password.'
            session['message_type'] = 'danger'
            return redirect(url_for('index'))

    return render_template('index.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@app.route('/settings')
@login_required
def settings():
    """Display user settings page."""
    user_identifier = str(current_user.id)
    json_file = f"parsed_data_{user_identifier}.json"
    json_data = "{}"  # Default empty data
    current_api_key = current_user.gemini_api_key
    genai.configure(api_key=current_api_key)

    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding="utf-8") as f:
                parsed_data = json.load(f)
                json_data = json.dumps(parsed_data, indent=2)
        except json.JSONDecodeError:
            json_data = "{}"  # Reset to empty if corrupted

    return render_template('settings.html', json_data=json_data, current_api_key=current_api_key)


@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile information."""
    name = request.form.get('name', '').strip()

    if not name:
        session['message'] = 'Name cannot be empty.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    try:
        current_user.name = name
        db.session.commit()
        session['name'] = name  
        session['message'] = 'Profile updated successfully.'
        session['message_type'] = 'success'
    except Exception as e:
        db.session.rollback()
        session['message'] = f'Error updating profile: {str(e)}'
        session['message_type'] = 'danger'

    return redirect(url_for('settings'))


@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    """Change user password."""
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    # Validate inputs
    if not current_password or not new_password or not confirm_password:
        session['message'] = 'All password fields are required.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    if new_password != confirm_password:
        session['message'] = 'New passwords do not match.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    # Check current password
    if not bcrypt.check_password_hash(current_user.password, current_password):
        session['message'] = 'Current password is incorrect.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    try:
        # Update password
        hashed_password = bcrypt.generate_password_hash(
            new_password).decode('utf-8')
        current_user.password = hashed_password
        db.session.commit()

        session['message'] = 'Password changed successfully.'
        session['message_type'] = 'success'
    except Exception as e:
        db.session.rollback()
        session['message'] = f'Error changing password: {str(e)}'
        session['message_type'] = 'danger'

    return redirect(url_for('settings'))


@app.route('/change_api_key', methods=['POST'])
@login_required
def change_api_key():
    """Handle API key change."""
    current_api_key = request.form.get('current_api_key')
    new_api_key = request.form.get('new_api_key')
    confirm_api_key = request.form.get('confirm_api_key')

    # Validate the input
    if not current_api_key or not new_api_key or not confirm_api_key:
        session['message'] = 'All API key fields are required.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    if new_api_key != confirm_api_key:
        session['message'] = 'New API keys do not match.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    if current_api_key != current_user.gemini_api_key:
        session['message'] = 'Current API key is incorrect.'
        session['message_type'] = 'danger'
        return redirect(url_for('settings'))

    try:
        
        current_user.gemini_api_key = new_api_key
        db.session.commit()

        session['message'] = 'API key changed successfully.'
        session['message_type'] = 'success'
    except Exception as e:
        db.session.rollback()
        session['message'] = f'Error changing API key: {str(e)}'
        session['message_type'] = 'danger'

    return redirect(url_for('settings'))


@app.route('/save_json_data', methods=['POST'])
@login_required
def save_json_data():
    """Save edited JSON data."""
    user_identifier = str(current_user.id)
    json_file = f"parsed_data_{user_identifier}.json"

    try:
        # Get JSON data from request
        data = request.json.get('data')
        parsed_json = json.loads(data)

        # Save to file
        with open(json_file, 'w', encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=2)

        # Also delete any ranking results since data has changed
        result_file = f"results_{user_identifier}.json"
        if os.path.exists(result_file):
            os.remove(result_file)

        return jsonify({
            'status': 'success',
            'message': 'Data saved successfully'
        })
    except json.JSONDecodeError:
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON format'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/dashboard')
@login_required
def dashboard():
    alert_message = session.pop('alert_message', None)
    # ranked_resumes = []

    # 🧠 Use user ID or email to create a unique file name
    user_identifier = str(current_user.id)

    json_file_1 = f"parsed_data_{user_identifier}.json"

    if os.path.exists(json_file_1) and os.path.getsize(json_file_1) > 0:
        try:
            with open(json_file_1, 'r', encoding="utf-8") as f:
                results = json.load(f)
                parsed_resumes = results
                # print(parsed_resumes)
        except json.JSONDecodeError:
            # print("Error: JSON file is corrupted!")
            parsed_resumes = []
    else:
        # print("Warning: JSON file is missing or empty!")
        parsed_resumes = []

    return render_template('dashboard.html',
                           parsed_resumes=parsed_resumes,
                           alert_message=alert_message,)


@app.route('/dashboard_candidate')
@login_required
def dashboard_candidate():

    # ranked_resumes = []

    # 🧠 Use user ID or email to create a unique file name
    user_identifier = str(current_user.id)

    json_file_1 = f"parsed_data_{user_identifier}.json"

    if os.path.exists(json_file_1) and os.path.getsize(json_file_1) > 0:
        try:
            with open(json_file_1, 'r', encoding="utf-8") as f:
                results = json.load(f)
                parsed_resumes = results
                # print(parsed_resumes)
        except json.JSONDecodeError:
            # print("Error: JSON file is corrupted!")
            parsed_resumes = []
    else:
        # print("Warning: JSON file is missing or empty!")
        parsed_resumes = []

    return render_template('dashboard_candidate.html',
                           parsed_resumes=parsed_resumes,
                           )


@app.route('/dashboard_rank')
@login_required
def dashboard_rank():

    alert_message = session.pop('alert_message', None)
    ranked_resumes = []

    # 🧠 Use user ID or email to create a unique file name
    user_identifier = str(current_user.id)
    # print(user_identifier)
    json_file = f"results_{user_identifier}.json"
    json_file_1 = f"parsed_data_{user_identifier}.json"
    # print(f"JSON file path: {json_file}")
    form_data = {
        'jd_text': session.get('jd_text', ''),
        'exp_weight': session.get('exp_weight', 5),
        'edu_weight': session.get('edu_weight', 3),
        'skill_weight': session.get('skill_weight', 4),
    }

    if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
        try:
            with open(json_file, 'r', encoding="utf-8") as f:
                results = json.load(f)
                ranked_resumes = results.get('ranked_resumes', [])

        except json.JSONDecodeError:
            # print("Error: JSON file is corrupted!")
            ranked_resumes = []
    else:
        # print("Warning: JSON file is missing or empty!")
        ranked_resumes = []

    if os.path.exists(json_file_1) and os.path.getsize(json_file_1) > 0:
        try:
            with open(json_file_1, 'r', encoding="utf-8") as f:
                results = json.load(f)
                parsed_resumes = results
                # print(parsed_resumes)
        except json.JSONDecodeError:
            # print("Error: JSON file is corrupted!")
            parsed_resumes = []
    else:
        # print("Warning: JSON file is missing or empty!")
        parsed_resumes = []

    return render_template('dashboard_rank.html',
                           parsed_resumes=parsed_resumes,
                           ranked_resumes=ranked_resumes,
                           alert_message=alert_message,
                           form_data=form_data)


def load_resources():
    """Loads NLP model, layout processor, and skill extractor once."""
    nlp = spacy.load("en_core_web_lg")
    layout = spaCyLayout(nlp)
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

    # Store in `app.config`
    app.config['NLP'] = nlp
    app.config['LAYOUT'] = layout
    app.config['SKILL_EXTRACTOR'] = skill_extractor


@app.before_request
def set_global_resources():
    """Attach preloaded models to `g` for request-level access."""
    g.nlp = app.config['NLP']
    g.layout = app.config['LAYOUT']
    g.skill_extractor = app.config['SKILL_EXTRACTOR']


@app.route('/parse', methods=['POST'])
@login_required
def parse():
    resumes = []
    for file in request.files.getlist('resumes'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        resumes.append(filepath)

    resume_data_list = get_resume_text(resumes, g.layout)

    current_api_key = current_user.gemini_api_key
    genai.configure(api_key=current_api_key)
    gem_data = asyncio.run(process_resumes_gemini(resume_data_list))

    parsed_data = parse_resume(resume_data_list, g.skill_extractor, gem_data)

    # 🔐 Store per-user file using user ID or email
    user_identifier = str(current_user.id)

    user_parsed_json = f"parsed_data_{user_identifier}.json"

    append_to_json(user_parsed_json, parsed_data)

    return jsonify({
        'status': 'success',
        'message': 'Resumes parsed successfully! You can upload more resumes if needed.'
    })


@app.route('/rank', methods=['POST'])
@login_required
def rank():
    user_identifier = str(current_user.id)

    parsed_file = f"parsed_data_{user_identifier}.json"
    # parsed_file = 'parsed_data.json'
    session['jd_text'] = request.form['jd']
    session['exp_weight'] = request.form['exp_weight']
    session['edu_weight'] = request.form['edu_weight']
    session['skill_weight'] = request.form['skill_weight']

    # 🔍 Check if parsing is done
    if not os.path.exists(parsed_file):
        session['alert_message'] = "Error: Parsing must be completed before ranking resumes."
        return redirect(url_for('dashboard'))

    try:
        with open(parsed_file, 'r') as f:
            parsed_resumes = json.load(f)

        if not parsed_resumes:  # Ensure the file is not empty
            session['alert_message'] = "Error: No parsed resumes found. Please upload resumes first."
            return redirect(url_for('dashboard'))
    except json.JSONDecodeError:
        session['alert_message'] = "Error: Parsed data is corrupted. Please re-upload resumes."
        return redirect(url_for('dashboard'))

    # ✅ Parsing is done, proceed with ranking
    jd_text = request.form['jd']
    exp_weight = int(request.form['exp_weight'])
    edu_weight = int(request.form['edu_weight'])
    skill_weight = int(request.form['skill_weight'])
    # Get sorting preference if provided
    # sort_preference = request.form.get('sort_by', 'total')

    # Parse skills from job description
    jd_skills = [skill.strip()
                 for skill in jd_text.split(',') if skill.strip()]

    # Get minimum education score
    min_edu_score = None
    if request.form.get('min_edu_score') == 'custom_set' and request.form.get('custom_min_edu_score'):
        # Use custom score if available
        min_edu_score = float(request.form.get('custom_min_edu_score'))
    elif request.form.get('min_edu_score') and request.form.get('min_edu_score') != 'custom':
        # Use predefined score if selected
        min_edu_score = float(request.form.get('min_edu_score'))

    # Get required degrees
    required_degrees = None
    if request.form.get('required_degrees'):
        try:
            required_degrees = json.loads(request.form.get('required_degrees'))
            # Ensure it's a valid list (even if empty)
            if not isinstance(required_degrees, list):
                required_degrees = None
        except json.JSONDecodeError:
            required_degrees = None  # Handle invalid JSON input

    # Prepare the resume data for ranking
    ranking_data = [
        {
            "name": resume["name"],
            "experience": resume.get("experience", []),
            "education": resume.get("education", []),
            "skills": resume.get("skills", [])
        }
        for resume in parsed_resumes
    ]
    print(request.form["job_role"])
    print(jd_skills)
    print(skill_weight, edu_weight, exp_weight)
    print(required_degrees)
    print(min_edu_score)
    # Rank resumes (this now returns already sorted results)
    ranked_resumes = rank_resumes(
        ranking_data, jd_skills, exp_weight, edu_weight, skill_weight,
        min_edu_score, required_degrees, g.skill_extractor
    )

    user_result_file = (f'results_{user_identifier}.json')

    # Save ranked results
    with open(user_result_file, 'w') as f:
        json.dump(
            {
                'ranked_resumes': json.loads(json.dumps(ranked_resumes, default=lambda o: int(o) if isinstance(o, np.integer) else o))
            },
            f,
            indent=4
        )

    flash("Ranking completed successfully!", "success")
    return redirect(url_for('dashboard_rank'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/resume/<int:index>')
@login_required
def resume_detail(index):
    """Display detailed resume information along with ranking details."""
    user_id = str(current_user.id)
    parsed_file = (f'parsed_data_{user_id}.json')
    ranking_file = (f'results_{user_id}.json')

    if not os.path.exists(parsed_file):
        return redirect(url_for('dashboard'))  # Redirect if no resumes exist

    with open(parsed_file, 'r') as f:
        parsed_resumes = json.load(f)  # Load resumes

    if index < 0 or index >= len(parsed_resumes):
        return "Resume not found", 404  # Handle invalid index

    # Get the resume
    resume = parsed_resumes[index]

    # Load ranking details
    ranking = None
    matched_skills = []
    skill_breakdown = {}

    if os.path.exists(ranking_file):
        with open(ranking_file, 'r') as f:
            ranked_resumes = json.load(f)

        # Find the ranking info for this resume
        ranking = next((r for r in ranked_resumes.get(
            "ranked_resumes", []) if r["index"] == index), None)

        if ranking:
            # Get matched skills if available
            if 'matched_skills' in ranking:
                matched_skills = ranking['matched_skills']

            # Get skill breakdown from ranking if available
            if 'skill_breakdown' in ranking and ranking['skill_breakdown']:
                skill_breakdown = ranking['skill_breakdown']

    # Either use the existing skill_breakdown or the one from ranking
    if not 'skill_breakdown' in resume or not resume['skill_breakdown']:
        resume['skill_breakdown'] = skill_breakdown

    # print(f"Skill breakdown for chart: {resume['skill_breakdown']}")

    return render_template('resume_detail.html',
                           resume=resume,
                           ranking=ranking,
                           matched_skills=matched_skills)


@app.route('/clear_results')
@login_required
def clear_results():
    """Clears session and stored results for the current user."""
    session.pop('ranked_resumes', None)

    user_id = str(current_user.id)
    result_file = (f'results_{user_id}.json')
    parsed_file = (f'parsed_data_{user_id}.json')

    if os.path.exists(result_file):
        os.remove(result_file)
    # if os.path.exists(parsed_file):
    #     os.remove(parsed_file)

    flash("Results cleared successfully.", "info")
    return redirect(url_for('dashboard_rank'))


@app.route('/delete_candidate/<int:index>', methods=['POST'])
@login_required
def delete_candidate(index):
    user_id = str(current_user.id)
    parsed_file = f'parsed_data_{user_id}.json'

    if os.path.exists(parsed_file):
        with open(parsed_file, 'r') as f:
            data = json.load(f)

        if 0 <= index < len(data):
            del data[index]

            with open(parsed_file, 'w') as f:
                json.dump(data, f, indent=4)

    result_file = (f'results_{user_id}.json')
    if os.path.exists(result_file):
        os.remove(result_file)

    return redirect(url_for('dashboard_candidate'))


load_resources()


if __name__ == '__main__':
    app.run(debug=False)

