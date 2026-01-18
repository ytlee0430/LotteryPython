from flask import Flask, json, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from lotterypython.logic import get_data_from_gsheet, run_predictions
from lotterypython.update_data import main as update_lottery_data
from lotterypython.analysis_sheet import append_analysis_results
from predict.lotto_predict_astrology import (
    add_profile, get_profile, get_all_profiles, delete_profile,
    predict_ziwei, predict_zodiac, has_profiles,
    get_profiles_by_family, get_all_family_groups,
    get_cache_stats, clear_all_prediction_cache
)
from predict.astrology.profiles import AllPredictionsCacheManager, UserManager, User
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'lottery-python-secret-key-change-in-production')

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '請先登入'

# User manager singleton
_user_manager = None

def get_user_manager():
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager

@login_manager.user_loader
def load_user(user_id):
    return get_user_manager().load_user(int(user_id))

@app.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        data = request.form
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            flash('請輸入帳號和密碼', 'error')
            return render_template('login.html', tab='login')

        user = get_user_manager().authenticate(username, password)
        if user:
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('帳號或密碼錯誤', 'error')
            return render_template('login.html', tab='login')

    return render_template('login.html', tab='login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        data = request.form
        username = data.get('username', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        email = data.get('email', '').strip()

        if not username or not password:
            flash('請輸入帳號和密碼', 'error')
            return render_template('login.html', tab='register')

        if password != confirm_password:
            flash('兩次密碼輸入不一致', 'error')
            return render_template('login.html', tab='register')

        if len(password) < 4:
            flash('密碼至少需要4個字元', 'error')
            return render_template('login.html', tab='register')

        try:
            user_data = get_user_manager().create_user(username, password, email or None)
            flash('註冊成功！請登入', 'success')
            return redirect(url_for('login'))
        except ValueError as e:
            flash(str(e), 'error')
            return render_template('login.html', tab='register')

    return render_template('login.html', tab='register')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('已登出', 'info')
    return redirect(url_for('login'))

@app.route('/profiles-ui')
@login_required
def profiles_ui():
    """Render the profile management UI page."""
    return render_template('profiles.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    lotto_type = data.get('type', 'big')
    if lotto_type not in ['big', 'super']:
        return jsonify({"error": "Invalid lotto type. Must be 'big' or 'super'"}), 400

    try:
        df = get_data_from_gsheet(lotto_type)
        if df.empty:
            return jsonify({"error": "Failed to fetch data or data is empty"}), 500

        results = run_predictions(df, user_id=current_user.id)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update', methods=['POST'])
@login_required
def update():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    lotto_type = data.get('type', 'big')
    if lotto_type not in ['big', 'super']:
        return jsonify({"error": "Invalid lotto type. Must be 'big' or 'super'"}), 400

    try:
        update_lottery_data(lotto_type)
        return jsonify({"message": f"Successfully updated {lotto_type} lottery data."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/combin', methods=['POST'])
@login_required
def combin():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    lotto_type = data.get('type', 'big')
    if lotto_type not in ['big', 'super']:
        return jsonify({"error": "Invalid lotto type. Must be 'big' or 'super'"}), 400

    try:
        # 1. Fetch data
        df = get_data_from_gsheet(lotto_type)
        if df.empty:
            return jsonify({"error": "Failed to fetch data or data is empty"}), 500

        # 2. Predict
        results = run_predictions(df, user_id=current_user.id)

        # 3. Prepare for saving
        predictions_list = []
        for key, val in results.items():
            if not isinstance(val, dict): continue
            if "error" in val: continue

            predictions_list.append((key, val["next_period"], val["numbers"], val["special"]))

        # 4. Save
        append_analysis_results(predictions_list, lotto_type)

        return jsonify({
            "message": "Predictions generated and saved to analysis sheet.",
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['POST'])
def history():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    lotto_type = data.get('type', 'big')
    if lotto_type not in ['big', 'super']:
        return jsonify({"error": "Invalid lotto type. Must be 'big' or 'super'"}), 400
    
    try:
        df = get_data_from_gsheet(lotto_type)
        if df.empty:
            return jsonify({"error": "Failed to fetch data or data is empty"}), 500
        
        # Convert to records
        # Sort by Date descending, fallback to Period
        try:
            if 'Date' in df.columns:
                # Ensure we work on a copy to avoid SettingWithCopy warnings if any
                df = df.copy()
                # Coerce errors to NaT
                df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # If we have valid dates
                if df['Date_Obj'].notna().any():
                    df = df.sort_values(by='Date_Obj', ascending=False)
                elif 'Period' in df.columns:
                    # Fallback to Period if dates are all bad
                    df['Period'] = pd.to_numeric(df['Period'], errors='coerce')
                    df = df.sort_values(by='Period', ascending=False)
                
                # Drop helper column
                if 'Date_Obj' in df.columns:
                    df = df.drop(columns=['Date_Obj'])
            elif 'Period' in df.columns:
                df['Period'] = pd.to_numeric(df['Period'], errors='coerce')
                df = df.sort_values(by='Period', ascending=False)

        except Exception as e:
            print(f"Sort failed: {e}")
            # Fallback simple sort by Period if possible
            if 'Period' in df.columns:
                try:
                    df = df.sort_values(by='Period', ascending=False)
                except:
                    pass
                
        records = df.to_dict(orient='records')
        return jsonify({"history": records})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ Birth Profile Management APIs ============

@app.route('/profiles', methods=['GET'])
@login_required
def list_profiles():
    """List all birth profiles for current user."""
    try:
        profiles = get_all_profiles(user_id=current_user.id)
        return jsonify({"profiles": profiles, "count": len(profiles)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/profiles', methods=['POST'])
@login_required
def create_profile():
    """Create a new birth profile for current user."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    required = ['name', 'birth_year', 'birth_month', 'birth_day', 'birth_hour']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        profile = add_profile(
            name=data['name'],
            birth_year=int(data['birth_year']),
            birth_month=int(data['birth_month']),
            birth_day=int(data['birth_day']),
            birth_hour=int(data['birth_hour']),
            family_group=data.get('family_group', 'default'),
            relationship=data.get('relationship', ''),
            user_id=current_user.id
        )
        return jsonify({"message": "Profile created", "profile": profile}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/profiles/<name>', methods=['GET'])
@login_required
def get_profile_by_name(name):
    """Get a specific birth profile by name for current user."""
    try:
        profile = get_profile(name, user_id=current_user.id)
        if not profile:
            return jsonify({"error": f"Profile '{name}' not found"}), 404
        return jsonify({"profile": profile})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/profiles/<name>', methods=['DELETE'])
@login_required
def delete_profile_by_name(name):
    """Delete a birth profile by name for current user."""
    try:
        deleted = delete_profile(name, user_id=current_user.id)
        if not deleted:
            return jsonify({"error": f"Profile '{name}' not found"}), 404
        return jsonify({"message": f"Profile '{name}' deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ Family Group APIs ============

@app.route('/families', methods=['GET'])
@login_required
def list_families():
    """List all family groups with member counts for current user."""
    try:
        families = get_all_family_groups(user_id=current_user.id)
        return jsonify({"families": families})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/families/<family_group>/members', methods=['GET'])
@login_required
def get_family_members(family_group):
    """Get all members in a specific family group for current user."""
    try:
        members = get_profiles_by_family(family_group, user_id=current_user.id)
        return jsonify({"family": family_group, "members": members, "count": len(members)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ Astrology Prediction APIs ============

@app.route('/predict/astrology', methods=['POST'])
@login_required
def predict_astrology():
    """Get astrology-based predictions (both Ziwei and Zodiac) for current user."""
    data = request.get_json() or {}
    lotto_type = data.get('type', 'big')
    profile_name = data.get('profile_name')  # Optional: specific profile

    if lotto_type not in ['big', 'super']:
        return jsonify({"error": "Invalid lotto type"}), 400

    if not has_profiles(user_id=current_user.id):
        return jsonify({"error": "No birth profiles found. Please add a profile first."}), 400

    results = {}

    # Ziwei prediction
    try:
        nums, special, details = predict_ziwei(lotto_type, profile_name, user_id=current_user.id)
        results["Astrology-Ziwei"] = {
            "numbers": nums,
            "special": special,
            "method": "紫微斗數",
            "details": details.get("predictions", []),
            "from_cache": details.get("from_cache", False),
            "period": details.get("period")
        }
    except Exception as e:
        results["Astrology-Ziwei"] = {"error": str(e)}

    # Zodiac prediction
    try:
        nums, special, details = predict_zodiac(lotto_type, profile_name, user_id=current_user.id)
        results["Astrology-Zodiac"] = {
            "numbers": nums,
            "special": special,
            "method": "西洋星座",
            "details": details.get("predictions", []),
            "from_cache": details.get("from_cache", False),
            "period": details.get("period")
        }
    except Exception as e:
        results["Astrology-Zodiac"] = {"error": str(e)}

    return jsonify(results)

@app.route('/predict/ziwei', methods=['POST'])
@login_required
def predict_ziwei_only():
    """Get Ziwei (紫微斗數) prediction only for current user."""
    data = request.get_json() or {}
    lotto_type = data.get('type', 'big')
    profile_name = data.get('profile_name')

    if not has_profiles(user_id=current_user.id):
        return jsonify({"error": "No birth profiles found"}), 400

    try:
        nums, special, details = predict_ziwei(lotto_type, profile_name, user_id=current_user.id)
        return jsonify({
            "numbers": nums,
            "special": special,
            "method": "紫微斗數",
            "details": details
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/zodiac', methods=['POST'])
@login_required
def predict_zodiac_only():
    """Get Zodiac (星座) prediction only for current user."""
    data = request.get_json() or {}
    lotto_type = data.get('type', 'big')
    profile_name = data.get('profile_name')

    if not has_profiles(user_id=current_user.id):
        return jsonify({"error": "No birth profiles found"}), 400

    try:
        nums, special, details = predict_zodiac(lotto_type, profile_name, user_id=current_user.id)
        return jsonify({
            "numbers": nums,
            "special": special,
            "method": "西洋星座",
            "details": details
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ Prediction Cache APIs ============

@app.route('/cache/stats', methods=['GET'])
@login_required
def cache_stats():
    """Get prediction cache statistics for current user."""
    try:
        astrology_stats = get_cache_stats(user_id=current_user.id)
        all_cache = AllPredictionsCacheManager()
        all_stats = all_cache.get_cache_stats(user_id=current_user.id)
        return jsonify({
            "astrology_cache": astrology_stats,
            "all_predictions_cache": all_stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cache/clear', methods=['POST'])
@login_required
def clear_cache():
    """Clear all prediction cache for current user."""
    try:
        astrology_count = clear_all_prediction_cache(user_id=current_user.id)
        all_cache = AllPredictionsCacheManager()
        all_count = all_cache.clear_all_cache(user_id=current_user.id)
        return jsonify({
            "message": f"Cleared {astrology_count} astrology + {all_count} all-predictions cached entries"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
