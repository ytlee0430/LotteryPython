from flask import Flask, json, request, jsonify, render_template
from lotterypython.logic import get_data_from_gsheet, run_predictions
from lotterypython.update_data import main as update_lottery_data
from lotterypython.analysis_sheet import append_analysis_results
import numpy as np
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
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
        
        results = run_predictions(df)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update', methods=['POST'])
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
        results = run_predictions(df)
        
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
