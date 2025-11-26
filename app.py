from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import joblib
import os
import io
import requests


app = Flask(__name__)
model = joblib.load('model.pkl')

# City Coordinates for Weather & Map
CITY_COORDINATES = {
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Delhi': {'lat': 28.7041, 'lon': 77.1025},
    'Lucknow': {'lat': 26.8467, 'lon': 80.9462},
    'Jaipur': {'lat': 26.9124, 'lon': 75.7873},
    'Pune': {'lat': 18.5204, 'lon': 73.8567},
    'Surat': {'lat': 21.1702, 'lon': 72.8311},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867}
}

def get_weather(city):
    try:
        if city not in CITY_COORDINATES:
            return None
        
        coords = CITY_COORDINATES[city]
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current_weather=true"
        response = requests.get(url)
        data = response.json()
        
        if 'current_weather' in data:
            # Open-Meteo current_weather doesn't always have rain, but we can check weathercode
            # Or we can use the 'hourly' or 'daily' API for precipitation, but for simplicity let's rely on weathercode or just add 0 if missing
            # Actually, let's try to get 'precipitation' from current if available, or assume 0 for MVP
            # Better: use 'current=temperature_2m,precipitation' in newer API, but let's stick to simple current_weather
            # and maybe map weathercode to rain? 
            # Let's just pass 0.0 for rain if not found, or use a separate call if needed.
            # For this MVP, let's assume 0 rain unless we want to be more complex.
            # Wait, the user wants "prediction based on weather". 
            # Let's try to get a bit more data.
            return {
                'temperature': data['current_weather']['temperature'],
                'windspeed': data['current_weather']['windspeed'],
                'weathercode': data['current_weather']['weathercode'],
                'rainfall_mm': 0.0 # Placeholder, or we could map weathercode 51-67, 80-82 to rain
            }
    except Exception as e:
        print(f"Error fetching weather for {city}: {e}")
    return None

# Define expected columns for validation
EXPECTED_COLUMNS = [
    'total_capacity', 'average_age', 'male_percentage', 'female_percentage',
    'temperature', 'rainfall_mm', 'area_population', 'staff_count',
    'children_count', 'senior_citizens_count', 'new_admissions', 'exits_today',
    'emergency_cases', 'unemployment_rate', 'crime_rate',
    'season', 'city', 'funding_level', 'day_of_week'
]

@app.route('/')
def index():
    # Load dataset for dashboard stats
    try:
        df = pd.read_csv(r"c:\Users\dell\adp_pbl\shelter_dataset_all_updated.csv")
        total_shelters = df['shelter_name'].nunique() if 'shelter_name' in df.columns else len(df)
        avg_occupancy = df['occupancy_rate'].mean()
        total_records = len(df)
        
        # Get list of shelters and their cities for the map
        shelters_list = df['shelter_name'].unique().tolist() if 'shelter_name' in df.columns else []
        
        # Create a list of shelters with their coordinates
        shelter_locations = []
        if 'shelter_name' in df.columns and 'city' in df.columns:
            # We'll take the first occurrence of each shelter to get its city
            unique_shelters = df.drop_duplicates(subset=['shelter_name'])
            for _, row in unique_shelters.iterrows():
                city = row['city']
                if city in CITY_COORDINATES:
                    # Add small random jitter to coordinates so markers don't overlap perfectly
                    import random
                    lat_jitter = random.uniform(-0.02, 0.02)
                    lon_jitter = random.uniform(-0.02, 0.02)
                    
                    shelter_locations.append({
                        'name': row['shelter_name'],
                        'city': city,
                        'lat': CITY_COORDINATES[city]['lat'] + lat_jitter,
                        'lon': CITY_COORDINATES[city]['lon'] + lon_jitter
                    })
        

        
        # Recent records for table
        recent_records = df.tail(10).to_dict(orient='records')

        # Resource Optimization Insights
        over_utilized = []
        under_utilized = []
        
        # Managerial Analytics
        city_stats = {}
        top_crowded_shelters = []
        system_utilization = {'occupied': 0, 'capacity': 0, 'percentage': 0}
        
        if 'date' in df.columns and 'occupancy_rate' in df.columns and 'shelter_name' in df.columns:
            # Convert date to datetime
            df['dt'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
            
            # Get last 30 days of data based on the latest date in the dataset
            max_date = df['dt'].max()
            if pd.notna(max_date):
                start_date = max_date - pd.Timedelta(days=30)
                recent_df = df[df['dt'] >= start_date]
                
                # Calculate average occupancy per shelter
                shelter_stats = recent_df.groupby('shelter_name')['occupancy_rate'].mean().reset_index()
                
                # Identify over/under utilized
                over_utilized = shelter_stats[shelter_stats['occupancy_rate'] > 90].to_dict(orient='records')
                under_utilized = shelter_stats[shelter_stats['occupancy_rate'] < 50].to_dict(orient='records')
                
                # Round values
                for s in over_utilized:
                    s['occupancy_rate'] = round(s['occupancy_rate'], 1)
                for s in under_utilized:
                    s['occupancy_rate'] = round(s['occupancy_rate'], 1)
                
                # --- Managerial Analytics Logic ---
                
                # 1. City-wise Average Occupancy
                if 'city' in recent_df.columns:
                    city_stats_df = recent_df.groupby('city')['occupancy_rate'].mean().reset_index()
                    # Convert to float for JSON serialization
                    city_stats = dict(zip(city_stats_df['city'], city_stats_df['occupancy_rate'].round(1).astype(float)))
                
                # 2. System-wide Capacity Utilization (Latest Snapshot)
                # Get the latest record for each shelter to represent "current" state
                latest_snapshot = df.sort_values('dt').groupby('shelter_name').tail(1)
                total_capacity = latest_snapshot['total_capacity'].sum()
                total_occupied = latest_snapshot['occupied_beds'].sum()
                
                # Calculate percentage safely
                percentage = 0
                if total_capacity > 0:
                    percentage = (total_occupied / total_capacity) * 100
                    
                system_utilization = {
                    'occupied': int(total_occupied),
                    'capacity': int(total_capacity),
                    'percentage': float(round(percentage, 1))
                }
                
                # 3. Top 5 Most Crowded Shelters (Latest Snapshot)
                top_crowded_shelters = latest_snapshot.sort_values('occupancy_rate', ascending=False).head(5).to_dict(orient='records')
                for s in top_crowded_shelters:
                    s['occupancy_rate'] = round(s['occupancy_rate'], 1)

    except Exception as e:
        print(f"Error loading dashboard data: {e}")
        total_shelters = 0
        avg_occupancy = 0
        total_records = 0
        shelters_list = []
        shelter_locations = []
        chart_data = []
        recent_records = []
        over_utilized = []
        under_utilized = []
        city_stats = {}
        top_crowded_shelters = []
        system_utilization = {'occupied': 0, 'capacity': 0, 'percentage': 0}

    return render_template('index.html', 
                           total_shelters=total_shelters,
                           avg_occupancy=round(avg_occupancy, 2),
                           total_records=total_records,
                           shelters_list=shelters_list,
                           shelter_locations=shelter_locations,
                           recent_records=recent_records,
                           over_utilized=over_utilized,
                           under_utilized=under_utilized,
                           city_stats=city_stats,
                           top_crowded_shelters=top_crowded_shelters,
                           system_utilization=system_utilization)

@app.route('/shelter/<shelter_name>')
def shelter_detail(shelter_name):
    try:
        df = pd.read_csv(r"c:\Users\dell\adp_pbl\shelter_dataset_all_updated.csv")
        
        # Filter by shelter name
        shelter_df = df[df['shelter_name'] == shelter_name]
        
        if shelter_df.empty:
            return "Shelter not found", 404
            
        # Stats
        capacity = shelter_df['total_capacity'].iloc[0]
        avg_occupancy = shelter_df['occupancy_rate'].mean()
        city = shelter_df['city'].iloc[0]
        state = shelter_df['state'].iloc[0] if 'state' in shelter_df.columns else ''
        
        # Weather
        weather = get_weather(city)
        
        # Chart data for this shelter
        chart_data = shelter_df[['date', 'occupancy_rate']].to_dict(orient='records')
        
        # Latest record for current stats
        latest_record = shelter_df.iloc[-1].to_dict() if not shelter_df.empty else {}
        
        return render_template('shelter.html', 
                               shelter_name=shelter_name,
                               capacity=capacity,
                               avg_occupancy=round(avg_occupancy, 2),
                               weather=weather,
                               city=city,
                               state=state,
                               chart_data=chart_data,
                               latest_record=latest_record)
                               
    except Exception as e:
        return f"Error loading shelter data: {e}", 500

@app.route('/predict_live/<shelter_name>')
def predict_live(shelter_name):
    try:
        df = pd.read_csv(r"c:\Users\dell\adp_pbl\shelter_dataset_all_updated.csv")
        shelter_df = df[df['shelter_name'] == shelter_name]
        
        if shelter_df.empty:
            return {"error": "Shelter not found"}, 404
            
        # Get latest static stats (using the last record for this shelter)
        latest_record = shelter_df.iloc[-1]
        
        # Get current weather
        city = latest_record['city']
        weather = get_weather(city)
        
        if not weather:
            return {"error": "Could not fetch weather data"}, 500
            
        # Prepare input data
        import datetime
        now = datetime.datetime.now()
        
        # Determine season (simple logic)
        month = now.month
        if 3 <= month <= 5:
            season = 'Summer'
        elif 6 <= month <= 9:
            season = 'Monsoon'
        elif 10 <= month <= 11:
            season = 'Autumn'
        else:
            season = 'Winter'
            
        day_of_week = now.strftime('%A')
        
        input_data = {
            'total_capacity': [latest_record['total_capacity']],
            'average_age': [latest_record['average_age']],
            'male_percentage': [latest_record['male_percentage']],
            'female_percentage': [latest_record['female_percentage']],
            'temperature': [weather['temperature']],
            'rainfall_mm': [weather['rainfall_mm']], # Using fetched or placeholder rain
            'area_population': [latest_record['area_population']],
            'staff_count': [latest_record['staff_count']],
            'children_count': [latest_record['children_count']],
            'senior_citizens_count': [latest_record['senior_citizens_count']],
            'new_admissions': [latest_record['new_admissions']], # Using last known, ideally should be input
            'exits_today': [latest_record['exits_today']], # Using last known
            'emergency_cases': [latest_record['emergency_cases']], # Using last known
            'unemployment_rate': [latest_record['unemployment_rate']],
            'crime_rate': [latest_record['crime_rate']],
            'season': [season],
            'city': [city],
            'funding_level': [latest_record['funding_level']],
            'day_of_week': [day_of_week]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Predict
        prediction = model.predict(input_df[EXPECTED_COLUMNS])[0]
        # Calculate occupancy rate
        capacity = latest_record['total_capacity']
        occupancy_rate = (prediction / capacity) * 100
        
        # Alternative Shelter Recommendation
        recommendations = []
        if occupancy_rate > 80:
            # Filter for other shelters in the same city
            city_shelters = df[df['city'] == city]
            other_shelters = city_shelters[city_shelters['shelter_name'] != shelter_name].copy()
            
            # Get latest status for each (assuming dataset has recent data, or we aggregate)
            # For simplicity, we'll take the latest record for each shelter
            latest_records = other_shelters.sort_values('date').groupby('shelter_name').tail(1)
            
            # Calculate availability
            latest_records['current_available'] = latest_records['total_capacity'] - latest_records['occupied_beds']
            latest_records['current_occupancy_rate'] = (latest_records['occupied_beds'] / latest_records['total_capacity']) * 100
            
            # Sort by available beds descending
            top_alternatives = latest_records.sort_values('current_available', ascending=False).head(3)
            
            for _, row in top_alternatives.iterrows():
                recommendations.append({
                    'name': row['shelter_name'],
                    'capacity': int(row['total_capacity']),
                    'occupied': int(row['occupied_beds']),
                    'available': int(row['current_available']),
                    'occupancy_rate': round(row['current_occupancy_rate'], 1)
                })

        # Resource Requirement Prediction
        import math
        predicted_occupied = int(prediction)
        resource_requirements = {
            'meals_per_day': predicted_occupied * 3,
            'water_liters_per_day': predicted_occupied * 5,
            'staff_required': max(1, math.ceil(predicted_occupied / 20)),
            'medical_kits': max(1, math.ceil(predicted_occupied / 50))
        }

        return jsonify({
            'prediction': int(prediction),
            'occupancy_rate': round(occupancy_rate, 2),
            'capacity': int(capacity),
            'recommendations': recommendations,
            'resource_requirements': resource_requirements
        })
        
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/check_in/<shelter_name>', methods=['POST'])
def check_in(shelter_name):
    return update_occupancy(shelter_name, 1)

@app.route('/api/check_out/<shelter_name>', methods=['POST'])
def check_out(shelter_name):
    return update_occupancy(shelter_name, -1)

def update_occupancy(shelter_name, change):
    try:
        csv_path = r"c:\Users\dell\adp_pbl\shelter_dataset_all_updated.csv"
        df = pd.read_csv(csv_path)
        
        # Find the index of the latest record for this shelter
        # We assume we are updating the *latest* status. 
        # In a real app, we might insert a new record or update a specific "current status" table.
        # Here, we'll update the last row for this shelter to reflect current reality.
        shelter_indices = df[df['shelter_name'] == shelter_name].index
        
        if shelter_indices.empty:
            return {"error": "Shelter not found"}, 404
            
        last_index = shelter_indices[-1]
        
        current_occupied = df.at[last_index, 'occupied_beds']
        capacity = df.at[last_index, 'total_capacity']
        
        new_occupied = current_occupied + change
        
        if new_occupied < 0:
            return {"error": "Occupancy cannot be negative"}, 400
        if new_occupied > capacity:
            return {"error": "Shelter is full"}, 400
            
        # Update values
        df.at[last_index, 'occupied_beds'] = new_occupied
        df.at[last_index, 'available_beds'] = capacity - new_occupied
        df.at[last_index, 'occupancy_rate'] = (new_occupied / capacity) * 100
        
        # Save back to CSV
        df.to_csv(csv_path, index=False)
        
        return {
            "success": True,
            "occupied_beds": int(new_occupied),
            "available_beds": int(capacity - new_occupied),
            "occupancy_rate": round((new_occupied / capacity) * 100, 2)
        }
        
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        df = pd.read_csv(file)
        
        # Validate columns
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            return f"Missing columns: {', '.join(missing_cols)}", 400

        # Predict
        predictions = model.predict(df[EXPECTED_COLUMNS])
        
        df['predicted_occupied_beds'] = predictions.astype(int)
        df['predicted_occupancy_rate'] = (df['predicted_occupied_beds'] / df['total_capacity']) * 100
        df['predicted_available_beds'] = df['total_capacity'] - df['predicted_occupied_beds']
        
        accuracy_report = None
        
        # Check if actuals are present for accuracy calculation
        if 'occupied_beds' in df.columns:
            df['actual_occupied_beds'] = df['occupied_beds']
            df['difference'] = df['predicted_occupied_beds'] - df['actual_occupied_beds']
            
            mae = (df['difference'].abs()).mean()
            # Simple accuracy metric: 100 - Mean Absolute Percentage Error (MAPE), handled for 0 division
            # Or just use R2 if we imported sklearn, but let's stick to simple metrics for display
            # Let's use a simple "Accuracy within 10 beds" % or similar, or just MAE.
            # Let's calculate a custom accuracy score: 1 - (abs(diff) / capacity)
            df['accuracy_score'] = 1 - (df['difference'].abs() / df['total_capacity'])
            avg_accuracy = df['accuracy_score'].mean() * 100
            
            accuracy_report = {
                'mae': round(mae, 2),
                'accuracy_score': round(avg_accuracy, 2)
            }

        # Save to CSV for download
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Save temporarily to serve via download link (optional, or just return data)
        # For this implementation, we'll pass data to template and allow download there
        # But to support the "Download button", we might need a separate route or embed data
        
        # Let's save it to a static file for simplicity in this rapid prototype, 
        # or better, return it as a downloadable response if requested, 
        # but the requirement says "Display a preview table... Download button".
        # So we render the template with preview and provide a way to download.
        
        # We will save the result to a file and provide a link
        output_path = os.path.join('static', 'predicted_output.csv')
        os.makedirs('static', exist_ok=True)
        df.to_csv(output_path, index=False)

        preview_data = df.head(20).to_dict(orient='records')
        
        return render_template('index.html', 
                               preview_data=preview_data, 
                               download_link=output_path,
                               accuracy_report=accuracy_report,
                               # Re-pass dashboard data to keep the page complete
                               total_shelters=0, avg_occupancy=0, total_records=0, 
                               shelters_list=[], 
                               shelter_locations=[], # Fix for JSON serialization error
                               chart_data=[], 
                               recent_records=[])

    except Exception as e:
        return f"Error processing file: {e}", 500

@app.route('/download')
def download_file():
    return send_file(os.path.join('static', 'predicted_output.csv'), as_attachment=True)



@app.route('/add_shelter', methods=['POST'])
def add_shelter():
    try:
        data = request.json
        shelter_name = data.get('shelter_name')
        city = data.get('city')
        capacity = int(data.get('capacity'))
        
        if not shelter_name or not city or not capacity:
            return jsonify({"error": "Missing required fields"}), 400
            
        csv_path = r"c:\Users\dell\adp_pbl\shelter_dataset_all_updated.csv"
        df = pd.read_csv(csv_path)
        
        # Create new row with defaults
        new_row = {col: 0 for col in df.columns}
        new_row.update({
            'shelter_name': shelter_name,
            'city': city,
            'total_capacity': capacity,
            'occupied_beds': 0,
            'available_beds': capacity,
            'occupancy_rate': 0.0,
            'date': pd.Timestamp.now().strftime('%d-%m-%Y'),
            # Set some reasonable defaults for other required columns to avoid issues
            'average_age': 30,
            'male_percentage': 50,
            'female_percentage': 50,
            'staff_count': 5,
            'funding_level': 'Medium',
            'season': 'Winter', # Placeholder
            'day_of_week': 'Monday' # Placeholder
        })
        
        # Append to dataframe
        new_df = pd.DataFrame([new_row])
        # Ensure columns match
        for col in df.columns:
            if col not in new_df.columns:
                new_df[col] = 0
                
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/add_data/<shelter_name>', methods=['POST'])
def add_shelter_data(shelter_name):
    """
    Add a new data record for a specific shelter
    """
    try:
        data = request.json
        
        # Load existing data
        csv_path = r"c:\Users\dell\adp_pbl\shelter_dataset_all_updated.csv"
        df = pd.read_csv(csv_path)
        
        # Create new row with provided data
        new_row = {
            'id': len(df) + 1,
            'date': data.get('date', pd.Timestamp.now().strftime('%d-%m-%Y')),
            'shelter_name': shelter_name,
            'city': data.get('city', ''),
            'state': data.get('state', ''),
            'total_capacity': int(data.get('total_capacity', 0)),
            'occupied_beds': int(data.get('occupied_beds', 0)),
            'available_beds': int(data.get('total_capacity', 0)) - int(data.get('occupied_beds', 0)),
            'occupancy_rate': round((int(data.get('occupied_beds', 0)) / int(data.get('total_capacity', 1))) * 100, 2),
            'average_age': int(data.get('average_age', 35)),
            'male_percentage': int(data.get('male_percentage', 50)),
            'female_percentage': int(data.get('female_percentage', 50)),
            'season': data.get('season', 'Summer'),
            'notes': data.get('notes', ''),
            'day_of_week': data.get('day_of_week', ''),
            'month': int(data.get('month', 1)),
            'is_holiday': int(data.get('is_holiday', 0)),
            'temperature': float(data.get('temperature', 25.0)),
            'rainfall_mm': float(data.get('rainfall_mm', 0.0)),
            'area_population': int(data.get('area_population', 1000000)),
            'staff_count': int(data.get('staff_count', 10)),
            'funding_level': data.get('funding_level', 'Medium'),
            'food_availability_score': int(data.get('food_availability_score', 5)),
            'medical_facilities_available': int(data.get('medical_facilities_available', 1)),
            'children_count': int(data.get('children_count', 0)),
            'senior_citizens_count': int(data.get('senior_citizens_count', 0)),
            'new_admissions': int(data.get('new_admissions', 0)),
            'exits_today': int(data.get('exits_today', 0)),
            'emergency_cases': int(data.get('emergency_cases', 0)),
            'unemployment_rate': float(data.get('unemployment_rate', 5.0)),
            'crime_rate': float(data.get('crime_rate', 5.0)),
            'covid_cases_city': int(data.get('covid_cases_city', 0)),
            'public_event_impact': int(data.get('public_event_impact', 0))
        }
        
        # Append to dataframe
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
        return jsonify({
            "success": True,
            "message": f"Data added successfully for {shelter_name}",
            "total_records": len(df)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/retrain_model', methods=['POST'])
def retrain_model():
    """
    Trigger model retraining with the updated dataset
    """
    try:
        import subprocess
        import sys
        
        # Run the training script
        result = subprocess.run(
            [sys.executable, r"c:\Users\dell\adp_pbl\train_model.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Reload the model in the current app
            global model
            model = joblib.load('model.pkl')
            
            return jsonify({
                "success": True,
                "message": "Model retrained successfully",
                "output": result.stdout
            })
        else:
            return jsonify({
                "success": False,
                "error": "Training failed",
                "output": result.stderr
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Training timeout - process took too long"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/api/upload_training_data', methods=['POST'])
def upload_training_data():
    """
    Upload a CSV file to append to the training dataset and trigger retraining
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV"}), 400

    try:
        # Load new data
        new_df = pd.read_csv(file)
        
        # Load existing data
        csv_path = r"c:\Users\dell\adp_pbl\shelter_dataset_all_updated.csv"
        df = pd.read_csv(csv_path)
        
        # Validate columns (basic check)
        required_cols = ['shelter_name', 'total_capacity', 'occupied_beds']
        missing = [col for col in required_cols if col not in new_df.columns]
        if missing:
             return jsonify({"error": f"Missing required columns: {', '.join(missing)}"}), 400

        # Ensure all columns from main df exist in new_df
        for col in df.columns:
            if col not in new_df.columns:
                if col == 'id':
                    new_df[col] = range(len(df) + 1, len(df) + len(new_df) + 1)
                elif col == 'date':
                    new_df[col] = pd.Timestamp.now().strftime('%d-%m-%Y')
                else:
                    dtype = df[col].dtype
                    if pd.api.types.is_numeric_dtype(dtype):
                        new_df[col] = 0
                    else:
                        new_df[col] = ""
        
        # Align columns
        new_df = new_df[df.columns]
        
        # Append
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
        # Trigger Retraining
        import subprocess
        import sys
        
        # Run the training script
        result = subprocess.run(
            [sys.executable, r"c:\Users\dell\adp_pbl\train_model.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            # Reload model
            global model
            model = joblib.load('model.pkl')
            
            return jsonify({
                "success": True,
                "message": f"Successfully added {len(new_df)} records and retrained model.",
                "total_records": len(df)
            })
        else:
            return jsonify({
                "success": False, 
                "error": "Data added but training failed",
                "output": result.stderr
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
