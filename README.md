# Smart Shelter Management System

## About the Project
The **Smart Shelter Management System** is a comprehensive digital solution designed to optimize the operations of homeless shelters. By replacing manual record-keeping with a centralized dashboard and integrating advanced predictive analytics, the system empowers administrators to make data-driven decisions.

Key capabilities include forecasting future occupancy rates using Machine Learning (Random Forest Regressor), automatically estimating daily resource requirements (meals, medical kits, staff), and providing actionable recommendations to prevent overcrowding and ensure efficient resource allocation.

## Key Features
*   **Centralized Dashboard:** Real-time visualization of occupancy, capacity, and demographics.
*   **Predictive Analytics:** AI-powered forecasting of shelter occupancy with high accuracy.
*   **Resource Optimization:** Automated calculation of daily needs for food, water, and medical supplies.
*   **Smart Recommendations:** Actionable insights for staffing and maintenance based on data trends.
*   **Bulk Data Management:** Easy CSV upload feature for updating records and retraining the model.

## Installation & Setup

### Prerequisites
*   Python 3.8 or higher
*   pip (Python package installer)

### Step 1: Clone or Download
Download the project files to your local machine.

### Step 2: Install Dependencies
Open your terminal or command prompt, navigate to the project directory, and install the required Python packages:

```bash
pip install flask pandas scikit-learn joblib requests
```

### Step 3: Run the Application
Start the Flask server by running the following command:

```bash
python app.py
```

### Step 4: Access the Dashboard
Once the server is running, open your web browser and go to:

```
http://127.0.0.1:5000
```

## Usage
1.  **Dashboard:** View current stats and occupancy trends on the home page.
2.  **Shelter Details:** Click on a shelter to view detailed analytics, forecasts, and resource requirements.
3.  **Upload Data:** Use the "Bulk Upload" feature in the shelter detail page to add new data via CSV and automatically retrain the prediction model.
