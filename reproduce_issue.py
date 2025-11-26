import requests
import pandas as pd
import time

url = 'http://127.0.0.1:5000/add_shelter'
csv_path = r"c:\Users\dell\adp_pbl\shelter_dataset_all_updated.csv"

def get_row_count():
    df = pd.read_csv(csv_path)
    return len(df)

initial_count = get_row_count()
print(f"Initial row count: {initial_count}")

# Add Shelter 1
shelter1 = {"shelter_name": f"Test 1 {time.time()}", "city": "Bangalore", "capacity": 100}
requests.post(url, json=shelter1)
count_after_1 = get_row_count()
print(f"Count after adding 1: {count_after_1}")

# Add Shelter 2
shelter2 = {"shelter_name": f"Test 2 {time.time()}", "city": "Bangalore", "capacity": 100}
requests.post(url, json=shelter2)
count_after_2 = get_row_count()
print(f"Count after adding 2: {count_after_2}")

if count_after_2 == count_after_1 + 1 == initial_count + 2:
    print("SUCCESS: Rows added correctly.")
else:
    print("FAILURE: Row counts do not match expected increase.")
