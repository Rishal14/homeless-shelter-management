import pandas as pd
import random

# Mock data
data = {
    'shelter_name': ['Shelter A', 'Shelter B', 'Shelter C'],
    'city': ['Bangalore', 'Bangalore', 'Bangalore']
}
df = pd.DataFrame(data)

CITY_COORDINATES = {
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946}
}

shelter_locations = []
unique_shelters = df.drop_duplicates(subset=['shelter_name'])

print("Generating coordinates with jitter...")
for _, row in unique_shelters.iterrows():
    city = row['city']
    if city in CITY_COORDINATES:
        lat_jitter = random.uniform(-0.02, 0.02)
        lon_jitter = random.uniform(-0.02, 0.02)
        
        loc = {
            'name': row['shelter_name'],
            'city': city,
            'lat': CITY_COORDINATES[city]['lat'] + lat_jitter,
            'lon': CITY_COORDINATES[city]['lon'] + lon_jitter
        }
        shelter_locations.append(loc)
        print(f"{loc['name']}: {loc['lat']}, {loc['lon']}")

# Verify they are different
lats = [x['lat'] for x in shelter_locations]
if len(set(lats)) == len(lats):
    print("\nSUCCESS: All latitudes are unique.")
else:
    print("\nFAILURE: Latitudes are not unique.")
