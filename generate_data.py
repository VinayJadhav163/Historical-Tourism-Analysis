import pandas as pd
import random
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Generate Fake Data with 20 distinct places

places = [
    # Original 10
    "Ajanta Caves", "Ellora Caves", "Daulatabad Fort", 
    "Shani Shingnapur", "Trimbakeshwar Temple", 
    "Gateway of India", "Raigad Fort", 
    "Sindhudurg Fort", "Mahabaleshwar", "Shirdi",
    
    # 10 New Additions
    "Elephanta Caves", "Lonavala", "Khandala", 
    "Panchgani", "Pratapgad Fort", "Bhimashankar Temple",
    "Aga Khan Palace", "Matheran", "Kaas Plateau", "Tarkarli Beach"
]

districts = {
    "Ajanta Caves": "Aurangabad", "Ellora Caves": "Aurangabad", "Daulatabad Fort": "Aurangabad",
    "Shani Shingnapur": "Ahmednagar", "Trimbakeshwar Temple": "Nashik",
    "Gateway of India": "Mumbai", "Raigad Fort": "Raigad",
    "Sindhudurg Fort": "Sindhudurg", "Mahabaleshwar": "Satara", "Shirdi": "Ahmednagar",
    "Elephanta Caves": "Mumbai", "Lonavala": "Pune", "Khandala": "Pune",
    "Panchgani": "Satara", "Pratapgad Fort": "Satara", "Bhimashankar Temple": "Pune",
    "Aga Khan Palace": "Pune", "Matheran": "Raigad", "Kaas Plateau": "Satara", "Tarkarli Beach": "Sindhudurg"
}

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

seasons = {
    "January": "Winter", "February": "Winter", 
    "March": "Summer", "April": "Summer", "May": "Summer",
    "June": "Monsoon", "July": "Monsoon", "August": "Monsoon", "September": "Monsoon",
    "October": "Post-Monsoon", "November": "Post-Monsoon", "December": "Winter"
}

weather_map = {
    "Winter": ["Cool", "Pleasant"],
    "Summer": ["Hot", "Warm", "Humid"],
    "Monsoon": ["Rainy", "Humid"],
    "Post-Monsoon": ["Pleasant", "Clear"]
}

data = []
# Create vastly more deterministic data so the model has undeniable patterns to find
for _ in range(10000):
    place = random.choice(places)
    month = random.choice(months)
    season = seasons[month]
    weekend = random.choice(["Yes", "No"])
    
    # 1. CROWD LOGIC (Highly dependent on Weekend and Season)
    crowd_chance = 0.0
    if weekend == "Yes":
        crowd_chance += 0.5
    if season == "Winter" or season == "Post-Monsoon":
        crowd_chance += 0.4
    if place in ["Mahabaleshwar", "Lonavala", "Khandala"] and season == "Monsoon":
        crowd_chance += 0.8  # Very popular in monsoon
        
    if crowd_chance >= 0.8:
        visitor_count_level = "High"
    elif crowd_chance >= 0.4:
        visitor_count_level = "Medium"
    else:
        visitor_count_level = "Low"
        
    # 2. KIDS VISITORS (Dependent on Summer Vacation and Weekends)
    kids_chance = 0.0
    if season == "Summer":
        kids_chance += 0.6  # Summer vacation
    if weekend == "Yes":
        kids_chance += 0.3
        
    if kids_chance >= 0.7:
        kids = "High"
    elif kids_chance >= 0.3:
        kids = "Medium"
    else:
        kids = "Low"
        
    # 3. SENIOR CITIZENS (Highly dependent on Religious/Pilgrimage places)
    senior_chance = 0.0
    religious_places = ["Shirdi", "Shani Shingnapur", "Trimbakeshwar Temple", "Bhimashankar Temple"]
    if place in religious_places:
        senior_chance += 0.7
    if season not in ["Summer", "Monsoon"]:
        senior_chance += 0.2 # Prefer good weather
    
    if senior_chance >= 0.7:
        seniors = "High"
    elif senior_chance >= 0.3:
        seniors = "Medium"
    else:
        seniors = "Low"
        
    # 4. FOREIGN TOURISTS (Highly dependent on UNESCO/Famous places, mostly in good weather)
    foreign_chance = 0.0
    famous_places = ["Ajanta Caves", "Ellora Caves", "Gateway of India", "Elephanta Caves", "Aga Khan Palace"]
    if place in famous_places:
        foreign_chance += 0.8
    if season == "Monsoon":
        foreign_chance -= 0.4 # Avoid heavy rain
        
    if foreign_chance >= 0.6:
        foreign = "Yes"
    else:
        foreign = "No"

    data.append({
        "Place": place,
        "District": districts[place],
        "Month": month,
        "Season": season,
        "Weekend": weekend,
        "Weather": random.choice(weather_map[season]),
        "Visitor_Count_Level": visitor_count_level,
        "Visitor_Type": random.choice(["Families", "Students", "Foreign Tourists", "Couples", "Solo Travelers"]),
        "Foreign_Tourists": foreign,
        "Kids_Visitors": kids,
        "Senior_Citizens": seniors,
        "Business_Activity": random.choice(["High", "Medium", "Low"])
    })

df = pd.DataFrame(data)
df.to_csv("tourism_maharashtra_pattern_dataset.csv", index=False)
print(f"Generated data with shape {df.shape}")

# 2. Train and Save the model

X = df[["Place", "Month", "Season", "Weekend"]].copy()
targets = ["Visitor_Count_Level", "Kids_Visitors", "Senior_Citizens", "Foreign_Tourists"]
Y = df[targets]

print("Started encoding...")
encoders = {}
for column in X.columns:
    le = LabelEncoder()
    # explicitly fit and transform to avoid annoying warnings
    X[column] = le.fit_transform(X[column])
    encoders[column] = le
    
target_encoders = {}
Y_encoded = pd.DataFrame()
for col in targets:
    tle = LabelEncoder()
    Y_encoded[col] = tle.fit_transform(Y[col])
    target_encoders[col] = tle

X_train, X_test, y_train, y_test = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=42
)

print("Training models...")
models = {}
for target in targets:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[target])
    
    train_acc = model.score(X_train, y_train[target])
    test_acc = model.score(X_test, y_test[target])
    print(f"Model for {target} -> Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")
    
    models[target] = model

pickle.dump(
    (models, encoders, target_encoders),
    open("tourism_multi_prediction_model.pkl", "wb")
)

print("Saved model: tourism_multi_prediction_model.pkl!")
