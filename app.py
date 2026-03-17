from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

models, encoders, target_encoders = pickle.load(open("tourism_multi_prediction_model.pkl","rb"))

places = [
    "Ajanta Caves", "Ellora Caves", "Daulatabad Fort", 
    "Shani Shingnapur", "Trimbakeshwar Temple", 
    "Gateway of India", "Raigad Fort", 
    "Sindhudurg Fort", "Mahabaleshwar", "Shirdi",
    "Elephanta Caves", "Lonavala", "Khandala", 
    "Panchgani", "Pratapgad Fort", "Bhimashankar Temple",
    "Aga Khan Palace", "Matheran", "Kaas Plateau", "Tarkarli Beach"
]

months = [
"January","February","March","April","May","June",
"July","August","September","October","November","December"
]

seasons = ["Winter","Summer","Monsoon","Post-Monsoon"]

weekends = ["Yes","No"]

@app.route("/", methods=["GET","POST"])
def index():

    result = None
    graph_data = None

    if request.method == "POST":

        place = request.form["place"]
        month = request.form["month"]
        season = request.form["season"]
        weekend = request.form["weekend"]

        data = pd.DataFrame([{
            "Place":place,
            "Month":month,
            "Season":season,
            "Weekend":weekend
        }])

        for col in encoders:
            data[col] = encoders[col].transform(data[col])

        results = {}

        for target in models:
            pred = models[target].predict(data)
            results[target] = target_encoders[target].inverse_transform(pred)[0]

        # Calculate demographic percentages (out of 100 tourists)
        kids_pct = 38 if results["Kids_Visitors"] == 'High' else 18 if results["Kids_Visitors"] == 'Medium' else 8
        senior_pct = 25 if results["Senior_Citizens"] == 'High' else 14 if results["Senior_Citizens"] == 'Medium' else 6
        foreign_pct = 18 if results["Foreign_Tourists"] == 'Yes' else 4
        adults_pct = 100 - (kids_pct + senior_pct + foreign_pct)

        result = {
            "place": place,
            "crowd": results["Visitor_Count_Level"],
            "kids": results["Kids_Visitors"],
            "senior": results["Senior_Citizens"],
            "foreign": results["Foreign_Tourists"],
            "kids_pct": kids_pct,
            "senior_pct": senior_pct,
            "foreign_pct": foreign_pct,
            "adults_pct": adults_pct
        }

        graph_data = [kids_pct, senior_pct, foreign_pct, adults_pct]

    return render_template(
        "index.html",
        places=places,
        months=months,
        seasons=seasons,
        weekends=weekends,
        result=result,
        graph_data=graph_data
    )

if __name__ == "__main__":
    app.run(debug=True)