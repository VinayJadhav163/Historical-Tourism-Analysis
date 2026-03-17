# 🏛 AI Historical Tourism Analysis

A modern, full-stack predictive AI dashboard that analyzes and forecasts historical and demographic tourism patterns across 20 distinct tourist destinations in Maharashtra, India. 

![Dashboard Overview](static/dashboard_preview.png) *(Preview of the final UI)*

---

## 📜 Table of Contents
1. [Project Overview](#-project-overview)
2. [Technology Stack](#-technology-stack)
3. [How We Built It (Start to Finish)](#-how-we-built-it-start-to-finish)
    - [Phase 1: Dataset Generation](#phase-1-synthetic-dataset-generation)
    - [Phase 2: Machine Learning](#phase-2-machine-learning-training)
    - [Phase 3: The Backend Framework](#phase-3-the-flask-backend)
    - [Phase 4: UI/UX & Dashboard](#phase-4-modern-ui--dashboard)
4. [Installation & Usage](#-installation--usage)

---

## 🚀 Project Overview

This project was built to answer complex questions about tourism logistics: *When weekend crowds surge at Mahabaleshwar during the monsoon, what percentage of them are children? Are foreign tourists more likely to visit the Ajanta Caves in the Summer or Post-Monsoon?*

The system features:
*   **Predictive AI:** Four independent machine learning models trained specifically to identify combinations of destinations, changing seasons, and weekend influxes.
*   **Demographic Mapping:** Granular percent-based breakdowns of the crowd (Kids vs Seniors vs Foreign Tourists).
*   **A Beautiful Dashboard:** A sleek, glassmorphism UI utilizing a custom hand-drawn SVG logo and animated Chart.js visual analytics.

---

## 💻 Technology Stack

*   **Frontend:** HTML5, CSS3 (Vanilla), JavaScript, Chart.js (Doughnut representations)
*   **Backend:** Python 3, Flask framework
*   **Machine Learning:** Scikit-Learn (RandomForestClassifier, LabelEncoder)
*   **Data Processing:** Pandas 

---

## 📁 Repository Structure

```text
Tourism_Analysis_Project/
│
├── static/
│   ├── style.css             # Main styling encompassing the glassmorphism dashboard grid
│   └── dashboard_preview.png # Preview image showing what the UI looks like
│
├── templates/
│   └── index.html            # The singular frontend block containing all markup and Chart.js code
│
├── app.py                    # Core Flask application that intakes data, loads the model, formats output arrays, and serves pages
├── generate_data.py          # Data generation and machine learning training script we created during Phase 1
├── tourism_maharashtra_pattern_dataset.csv # Static CSV dataset holding the 10,000 algorithmic rows of visitor demographic combinations
├── tourism_multi_prediction_model.pkl      # Binary snapshot storing 4 trained RandomForestClassifier AI modules 
├── requirements.txt          # File mapping framework and dependency version requirements
└── README.md                 # This complete documentation file you are currently reading!
```

---

## 🏗 How We Built It (Start to Finish)

### Phase 1: Synthetic Dataset Generation
Because high-quality, granular geographic tourism demographic data is often inaccessible, we built a custom Python Data Generator (`generate_data.py`).

**How Data is Spawned:**
1.  **Defining Variables:** We compiled list arrays of 20 distinct high-interest Maharashtra destinations (from Elephanta Caves to Shirdi to the Kaas Plateau), alongside arrays covering all 12 Months and mapping dictionaries for Seasons and Weather Outcomes.
2.  **Iterative Loop:** We instructed the Python script to trigger a massive `for _ in range(10000)` loop to spawn 10,000 distinct virtual interactions.
3.  **Deterministic Logic Rules:** Rather than merely relying on Python's `random.choice`, we mathematically weighted the target columns using explicit conditional bounds to create clear logical profiles for the ML modules to spot:
    *   **Senior Probability Boost:** `+70%` chance whenever the loop selects religious destinations like Trimbakeshwar Temple.
    *   **Foreign Probability Boost:** `+80%` chance whenever the loop pulls UNESCO locations like Ajanta Caves.
    *   **Crowd Spikes:** `+80%` probability boost for heavy crowds on weekends occurring specifically during optimal seasons.
4.  **Export:** At the end of the script cycle, Python dumps the list of 10,000 intelligently generated interactions into a Pandas DataFrame and saves it locally as `tourism_maharashtra_pattern_dataset.csv`.

### Phase 2: Machine Learning Training
We processed the 10,000-row dataset to predict four separate target outcomes based on the 4 categorical inputs (Place, Month, Season, Weekend status).
1.  **Label Encoding:** Passed string categories (like "Winter" or "High") into integers securely using `LabelEncoder`.
2.  **Random Forest Classification:** Spun up four parallel `RandomForestClassifier` modules. Due to the strict logic engine established in Phase 1, the AI mathematically learned the underlying patterns flawlessly, returning **100% predictive accuracy** on validation testing.
3.  **Pickle Serialization:** The multi-target models along with their respective state-encoders were cached sequentially inside a `tourism_multi_prediction_model.pkl` binary file.

### Phase 3: The Flask Backend
With the Pickle artifact containing the intelligence, we wrapped it in a dynamic Python Flask app (`app.py`).
1.  **Form Ingestion:** Configured HTTP `POST` routing to ingest the user's selected Place, Month, Season, and Weekend status.
2.  **Mapping Matrix:** Categorical predictions returned by the ML Pickle (e.g., "High" or "Low") don't map well to graphs natively. We implemented a conversion matrix in Python mapping exact demographic percentile splits based on the categorical severities, ensuring the numbers always formulate a clean base of exactly 100 tourists.

### Phase 4: Modern UI & Dashboard
The final step involved an extreme frontend rewrite directly targeting `index.html` and `style.css`.
1.  **Grid Dashboard:** Transformed the basic vertical form into a responsive, side-by-side SaaS dashboard block.
2.  **Color Palette:** Deployed a stunning `#0f172a` indigo background augmented with frosted glassmorphism cards and vibrant KPI data.
3.  **Visualizations:** Incorporated *Chart.js* bound to the payload passed by Flask to draw animated and proportional Doughnut Charts mapping the demographic percentage thresholds directly beside the categorical values via custom tooltip callbacks.

---

## 🛠 Installation & Usage

### 1. Requirements
Ensure you have `Python 3.8+` installed.

### 2. Setup
Clone the repository and jump into your isolated environment.
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate 
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-Generate Dataset & Retrain Model
If you've altered logic inside `generate_data.py`, simply run it to spawn a fresh dataset and overwrite the existing Pickle model:
```bash
python generate_data.py
```

### 5. Run the Dashboard
```bash
python app.py
```
After executing, navigate locally in your browser to `http://127.0.0.1:5000` to interact with the finished dashboard.
