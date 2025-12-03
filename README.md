# Made by: Aarav Agarwal, IBCP-XI
## Summative Assessment 6

### Project Overview

This project is an interactive Streamlit dashboard built for the Summative Assessment of Mathematics for AI-II. <br/>
The goal is to analyze hospital inpatient discharge data and provide actionable insights for hospital administrators, clinicians, and operations managers. <br/>

The dashboard allows users to explore:
- Length of Stay (LOS) trends
- Hospital charges distribution
- High-utilization departments
- Severity-based cost patterns
- Diagnosis-related LOS differences
- Patient demographics and payment types
- County and facility performance patterns

The dashboard is fully interactive and built using:
- Python (pandas, numpy)
- Plotly / Matplotlib
- Streamlit
- GitHub + Streamlit Cloud Deployment
  
### Features Included
This dashboard satisfies the Summative Assessment requirements with 5+ interactive visualizations:
1. Bar Chart — Average Length of Stay per Diagnosis
2. Boxplot — Total Charges segmented by Severity
3. Heatmap — Average Length of Stay by Facility × County
4. Pie Chart — Patient distribution by Payment Type
5. Histogram — Distribution of Length of Stay

### Running the Project Locally
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies using the requirements.txt
4. Place your dataset in the data/ folder
5. Run the app using:
streamlit run app.py

This will launch the dashboard locally on your browser.

### Deployment Instructions (Streamlit Cloud)
1. Push your repo to GitHub
2. Visit Streamlit Cloud and select Deploy an App
3. Connect GitHub → choose your repository
4. Set app.py as the startup file
5. Deploy
Streamlit will build the app and provide a public URL.

### Data Preprocessing Performed
The preprocessing pipeline includes:
- Handling missing values in LOS, charges, severity
- Converting diagnosis codes to readable labels (where applicable)
- Converting charges from string → float
- Creating age groups (0–20, 21–40, 41–60, 60+)
- Standardizing category columns (county, facility, gender, race)
- Removing outliers in length of stay and charges
- Creating grouped summaries needed for visuals
- Creating pivot tables for heatmaps

### Exploratory Data Analysis (EDA)
The following questions were answered using EDA:
- Which diagnosis codes lead to longest hospital stays?
- Which age groups incur the highest charges?
- Do higher-severity patients stay longer or cost more?
- Are specific counties or facilities responsible for most long-stay cases?
- How do different payment types influence total billing?
- EDA includes statistical summaries, grouped metrics, trends, histograms, boxplots, and heatmaps.

### Live Streamlit link: https://iadai102-1000197-aarav-agarwal-n49gmgfujgnf7p3pqd5efs.streamlit.app/

### Citations <br/>
https://chatgpt.com/ <br/>
https://docs.streamlit.io/develop/api-reference/status <br/>
https://blog.streamlit.io/improving-healthcare-management-with-streamlit/ <br/>


