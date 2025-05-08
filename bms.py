import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.title("Battery Voltage Discharge Analysis with Classification")

if st.button("Analyse & Classify"):

    try:
        # Connect to MySQL
        conn = mysql.connector.connect(
            host="82.180.143.66",
            user="u263681140_students",
            password="testStudents@123",
            database="u263681140_students"
        )

        # SQL query
        query = """
        SELECT 
            DATE(dateTime) AS date,
            MAX(CAST(vtg AS DECIMAL(10, 2))) AS max_voltage,
            MIN(CAST(vtg AS DECIMAL(10, 2))) AS min_voltage,
            (MAX(CAST(vtg AS DECIMAL(10, 2))) - MIN(CAST(vtg AS DECIMAL(10, 2)))) AS daily_discharge
        FROM BMS1
        WHERE CAST(vtg AS DECIMAL(10, 2)) > 9
        GROUP BY DATE(dateTime)
        ORDER BY DATE(dateTime);
        """

        # Read data into DataFrame
        df = pd.read_sql(query, conn)

        # Label abnormal discharge (> 0.5V drop)
        df['label'] = df['daily_discharge'].apply(lambda x: 1 if x > 0.5 else 0)

        # Features and target
        X = df[['max_voltage', 'min_voltage', 'daily_discharge']]
        y = df['label']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Add predictions
        df['prediction'] = clf.predict(X)

        # ✅ Map numerical labels to text
        df['label'] = df['label'].map({0: 'Normal', 1: 'Abnormal'})
        df['prediction'] = df['prediction'].map({0: 'Normal', 1: 'Abnormal'})

        # Display DataFrame
        st.success("Data Retrieved and Classified Successfully!")
        st.dataframe(df)

        # Chart
        st.line_chart(df.set_index('date')['daily_discharge'])

        # Classification Report
        y_pred = clf.predict(X_test)
        st.text("Classification Report:")
        st.text(classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Abnormal"],
            zero_division=0
        ))

        conn.close()

    except Exception as e:
        st.error(f"Error: {e}")
