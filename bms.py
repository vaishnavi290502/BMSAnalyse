import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Streamlit title
st.title("🔋 Battery Voltage Discharge Analysis with Classification")

# Run on button click
if st.button("Analyse & Classify"):

    try:
        # Connect to MySQL
        conn = mysql.connector.connect(
            host="82.180.143.66",
            user="u263681140_students",
            password="testStudents@123",
            database="u263681140_students"
        )

        # SQL query to fetch daily voltage stats
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

        # Load data into DataFrame
        df = pd.read_sql(query, conn)

        # Label: 1 for abnormal (discharge > 0.5V), 0 for normal
        df['label'] = df['daily_discharge'].apply(lambda x: 1 if x > 0.5 else 0)

        # Prepare features and target
        X = df[['max_voltage', 'min_voltage', 'daily_discharge']]
        y = df['label']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Make predictions
        df['prediction'] = clf.predict(X)

        # Convert labels to text
        df['label'] = df['label'].map({0: 'Normal', 1: 'Abnormal'})
        df['prediction'] = df['prediction'].map({0: 'Normal', 1: 'Abnormal'})

        # Calculate average daily discharge
        avg_discharge = df['daily_discharge'].mean()

        # Estimate battery life based on average discharge
        if avg_discharge <= 0.8:
            battery_life = "4 years"
        elif avg_discharge <= 1.0:
            battery_life = "3 years"
        elif avg_discharge <= 1.5:
            battery_life = "2 years"
        else:
            battery_life = "1 year"

        # Show results
        st.success("✅ Data Retrieved and Classified Successfully!")
        st.dataframe(df)

        st.line_chart(df.set_index('date')['daily_discharge'])

        st.info(f"🔋 **Average Daily Discharge:** {avg_discharge:.2f} V")
        st.warning(f"⏳ **Estimated Battery Life:** {battery_life}")

        # Show classification report
        y_pred = clf.predict(X_test)
        st.text("📊 Classification Report:")
        st.text(classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Abnormal"],
            zero_division=0
        ))

        conn.close()

    except Exception as e:
        st.error(f"❌ Error: {e}")
