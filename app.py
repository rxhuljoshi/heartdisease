import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import librosa.display
from model import load_model, predict
from utils import process_audio_file, get_prediction_label, get_condition_description
from datetime import datetime
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Heart Sound Classifier",
    page_icon="❤️",
    layout="wide"
)

def save_results(patient_name, patient_id, age, gender, date, symptoms, prediction, confidence):
    """Save patient results to a CSV file."""
    # Create records directory if it doesn't exist
    os.makedirs('data/records', exist_ok=True)
    
    # Prepare the record
    record = {
        'patient_name': patient_name,
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'date': date,
        'symptoms': symptoms,
        'prediction': prediction,
        'confidence': f"{confidence:.2%}",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Convert to DataFrame
    df_record = pd.DataFrame([record])
    
    # Path to records file
    records_file = 'data/records/patient_records.csv'
    
    # Append to existing file or create new one
    if os.path.exists(records_file):
        df_record.to_csv(records_file, mode='a', header=False, index=False)
    else:
        df_record.to_csv(records_file, index=False)

def main():
    st.title("Heart Sound Classification System")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Classification", "Saved Records"])
    
    with tab1:
        st.write("Upload a heart sound recording to classify it.")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
        
        # Patient details section
        st.subheader("Patient Details")
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        with col2:
            patient_id = st.text_input("Patient ID")
            date = st.date_input("Date")
            symptoms = st.text_area("Symptoms")
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process audio file
            try:
                # Extract features
                features = process_audio_file(temp_path)
                
                if features is not None:
                    # Display waveform
                    st.subheader("Audio Waveform")
                    audio, sr = librosa.load(temp_path)
                    fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
                    librosa.display.waveshow(audio, sr=sr)
                    plt.title("Waveform")
                    plt.tight_layout()
                    st.pyplot(fig_wave)
                    
                    # Load model
                    model = load_model('data/models/heart_sound_model.joblib')
                    
                    # Make prediction
                    prediction, probabilities = predict(model, features)
                    
                    # Display results
                    st.subheader("Classification Results")
                    
                    # Create two columns for results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create a bar chart of probabilities
                        classes = ['Normal', 'Noisy Normal', 'Murmur', 'Noisy Murmur', 'Extrasystole']
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.bar(classes, probabilities)
                        
                        # Color the bars based on prediction
                        for i, bar in enumerate(bars):
                            if i == prediction:
                                bar.set_color('green')
                            else:
                                bar.set_color('lightgray')
                        
                        plt.title("Prediction Probabilities")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        # Calculate risk score (higher for abnormal conditions)
                        risk_score = 0
                        if classes[prediction] in ['Murmur', 'Noisy Murmur', 'Extrasystole']:
                            risk_score = probabilities[prediction] * 100
                        elif classes[prediction] == 'Noisy Normal':
                            risk_score = probabilities[prediction] * 50
                        
                        # Create circular gauge with smaller size
                        fig_gauge, ax_gauge = plt.subplots(figsize=(4, 4))
                        
                        # Create the gauge
                        gauge_color = 'red' if risk_score > 50 else 'green'
                        
                        # Draw the gauge background (full circle)
                        circle = Circle((0.5, 0.5), 0.35, color='lightgray', transform=ax_gauge.transAxes)
                        ax_gauge.add_patch(circle)
                        
                        # Draw the gauge value (partial circle)
                        angle = risk_score * 3.6  # Convert percentage to degrees (360 * percentage/100)
                        arc = Arc((0.5, 0.5), 0.7, 0.7, theta1=90, theta2=90-angle, 
                                color=gauge_color, transform=ax_gauge.transAxes)
                        ax_gauge.add_patch(arc)
                        
                        # Set gauge properties
                        ax_gauge.set_xlim(0, 1)
                        ax_gauge.set_ylim(0, 1)
                        ax_gauge.axis('off')
                        
                        # Add value text
                        ax_gauge.text(0.5, 0.5, f'{risk_score:.1f}%', 
                                    ha='center', va='center', 
                                    fontsize=16, fontweight='bold')
                        
                        # Add title
                        plt.title("Heart Disease Risk Level", pad=20)
                        plt.tight_layout()
                        st.pyplot(fig_gauge)
                        
                        # Add risk level description
                        if risk_score > 75:
                            st.warning("High risk of heart disease. Immediate medical attention recommended.")
                        elif risk_score > 50:
                            st.warning("Moderate risk of heart disease. Medical consultation advised.")
                        elif risk_score > 25:
                            st.info("Low to moderate risk. Regular check-ups recommended.")
                        else:
                            st.success("Low risk of heart disease. Maintain regular heart health monitoring.")
                    
                    # Display prediction and confidence
                    st.write(f"Prediction: {classes[prediction]}")
                    st.write(f"Confidence: {probabilities[prediction]:.2%}")
                    
                    # Display condition description
                    st.subheader("Condition Description")
                    st.write(get_condition_description(classes[prediction]))
                    
                    # Save results
                    if st.button("Save Results"):
                        save_results(patient_name, patient_id, age, gender, date, 
                                   symptoms, classes[prediction], probabilities[prediction])
                        st.success("Results saved successfully!")
                
            except Exception as e:
                st.error(f"Error processing audio file: {str(e)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    with tab2:
        st.subheader("Saved Patient Records")
        
        # Create directory for saved records if it doesn't exist
        os.makedirs('data/records', exist_ok=True)
        
        # Load and display saved records
        records_file = 'data/records/patient_records.csv'
        if os.path.exists(records_file):
            df = pd.read_csv(records_file)
            
            # Add search functionality
            search_term = st.text_input("Search records by patient name or ID")
            if search_term:
                df = df[df['patient_name'].str.contains(search_term, case=False) | 
                       df['patient_id'].str.contains(search_term, case=False)]
            
            # Display records in a table
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                # Add download button for records
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Records",
                    data=csv,
                    file_name="patient_records.csv",
                    mime="text/csv"
                )
            else:
                st.info("No records found matching your search.")
        else:
            st.info("No saved records found.")

if __name__ == "__main__":
    main()

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created with ❤️ for heart sound classification</p>
    <p>Using machine learning to help diagnose heart conditions</p>
</div>
""", unsafe_allow_html=True) 