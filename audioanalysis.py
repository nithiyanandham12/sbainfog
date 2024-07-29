import streamlit as st
import pandas as pd
import numpy as np
import librosa

def analyze_audio(file):
    # Load audio file
    y, sr = librosa.load(file, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Split the audio into 10 segments
    segment_length = len(y) // 10
    segments = [y[i*segment_length:(i+1)*segment_length] for i in range(10)]
    
    # Manually set segment durations for demonstration
    segment_durations = np.linspace(0.5, total_duration, 10)
    
    data = {
        'Speaker': [],
        'Segment Duration': [],
        'RMS Energy': [],
        'Zero Crossing Rate': [],
        'Spectral Centroid': [],
        'Spectral Bandwidth': [],
        'Sentiment Analysis': []
    }
    
    for i, segment in enumerate(segments):
        rms = librosa.feature.rms(y=segment).mean()
        zcr = librosa.feature.zero_crossing_rate(segment).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr).mean()
        
        # Mock sentiment analysis for example purposes
        sentiment = np.random.choice(['Positive', 'Neutral', 'Negative'])
        speaker = np.random.choice(['Agent', 'Customer'])
        
        data['Speaker'].append(speaker)
        data['Segment Duration'].append(segment_durations[i])
        data['RMS Energy'].append(rms)
        data['Zero Crossing Rate'].append(zcr)
        data['Spectral Centroid'].append(spectral_centroid)
        data['Spectral Bandwidth'].append(spectral_bandwidth)
        data['Sentiment Analysis'].append(sentiment)
    
    df = pd.DataFrame(data)
    return df

st.title('Audio File Analysis')

uploaded_file = st.file_uploader('Upload an audio file', type=['wav', 'mp3'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    df = analyze_audio(uploaded_file)

    st.write('Analysis Results:')
    st.dataframe(df)

    # Convert DataFrame to Excel
    excel_file = 'analysis_results.xlsx'
    df.to_excel(excel_file, index=False)

    with open(excel_file, 'rb') as f:
        st.download_button('Download Excel file', f, file_name='analysis_results.xlsx')
