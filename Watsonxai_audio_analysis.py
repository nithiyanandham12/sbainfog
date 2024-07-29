import streamlit as st
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import requests
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import base64
import io

# IBM Watsonx.ai text generation API details
api_key = "V-4HX4rNj55Qyt1b6NsV-gkHACibNleI4Tq1qJ9qdxj3"
url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
project_id = "8a699899-a177-47e4-8c9f-1c018d07dc8c"
model_id = "ibm/granite-13b-chat-v2"
stt_api_key = "FL2nWrD69WEz1J_P9c4Z2IaWXdPUodKvJXyowYoOkbE0"
stt_url = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/9c5cbeda-0cad-4f48-a55b-13e57908860c"

def get_access_token(api_key):
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }

    response = requests.post(auth_url, headers=headers, data=data)

    if response.status_code != 200:
        raise Exception("Failed to get access token: " + str(response.text))

    token_info = response.json()
    return token_info['access_token']

access_token = get_access_token(api_key)

def get_watson_sentiment_analysis(text, access_token):
    body = {
        "input": f"""
Analyze the sentiment of the following text: \"{text}\"
Provide the sentiment as Positive, Negative or Neutral.
""",
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 50,
            "repetition_penalty": 1.05
        },
        "model_id": model_id,
        "project_id": project_id
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.post(url, headers=headers, json=body)

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    sentiment = data['results'][0]['generated_text'].strip()
    return sentiment

def transcribe_audio(file_path):
    authenticator = IAMAuthenticator(stt_api_key)
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(stt_url)

    with open(file_path, 'rb') as audio_file:
        response = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav',  # Ensure the file is in WAV format
            timestamps=True,
            speaker_labels=True
        ).get_result()

    return response

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
        
        # Convert the segment to text using IBM Watson Speech-to-Text
        segment_file_path = f"temp_segment_{i}.wav"
        sf.write(segment_file_path, segment, sr)

        response = transcribe_audio(segment_file_path)

        segment_text = ""
        if 'results' in response and len(response['results']) > 0:
            segment_text = response['results'][0]['alternatives'][0]['transcript']

        sentiment = get_watson_sentiment_analysis(segment_text, access_token)
        
        speaker = np.random.choice(['Agent', 'Customer'])  # Placeholder for actual speaker identification
        
        data['Speaker'].append(speaker)
        data['Segment Duration'].append(segment_durations[i])
        data['RMS Energy'].append(rms)
        data['Zero Crossing Rate'].append(zcr)
        data['Spectral Centroid'].append(spectral_centroid)
        data['Spectral Bandwidth'].append(spectral_bandwidth)
        data['Sentiment Analysis'].append(sentiment)
    
    df = pd.DataFrame(data)
    return df

def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load logo image and encode it as base64
logo_path = "sba.jpg"
logo_base64 = get_image_as_base64(logo_path)

st.sidebar.markdown(
    f"""
    <div style="text-align:center;">
        <img src="data:image/jpeg;base64,{logo_base64}" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.title('Upload an audio file')
uploaded_file = st.sidebar.file_uploader('Choose a file', type=['wav', 'mp3'])

st.title('Audio File Analysis - SBA Info Solutions')

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
