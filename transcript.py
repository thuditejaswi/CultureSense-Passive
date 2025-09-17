import assemblyai as aai
import json
from datetime import datetime
from transformers import pipeline
import streamlit as st

# ----------------------------------------------------
aai.settings.api_key = "27879e3826214e3885ad655d718e41dc"

# Hugging Face sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

def analyze_sentiment(text):
    """Analyze sentiment of text using Hugging Face Transformers"""
    result = sentiment_pipeline(text[:512])[0]  # truncate to 512 tokens max
    label = result['label']
    score = result['score']

    if label.upper() == "LABEL_0":
        label = "NEGATIVE"
    elif label.upper() == "LABEL_1":
        label = "NEUTRAL"
    elif label.upper() == "LABEL_2":
        label = "POSITIVE"

    return label, score

def transcribe(audio_file_path):
    """Transcribe audio file with speaker diarization"""
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.universal,
        speaker_labels=True
    )
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_file_path)
    return transcript

# --------------------------
# Streamlit Dashboard
# --------------------------
st.set_page_config(page_title="Meeting Transcription Dashboard", layout="wide")
st.title("🎙 Meeting Transcription with Sentiment Analysis")

uploaded_file = st.file_uploader("Upload an audio file (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save uploaded file locally
    audio_path = f"uploaded_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Transcribing... please wait ⏳"):
        transcript = transcribe(audio_path)

    if transcript.status == aai.TranscriptStatus.error:
        st.error(f"Transcription failed: {transcript.error}")
    else:
        st.success("✅ Transcription completed successfully!")

        # Display utterances with sentiment
        st.subheader("Conversation Details")
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

        for utterance in transcript.utterances:
            sentiment, score = analyze_sentiment(utterance.text)
            sentiment_counts[sentiment] += 1

            st.markdown(
                f"""
                **Speaker {utterance.speaker}:**  
                {utterance.text}  
                _Sentiment: **{sentiment}** (Confidence: {score:.2f})_
                ---
                """
            )

        # Summary
        st.subheader("📊 Sentiment Summary")
        total_utterances = len(transcript.utterances)
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_utterances * 100) if total_utterances > 0 else 0
            st.write(f"- {sentiment}: {count} utterances ({percentage:.1f}%)")

        # Chart
        st.bar_chart(sentiment_counts)
