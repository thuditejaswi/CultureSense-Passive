import streamlit as st
import assemblyai as aai
from transformers import pipeline
import pandas as pd

# --------------------------
# Setup AssemblyAI
# --------------------------
aai.settings.api_key = "APIKEY"

# Hugging Face sentiment pipeline (text-based)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# Hugging Face SER pipeline (tone-based audio emotion recognition)
ser_pipeline = pipeline(
    task="audio-classification",
    model="superb/hubert-large-superb-er"
)

# Map Hugging Face sentiment labels to richer emotion descriptors
emotion_map = {
    "LABEL_0": "Negative 😠 (Frustration, Sadness, Anger)",
    "LABEL_1": "Neutral 😐 (Calm, Informative, Balanced)",
    "LABEL_2": "Positive 😊 (Excitement, Happiness, Agreement)"
}

# --------------------------
# Streamlit UI
# --------------------------
st.title("🎙️ Speech Emotion & Sentiment Analyzer (Text + Tone)")
st.write("Upload an audio file to analyze **per-speaker transcription, text sentiment, and vocal tone emotions**.")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save uploaded file locally
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    st.info("⏳ Transcribing... please wait.")

    # --------------------------
    # Transcription with AssemblyAI
    # --------------------------
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.universal,
        speaker_labels=True
    )

    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe("temp_audio.mp3")

    # --------------------------
    # Sentiment + SER Analysis
    # --------------------------
    data = []

    for utt in transcript.utterances:
        # Text sentiment
        sentiment = sentiment_pipeline(utt.text)[0]
        emotion_desc = emotion_map.get(sentiment["label"], "Unknown")

        # Tone-based SER (only analyze the audio segment)
        ser_result = ser_pipeline("temp_audio.mp3")
        ser_top = max(ser_result, key=lambda x: x["score"])  # best match

        data.append({
            "Speaker": f"Speaker {utt.speaker}",
            "Text": utt.text,
            "Text Sentiment": sentiment["label"],
            "Sentiment Score": round(sentiment["score"], 2),
            "Emotion Descriptor (Text)": emotion_desc,
            "SER Emotion (Tone)": ser_top["label"],
            "SER Confidence": round(ser_top["score"], 2)
        })

    df = pd.DataFrame(data)

    # --------------------------
    # Dashboard Display
    # --------------------------
    st.success("✅ Analysis Complete!")

    st.subheader("🗣️ Conversation with Emotions")
    for row in data:
        st.markdown(
            f"**{row['Speaker']}**: {row['Text']}  "
            f"→ *Text:* {row['Emotion Descriptor (Text)']} (conf {row['Sentiment Score']}) "
            f"| *Tone:* {row['SER Emotion (Tone)']} (conf {row['SER Confidence']})"
        )

    st.subheader("📊 Sentiment & Tone Dashboard")
    st.dataframe(df)

    # Group sentiment by speaker for chart
    sentiment_counts = df.groupby("Speaker")["Text Sentiment"].value_counts().unstack().fillna(0)
    st.bar_chart(sentiment_counts)

    tone_counts = df.groupby("Speaker")["SER Emotion (Tone)"].value_counts().unstack().fillna(0)
    st.bar_chart(tone_counts)

