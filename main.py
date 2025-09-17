import os
import assemblyai as aai
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
import shutil
import uvicorn

# -------------------------------
# Load API Key from environment
# -------------------------------
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not aai.settings.api_key:
    raise RuntimeError("❌ AssemblyAI API key is missing. Set ASSEMBLYAI_API_KEY env variable.")

# Hugging Face sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

app = FastAPI(title="Meeting Transcription & Sentiment API")

# -------------------------------
# Helper Functions
# -------------------------------
def analyze_sentiment(text: str):
    """Analyze sentiment of text using Hugging Face Transformers"""
    result = sentiment_pipeline(text[:512])[0]
    label = result['label']
    score = result['score']

    if label.upper() == "LABEL_0":
        label = "NEGATIVE"
    elif label.upper() == "LABEL_1":
        label = "NEUTRAL"
    elif label.upper() == "LABEL_2":
        label = "POSITIVE"

    return {"sentiment": label, "confidence": score}


def transcribe_audio(audio_path: str):
    """Transcribe audio file with AssemblyAI"""
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.universal,
        speaker_labels=True
    )
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)
    return transcript


# -------------------------------
# API Endpoints
# -------------------------------

@app.get("/")
def root():
    return {"message": "🚀 FastAPI backend for transcription + sentiment is running!"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Save uploaded file locally
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe with AssemblyAI
        transcript = transcribe_audio(file_path)

        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")

        # Analyze sentiment for each utterance
        utterance_results = []
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

        for utt in transcript.utterances:
            sent = analyze_sentiment(utt.text)
            sentiment_counts[sent["sentiment"]] += 1
            utterance_results.append({
                "speaker": utt.speaker,
                "text": utt.text,
                "sentiment": sent["sentiment"],
                "confidence": round(sent["confidence"], 2)
            })

        total = len(transcript.utterances)
        summary = {
            s: {"count": c, "percentage": round((c / total) * 100, 1) if total > 0 else 0}
            for s, c in sentiment_counts.items()
        }

        return {
            "status": "success",
            "utterances": utterance_results,
            "summary": summary
        }

    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


# -------------------------------
# Run with: uvicorn main:app --reload
# -------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
