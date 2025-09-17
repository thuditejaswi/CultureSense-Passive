import os
import assemblyai as aai
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
import shutil
import uvicorn
from collections import defaultdict

# -------------------------------
# Load API Key
# -------------------------------
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not aai.settings.api_key:
    raise RuntimeError("❌ AssemblyAI API key is missing. Set ASSEMBLYAI_API_KEY env variable.")

# Hugging Face pipelines
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

emotion_pipeline = pipeline(
    task="audio-classification",
    model="superb/hubert-large-superb-er"
)

app = FastAPI(title="Meeting Transcription, Sentiment & Emotion API")

# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def root():
    return {
        "message": "Welcome 👋 This is the Meeting Transcription, Sentiment & Emotion API. Go to /docs to try it out."
    }

# -------------------------------
# Helpers
# -------------------------------
def analyze_sentiment(text: str):
    result = sentiment_pipeline(text[:512])[0]
    label = result['label']
    score = result['score']

    if label.upper() == "LABEL_0":
        label = "NEGATIVE"
    elif label.upper() == "LABEL_1":
        label = "NEUTRAL"
    elif label.upper() == "LABEL_2":
        label = "POSITIVE"

    return {"sentiment": label, "confidence": round(score, 2)}


def analyze_emotion(audio_path: str):
    results = emotion_pipeline(audio_path)
    if isinstance(results, list) and len(results) > 0:
        top = results[0]
        return {"emotion": top["label"], "confidence": round(top["score"], 2)}
    return {"emotion": "UNKNOWN", "confidence": 0.0}


def give_feedback(sentiment_stats: dict):
    """Generate textual feedback from sentiment breakdown"""
    if sentiment_stats["POSITIVE"]["percentage"] > 50:
        return "Mostly positive and encouraging tone 👍"
    elif sentiment_stats["NEGATIVE"]["percentage"] > 40:
        return "Frequent negative expressions — consider being more constructive ⚠️"
    elif sentiment_stats["NEUTRAL"]["percentage"] > 60:
        return "Neutral and factual, but try adding more positivity 🙂"
    else:
        return "Balanced communication."


def transcribe_audio(audio_path: str):
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.universal,
        speaker_labels=True
    )
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)
    return transcript


# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Save locally
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Transcribe
        transcript = transcribe_audio(file_path)
        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")

        # 2. Collect sentiment per utterance
        utterance_results = []
        speaker_stats = defaultdict(lambda: {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0, "total": 0})

        for utt in transcript.utterances:
            sent = analyze_sentiment(utt.text)
            speaker_stats[utt.speaker][sent["sentiment"]] += 1
            speaker_stats[utt.speaker]["total"] += 1

            utterance_results.append({
                "speaker": utt.speaker,
                "text": utt.text,
                "sentiment": sent["sentiment"],
                "sentiment_confidence": sent["confidence"]
            })

        # 3. Summaries per speaker
        speaker_feedback = {}
        for speaker, stats in speaker_stats.items():
            total = stats["total"]
            breakdown = {
                s: {
                    "count": stats[s],
                    "percentage": round((stats[s] / total) * 100, 1) if total > 0 else 0
                }
                for s in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
            }
            speaker_feedback[speaker] = {
                "sentiment_breakdown": breakdown,
                "feedback": give_feedback(breakdown)
            }

        # 4. Overall emotion (whole audio)
        emotion = analyze_emotion(file_path)

        return {
            "status": "success",
            "utterances": utterance_results,
            "per_speaker_feedback": speaker_feedback,
            "overall_emotion": emotion
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# -------------------------------
# Run with: uvicorn main:app --reload
# -------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
