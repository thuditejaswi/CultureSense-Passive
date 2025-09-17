from pyannote.audio import Pipeline
import os

# Put your Hugging Face token here (or set it as environment variable HF_TOKEN)
HF_TOKEN = os.environ.get("HF_TOKEN") or "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"

# Load the pretrained diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# Run diarization on your audio file
diarization = pipeline("c:/Users/Tejaswi/Desktop/Unity/meeting.mp3")
 # replace with your file path (wav/mp3/mp4)

# Print speaker segments
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s - {turn.end:.1f}s: {speaker}")
