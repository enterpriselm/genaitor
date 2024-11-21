import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import os

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()

    if file_path.endswith('.mp4'):
        video = VideoFileClip(file_path)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path)
    elif file_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(file_path)
        audio.export("temp_audio.wav", format="wav")
        audio_path = "temp_audio.wav"
    else:
        raise ValueError("Unsupported file format")

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data, language="en-US")

    os.remove(audio_path)
    return transcript
