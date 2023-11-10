import whisper
from googletrans import Translator

def speech_to_text(audio_path):

    # Load the Whisper model
    model = whisper.load_model("base")

    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Make a log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)

    # Decode the audio to text
    result = model.transcribe(audio_path)
    transcribe_text = result["text"]

    # Print the detected language and transcribed text
    print(f"Detected language: {detected_language}")
    print(f"Transcribed text: {transcribe_text}")

    # Translate to Urdu
    translator = Translator()
    translation = translator.translate(transcribe_text, dest='ur')
    
    # Print the translated text
    print(f"Translated to Urdu: {translation.text}")

# Function calling with the audio file path argument
speech_to_text('audio.mp3')
