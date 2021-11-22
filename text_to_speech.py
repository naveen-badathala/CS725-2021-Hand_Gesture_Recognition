#Import Speak function and pass the text as argument

from gtts import gTTS
import playsound


def convert_text_to_audio(text):
    language = 'en'
    myobj = gTTS(text = text, lang = language, slow = True)
    saved_fname = 'prediction.mp3'
    myobj.save(saved_fname)
    return saved_fname

def play_audio(audio_path):  
    blocking = True
    playsound.playsound(audio_path, block=blocking)

def speak(input_text):
    path = convert_text_to_audio(input_text)
    play_audio(path)
