import argparse
import os
import pydub
import whisper
import time
import numpy as np

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

stt = whisper.load_model("base.en")

template = """
You are an AI specializing in summarizing transcribed voice notes. Below is a transcript of a spoken recording. Please generate concise notes in markdown format, prioritizing clarity and coherence. Reorganize content into appropriate sections with headers. Do not infer any additional context or information beyond the transcription. Make sure to list all names of software, plugin's, or integrations mentioned. Keep the content structured and readable in markdown format, but without using code blocks. Below is the transcribed audio: {history}{input}
"""

PROMPT = PromptTemplate(input_variables=["history","input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(model="llama3.2"),
)

def chunk_audio(audio_file):
    """
    Breaks an audio file into chunks of a specified duration.

    Parameters:
        audio_file (str): Path to the input audio file.

    Returns:
        Numpy array: Array of sound chunks.
    """

    # Load the audio file
    sound = pydub.AudioSegment.from_file(audio_file)
    sounds = pydub_to_np(sound)

    return sounds

def pydub_to_np(audio: pydub.AudioSegment):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.int16).astype(np.float32) / 32768.0

def transcribe(audio) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio: The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text

def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output of the transcription')
    parser.add_argument(
        '--audio_paths',
        nargs='+',
        required=True,
        metavar='PATH',
        help='Absolute paths to the audio files to be transcribed')
    args = parser.parse_args()

    for audio in args.audio_paths:
        # chunks = chunk_audio(audio)

        transcript = transcribe(audio)

        if (args.verbose):
            print(transcript)

        summary = get_llm_response(transcript)
        if (args.verbose):
            print(summary)

        notes = summary+"\n\n---\n\nFull Transcript:\n"+transcript

        timestr = time.strftime("%Y%m%d-%H%M%S")
        with open("meeting-summary-"+timestr+".md", "w") as f:
            f.write(notes)

if __name__ == '__main__':
    main()