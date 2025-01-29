import argparse

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import whisper
import time

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
        '--library_path',
        help='Absolute path to dynamic library. Default: using the library provided by `pvleopard`')
    parser.add_argument(
        '--model_path',
        help='Absolute path to Leopard model. Default: using the model provided by `pvleopard`')
    parser.add_argument(
        '--disable_automatic_punctuation',
        action='store_true',
        help='Disable insertion of automatic punctuation')
    parser.add_argument(
        '--disable_speaker_diarization',
        action='store_true',
        help='Disable identification of unique speakers')
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
    
    for audio_path in args.audio_paths:
            transcript = transcribe(audio_path)
            print(transcript)
                
            summary = get_llm_response(transcript)
            print(summary)

            notes = summary+"\n\n---\n\nFull Transcript:\n"+transcript

    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open("meeting-summary-"+timestr+".md", "w") as f:
        f.write(notes)

if __name__ == '__main__':
    main()