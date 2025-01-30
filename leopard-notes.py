import argparse
import os

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

from pvleopard import create, LeopardActivationLimitError
from tabulate import tabulate

template = """
You are an AI specializing in summarizing transcribed voice notes. Below is a transcript of a spoken recording. Please generate concise notes in markdown format, prioritizing clarity and coherence. Reorganize content into appropriate sections with headers. Do not infer any additional context or information beyond the transcription. Keep the content structured and readable in markdown format, but without using code blocks. Below is the transcribed audio: {history}{input}
"""

PROMPT = PromptTemplate(input_variables=["history","input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(model="llama3.3"),
)

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
        help='Absolute paths to `.wav` files to be transcribed')
    args = parser.parse_args()

    o = create(
        access_key=os.environ.get('ACCESS_KEY'),
        model_path=args.model_path,
        library_path=args.library_path,
        enable_automatic_punctuation=not args.disable_automatic_punctuation,
        enable_diarization=not args.disable_speaker_diarization)

    try:
        for audio_path in args.audio_paths:
            transcript, words = o.process_file(audio_path)
            print(transcript)
            if args.verbose:
                print(tabulate(
                    words,
                    headers=['word', 'start_sec', 'end_sec', 'confidence', 'speaker_tag'],
                    floatfmt='.2f'))
                
            summary = chain.predict(input=transcript)
            print(summary)

    except LeopardActivationLimitError:
        print('AccessKey has reached its processing limit.')


if __name__ == '__main__':
    main()