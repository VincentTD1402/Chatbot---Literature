import os
import random
import time
import numpy as np
import openai
import streamlit as st
from datetime import datetime
import asyncio
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from utils import read_file, sentence_to_generator, show_time_sleep_generator, show_async_generator
from create_chroma_db import LocalChromaDB
from temp_chroma_db import TempChromaDB
from templates.prompt import openai_for_summary, openai_for_answer
from config import config
from templates.prompt import openai_for_answer_with_context
from scipy.io.wavfile import write
from audio_recorder_streamlit import audio_recorder

load_dotenv()
# os.environ['OPENAI_API_KEY'] = "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"
# openai.api_key = os.environ['OPENAI_API_KEY']
# openai_api_key = os.getenv('openai_token')   
client = OpenAI(api_key="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")
list_stories_path = "collections/baivancap2.txt"
list_stories = read_file(list_stories_path)

chroma_db = LocalChromaDB()
temp_chroma_db = TempChromaDB()

# Define a CSS style with a custom font size
custom_css = f"""
<style>
    .st-emotion-cache-16idsys p {{
        font-size: 20px; /* You can adjust the font size as needed */
        font-weight: bold;
    }}

    .st-emotion-cache-1v0mbdj img {{
        
    }}
</style>
"""

# State management
CURRENT_PAGE = "main"

#search the data in dataset
def summary_search(story_name: str=None, 
        question: str="Tóm tắt câu chuyện", 
        num_of_answers: int=1):
    '''
    Return a list of summaries in a certain story in database
    Args:
        - story_name (str): name of the story
        - question (str): The question of stories
        - num_of_answers (int): The number of answer for stories
    '''
    summary_question = chroma_db.client.get_collection(
        name='tomtatvanhoccap2-vn-supsimcse',
        embedding_function=chroma_db.sentence_transformer_ef)

    results = summary_question.query(
        query_texts=[question],
        n_results=num_of_answers,
        where_document={"$contains": story_name}
    )
    result = []
    if results['metadatas'][0] == []:
        result = []
    else:
        for i in results['metadatas'][0]:
            # print(len(results['metadatas'][0]))
            result.append(i['Tóm tắt'])
    return result

def question_search(story_name: str,
        question: str,
        num_of_answers = 1):
    '''
    Search the similarity queries in the database
    Args:
        - question (str): The question of stories
        - num_of_answers (int): The number of answer for stories
    Returns:
        A list of similar questions
    '''
    literature_collection = chroma_db.client.get_collection(
        name='cauhoivanhoccap2-vn-supsimcse',
        embedding_function=chroma_db.sentence_transformer_ef)
    results = literature_collection.query(
        query_texts=[question],
        n_results=num_of_answers,
        where_document={"$contains": story_name}
    )
    for i in results['documents'][0]:
        print('Answer: ', i)
    return results['documents']


def data_summary_prepare(story_name: str):
    '''
    Finding data summary for a certain story name
    Args:
        - story_name (str): story name
    Return:
        short and long summaries of a story
    '''
    data_state = {}
    sumary_short = summary_search(story_name=story_name, 
                    question="Tóm tắt câu chuyện ngắn gọn",
                    num_of_answers=10)
    sumary_long = summary_search(story_name=story_name,
                    question="Tóm tắt câu chuyện đầy đủ và chi tiết",
                    num_of_answers=10)
    data_state['tóm tắt ngắn'] = sumary_short
    data_state['tóm tắt dài'] = sumary_long
    return data_state

def random_data_answer(list_answers):
    '''
    Select a random an answer from list answers
    '''
    return random.choice(list_answers)

def summary_answer(prompt, data_state):
    '''
    Summary answer by two way:
        - Shortly summarizing if prompt requires
        - Longly summarizing if prompt requires
    '''
    if prompt == "Ngắn":
        return (random_data_answer(data_state['tóm tắt ngắn']))
    elif prompt == "Dài":
        return (random_data_answer(data_state['tóm tắt dài']))
    else:
        return (random_data_answer(data_state['tóm tắt ngắn']))


def query_make(question_input: str, story_selected: str):
    '''
    Querying the question based on the story selected
    Args:
        - question_input (str): question input
        - story_selected (str): story is selected
    Returns:
        a full sentence based on template
    '''
    query_sen = "Trong truyện " + story_selected + ", " + question_input
    return query_sen

def save_data(prompt, data_accept):
    '''
    Saving the data
    Args:
        - prompt (str): Prompting data to save
        - data_accept (str): data_accept
    '''
    return 0


# def record_and_save(filename='output.wav', duration=3, samplerate=44100, channels=2):
#     """
#     Record audio for the specified duration and save it to a WAV file.

#     Parameters:
#         filename (str): The name of the WAV file to save the recording to.
#         duration (int): The duration of the recording in seconds.
#         samplerate (int): The sample rate of the recording.
#         channels (int): The number of audio channels (1 for mono, 2 for stereo).

#     Returns:
#         numpy.ndarray: The recorded audio data.
#     """
#     myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
#     sd.wait()  # Wait until recording is finished
#     write(filename, samplerate, myrecording)
    
#     return myrecording



def generate_text_to_speech(input_text):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=input_text
        )
        response.stream_to_file("output.mp3")
        audio_file = open("output.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        print(e)
        st.error(f"An error occurred: {e}")

# Example usage
# sample_rate = 44100
# note_la = generate_sine_wave(frequency=440, duration=2, sample_rate=sample_rate)
# st.audio(note_la, sample_rate=sample_rate)

async def main():
    story_selected = st.sidebar.selectbox(
        "🌟 Document Selection",
        list_stories
    )
    st.markdown(custom_css, unsafe_allow_html=True)
    if story_selected != "":
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/d/d9/Neurond.png",
            width=None,  # Manually Adjust the width of the image as per requirement
        )
        summary_type = st.sidebar.radio('📻 Summary Type', ['Short Summarization', 'Full Summarization'])
        if summary_type == 'Short Summarization':
            prompt = 'Tóm tắt ngắn gọn tác phẩm {}'.format(story_selected)
            list_summary = chroma_db.find_summary(
                name_collection='tomtatvanhoccap2-vn-supsimcse',
                story_name=story_selected,
                question=prompt,
                num_of_answer=1
            )
            summary = list_summary[0]

        if summary_type == 'Full Summarization':
            prompt = 'Tóm tắt đầy đủ và chi tiết tác phẩm {}'.format(story_selected)
            list_summary = chroma_db.find_summary(
                name_collection='tomtatvanhoccap2-vn-supsimcse',
                story_name=story_selected,
                question=prompt,
                num_of_answer=1
            )
            summary = list_summary[0]
            
        st.sidebar.text_area(label="🔊 Summarization", height=350, value=summary, placeholder="Area")

    st.title("💬 Vietnam Literary Chatbot")
    if "messages" not in st.session_state:
        FIRST_ASSISTANT_MESSAGE = "Xin chào, tôi là trợ lý văn học Việt Nam, tôi có thể giúp gì cho bạn?"
        st.session_state["messages"] = [{"role": "assistant", "content": FIRST_ASSISTANT_MESSAGE}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"], unsafe_allow_html=True)
    audio_bytes = audio_recorder(text="ghi âm",
                                recording_color="#e8b62c",
                                neutral_color="#6aa36f",
                                icon_name="microphone",
                                icon_size="2x",
                                )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)  # Replace with the actual sample rate used in recording
        write('test.wav',96000, audio_array)
        if os.path.exists("test.wav"):
            audio_file = open("test.wav", "rb")
            transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
            )
            assistant_message = """Có phải đây là câu hỏi bạn muốn hỏi không? Vui lòng nhập câu hỏi vào khung chat . 
            Bạn có thể chỉnh sửa lại câu hỏi nếu như câu hỏi chưa đúng với ý của bạn\n"""

            msger = assistant_message +"\n"+ transcript
            # st.session_state["messages"] = [{"role": "assistant", "content": msger}]
            st.chat_message("assistant").write(msger)
    
    answer = None
    full_response = ''
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        sim_anwer = chroma_db.find_sim_answer(
            name_collection='cauhoivanhoccap2-vn-supsimcse',
            story_name=story_selected,
            question=prompt,
            num_of_answer=1
        )
        answer=sim_anwer
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            start_time = datetime.now()  # Record start time
            if answer:
                generator = sentence_to_generator(answer)
                message_placeholder, full_response = \
                    show_time_sleep_generator(message_placeholder, generator)
                generate_text_to_speech(answer)
            else:
                st.markdown(f"🔎 searching external source")
                external_chunks = await temp_chroma_db.find_external_chunks(prompt)
                new_prompt = openai_for_answer_with_context(
                    question=prompt,
                    context=external_chunks['context']
                )
                print("new prompt",new_prompt)
                # Update the last message by adding context into prompt.
                update_messages = st.session_state.messages.copy()
                update_messages[-1] = {"role":"user", "content":new_prompt}
                # Request chatbot
                # openai.api_key = openai_api_key
                st.markdown(f"🚀 generating content")
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0613", 
                    temperature=0.01,
                    stream=True,
                    messages=update_messages)
                print("response:",response)
                for token in response:
                    # print(token)
                    try:
                        # value = token['choices'][0]['delta']['content']
                        value = token.choices[0].delta.content
                        # print(value)
                        full_response += value
                        message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    except Exception as e:
                        print(e)
                        pass
                # print("full: ", full_response)
                generate_text_to_speech(full_response)
            # Create a button to open the text input when clicked
            if st.button("Response"):
                CURRENT_PAGE = "response"
                response_page()

            end_time = datetime.now()
            response_time = end_time - start_time
            role_of_last_anwser = st.session_state["messages"][-1]['role']
            if role_of_last_anwser == "user":
                st.markdown(f"🕒 Bot response time: {response_time.total_seconds()} seconds")
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
# Response page content
def response_page():
    st.title("Response Page")
    user_input = st.text_input("Enter your response:")
    if user_input:
        st.write(f"You entered: {user_input}")
        if st.button("Back to Main"):
            CURRENT_PAGE = "main"

async def start_app():
    # Determine which page to display (e.g., CURRENT_PAGE)
    if CURRENT_PAGE == 'response':
        response_page()
    else:
        await main()

if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
    asyncio.run(start_app())