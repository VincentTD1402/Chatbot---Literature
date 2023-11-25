import os
import openai
from openai import OpenAI
client = OpenAI(api_key="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")
# os.environ['OPENAI_API_KEY'] = "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"
# openai.api_key = os.environ['OPENAI_API_KEY']
# openai_api_key = os.getenv('openai_token')   

update_messages = [{"role":"assistant", "content":"xin chào, tôi là trợ lí văn học việt nam"},
                   {"role":"user", "content":"bạn trả lời giúp tôi câu hỏi sau được không?"}]

# update_messages = st.session_state.messages.copy()
update_messages[-1] = {"role":"user", "content":"Tóm tắt truyện ngắn ông lão đánh cá và con cá vàng"}
# Request chatbot
# openai.api_key = openai_api_key
# st.markdown(f"🚀 generating content")
response = client.chat.completions.create(
    model="gpt-3.5-turbo-0613", 
    temperature=0.01,
    stream=True,
    messages=update_messages)
for token in response:
    
print("response:",response)