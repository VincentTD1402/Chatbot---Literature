import openai
from langchain.prompts import ChatPromptTemplate

def openai_for_summary(info):
    prompt = """
    Bạn là một học sinh cấp 2 và được yêu cầu viết lại đoạn văn dựa trên đoạn văn tóm dưới đây.
    Dựa vào đoạn văn tóm tắt được cho, hãy viết lại một đoạn văn khác sử dụng những từ ngữ có sẵn trong đoạn văn, giữ nguyên ý nghĩa, văn phong và từ ngữ thuộc thể loại văn học đoạn văn gốc sử dụng, không được sử dụng từ ngữ hiện đại hoặc khác văn phong để thay thế. Không được tùy ý bổ sung các thông tin sai lệch so với văn bản và phải giữ nguyên các câu trích dẫn trong dấu " ".
    Đoạn văn gốc: {}
    """.format(info)
    completion = client.chat.completions.create(model="gpt-3.5-turbo-0613", messages=[{"role": "user", "content": prompt}])
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def openai_for_answer(info):
    prompt = """
    Bạn là một học sinh và được yêu cầu viết lại đoạn văn dựa trên đoạn văn hoặc danh sách các ý được cho sẵn dưới đây
    Dựa vào câu hỏi được và câu trả lời được cho, hãy viết lại câu trả lời một cách đầy đủ nhất, vẫn giữ nguyên ý nghĩa, văn phong và độ chính xác của thông tin, không được sử dụng từ ngữ hiện đại thay thế. Không được tùy ý bổ sung các thông tin sai lệch so với văn bản và phải giữ nguyên các câu trích dẫn trong dấu " "
    Chỉ đưa ra kết quả.
    Thông tin gốc: {}
    """.format(info)
    completion = client.chat.completions.create(model="gpt-3.5-turbo-0613", messages=[{"role": "user", "content": prompt}])
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def openai_for_qna_with_context(question, context):
    prompt = """
    Bạn là một trợ lý văn học Việt Nam. Hãy trả lời câu hỏi dựa vào gợi ý từ nội dung được cung cấp.
    Câu hỏi: {question}
    Nội dung: {context}
    Câu trả lời:
    """.format(question, context)
    completion = client.chat.completions.create(model="gpt-3.5-turbo-0613", messages=[{"role": "user", "content": prompt}])
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def openai_for_answer_with_context(question, context):
    prompt = """
    Bạn là một trợ lý văn học Việt Nam. Hãy trả lời câu hỏi dựa vào gợi ý từ nội dung được cung cấp.
    Câu hỏi: {}
    Nội dung: {}
    Câu trả lời:
    """.format(question, context)
    return prompt

template_with_context="""Bạn là một trợ lý văn học Việt Nam. Hãy trả lời câu hỏi dựa vào gợi ý từ nội dung được cung cấp.
Câu hỏi: {question}
Nội dung: {context}
Câu trả lời:
"""

QA_PROMPT_WITH_CONTEXT = ChatPromptTemplate.from_template(template_with_context)


template_without_context = """Bạn là một trợ lý văn học Việt Nam. Hãy trả lời câu hỏi bên dưới.
Câu hỏi: {}
Câu trả lời:
"""

QA_PROMPT_WITHOUT_CONTEXT = ChatPromptTemplate.from_template(template_with_context)