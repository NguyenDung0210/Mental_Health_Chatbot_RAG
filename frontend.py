from main import ChatBot
import streamlit as st
bot = ChatBot()
    
st.set_page_config(page_title="Symptom-chatbot")
with st.sidebar:
    st.title('Hi there! I am mental health symptom analyzing chatbot!')
# Function for generating LLM response
def conv_past(inp):
    ret = []
    for num,comb in enumerate(inp):
        ret.append(f"Message {num%2} by the {comb['role']}: {comb['content']}\n")
    return ret
def generate_response(input):
    #result = bot.rag_chain.invoke(input)
    result = bot.rag_chain.invoke({"context":bot.docsearch.as_retriever(),"question":input,"pasts":str(conv_past(st.session_state.messages))})
    return result
def afterRes(input_string):
    # Find the index of "Question:"
    question_index = input_string.find("Question:")
    if question_index == -1:
        return input_string
        #return "No 'Question:' found in the input string"
    # Find the index of "Answer:"
    answer_index = input_string.find("Answer:", question_index)
    if answer_index == -1:
        return input_string
        #return "No 'Answer:' found in the input string"
    # Extract the text after "Answer:"
    text_after_answer = input_string[answer_index + len("Answer:"):].strip()
    return text_after_answer
        
        
    #return response
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi there!"}]
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating.."):
            response = generate_response(input) 
            response = afterRes(response)
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)