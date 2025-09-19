import streamlit as st
from main import ChatBot

bot = ChatBot()

st.set_page_config(page_title="Mental Health Chatbot")
st.sidebar.title("Hi! I'm a mental health symptom analyzer.")

# Function to convert past messages
def conv_past(messages):
    return "\n".join([f"Message {i}: {msg['role']}: {msg['content']}" for i, msg in enumerate(messages)])

# Create response
def generate_response(input_text, pasts):
    result = bot.rag_chain.invoke({"question": input_text, "pasts": pasts})
    answer_start = result.find("Answer:")
    return result[answer_start + 7:].strip() if answer_start != -1 else result

# Manage session
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help with your mental health today?"}]

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User's input
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            pasts = conv_past(st.session_state.messages[:-1])
            response = generate_response(user_input, pasts)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})