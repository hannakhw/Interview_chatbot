import streamlit as st
from utils.rag_utils import setup_chat_chain, create_and_save_vectorstore
from utils.chat_utils import process_chat
import os


def show_chat_page():
    st.title("제 페르소나를 가진 AI와의 인터뷰")
    st.write("저에 대해 궁금한 점을 물어보세요!")

    # 최초 실행 시에만 벡터스토어 생성
    if 'vectorstore_initialized' not in st.session_state:
        with st.spinner('첫 실행이라 벡터스토어를 생성하고 있습니다... 잠시만 기다려주세요.'):
            if not os.path.exists("data/vectorstore"):
                create_and_save_vectorstore()
            st.session_state.vectorstore_initialized = True

    # chain과 memory는 세션에 없을 때만 초기화
    if 'chain' not in st.session_state or 'memory' not in st.session_state:
        chain, memory = setup_chat_chain()  # 이미 저장된 vectorstore 사용
        st.session_state['chain'] = chain
        st.session_state['memory'] = memory

    # 채팅 히스토리 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    
    
    # 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 챗봇 응답 생성
        with st.chat_message("assistant"):
            if st.session_state['chain']:
                result = st.session_state['chain'].invoke(prompt)
                response = result['answer']
                st.session_state['memory'].save_context(
                    {'query': prompt}, 
                    {'answer': response}
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("챗봇 초기화에 실패했습니다.")