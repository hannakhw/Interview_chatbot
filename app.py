import streamlit as st
from pages.home_page import show_home_page
from pages.portfolio_page import show_portfolio_page
from pages.chat_page import show_chat_page
import os
from dotenv import load_dotenv

load_dotenv()
PASSWORD = os.getenv("STREAMLIT_PASSWORD")

pages = {
    "Home": show_home_page,
    "Portfolio": show_portfolio_page,
    "Chat with Me": show_chat_page
}

# 비밀번호 확인 함수
def check_password():
    def password_entered():
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
            st.error("비밀번호가 잘못되었습니다. 다시 시도해주세요.")  

    if "password_correct" not in st.session_state:
        st.title("Welcome!")
        st.text_input(
            "비밀번호를 입력하세요", 
            type="password", 
            on_change=password_entered, 
            key="password",
            placeholder="Enter password"  
        )
        return False
    
    return st.session_state["password_correct"]


def main():
    st.title("제 페르소나에게 질문해보세요!")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", pages.keys())
    pages[page]()

if __name__ == "__main__":
    if check_password():
        main()