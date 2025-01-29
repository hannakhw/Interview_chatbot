import streamlit as st
from pages.home_page import show_home_page
from pages.portfolio_page import show_portfolio_page
from pages.chat_page import show_chat_page

pages = {
    "Home": show_home_page,
    "Portfolio": show_portfolio_page,
    "Chat with Me": show_chat_page
}

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", pages.keys())
    pages[page]()

if __name__ == "__main__":
    main()