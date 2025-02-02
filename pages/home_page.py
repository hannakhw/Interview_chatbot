import streamlit as st


def show_home_page():
    # 홈페이지 내용
    st.title("AI 엔지니어 Day의 포트폴리오")
    st.write("""안녕하세요! AI 개발자 김효원입니다.
            저의 자세한 포트폴리오를 보시려면 Navigation에서 포트폴리오로 이동하세요.
            저에 대해 궁금한 게 있다면 chat으로 이동해서 질문해주세요.
            챗봇은 RAG를 활용해 만들었고요, 어떤 질문이라도 환영합니다!
            """)
    
    # 간단한 소개
    st.markdown("""
    ### About Me
    - 2024. 07. 16 ~ 2025. 02. 14 패스트캠퍼스x업스테이지 AI 부트캠프 4기 수료(예정)
    - 과학, 기술, 공학 분야 콘텐츠(기사, 책, 유튜브 대본, 교육 프로그램 등) 기획 및 제작 업무 경력 7년 이상
    - 현재는 AI 엔지니어이자 콘텐츠 기획자로 이직하고자 준비 중
    """)