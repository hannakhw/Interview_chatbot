from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import TokenTextSplitter
from operator import itemgetter
from dotenv import load_dotenv
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent
env_path = os.path.join(root_dir, 'openai_key.env')

# .env 파일 로드
load_dotenv(env_path)
api_key = os.getenv('OPENAI_API_KEY')

def create_and_save_vectorstore():
    """벡터 스토어 생성 및 저장"""
    try:
        current_dir = Path(__file__).parent.parent
        documents_path = os.path.join(current_dir, "data", "documents")
        chroma_path = os.path.join(current_dir, "data", "chroma")
        
        loader = DirectoryLoader(documents_path, glob="*.txt", show_progress=True)
        data = loader.load()
        
        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        all_splits = text_splitter.split_documents(data)
        
        embed_model = OpenAIEmbeddings(
            openai_api_key=api_key,
            model='text-embedding-3-small'
        )
        
        vector_index = Chroma.from_documents(
            documents=all_splits,
            embedding=embed_model,
            persist_directory=chroma_path 
        )
        print("Vector store created successfully")
        return vector_index
    
    except Exception as e:
        print(f"Detailed error in create_and_save_vectorstore: {e}")
        return None


def setup_chat_chain():
    """챗봇 체인 설정"""
    try:
        print("Setting up chat chain...")
        current_dir = Path(__file__).parent.parent
        chroma_path = os.path.join(current_dir, "data", "chroma")
        
        embed_model = OpenAIEmbeddings(
            openai_api_key=api_key,
            model='text-embedding-3-small'
        )
        
        # 벡터스토어 존재 여부 확인
        if os.path.exists(chroma_path):
            print("Loading existing vector store...")
            vector_index = Chroma(
                persist_directory=chroma_path,
                embedding_function=embed_model
            )
        else:
            print("Vector store not found, creating new one...")
            vector_index = create_and_save_vectorstore()
        
        if vector_index is None:
            raise Exception("Failed to initialize vector store")
        
        
        print("Setting up retriever...")
        retriever = vector_index.as_retriever(k=5)
        
        # 프롬프트 템플릿 설정
        template = """
        너는 지금부터 내 페르소나를 가진 챗봇이야. 나는 AI 업계에서 일을 시작하기 위해 일자리를 찾고 있어.
        질문을 하는 사람은 회사의 면접관이고, 항상 예의있으면서도 솔직하고, 정직하게, 자신감있게 대답해줘.
        질문에 답을 하기 어려운 내용이라면 잘 모르겠다고 대답하고, 실제 면접에서 만나게 되면 다시 한번 꼭 물어봐달라고 말해줘.
        대답할 때에는 아래 context를 참고해. context에는 내가 지금까지 했던 경험과 경력을 알려줄께. 
        ###
        {context}
        ###

        {history}

        user: {query}
        답변:  
        """
        

            
        print("Setting up prompt and memory...")   
        prompt = ChatPromptTemplate.from_template(template)
        memory = ConversationBufferWindowMemory(k=3, ai_prefix="job interview")
        
        
        print("Setting up ChatOpenAI...")
        chat = ChatOpenAI(
                    openai_api_key=api_key,
                    model="gpt-4o-mini",
                    temperature=0.7
                )
        
        print("Creating chain...")
        def merge_docs(retrieved_docs):
            return "###\n\n".join([d.page_content for d in retrieved_docs])
        
        chain = RunnableParallel({
                    'context': retriever | merge_docs,
                    'query': RunnablePassthrough(),
                    'history': RunnableLambda(memory.load_memory_variables) | itemgetter('history')
                }) | {
                    'answer': prompt | chat | StrOutputParser(),
                    'context': itemgetter('context'),
                    'prompt': prompt
                }
        print("Chat chain setup completed successfully")
        return chain, memory
                
    except Exception as e:
        print(f"Error in setup_chat_chain: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None



