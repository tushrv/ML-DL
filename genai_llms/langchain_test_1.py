import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

class ChatbotApp:
    def __init__(self):
        self.setup_config()
        self.setup_ui()
        
    def setup_config(self):
        """Initialize the LLM model and prompt template"""
        self.model = OllamaLLM(
            model="llama3.2",
            temperature=0.2,
            top_p=0.9,
        )
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        Question: {question}
        Please provide a clear and concise response.""")
        
        self.chain = self.prompt_template | self.model

    def setup_ui(self):
        """Setup the Streamlit user interface"""
        # Custom CSS for better aesthetics
        st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput > div > div > input {
            background-color: black;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #ddd;
        }
        .response-box {
            background-color: #e3f2fd;
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 1rem;
            border: 1px solid #90caf9;
        }
        </style>
        """, unsafe_allow_html=True)

        # Main UI components
        st.title("🤖 Simple AI QA Bot")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Settings")
            temperature = st.slider(
                "Creativity (Temperature)", 
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Higher values (closer to 1.0) make the output more creative but less focused. Lower values (closer to 0.0) make it more focused and deterministic."
            )
            self.model.temperature = temperature
            
            st.markdown("---")
            st.markdown("""
            ### About
            This AI assistant uses the Llama 3.2 model with:
            - Adjustable creativity setting
            - Clean and responsive interface
            - Fast response time
            """)

        # Main chat interface
        self.handle_user_input()

    def handle_user_input(self):
        """Handle user input and generate response"""
        question = st.text_input("What would you like to know?", key="user_input")
        
        if question:
            try:
                with st.spinner('Thinking...'):
                    # Get response
                    response = self.chain.invoke({
                        "question": question
                    })
                    
                    # Display response in a styled container
                    st.markdown("**Response:**")
                    st.write(response)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.info("Please enter a question to get started!")

if __name__ == "__main__":
    app = ChatbotApp()