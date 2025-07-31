from flask import Flask, render_template, request, jsonify
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# --- Load Model ---
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=200,
)

model = ChatHuggingFace(llm=llm)

# --- Init app ---
app = Flask(__name__)
chat_history = [SystemMessage(content="You are a helpful AI assistant")]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/chat', methods=["POST"])
def chat():
    user_input = request.json["message"]
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    return jsonify({"response": result.content})

if __name__ == "__main__":
    app.run(debug=True)
