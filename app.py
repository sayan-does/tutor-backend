import os
import torch
import chromadb
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx


class DocumentQASystem:
    def __init__(self):
        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(path="./vector_db")

        # Collection name
        collection_name = "document_collection"

        # Modify collection handling to prevent UniqueConstraintError
        try:
            # Try to get the existing collection
            self.collection = self.chroma_client.get_collection(
                name=collection_name)
        except Exception:
            # If collection doesn't exist, create it
            self.collection = self.chroma_client.create_collection(
                name=collection_name)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Qwen LLM
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B")

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.llm_model = self.llm_model.to("cuda")

    def extract_text_from_file(self, file):
        """Extract text from different file types"""
        filename = file.filename.lower()
        if filename.endswith('.pdf'):
            reader = PdfReader(file)
            return " ".join([page.extract_text() for page in reader.pages])
        elif filename.endswith('.docx'):
            doc = docx.Document(file)
            return " ".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            return file.read().decode('utf-8')

    def create_embeddings(self, text):
        """Create embeddings for text"""
        return self.embedding_model.encode(text).tolist()

    def store_document(self, user_id, file):
        """Process and store document in vector database"""
        text = self.extract_text_from_file(file)
        embeddings = self.create_embeddings(text)

        # Store document with user-specific metadata
        self.collection.add(
            embeddings=[embeddings],
            documents=[text],
            metadatas=[{"user_id": user_id}]
        )
        return len(text)

    def retrieve_relevant_context(self, query, user_id, top_k=3):
        """Retrieve most relevant document sections"""
        query_embedding = self.create_embeddings(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": user_id}
        )
        return results['documents'][0] if results['documents'] else ""

    def generate_llm_text(self, prompt, max_length=200):
        """Generate text using Qwen LLM"""
        try:
            # Prepare input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate text
            outputs = self.llm_model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )

            # Decode and return
            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            return f"Error generating text: {str(e)}"

    def generate_response(self, query, context):
        """Generate detailed response using Qwen LLM"""
        prompt = (
            f"Context: {context}\n"
            f"Query: {query}\n"
            f"Answer:"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = self.llm_model.generate(
            inputs["input_ids"],
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Generate additional context and trivia
        trivia = self.generate_trivia(context)

        return {
            "answer": answer,
            "explanation": "The answer is generated using the Qwen LLM based on the provided context.",
            "trivia": trivia
        }

    def generate_trivia(self, context):
        """Generate random trivia from context"""
        sentences = context.split('.')
        return sentences[0] if sentences else "No additional trivia found."


# Flask Application Setup
app = Flask(__name__)
CORS(app)
qa_system = DocumentQASystem()


@app.route('/upload', methods=['POST'])
def upload_document():
    # Default user_id if not provided
    user_id = request.form.get('user_id', 'default_user')

    # Validate if the file is included in the request
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files['file']

    # Ensure the file has a valid name and content
    if file.filename == '':
        return jsonify({"status": "error", "message": "File name is missing"}), 400

    try:
        # Process and store the document
        document_length = qa_system.store_document(user_id, file)
        return jsonify({
            "status": "success",
            "message": f"Document processed. Length: {document_length} characters"
        }), 200
    except Exception as e:
        # Log the exception for debugging
        app.logger.error(f"Error processing document: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/query', methods=['POST'])
def process_query():
    data = request.json
    user_id = data.get('user_id', 'default_user')
    query = data.get('query', '')

    try:
        context = qa_system.retrieve_relevant_context(query, user_id)
        if not context:
            return jsonify({"error": "No relevant context found."}), 404

        response = qa_system.generate_response(query, context)
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/llm_text', methods=['POST'])
def llm_text():
    """Generate LLM text based on a given prompt"""
    data = request.json
    prompt = data.get('prompt', 'Tell me a short story')
    max_length = data.get('max_length', 200)

    try:
        generated_text = qa_system.generate_llm_text(prompt, max_length)
        return jsonify({
            "status": "success",
            "generated_text": generated_text
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
