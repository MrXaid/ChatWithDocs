# ChatWithDocs 📚💬

![ChatWithDocs Demo](/demoimage.png)

ChatWithDocs is an interactive document analysis tool that allows users to upload multiple documents of any type and engage in a conversation about their content. Powered by advanced AI technologies, it enables users to extract specific information and gain insights from their documents effortlessly.

## Features 🌟

- **Multi-document Upload**: Support for uploading multiple documents of various types.
- **AI-Powered Chat**: Engage in conversations about your documents using state-of-the-art language models.
- **Information Extraction**: Easily extract specific information from your PDFs and other documents.
- **User-friendly Interface**: Built with Streamlit for a smooth and intuitive user experience.
- **Advanced Search**: Utilizes FAISS indexing for fast and efficient document searching.

## Technologies Used 🛠️

- [Streamlit](https://streamlit.io/): For the graphical user interface
- [LangChain](https://python.langchain.com/): For building applications with large language models
- [FAISS](https://github.com/facebookresearch/faiss): For efficient similarity search and clustering of dense vectors
- [Google Gemini](https://deepmind.google/technologies/gemini/): As the underlying language model
- Tokenizers and Embeddings: For processing and representing text data

## Installation 🚀

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ChatWithDocs.git
   cd ChatWithDocs
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file in the project root directory
   - Add your Gemini API key to the `.env` file:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage 🖥️

1. Start the Streamlit app:
   ```bash
   streamlit run mainfunv.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload your documents using the file uploader in the sidebar.

4. Start chatting with your documents! Ask questions, request information, or use any of the available features to analyze your documents.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.
