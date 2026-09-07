import os
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB Limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache dataset in memory
CURRENT_DF = None
CURRENT_FILENAME = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Full HTML & Styling directly inside app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA & LangChain AI Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col font-sans">

    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-950/50 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <i class="fa-solid fa-chart-line text-indigo-500 text-2xl"></i>
                <h1 class="text-xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                    LangChain EDA Assistant
                </h1>
            </div>
            <span class="text-xs font-semibold px-3 py-1 bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 rounded-full">
                Render Ready
            </span>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-6 py-8 flex-1 w-full grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <!-- Left Panel: Upload & Config -->
        <div class="lg:col-span-4 space-y-6">
            
            <!-- File Upload Box -->
            <div class="bg-slate-800/50 border border-slate-700/60 rounded-xl p-6 shadow-xl">
                <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
                    <i class="fa-solid fa-file-csv text-indigo-400"></i> Upload Dataset
                </h2>
                <form id="uploadForm" class="space-y-4">
                    <div class="border-2 border-dashed border-slate-600 hover:border-indigo-500 rounded-lg p-6 text-center transition cursor-pointer" onclick="document.getElementById('fileInput').click()">
                        <i class="fa-solid fa-cloud-arrow-up text-3xl text-slate-400 mb-2"></i>
                        <p class="text-sm text-slate-300">Click to upload CSV or XLSX</p>
                        <p class="text-xs text-slate-500 mt-1">Max size: 16MB</p>
                        <input type="file" id="fileInput" name="file" accept=".csv, .xlsx" class="hidden" onchange="updateFileName()">
                    </div>
                    <p id="fileNameDisplay" class="text-xs text-cyan-400 truncate hidden"></p>
                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2 px-4 rounded-lg transition shadow-lg shadow-indigo-600/30">
                        Process Dataset
                    </button>
                </form>
            </div>

            <!-- API Key Input -->
            <div class="bg-slate-800/50 border border-slate-700/60 rounded-xl p-6 shadow-xl">
                <h2 class="text-lg font-semibold mb-2 flex items-center gap-2">
                    <i class="fa-solid fa-key text-amber-400"></i> OpenAI Key
                </h2>
                <p class="text-xs text-slate-400 mb-3">Optional if set in Render Environment Variables.</p>
                <input type="password" id="apiKey" placeholder="sk-..." class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500">
            </div>
        </div>

        <!-- Right Panel: Data Overview & Chat -->
        <div class="lg:col-span-8 space-y-6">
            
            <!-- Dataset Overview -->
            <div id="edaSummary" class="bg-slate-800/50 border border-slate-700/60 rounded-xl p-6 shadow-xl hidden">
                <h2 class="text-lg font-semibold mb-4 text-slate-200">Dataset Overview</h2>
                <div class="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-4">
                    <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-700/40">
                        <span class="text-xs text-slate-400 block">Total Rows</span>
                        <span id="statRows" class="text-lg font-bold text-indigo-400">-</span>
                    </div>
                    <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-700/40">
                        <span class="text-xs text-slate-400 block">Total Columns</span>
                        <span id="statCols" class="text-lg font-bold text-cyan-400">-</span>
                    </div>
                    <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-700/40 col-span-2 sm:col-span-1">
                        <span class="text-xs text-slate-400 block">File Name</span>
                        <span id="statFile" class="text-sm font-bold text-slate-200 truncate block">-</span>
                    </div>
                </div>
            </div>

            <!-- Chat Interface -->
            <div class="bg-slate-800/50 border border-slate-700/60 rounded-xl p-6 shadow-xl flex flex-col h-[500px]">
                <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
                    <i class="fa-solid fa-robot text-cyan-400"></i> LangChain Data Assistant
                </h2>

                <div id="chatBox" class="flex-1 overflow-y-auto space-y-4 mb-4 p-4 bg-slate-900/80 rounded-lg border border-slate-700/40">
                    <div class="text-sm text-slate-400 italic">Upload a dataset and ask questions about your data...</div>
                </div>

                <form id="queryForm" class="flex gap-2">
                    <input type="text" id="userQuery" placeholder="Ask anything about your data..." class="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                    <button type="submit" class="bg-cyan-600 hover:bg-cyan-500 text-white font-medium px-5 py-2.5 rounded-lg transition shadow-lg shadow-cyan-600/30">
                        Ask
                    </button>
                </form>
            </div>

        </div>
    </main>

    <script>
        function updateFileName() {
            const input = document.getElementById('fileInput');
            const display = document.getElementById('fileNameDisplay');
            if (input.files.length > 0) {
                display.innerText = "Selected: " + input.files[0].name;
                display.classList.remove('hidden');
            }
        }

        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) return alert('Please select a file first.');

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.success) {
                document.getElementById('edaSummary').classList.remove('hidden');
                document.getElementById('statRows').innerText = data.summary.rows;
                document.getElementById('statCols').innerText = data.summary.columns;
                document.getElementById('statFile').innerText = data.summary.filename;
                
                appendMessage('System', 'Dataset loaded successfully! What would you like to analyze?', 'text-indigo-400');
            } else {
                alert(data.error);
            }
        });

        document.getElementById('queryForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const queryInput = document.getElementById('userQuery');
            const query = queryInput.value.trim();
            const apiKey = document.getElementById('apiKey').value.trim();

            if (!query) return;

            appendMessage('You', query, 'text-slate-200');
            queryInput.value = '';

            const res = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, api_key: apiKey })
            });

            const data = await res.json();
            if (data.success) {
                appendMessage('LangChain', data.response, 'text-cyan-400');
            } else {
                appendMessage('Error', data.error, 'text-red-400');
            }
        });

        function appendMessage(sender, text, colorClass) {
            const chatBox = document.getElementById('chatBox');
            const msgDiv = document.createElement('div');
            msgDiv.className = 'p-3 bg-slate-800/80 rounded-lg border border-slate-700/50';
            msgDiv.innerHTML = `<strong class="${colorClass}">${sender}:</strong> <p class="text-sm mt-1 text-slate-300 whitespace-pre-wrap">${text}</p>`;
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_DF, CURRENT_FILENAME
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Upload CSV or XLSX.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        if filename.endswith('.csv'):
            CURRENT_DF = pd.read_csv(filepath)
        else:
            CURRENT_DF = pd.read_excel(filepath)
        
        CURRENT_FILENAME = filename

        summary = {
            'filename': filename,
            'rows': CURRENT_DF.shape[0],
            'columns': CURRENT_DF.shape[1]
        }

        return jsonify({'success': True, 'summary': summary})

    except Exception as e:
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_langchain():
    global CURRENT_DF
    
    if CURRENT_DF is None:
        return jsonify({'error': 'Please upload a dataset first.'}), 400

    data = request.get_json()
    query = data.get('query')
    api_key = data.get('api_key') or os.getenv('OPENAI_API_KEY')

    if not query:
        return jsonify({'error': 'Query string is required.'}), 400

    if not api_key:
        return jsonify({'error': 'OpenAI API Key is required.'}), 400

    try:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=api_key)
        agent = create_pandas_dataframe_agent(
            llm,
            CURRENT_DF,
            verbose=True,
            allow_dangerous_code=True
        )

        response = agent.run(query)
        return jsonify({'success': True, 'response': str(response)})

    except Exception as e:
        return jsonify({'error': f'LangChain Execution Error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
