from flask import Flask, render_template, request, jsonify
import subprocess
import os
import json
import plotly
import plotly.graph_objs as go

app = Flask(__name__)

# Relative paths
SCRIPTS_DIR = 'scripts'
RESULTS_DIR = 'results'
DATA_COLLECTION_DIR = 'P1_Data_Collection'
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
VENV_PYTHON = os.path.join('venv', 'Scripts', 'python.exe')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check if VENV_PYTHON exists
if not os.path.exists(VENV_PYTHON):
    raise FileNotFoundError(f"Virtual environment Python executable not found at: {os.path.abspath(VENV_PYTHON)}")

@app.route('/')
def index():
    # Dropdown options
    cryptocurrencies = ['binance-coin', 'bitcoin', 'cardano', 'dogecoin', 'ethereum', 'terra-luna', 'solana', 'ripple']
    sentiment_methods = ['Loughran McDonald', 'AFINN', 'Vader', 'LSTM', 'NBoW', 'Transformers']
    sentiment_sources = ['Reddit only', 'Alpha Vantage only', 'Both']
    trading_options = ['Long only', 'Long and short']
    
    return render_template(
        'index.html',
        cryptocurrencies=cryptocurrencies,
        sentiment_methods=sentiment_methods,
        sentiment_sources=sentiment_sources,
        trading_options=trading_options
    )

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    try:
        data = request.json
        crypto = data.get('crypto')
        method = data.get('method')
        source = data.get('source')
        trading = data.get('trading')
        
        if not all([crypto, method, source, trading]):
            return jsonify({'error': 'All dropdown selections are required'})

        # Map UI values to script arguments
        source_map = {
            'Reddit only': 'reddit',
            'Alpha Vantage only': 'alpha',
            'Both': 'both'
        }
        trading_map = {
            'Long only': 'long',
            'Long and short': 'long_short'
        }
        
        # Run analysis script
        script_path = os.path.join(SCRIPTS_DIR, 'analysis.py')
        if not os.path.exists(os.path.join(PROJECT_ROOT, script_path)):
            return jsonify({'error': f"Analysis script not found at: {script_path}"})
        
        cmd = [
            VENV_PYTHON, script_path,
            '--crypto', crypto,
            '--method', method.replace(' ', '_').lower(),
            '--source', source_map[source],
            '--trading', trading_map[trading]
        ]
        result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
        
        # Read outputs from results folder
        metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
        graph_path = os.path.join(RESULTS_DIR, 'graph.json')
        
        if not os.path.exists(metrics_path) or not os.path.exists(graph_path):
            return jsonify({'error': 'Analysis failed to produce output files'})
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        # Convert graph data to Plotly format
        fig = go.Figure(data=[go.Scatter(
            x=graph_data['x'],
            y=graph_data['y'],
            mode='lines',
            name='Portfolio Value'
        )])
        fig.update_layout(
            title=f'Trading Simulation for {crypto}',
            xaxis_title='Date',
            yaxis_title='Portfolio Value'
        )
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'output': result,
            'metrics': metrics,
            'graph': graph_json,
            'results_path': RESULTS_DIR
        })
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f"Analysis failed: {e.output}"})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/update_data', methods=['POST'])
def update_data():
    try:
        data = request.json
        include_reddit = data.get('include_reddit', False)
        
        # List of modules to run
        modules = [
            'mod_tokeninsight.py',
            'mod_alphavantage.py'
        ]
        if include_reddit:
            modules.extend([
                'mod_reddit_posts.py',
                'mod_reddit_comments.py'
            ])
        
        # Run each module in order
        outputs = []
        for module in modules:
            script_path = os.path.join(DATA_COLLECTION_DIR, module)
            if not os.path.exists(os.path.join(PROJECT_ROOT, script_path)):
                outputs.append(f"{module} failed: Script not found at {script_path}")
                continue
            
            cmd = [VENV_PYTHON, script_path]
            try:
                result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
                outputs.append(f"{module}: {result}")
            except subprocess.CalledProcessError as e:
                outputs.append(f"{module} failed: {e.output}")
        
        # Combine outputs
        output = '\n'.join(outputs)
        
        return jsonify({'output': output})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)