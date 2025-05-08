from flask import Flask, render_template, request, jsonify
import subprocess
import os
import json
import plotly
import plotly.graph_objs as go

app = Flask(__name__)

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, 'scripts')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route('/')
def index():
    # Dropdown options
    cryptocurrencies = ['BNB', 'BTC', 'ADA', 'DOGE', 'ETH', 'LUNA', 'SOL', 'XRP']
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
        cmd = [
            'python', script_path,
            '--crypto', crypto,
            '--method', method.replace(' ', '_').lower(),  # e.g., 'Loughran McDonald' -> 'loughran_mcdonald'
            '--source', source_map[source],
            '--trading', trading_map[trading]
        ]
        result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, cwd=PROJECT_DIR)
        
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
            'results_path': os.path.abspath(RESULTS_DIR)
        })
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Analysis failed: {e.output}'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/update_data', methods=['POST'])
def update_data():
    try:
        data = request.json
        include_reddit = data.get('include_reddit', False)
        
        # Run update data script
        script_path = os.path.join(SCRIPTS_DIR, 'update_data.py')
        cmd = [
            'python', script_path,
            '--reddit', str(include_reddit).lower()
        ]
        result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, cwd=PROJECT_DIR)
        
        return jsonify({'output': result})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Data update failed: {e.output}'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)