import os
import time
import csv
from dotenv import load_dotenv
from groq import Groq
from ollama import Client

# Load environment variables
load_dotenv()

# Define the models to evaluate
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b", 
    "meta-llama/llama-4-scout-17b-16e-instruct"
]

OLLAMA_MODELS = [
    "gpt-oss:120b-cloud",
    "llama3.3:70b",
    "qwen2.5:72b" # Or equivalent massive models you deploy on Ollama Cloud
]

# Payloads modeled after a dual-model RAG and coding agent architecture
TEST_CASES = {
    "Fast_Routing": {
        "system": "You are a fast routing agent. Return ONLY the exact target node name: [VECTOR_DB, CODE_ANALYZER, API_FETCHER].",
        "user": "I need to search the codebase for the implementation of the DCLA contrastive logit adjustment function."
    },
    "Heavy_Analysis": {
        "system": "You are an expert diagnostic coding agent. Analyze the architectural logic.",
        "user": "Evaluate this text decoder output design for potential hallucinations and logical inconsistencies when pruning attention mechanisms."
    }
}

def benchmark_groq(client, model, test_name, prompt):
    print(f"  [Groq] Testing {model}...")
    start_time = time.time()
    ttft = None
    token_count = 0
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ],
            stream=True
        )
        
        for chunk in stream:
            if ttft is None:
                ttft = time.time() - start_time
            if chunk.choices and chunk.choices[0].delta.content:
                token_count += 1
                
        total_time = time.time() - start_time
        tps = token_count / total_time if total_time > 0 else 0
        
        return {
            "Provider": "Groq", "Model": model, "Workload": test_name, 
            "TTFT (s)": round(ttft, 4), "Total Time (s)": round(total_time, 4), 
            "Est. TPS": round(tps, 2), "Status": "Success"
        }
    except Exception as e:
        return {
            "Provider": "Groq", "Model": model, "Workload": test_name, 
            "TTFT (s)": "Error", "Total Time (s)": "Error", 
            "Est. TPS": "Error", "Status": f"Failed: {str(e)}"
        }

def benchmark_ollama(client, model, test_name, prompt):
    print(f"  [Ollama] Testing {model}...")
    start_time = time.time()
    ttft = None
    token_count = 0
    
    try:
        stream = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ],
            stream=True
        )
        
        for chunk in stream:
            if ttft is None:
                ttft = time.time() - start_time
            if 'message' in chunk and 'content' in chunk['message']:
                token_count += 1
                
        total_time = time.time() - start_time
        tps = token_count / total_time if total_time > 0 else 0
        
        return {
            "Provider": "Ollama Cloud", "Model": model, "Workload": test_name, 
            "TTFT (s)": round(ttft, 4), "Total Time (s)": round(total_time, 4), 
            "Est. TPS": round(tps, 2), "Status": "Success"
        }
    except Exception as e:
        return {
            "Provider": "Ollama Cloud", "Model": model, "Workload": test_name, 
            "TTFT (s)": "Error", "Total Time (s)": "Error", 
            "Est. TPS": "Error", "Status": f"Failed: {str(e)}"
        }

def run_suite():
    print("Initializing Agentic Benchmark Suite...\n")
    
    # Verify environment keys are available
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("OLLAMA_API_KEY"):
        print("Error: Missing API keys in .env file.")
        return

    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    ollama_client = Client(
        host="https://ollama.com", 
        headers={'Authorization': f"Bearer {os.environ.get('OLLAMA_API_KEY')}"}
    )
    
    results = []
    
    for test_name, prompt in TEST_CASES.items():
        print(f"\n--- Running Workload: {test_name} ---")
        
        # Benchmarking Groq Models
        for model in GROQ_MODELS:
            res = benchmark_groq(groq_client, model, test_name, prompt)
            results.append(res)
            
        # Benchmarking Ollama Cloud Models
        for model in OLLAMA_MODELS:
            res = benchmark_ollama(ollama_client, model, test_name, prompt)
            results.append(res)
            
    # Exporting the report
    csv_file = "benchmark_report.csv"
    keys = ["Provider", "Model", "Workload", "TTFT (s)", "Total Time (s)", "Est. TPS", "Status"]
    
    with open(csv_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
        
    print(f"\nBenchmarking Complete. Detailed report saved to: {csv_file}")

if __name__ == "__main__":
    run_suite()