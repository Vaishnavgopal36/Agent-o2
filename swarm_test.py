import asyncio
import time
import os
from dotenv import load_dotenv
from groq import AsyncGroq

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from the .env file.")

# --- 2. MOE ARCHITECTURE CONFIGURATION (GROQ ONLY) ---
SWARM_NODES = {
    "Router": "llama-3.1-8b-instant", 
    "Code_Generation": "openai/gpt-oss-120b", # Assuming this alias resolves correctly on your Groq tier
    "Architecture_RAG": "meta-llama/llama-4-scout-17b-16e-instruct", 
    "Critic_Execution": "qwen/qwen3-32b"     
}

# --- 3. COMPREHENSIVE WORKLOADS ---
TEST_CASES = {
    "Router": {
        "system": "You are a low-latency gating router. Output EXACTLY ONE of the following node destinations based on the user prompt: [CODE_GEN, RAG_SEARCH, CRITIC_REVIEW, GENERAL_CHAT]. Do not output any other text.",
        "user": "The Docker container exited with code 137. I need you to look at the stack trace and tell me if the distributed training script caused an OOM error."
    },
    "Code_Generation": {
        "system": "You are an expert Backend Systems Architect. Write production-ready, highly optimized code.",
        "user": "Implement a high-performance gRPC server in Python using asyncio. Define a service that accepts a stream of image embeddings (float32 arrays) and returns a normalized vector."
    },
    "Architecture_RAG": {
        "system": "You are an AI research assistant specializing in hallucination mitigation architectures.",
        "user": "Compare and contrast the implementation complexities of Self-Play Fine-Tuning (SPIN) versus Dynamic Contrastive Logit Adjustment (DCLA) when applied to a vision-language model."
    },
    "Critic_Execution": {
        "system": "You are an autonomous code reviewer and debugging agent. Analyze the provided logic and identify vulnerabilities.",
        "user": "Review the following C code for a custom memory allocator. Identify any potential memory leaks or pointer arithmetic errors: \nvoid* custom_malloc(size_t size) {\n    void* ptr = sbrk(0);\n    if (sbrk(size) == (void*)-1) return NULL;\n    return ptr;\n}"
    }
}

# --- 4. ASYNC SDK BENCHMARKER ---
async def test_groq_endpoint(client, role, model):
    prompt_data = TEST_CASES[role]
    start_time = time.time()
    ttft = None
    token_count = 0

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_data["system"]},
                {"role": "user", "content": prompt_data["user"]}
            ],
            stream=True,
            max_tokens=1024
        )
        
        async for chunk in stream:
            if ttft is None:
                ttft = time.time() - start_time
            if chunk.choices and chunk.choices[0].delta.content:
                token_count += 1
                
        end_time = time.time()
        req_time = end_time - start_time
        speed = token_count / req_time if req_time > 0 else 0
        
        return {"role": role, "model": model, "status": "200", "ttft": ttft or 0, "speed": speed}
        
    except Exception as e:
        error_msg = str(e)
        status = "429" if "429" in error_msg else "Failed"
        return {"role": role, "model": model, "status": status, "error": error_msg}

# --- 5. LOAD TESTING EXECUTION ---
async def run_swarm_test():
    print(f"\nInitiating Groq-Only MoE Endpoint Stress Test...")
    print(f"Groq Key Loaded: {bool(GROQ_API_KEY)}")
    print("-" * 75)
    
    # Initialize the official async client
    groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    
    tasks = []
    
    # Queue up all nodes concurrently
    for role, model in SWARM_NODES.items():
        tasks.append(test_groq_endpoint(groq_client, role, model))
        
    results = await asyncio.gather(*tasks)
    
    print(f"{'NODE ROLE':<18} | {'MODEL':<40} | {'STATUS':<8} | {'TTFT (s)':<8} | {'TPS':<8}")
    print("-" * 85)
    
    for res in results:
        if res['status'] == "200":
            print(f"{res['role']:<18} | {res['model']:<40} | {res['status']:<8} | {res['ttft']:<8.3f} | {res['speed']:<8.2f}")
        elif res['status'] == "429":
            print(f"{res['role']:<18} | {res['model']:<40} | {res['status']:<8} | RATE LIMITED (429)")
        else:
            print(f"{res['role']:<18} | {res['model']:<40} | {res['status']:<8} | ERROR: {res.get('error')[:45]}...")

if __name__ == "__main__":
    asyncio.run(run_swarm_test())