from langchain_community.llms import LlamaCpp
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,  
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

print(llm('tell me a joke'))