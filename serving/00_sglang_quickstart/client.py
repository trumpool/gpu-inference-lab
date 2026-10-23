"""给本地 SGLang 服务发几个 OpenAI 兼容请求，顺便看 radix cache 复用前缀的效果。

用法（先用 launch.sh 把服务起起来）：
    ~/venvs/sglang/bin/python serving/00_sglang_quickstart/client.py
发完以后看服务器日志里的 `Prefill batch` 行：第二个共享前缀的请求，#cached-token 应该接近整个前缀长度。
"""
import json
import os
import time
import urllib.request

BASE = f"http://127.0.0.1:{os.environ.get('PORT', '30000')}/v1"


def get_model():
    with urllib.request.urlopen(f"{BASE}/models") as r:
        return json.load(r)["data"][0]["id"]


def chat(model, messages, max_tokens=128, temperature=0.0):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0


def main():
    model = get_model()
    print("model:", model)

    # 1. 普通请求。前一两个会慢：FlashInfer 第一次用到某些 kernel 时要现场编译
    for q in ["用一句话解释什么是 KV cache。",
              "What does cudaDeviceSynchronize() do? Answer in one sentence."]:
        out, dt = chat(model, [{"role": "user", "content": q}])
        u = out["usage"]
        print(f"\nQ: {q}\nA: {out['choices'][0]['message']['content'].strip()}")
        print(f"   prompt_tokens={u['prompt_tokens']} completion_tokens={u['completion_tokens']} time={dt:.3f}s")

    # 2. 两个请求共用一段很长的 system prompt：第二个应该命中 radix cache
    system = "You are a helpful assistant. " + (
        "Background: GPUs have many SMs; each SM runs warps of 32 threads. " * 60)
    for i, q in enumerate(["Summarize the background in 5 words.", "How many threads are in a warp?"]):
        out, dt = chat(model, [{"role": "system", "content": system}, {"role": "user", "content": q}],
                       max_tokens=32)
        print(f"\n[shared-prefix request {i + 1}] prompt_tokens={out['usage']['prompt_tokens']} time={dt:.3f}s")
        print("   A:", out["choices"][0]["message"]["content"].strip()[:120])


if __name__ == "__main__":
    main()
