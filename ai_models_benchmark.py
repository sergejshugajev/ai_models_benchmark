import json
import time
import urllib.request
from datetime import datetime


VERSION = "1"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = [
    "qwen3-coder",
    "devstral",
]
PROMPT = """
Напиши на Python функцию, которая принимает список целых чисел и возвращает:
1. минимальное значение;
2. максимальное значение;
3. среднее арифметическое;
4. медиану.

Не используй сторонние библиотеки.
Сначала кратко объясни решение, затем покажи код.
"""
RESULT_FILE = "ollama_benchmark_results.txt"


def test_model(model):
    print("\n" + "=" * 70)
    print(f"ТЕСТ: {model}")
    print("=" * 70)

    data = json.dumps(
        {
            "model": model,
            "prompt": PROMPT,
            "stream": True,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_time = time.perf_counter()
    first_token_time = None
    full_response = ""
    eval_count = None
    eval_duration = None
    prompt_eval_count = None
    load_duration = None

    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw_line in response:
                if not raw_line:
                    continue

                chunk = json.loads(raw_line.decode("utf-8"))
                text = chunk.get("response", "")

                if text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    full_response += text
                    print(text, end="", flush=True)

                if chunk.get("done"):
                    eval_count = chunk.get("eval_count")
                    eval_duration = chunk.get("eval_duration")
                    prompt_eval_count = chunk.get("prompt_eval_count")
                    load_duration = chunk.get("load_duration")
    except Exception as error:
        return {
            "model": model,
            "error": str(error),
        }

    end_time = time.perf_counter()
    total_seconds = end_time - start_time

    if first_token_time is not None:
        first_token_seconds = first_token_time - start_time
        generation_seconds = end_time - first_token_time
    else:
        first_token_seconds = None
        generation_seconds = None

    if eval_count and eval_duration:
        official_tps = eval_count / (eval_duration / 1_000_000_000)
    else:
        official_tps = None

    return {
        "model": model,
        "first_token_seconds": first_token_seconds,
        "total_seconds": total_seconds,
        "generation_seconds": generation_seconds,
        "tokens_generated": eval_count,
        "tokens_per_second": official_tps,
        "prompt_tokens": prompt_eval_count,
        "load_seconds": (
            load_duration / 1_000_000_000
            if load_duration is not None
            else None
        ),
        "response": full_response,
    }


def format_number(value, digits=2):
    if value is None:
        return "нет данных"
    return f"{value:.{digits}f}"


def main():
    print(f"OLLAMA BENCHMARK v{VERSION}")
    print("Модели:", ", ".join(MODELS))
    print()
    print("Ничего не трогай — тест выполнится автоматически.")

    results = []

    for model in MODELS:
        result = test_model(model)
        results.append(result)

        print("\n\nРезультат:")
        if "error" in result:
            print("ОШИБКА:", result["error"])
        else:
            print(
                "До первого токена:",
                format_number(result["first_token_seconds"]),
                "сек",
            )
            print(
                "Полное время:",
                format_number(result["total_seconds"]),
                "сек",
            )
            print(
                "Скорость:",
                format_number(result["tokens_per_second"]),
                "токен/сек",
            )
            print("Сгенерировано токенов:", result["tokens_generated"])
            print(
                "Загрузка модели:",
                format_number(result["load_seconds"]),
                "сек",
            )

    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ")
    print("=" * 70)

    good_results = [result for result in results if "error" not in result]
    for result in good_results:
        print(
            f"{result['model']:20} | "
            f"первый токен: {format_number(result['first_token_seconds']):>8} сек | "
            f"всего: {format_number(result['total_seconds']):>8} сек | "
            f"скорость: {format_number(result['tokens_per_second']):>8} ток/с"
        )

    with open(RESULT_FILE, "w", encoding="utf-8") as file:
        file.write("OLLAMA BENCHMARK\n")
        file.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write("PROMPT:\n")
        file.write(PROMPT.strip())
        file.write("\n\n")

        for result in results:
            file.write("=" * 70 + "\n")
            file.write(f"MODEL: {result['model']}\n")

            if "error" in result:
                file.write(f"ERROR: {result['error']}\n\n")
                continue

            file.write(
                f"First token: {format_number(result['first_token_seconds'])} s\n"
            )
            file.write(f"Total time: {format_number(result['total_seconds'])} s\n")
            file.write(
                "Generation speed: "
                f"{format_number(result['tokens_per_second'])} tokens/s\n"
            )
            file.write(f"Generated tokens: {result['tokens_generated']}\n")
            file.write(f"Prompt tokens: {result['prompt_tokens']}\n")
            file.write(
                f"Model load time: {format_number(result['load_seconds'])} s\n\n"
            )
            file.write("RESPONSE:\n")
            file.write(result["response"])
            file.write("\n\n")

    print()
    print(f"Полный отчёт сохранён: {RESULT_FILE}")


if __name__ == "__main__":
    main()
