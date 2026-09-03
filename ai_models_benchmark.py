import json
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path


VERSION = "7"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
PROMPT = """
Напиши на Python функцию, которая принимает список целых чисел и возвращает:
1. минимальное значение;
2. максимальное значение;
3. среднее арифметическое;
4. медиану.

Не используй сторонние библиотеки.
Сначала кратко объясни решение, затем покажи код.
"""
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


def get_installed_models():
    with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=10) as response:
        data = json.load(response)
    return [item.get("name") or item["model"] for item in data.get("models", [])]


def get_running_models():
    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    lines = [line for line in result.stdout.splitlines()[1:] if line.strip()]
    return [line.split()[0] for line in lines]


def prepare_model(selected_model):
    running_models = get_running_models()
    if not running_models:
        print("\nВ памяти нет загруженных моделей.")
        return

    print("\nСейчас в памяти:", ", ".join(running_models))
    other_models = [model for model in running_models if model != selected_model]

    if not other_models:
        print("Выбранная модель уже загружена. Продолжаю.")
        return

    for model in other_models:
        print(f"Останавливаю {model}...")
        subprocess.run(["ollama", "stop", model], check=True)
    print("Другие модели остановлены. Запускаю тест.")


def save_report(result):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model = result["model"].replace(":", "-")
    script_dir = Path(__file__).resolve().parent
    result_file = script_dir / f"ollama_test_{safe_model}_{timestamp}.txt"

    with open(result_file, "w", encoding="utf-8") as file:
        file.write(f"OLLAMA BENCHMARK v{VERSION}\n")
        file.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Модель: {result['model']}\n\n")
        file.write("ПРОМТ:\n")
        file.write(PROMPT.strip())
        file.write("\n\n")

        if "error" in result:
            file.write(f"ОШИБКА: {result['error']}\n")
        else:
            file.write(
                f"До первого токена: {format_number(result['first_token_seconds'])} сек\n"
            )
            file.write(
                f"Полное время: {format_number(result['total_seconds'])} сек\n"
            )
            file.write(
                "Скорость: "
                f"{format_number(result['tokens_per_second'])} токен/сек\n"
            )
            file.write(f"Сгенерировано токенов: {result['tokens_generated']}\n")
            file.write(f"Токенов в промпте: {result['prompt_tokens']}\n")
            file.write(
                f"Загрузка модели: {format_number(result['load_seconds'])} сек\n\n"
            )
            file.write("ОТВЕТ:\n")
            file.write(result["response"])
            file.write("\n")

    return result_file.name


def main():
    print(f"OLLAMA BENCHMARK v{VERSION}")
    print("\nПроверяю установленные модели Ollama...")

    models = get_installed_models()
    if not models:
        print("\nУстановленные модели не найдены.")
        return

    print()
    for number, model in enumerate(models, start=1):
        print(f"{number} - {model}")

    choice = input("\nВыбери номер модели для теста: ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(models):
        print("Неверный номер модели.")
        return

    model = models[int(choice) - 1]
    prepare_model(model)
    result = test_model(model)

    print("\n\nРезультат:")
    if "error" in result:
        print("ОШИБКА:", result["error"])
    else:
        print(
            "До первого токена:",
            format_number(result["first_token_seconds"]),
            "сек",
        )
        print("Полное время:", format_number(result["total_seconds"]), "сек")
        print(
            "Скорость:",
            format_number(result["tokens_per_second"]),
            "токен/сек",
        )

    result_file = save_report(result)
    print(f"\nПолный отчёт сохранён: {result_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nОШИБКА: {error}")
        result_file = save_report(
            {
                "model": "не выбрана",
                "error": str(error),
            }
        )
        print(f"Ошибка сохранена в отчёт: {result_file}")
    finally:
        input("\nНажми Enter для выхода...")
