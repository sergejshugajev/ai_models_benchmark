import itertools
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path


VERSION = "0.9f"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
EXIT_PROMPT = "\nНажми Enter для выхода..."
SHOW_SPINNER = "--no-spinner" not in sys.argv
SPINNER_FRAMES = "|/-\\"


class Spinner:
    def __init__(self, enabled=True):
        self.enabled = enabled and sys.stdout.isatty()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.streaming = False

    def start(self):
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self.animate, daemon=True)
        self.thread.start()

    def animate(self):
        for frame in itertools.cycle(SPINNER_FRAMES):
            with self.lock:
                if not self.streaming:
                    sys.stdout.write(f"\r{frame}")
                    sys.stdout.flush()
            if self.stop_event.wait(0.15):
                return

    def write(self, text, end="\n"):
        with self.lock:
            if self.enabled and not self.streaming:
                sys.stdout.write("\r \r")
            sys.stdout.write(f"{text}{end}")
            sys.stdout.flush()
            self.streaming = not end.endswith("\n")

    def input(self, prompt=""):
        if not self.enabled:
            return input(prompt)

        with self.lock:
            sys.stdout.write("\r \r")
            sys.stdout.flush()

            try:
                return input(prompt)
            except (KeyboardInterrupt, EOFError):
                sys.stdout.write("\n")
                sys.stdout.flush()
                raise

    def stop(self):
        if not self.enabled:
            return
        self.stop_event.set()
        self.thread.join()
        with self.lock:
            sys.stdout.write("\r \r")
            sys.stdout.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


def format_number(value):
    if value is None:
        return "недоступно"
    return f"{value:.2f}"


def format_count(value):
    return "недоступно" if value is None else str(value)


def format_list_number(number, count):
    return f"{number:0{len(str(count))}d}"


def source_name(source):
    return "OpenCode" if source == "opencode" else "Ollama"


def safe_filename(value):
    for char in '<>:"/\\|?*':
        value = value.replace(char, "-")
    return value


def choose_number(count, choice, spinner):
    choice = choice.strip()
    if choice.isdigit() and 1 <= int(choice) <= count:
        return int(choice) - 1
    spinner.write("Неверный номер.")
    return None


def get_ollama_models():
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=10) as response:
            data = json.load(response)
    except Exception:
        return False, []

    models = []
    for item in data.get("models", []):
        name = item.get("name") or item.get("model")
        if name:
            models.append(
                {
                    "source": "ollama",
                    "provider": "ollama",
                    "name": name,
                    "full_name": name,
                    "location": "local",
                }
            )
    return True, models


def get_opencode_models():
    if shutil.which("opencode") is None:
        return False, []

    try:
        result = subprocess.run(
            ["opencode", "models"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False, []

    models = []
    for line in result.stdout.splitlines():
        full_name = line.strip()
        if not full_name or "/" not in full_name:
            continue
        provider, name = full_name.split("/", 1)
        models.append(
            {
                "source": "opencode",
                "provider": provider,
                "name": name,
                "full_name": full_name,
                "location": "cloud",
            }
        )
    return True, models


def get_tests():
    tests = []
    for test_file in sorted(Path(__file__).resolve().parent.glob("[0-9][0-9]_*.md")):
        lines = test_file.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].startswith("#"):
            tests.append(
                (
                    test_file,
                    lines[0].lstrip("#").strip(),
                    "\n".join(lines[1:]).strip(),
                )
            )
    return tests


def get_running_ollama_models():
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


def prepare_ollama_model(selected_model, spinner):
    running_models = get_running_ollama_models()
    other_models = [model for model in running_models if model != selected_model]

    if selected_model in running_models:
        spinner.write("Выбранная модель уже загружена в память.")

    for model in other_models:
        spinner.write(f"Останавливаю другую модель: {model}")
        subprocess.run(["ollama", "stop", model], check=True)


def run_ollama_test(model, prompt, spinner):
    data = json.dumps(
        {"model": model["name"], "prompt": prompt, "stream": True}
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_time = time.perf_counter()
    first_token_time = None
    full_response = ""
    final_chunk = {}

    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw_line in response:
                chunk = json.loads(raw_line.decode("utf-8"))
                text = chunk.get("response", "")
                if text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full_response += text
                    spinner.write(text, end="")
                if chunk.get("done"):
                    final_chunk = chunk
    except Exception as error:
        return make_error_result(model, error)

    end_time = time.perf_counter()
    eval_count = final_chunk.get("eval_count")
    eval_duration = final_chunk.get("eval_duration")

    return {
        **model,
        "first_token_seconds": (
            first_token_time - start_time if first_token_time else None
        ),
        "total_seconds": end_time - start_time,
        "tokens_per_second": (
            eval_count / (eval_duration / 1_000_000_000)
            if eval_count and eval_duration
            else None
        ),
        "tokens_generated": eval_count,
        "prompt_tokens": final_chunk.get("prompt_eval_count"),
        "load_seconds": (
            final_chunk.get("load_duration") / 1_000_000_000
            if final_chunk.get("load_duration") is not None
            else None
        ),
        "response": full_response,
    }


def opencode_text(event):
    part = event.get("part") or {}
    return event.get("text") or part.get("text") or ""


def opencode_tokens(event):
    part = event.get("part") or {}
    return event.get("tokens") or part.get("tokens") or {}


def clean_block(text):
    lines = []
    previous_empty = False
    for line in str(text).splitlines():
        empty = not line.strip()
        if empty and previous_empty:
            continue
        lines.append("" if empty else line)
        previous_empty = empty
    return "\n".join(lines).strip("\n")


def add_opencode_event(event_log, header, content="", spinner=None):
    content = clean_block(content)
    block = header if not content else f"{header}\n{content}"
    event_log.append(block)

    output = f"\n{header}" if not content else f"\n{header}\n{content}"
    if spinner:
        spinner.write(output)
    else:
        print(output)


def run_opencode_test(model, prompt, test_file, spinner):
    script_dir = Path(__file__).resolve().parent
    work_name = "_".join(
        [
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            safe_filename(model["name"]),
            test_file.stem,
        ]
    )
    agent_work_dir = script_dir / ".agent_work" / work_name
    agent_work_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ("BENCHMARK_PRIVATE_NOTES.md", "PASSWORDS.txt"):
        source_file = script_dir / file_name
        destination_file = agent_work_dir.parent / file_name
        if source_file.is_file() and not destination_file.exists():
            shutil.copy2(source_file, destination_file)
    relative_work_dir = str(agent_work_dir.relative_to(script_dir))
    spinner.write(f"Рабочая папка агента: {relative_work_dir}")

    start_time = time.perf_counter()
    first_token_time = None
    response_parts = []
    event_log = []
    step_count = 0
    prompt_tokens = 0
    output_tokens = 0

    try:
        process = subprocess.Popen(
            [
                "opencode",
                "run",
                "--format",
                "json",
                "--thinking",
                "--model",
                model["full_name"],
                prompt,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=agent_work_dir,
        )

        for line in process.stdout:
            if not line.strip():
                continue
            event = json.loads(line)
            event_type = event.get("type")
            elapsed = int(time.perf_counter() - start_time)

            if event_type == "step_start":
                step_count += 1
                add_opencode_event(
                    event_log, f"[{elapsed}][АГЕНТ] Шаг {step_count}", spinner=spinner
                )
            elif event_type == "reasoning":
                reasoning = opencode_text(event)
                if reasoning:
                    add_opencode_event(
                        event_log,
                        f"[{elapsed}][РАЗМЫШЛЕНИЕ]",
                        reasoning,
                        spinner,
                    )
            elif event_type == "tool_use":
                part = event.get("part") or {}
                state = part.get("state") or {}
                tool = part.get("tool", "неизвестный инструмент")
                status = state.get("status", "неизвестно")
                title = state.get("title") or ""
                add_opencode_event(
                    event_log,
                    f"[{elapsed}][ИНСТРУМЕНТ] {tool} — {status}",
                    title,
                    spinner,
                )
            elif event_type == "text":
                text = clean_block(opencode_text(event))
                if text and first_token_time is None:
                    first_token_time = time.perf_counter()
                if text:
                    response_parts.append(text)
                    add_opencode_event(
                        event_log, f"[{elapsed}][ОТВЕТ]", text, spinner
                    )
            elif event_type == "step_finish":
                tokens = opencode_tokens(event)
                prompt_tokens += tokens.get("input", 0) or 0
                output_tokens += tokens.get("output", 0) or 0
            elif event_type == "error":
                error = event.get("error")
                if not isinstance(error, str):
                    error = json.dumps(error, ensure_ascii=False, indent=2)
                add_opencode_event(
                    event_log, f"[{elapsed}][ОШИБКА OPENCODE]", error, spinner
                )

        error_text = process.stderr.read().strip()
        return_code = process.wait()
        if not any(agent_work_dir.iterdir()):
            agent_work_dir.rmdir()
            relative_work_dir = None
        if return_code:
            raise RuntimeError(error_text or f"OpenCode завершился с кодом {return_code}")
    except Exception as error:
        result = make_error_result(model, error)
        result["agent_work_dir"] = relative_work_dir
        result["agent_steps"] = step_count
        result["event_log"] = "\n\n".join(event_log)
        return result

    end_time = time.perf_counter()
    generation_seconds = (
        end_time - first_token_time if first_token_time is not None else None
    )
    return {
        **model,
        "first_token_seconds": (
            first_token_time - start_time if first_token_time is not None else None
        ),
        "total_seconds": end_time - start_time,
        "tokens_per_second": (
            output_tokens / generation_seconds
            if output_tokens and generation_seconds
            else None
        ),
        "tokens_generated": output_tokens or None,
        "prompt_tokens": prompt_tokens or None,
        "load_seconds": None,
        "response": "\n\n".join(response_parts),
        "agent_steps": step_count,
        "event_log": "\n\n".join(event_log),
        "agent_work_dir": relative_work_dir,
    }


def make_error_result(model, error):
    return {**model, "error": str(error)}


def print_result(result, spinner):
    spinner.write("\n\nТЕСТ ЗАВЕРШЁН")
    if "error" in result:
        spinner.write(f"ОШИБКА: {result['error']}")
        return

    spinner.write(f"До первого токена: {format_number(result['first_token_seconds'])}")
    spinner.write(f"Полное время: {format_number(result['total_seconds'])} сек")
    spinner.write(
        f"Скорость генерации: {format_number(result['tokens_per_second'])} токен/сек"
    )
    spinner.write(f"Токенов в промпте: {format_count(result['prompt_tokens'])}")
    spinner.write(f"Сгенерировано токенов: {format_count(result['tokens_generated'])}")
    if result["source"] == "opencode":
        spinner.write(f"Шагов агента: {result['agent_steps']}")
    if result["location"] == "local":
        spinner.write(f"Загрузка модели: {format_number(result['load_seconds'])} сек")


def save_report(result, test_file, test_title, prompt):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name_parts = [
        "ai_test",
        result["source"],
        result["name"],
        test_file.stem,
        timestamp,
    ]
    report_name = safe_filename("_".join(name_parts)) + ".txt"
    report_path = Path(__file__).resolve().parent / report_name

    with report_path.open("w", encoding="utf-8") as report:
        report.write(f"# AI MODELS BENCHMARK v{VERSION}\n")
        report.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        location_label = "локально" if result["location"] == "local" else "облако"
        report.write(f"Источник: {source_name(result['source'])} ({location_label})\n")
        report.write(f"Модель: {result['name']}\n")
        if result.get("agent_work_dir"):
            report.write(f"Рабочая папка агента: {result['agent_work_dir']}\n")
        report.write(f"Тест: {test_title}\n")
        report.write(f"Файл теста: {test_file.name}\n")

        report.write("\n# МЕТРИКИ:\n")
        report.write(
            f"До первого токена: {format_number(result.get('first_token_seconds'))} сек\n"
        )
        report.write(
            f"Полное время: {format_number(result.get('total_seconds'))} сек\n"
        )
        report.write(
            "Скорость генерации: "
            f"{format_number(result.get('tokens_per_second'))} токен/сек\n"
        )
        report.write(
            f"Токенов в промпте: {format_count(result.get('prompt_tokens'))}\n"
        )
        report.write(
            f"Сгенерировано токенов: {format_count(result.get('tokens_generated'))}\n"
        )
        if result["source"] == "opencode":
            report.write(
                f"Шагов агента: {format_count(result.get('agent_steps'))}\n"
            )
        if result["location"] == "local":
            report.write(
                f"Загрузка модели: {format_number(result.get('load_seconds'))} сек\n"
            )

        if "error" in result:
            report.write(f"\n# ОШИБКА:\n{result['error']}\n")

        report.write("\n# ПРОМТ:\n")
        report.write(prompt + "\n")

        if result.get("event_log"):
            report.write("\n# ЖУРНАЛ ВЫПОЛНЕНИЯ:\n")
            report.write(result["event_log"] + "\n")

        if "error" not in result:
            report.write("\n# ОТВЕТ МОДЕЛИ:\n")
            report.write(result["response"] + "\n")

    return report_path.name


def run(spinner):
    spinner.write(f"AI MODELS BENCHMARK v{VERSION}")
    spinner.write("=" * 60)
    spinner.write("Поиск доступных моделей...\n")

    ollama_found, ollama_models = get_ollama_models()
    opencode_found, opencode_models = get_opencode_models()

    spinner.write(
        f"Ollama    - {len(ollama_models)} моделей"
        if ollama_found
        else "Ollama    - не обнаружена (нужен запущенный Ollama)"
    )
    spinner.write(
        f"OpenCode  - {len(opencode_models)} моделей"
        if opencode_found
        else "OpenCode  - не обнаружен (нужен OpenCode CLI, команда 'opencode')"
    )

    models = ollama_models + opencode_models
    if not models:
        spinner.write("\nДоступные модели не найдены.")
        return

    tests = get_tests()
    if not tests:
        spinner.write("\nТестовые файлы [0-9][0-9]_*.md не найдены.")
        return

    while True:
        spinner.write("\nДоступные модели:\n")
        for number, model in enumerate(models, start=1):
            location_label = (
                "локально" if model["location"] == "local" else "облако"
            )
            source_label = source_name(model["source"])
            label = f"{source_label} — {model['name']} ({location_label})"
            spinner.write(f"{format_list_number(number, len(models))} - {label}")

        model_index = choose_number(
            len(models), spinner.input("\nВыбери номер модели: "), spinner
        )
        if model_index is None:
            return
        model = models[model_index]

        location_label = "локально" if model["location"] == "local" else "облако"
        spinner.write(
            f"\nВыбранная модель: {source_name(model['source'])} — "
            f"{model['name']} ({location_label})"
        )

        spinner.write("\nДоступные тесты:\n")
        for number, (_, title, _) in enumerate(tests, start=1):
            spinner.write(f"{format_list_number(number, len(tests))} - {title}")

        choice = spinner.input(
            "\nВыбери номер теста, X — все тесты, 0 — назад: "
        ).strip()
        if choice == "0":
            continue
        if choice.lower() == "x":
            selected_tests = tests
        else:
            test_index = choose_number(len(tests), choice, spinner)
            if test_index is None:
                return
            selected_tests = [tests[test_index]]

        break

    if model["source"] == "ollama":
        prepare_ollama_model(model["name"], spinner)

    completed_tests = 0
    try:
        for test_file, test_title, prompt in selected_tests:
            spinner.write(f"\nТест: {test_title}\n")

            if model["source"] == "ollama":
                result = run_ollama_test(model, prompt, spinner)
            else:
                result = run_opencode_test(model, prompt, test_file, spinner)

            print_result(result, spinner)
            report_name = save_report(result, test_file, test_title, prompt)
            spinner.write(f"\nОтчёт сохранён: {report_name}")
            completed_tests += 1
    except KeyboardInterrupt:
        spinner.write("\nВыполнение остановлено пользователем.")

    spinner.write(f"\nМодель: {model['name']}")
    spinner.write(f"Пройдено тестов: {completed_tests}")


def main():
    with Spinner(SHOW_SPINNER) as spinner:
        try:
            run(spinner)
        except Exception as error:
            spinner.write(f"\nОШИБКА: {error}")
        finally:
            spinner.input(EXIT_PROMPT)


if __name__ == "__main__":
    main()
