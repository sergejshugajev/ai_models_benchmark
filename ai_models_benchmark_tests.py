import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import ai_models_benchmark as benchmark


class TerminalOutput(io.StringIO):
    def isatty(self):
        return True


class FakeHttpResponse:
    def __init__(self, chunks):
        self.lines = [
            (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
            for chunk in chunks
        ]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        return iter(self.lines)


class FakeProcess:
    def __init__(self, events):
        self.stdout = io.StringIO(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
        )
        self.stderr = io.StringIO("")

    def wait(self):
        return 0


class SpinnerTests(unittest.TestCase):
    def test_write_preserves_streaming_output(self):
        output = TerminalOutput()
        with patch.object(benchmark.sys, "stdout", output):
            spinner = benchmark.Spinner()
            spinner.write("one", end="")
            spinner.write(" two")
            spinner.write("three")

        self.assertEqual(output.getvalue(), "\r \rone two\n\r \rthree\n")
        self.assertFalse(spinner.streaming)

    def test_disabled_when_output_is_not_terminal(self):
        output = io.StringIO()
        with patch.object(benchmark.sys, "stdout", output):
            spinner = benchmark.Spinner()
            spinner.start()

        self.assertFalse(spinner.enabled)
        self.assertIsNone(spinner.thread)
        self.assertEqual(output.getvalue(), "")

    def test_interrupted_input_adds_newline_and_reraises(self):
        for error in (KeyboardInterrupt, EOFError):
            with self.subTest(error=error.__name__):
                output = TerminalOutput()
                with patch.object(benchmark.sys, "stdout", output):
                    spinner = benchmark.Spinner()
                    with patch("builtins.input", side_effect=error):
                        with self.assertRaises(error):
                            spinner.input("Prompt: ")

                self.assertEqual(output.getvalue(), "\r \r\n")


class RunTests(unittest.TestCase):
    def run_with_test_choice(self, choice, run_side_effect=None, input_values=None):
        spinner = Mock()
        spinner.input.side_effect = input_values or ["1", choice]
        model = {
            "source": "ollama",
            "provider": "ollama",
            "name": "test-model",
            "full_name": "test-model",
            "location": "local",
        }
        tests = [
            ("01_test.md", "Первый тест", "prompt 1"),
            ("02_test.md", "Второй тест", "prompt 2"),
        ]
        result = object()

        with (
            patch.object(benchmark, "get_ollama_models", return_value=(True, [model])),
            patch.object(benchmark, "get_opencode_models", return_value=(False, [])),
            patch.object(benchmark, "get_tests", return_value=tests),
            patch.object(benchmark, "prepare_ollama_model") as prepare_model,
            patch.object(benchmark, "run_ollama_test", return_value=result) as run_test,
            patch.object(benchmark, "print_result"),
            patch.object(benchmark, "save_report", return_value="report.txt") as save_report,
        ):
            if run_side_effect is not None:
                run_test.side_effect = run_side_effect
            benchmark.run(spinner)

        return spinner, model, tests, prepare_model, run_test, save_report

    def test_x_runs_every_test_with_separate_report(self):
        spinner, model, tests, prepare_model, run_test, save_report = (
            self.run_with_test_choice("X")
        )

        prepare_model.assert_called_once_with(model["name"], spinner)
        self.assertEqual(run_test.call_count, len(tests))
        self.assertEqual(save_report.call_count, len(tests))
        spinner.write.assert_any_call("Пройдено тестов: 2")

    def test_number_runs_only_selected_test(self):
        spinner, model, tests, _, run_test, save_report = self.run_with_test_choice("2")

        run_test.assert_called_once_with(model, tests[1][2], spinner)
        save_report.assert_called_once_with(
            run_test.return_value, tests[1][0], tests[1][1], tests[1][2]
        )
        spinner.write.assert_any_call("Пройдено тестов: 1")

    def test_interruption_stops_batch_and_shows_completed_count(self):
        spinner, _, _, _, run_test, save_report = self.run_with_test_choice(
            "X", [object(), KeyboardInterrupt]
        )

        self.assertEqual(run_test.call_count, 2)
        save_report.assert_called_once()
        spinner.write.assert_any_call("\nВыполнение остановлено пользователем.")
        spinner.write.assert_any_call("Пройдено тестов: 1")

    def test_zero_returns_from_tests_to_model_selection(self):
        spinner, _, _, _, run_test, save_report = self.run_with_test_choice(
            "1", input_values=["1", "0", "1", "1"]
        )

        self.assertEqual(spinner.input.call_count, 4)
        run_test.assert_called_once()
        save_report.assert_called_once()


class ProviderFlowIntegrationTests(unittest.TestCase):
    ollama_model = {
        "source": "ollama",
        "provider": "ollama",
        "name": "test-ollama",
        "full_name": "test-ollama",
        "location": "local",
    }
    opencode_model = {
        "source": "opencode",
        "provider": "test",
        "name": "test-agent",
        "full_name": "test/test-agent",
        "location": "cloud",
    }

    def run_isolated(self, model_choice, provider_patch, timer_values):
        spinner = Mock()
        spinner.input.side_effect = [model_choice, "1"]

        with tempfile.TemporaryDirectory() as directory:
            program_dir = Path(directory)
            (program_dir / "01_test.md").write_text(
                "# Интеграционный тест\nВерни однозначный ответ.", encoding="utf-8"
            )
            with (
                patch.object(benchmark, "PROGRAM_DIR", program_dir),
                patch.object(
                    benchmark,
                    "get_ollama_models",
                    return_value=(True, [self.ollama_model]),
                ),
                patch.object(
                    benchmark,
                    "get_opencode_models",
                    return_value=(True, [self.opencode_model]),
                ),
                patch.object(benchmark, "prepare_ollama_model"),
                patch.object(
                    benchmark.time, "perf_counter", side_effect=timer_values
                ),
                provider_patch,
            ):
                benchmark.run(spinner)

            reports = list(program_dir.glob("ai_test_*.txt"))
            self.assertEqual(len(reports), 1)
            return reports[0].read_text(encoding="utf-8")

    def test_ollama_flow_creates_expected_report(self):
        response = FakeHttpResponse(
            [
                {"response": "Тестовый ", "done": False},
                {
                    "response": "ответ",
                    "done": True,
                    "prompt_eval_count": 10,
                    "eval_count": 2,
                    "eval_duration": 1_000_000_000,
                    "load_duration": 500_000_000,
                },
            ]
        )

        report = self.run_isolated(
            "1",
            patch.object(benchmark.urllib.request, "urlopen", return_value=response),
            [100.0, 101.0, 104.0],
        )

        self.assertIn("Источник: Ollama (локально)", report)
        self.assertIn("Модель: test-ollama", report)
        self.assertIn("До первого токена: 1.00 сек", report)
        self.assertIn("Полное время: 4.00 сек", report)
        self.assertIn("Скорость генерации: 2.00 токен/сек", report)
        self.assertIn("Токенов в промпте: 10", report)
        self.assertIn("Сгенерировано токенов: 2", report)
        self.assertIn("# ОТВЕТ МОДЕЛИ:\nТестовый ответ", report)

    def test_opencode_flow_creates_expected_report(self):
        process = FakeProcess(
            [
                {"type": "step_start"},
                {"type": "reasoning", "text": "Проверяю условие"},
                {
                    "type": "tool_use",
                    "part": {
                        "tool": "read",
                        "state": {"status": "completed", "title": "Прочитан файл"},
                    },
                },
                {"type": "text", "text": "Однозначный тестовый ответ"},
                {
                    "type": "step_finish",
                    "tokens": {
                        "input": 10,
                        "output": 4,
                        "reasoning": 2,
                        "total": 19,
                        "cache": {"read": 3},
                    },
                },
            ]
        )

        report = self.run_isolated(
            "2",
            patch.object(benchmark.subprocess, "Popen", return_value=process),
            [100.0, 100.0, 101.0, 102.0, 103.0, 104.0, 104.5, 105.0],
        )

        self.assertIn("Источник: OpenCode (облако)", report)
        self.assertIn("Модель: test-agent", report)
        self.assertIn("До первого текста: 4.00 сек", report)
        self.assertIn("Полное время: 5.00 сек", report)
        self.assertIn("Эффективная скорость агента: 0.80 токен/сек", report)
        self.assertIn("Входных токенов без кэша: 10", report)
        self.assertIn("Сгенерировано токенов: 4", report)
        self.assertIn("Токенов размышления: 2", report)
        self.assertIn("Токенов из кэша: 3", report)
        self.assertIn("Всего токенов: 19", report)
        self.assertIn("Шагов агента: 1", report)
        self.assertIn("[0][АГЕНТ] Шаг 1", report)
        self.assertIn("[1][РАЗМЫШЛЕНИЕ]\nПроверяю условие", report)
        self.assertIn("[2][ИНСТРУМЕНТ] read — completed\nПрочитан файл", report)
        self.assertIn("[3][ОТВЕТ]\nОднозначный тестовый ответ", report)
        self.assertIn("# ОТВЕТ МОДЕЛИ:\nОднозначный тестовый ответ", report)

class FormatNumberTests(unittest.TestCase):
    def test_none_returns_unavailable(self):
        self.assertEqual(benchmark.format_number(None), "недоступно")

    def test_formats_to_two_decimals(self):
        cases = [
            (0, "0.00"),
            (1.5, "1.50"),
            (2.345, "2.35"),
            (-0.554, "-0.55"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(benchmark.format_number(value), expected)


class FormatCountTests(unittest.TestCase):
    def test_none_returns_unavailable(self):
        self.assertEqual(benchmark.format_count(None), "недоступно")

    def test_converts_to_string(self):
        cases = [(0, "0"), (11, "11"), (1986, "1986")]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(benchmark.format_count(value), expected)


class CalculateRateTests(unittest.TestCase):
    def test_calculates_rate(self):
        self.assertEqual(benchmark.calculate_rate(120, 4), 30)

    def test_unavailable_without_count_or_time(self):
        for count, seconds in [(None, 1), (1, None), (0, 1), (1, 0)]:
            with self.subTest(count=count, seconds=seconds):
                self.assertIsNone(benchmark.calculate_rate(count, seconds))


class PrintResultTests(unittest.TestCase):
    def test_opencode_uses_agent_metric_labels(self):
        spinner = Mock()
        result = {
            "source": "opencode", "location": "cloud",
            "first_token_seconds": 2, "total_seconds": 10,
            "tokens_per_second": 5, "prompt_tokens": 20,
            "tokens_generated": 50, "reasoning_tokens": 3,
            "cache_read_tokens": 40, "total_tokens": 113,
            "agent_steps": 2,
        }

        benchmark.print_result(result, spinner)

        spinner.write.assert_any_call("До первого текста: 2.00 сек")
        spinner.write.assert_any_call(
            "Эффективная скорость агента: 5.00 токен/сек"
        )
        spinner.write.assert_any_call("Входных токенов без кэша: 20")

    def test_ollama_keeps_generation_metric_labels(self):
        spinner = Mock()
        result = {
            "source": "ollama", "location": "local",
            "first_token_seconds": 2, "total_seconds": 10,
            "tokens_per_second": 5, "prompt_tokens": 20,
            "tokens_generated": 50, "load_seconds": 1,
        }

        benchmark.print_result(result, spinner)

        spinner.write.assert_any_call("До первого токена: 2.00 сек")
        spinner.write.assert_any_call("Скорость генерации: 5.00 токен/сек")
        spinner.write.assert_any_call("Токенов в промпте: 20")


class FormatListNumberTests(unittest.TestCase):
    def test_keeps_plain_numbers_for_short_list(self):
        self.assertEqual(benchmark.format_list_number(1, 9), "1")

    def test_pads_numbers_for_long_list(self):
        cases = [(1, 10, "01"), (10, 10, "10"), (1, 100, "001")]
        for number, count, expected in cases:
            with self.subTest(number=number, count=count):
                self.assertEqual(
                    benchmark.format_list_number(number, count), expected
                )


class ChooseNumberTests(unittest.TestCase):
    def test_accepts_leading_zero(self):
        spinner = Mock()
        cases = [("01", 0), ("00001", 0), ("005", 4)]

        for choice, expected in cases:
            with self.subTest(choice=choice):
                self.assertEqual(
                    benchmark.choose_number(5, choice, spinner), expected
                )
        spinner.write.assert_not_called()


class SourceNameTests(unittest.TestCase):
    def test_opencode(self):
        self.assertEqual(benchmark.source_name("opencode"), "OpenCode")

    def test_everything_else_is_ollama(self):
        for source in ["ollama", "", "other"]:
            with self.subTest(source=source):
                self.assertEqual(benchmark.source_name(source), "Ollama")


class SafeFilenameTests(unittest.TestCase):
    def test_replaces_forbidden_characters(self):
        self.assertEqual(
            benchmark.safe_filename('a<>:"/\\|?*b'), "a---------b"
        )

    def test_keeps_allowed_characters(self):
        self.assertEqual(
            benchmark.safe_filename("gpt-5.6-sol_05 test"), "gpt-5.6-sol_05 test"
        )


class CleanBlockTests(unittest.TestCase):
    def test_strips_outer_empty_lines(self):
        self.assertEqual(benchmark.clean_block("\n  text  \n\n"), "  text  ")
        self.assertEqual(
            benchmark.clean_block("a\n  indented  \nb"), "a\n  indented  \nb"
        )

    def test_keeps_first_line_indentation(self):
        self.assertEqual(
            benchmark.clean_block(" \t\n\n    code\n    next\n \t\n"),
            "    code\n    next",
        )

    def test_empty_content(self):
        for text in ["", "\n\n", " \t\n \t"]:
            with self.subTest(text=text):
                self.assertEqual(benchmark.clean_block(text), "")

    def test_collapses_consecutive_empty_lines(self):
        self.assertEqual(
            benchmark.clean_block("a\n\n\nb\n\nc"), "a\n\nb\n\nc"
        )

    def test_keeps_single_empty_lines(self):
        self.assertEqual(benchmark.clean_block("a\n\nb"), "a\n\nb")

    def test_non_string_input(self):
        self.assertEqual(benchmark.clean_block(None), "None")
        self.assertEqual(benchmark.clean_block(123), "123")


class OpencodeTextTests(unittest.TestCase):
    def test_event_text_has_priority(self):
        event = {"text": "from event", "part": {"text": "from part"}}
        self.assertEqual(benchmark.opencode_text(event), "from event")

    def test_falls_back_to_part_text(self):
        event = {"part": {"text": "from part"}}
        self.assertEqual(benchmark.opencode_text(event), "from part")

    def test_missing_text_returns_empty_string(self):
        for event in [{}, {"part": {}}, {"text": "", "part": {"text": ""}}]:
            with self.subTest(event=event):
                self.assertEqual(benchmark.opencode_text(event), "")


class OpencodeTokensTests(unittest.TestCase):
    def test_event_tokens_have_priority(self):
        event = {"tokens": {"input": 1}, "part": {"tokens": {"input": 2}}}
        self.assertEqual(benchmark.opencode_tokens(event), {"input": 1})

    def test_falls_back_to_part_tokens(self):
        event = {"part": {"tokens": {"output": 5}}}
        self.assertEqual(benchmark.opencode_tokens(event), {"output": 5})

    def test_missing_tokens_return_empty_dict(self):
        for event in [{}, {"part": {}}]:
            with self.subTest(event=event):
                self.assertEqual(benchmark.opencode_tokens(event), {})


class AddOpencodeEventTests(unittest.TestCase):
    def test_header_only_without_content(self):
        event_log = []
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.add_opencode_event(event_log, "[АГЕНТ] Шаг 1")
        self.assertEqual(event_log, ["[АГЕНТ] Шаг 1"])
        self.assertEqual(output.getvalue(), "\n[АГЕНТ] Шаг 1\n")

    def test_header_with_cleaned_content(self):
        event_log = []
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.add_opencode_event(
                event_log, "[ОТВЕТ]", "a\n\n\nb\n"
            )
        self.assertEqual(event_log, ["[ОТВЕТ]\na\n\nb"])
        self.assertEqual(output.getvalue(), "\n[ОТВЕТ]\na\n\nb\n")


if __name__ == "__main__":
    unittest.main()
