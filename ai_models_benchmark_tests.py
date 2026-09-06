import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import ai_models_benchmark as benchmark


class TerminalOutput(io.StringIO):
    def isatty(self):
        return True


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
