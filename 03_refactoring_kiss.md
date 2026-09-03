# Рефакторинг: KISS против переусложнения

Ниже дан рабочий Python-код. Он решает простую задачу, но написан избыточно сложно.

Твоя задача:

1. Упростить код, сохранив его внешнее поведение.
2. Удалить лишние абстракции, обёртки и повторения.
3. Следовать KISS и YAGNI.
4. Не добавлять новые возможности.
5. Не добавлять проверки невозможных состояний.
6. Не менять формат входных и выходных данных.
7. Не использовать сторонние библиотеки.
8. После нового варианта кратко объяснить:
   - что именно было переусложнено;
   - что удалено;
   - почему новый вариант проще;
   - какие части исходной архитектуры ты сознательно не сохранил и почему.
9. Если видишь несколько разумных вариантов, выбери самый простой достаточный.

Исходный код:

```python
class ValueValidator:
    def validate(self, value):
        if value is None:
            return False

        if isinstance(value, int):
            return True

        return False


class NumberWrapper:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value


class NumberFactory:
    def create(self, value):
        validator = ValueValidator()

        if not validator.validate(value):
            raise ValueError("Invalid value")

        return NumberWrapper(value)


class AdditionService:
    def add(self, first, second):
        factory = NumberFactory()

        first_number = factory.create(first)
        second_number = factory.create(second)

        result = (
            first_number.get_value()
            + second_number.get_value()
        )

        return NumberWrapper(result)


class ResultFormatter:
    def format(self, number_wrapper):
        return {
            "result": number_wrapper.get_value()
        }


class CalculatorApplication:
    def execute(self, first, second):
        service = AdditionService()
        formatter = ResultFormatter()

        result = service.add(first, second)

        return formatter.format(result)


def calculate(first, second):
    app = CalculatorApplication()
    return app.execute(first, second)
```

Ожидаемое внешнее поведение должно остаться таким:

```python
calculate(2, 3)
# {"result": 5}
```
