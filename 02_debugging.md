# Отладка: найти и исправить скрытые ошибки

Ниже дан Python-код. Он выглядит правдоподобно и частично работает, но содержит несколько ошибок разного типа.

Твоя задача:

1. Найти все реальные ошибки и слабые места, которые могут привести к неправильному результату или сбою.
2. Для каждой найденной проблемы кратко объяснить:
   - в чём ошибка;
   - когда она проявится;
   - почему это именно ошибка, а не просто вопрос стиля.
3. Исправить код.
4. Не переписывать программу полностью.
5. Не добавлять архитектурные слои, классы, абстракции и проверки, которые не нужны для исправления реальных проблем.
6. Следовать KISS и YAGNI.
7. После исправленного кода показать 3 коротких тестовых примера, которые демонстрируют, что найденные ошибки действительно исправлены.

Код:

```python
def load_numbers(filename):
    numbers = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                numbers.append(int(line))

    return numbers


def average(numbers):
    total = 0

    for number in numbers:
        total += number

    return total / len(numbers)


def remove_negative(numbers):
    for number in numbers:
        if number < 0:
            numbers.remove(number)

    return numbers


def find_top_three(numbers):
    numbers.sort(reverse=True)
    return numbers[:3]


def save_result(filename, numbers):
    with open(filename, "w", encoding="utf-8") as file:
        for number in numbers:
            file.write(number + "\n")


def process_file(input_file, output_file):
    numbers = load_numbers(input_file)

    remove_negative(numbers)

    avg = average(numbers)
    top_three = find_top_three(numbers)

    save_result(output_file, top_three)

    return {
        "average": avg,
        "top_three": top_three,
        "count": len(numbers),
    }
```

Не придумывай проблемы, которых в этом коде нет.
