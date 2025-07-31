import os
import time
import json
import argparse
from tqdm import tqdm
import pandas as pd
import openai


schema_precision_recall = {
    "type": "object",
    "properties": {
        "precision": {
            "type": "number",
            "minimum": 10,
            "maximum": 100
        },
        "recall": {
            "type": "number",
            "minimum": 10,
            "maximum": 100
        },
    },
    "required": ["precision", "recall"]
}


# Prompt rules as provided by user
RULES = """
Правила:

**Установка точности**
   - 100% если ответ полностью релевантен: Содержит подробный и полный ответ на запрос.
      Вопрос "что нужно сделать, чтобы подключить только один VK канал через API?"
      Ответ: Чтобы подключить только один VK канал через API, выполните следующие шаги:
         1. В настройках группы ВК в разделе "Управление -> Работа с API -> Callback API" выберите версию API 5.103 и укажите адрес:
         - Для подключения только API (комментарии): https://chat.autofaq.ai/vk-adapter/api/vk/webhooks?api=&code=<строка_которую_должен_вернуть_сервер>
         - Для подключения только Direct (личные сообщения): https://chat.autofaq.ai/vk-adapter/api/vk/webhooks?direct=&code=<строка_которую_должен_вернуть_сервер>
         где и — ID каналов из AutoFAQ, а <строка_которую_должен_вернуть_сервер> — строка из раздела группы ВК "Управление" -> "Работа с API" -> "Callback API".
         2. Выберите типы событий в зависимости от типа подключаемого канала:
         - Для API (комментарии): "Комментарии на стене: Добавление, Редактирование, Удаление, Восстановление"
         - Для Direct (личные сообщения): "Сообщения: Входящее сообщение, Исходящее сообщение, Редактирование сообщения"
   - 70% если ответ почти точен: ответ отвечает на запрос, но не содержит важных деталей
      Пояснение: не показал все открытые вакансии
      Пример:
         Вопрос "Есть ли открытые вакансии?"
         Ответ: Да, на сайте AutoFAQ открыты вакансии. Они ищут специалистов, готовых работать в IT-компании над сложными проектами. В частности, сейчас открыта вакансия инженера технической поддержки с зарплатой от 150,000 до 220,000 рублей в зависимости от опыта и навыков. Требования включают опыт работы от 3 лет, знание Linux, Docker, Bash, JavaScript и другие технические навыки. Их команда распределенная, поэтому возможна удаленная работа. Подробнее можно ознакомиться на странице вакансий AutoFAQ. Есть ли еще что-то, о чём вы хотите узнать?
   - 40% если ответ очень неточен: содержит крайне минимальную или косвенную связь с запросом (например ответил про вакансии, но указал вводящую в заблуждение инфу)
      Пояснение: показал несуществующие открытые вакансии
      Вопрос "Есть ли открытые вакансии?"
      Ответ: Да, в компании AutoFAQ есть открытые вакансии. На сайте компании можно найти информацию о различных позициях, включая:
               * **Специалист по управлению ИИ-системами**. Задача такого специалиста — настраивать чат-ботов, обучать их новым сценариям и корректировать модели.
               * **Аналитик клиентского опыта**. Аналитик изучает данные о взаимодействии с клиентами, выявляет узкие места и предлагает решения для улучшения сервиса.
               * **Куратор ИИ-помощников**. Такой специалист следит за качеством ответов ИИ, анализирует ошибки и обучает модели на реальных кейсах.
               * **Разработчик разговорных интерфейсов**. UX-специалист настраивает диалоги, сценарии взаимодействий и адаптирует их к различным аудиториям.
               * **Медиатор по сложным кейсам**. Такой человек способен находить компромиссы, сглаживать острые моменты и эффективно вести переговоры.
   - 10% если ответ полностью некорректен (спросил про вебинар, ответил про вакансии)
         Пояснение: нету ответа на вопрос, только общая инфа, ожидалось, что будет именно инструкция из документации, некорректная ссылка на None
         Вопрос "как подключить только Direct?"
         Ответ: В Confluence нашлась страница "ВКонтакте API, Direct", на которой есть инструкция по подключению только Direct. В частности, сказано, что подключение только одного из каналов ВКонтакте (API или Direct) отличается некоторыми шагами от предыдущей инструкции. Также есть прямая ссылка для подключения только Direct.
         Вот ссылка на страницу с подробностями: https://deephack.atlassian.net/wiki/spaces/None/pages/1277526073
         Хотите, чтобы я кратко пересказал шаги подключения или показал полный текст инструкции?

         Пояснение: вместо VK Direct ответил про instagram direct
         Ответ2: Для подключения только Instagram* Direct в качестве канала коммуникации с клиентами в системе AutoFAQ, следуйте следующим шагам:
               1. Убедитесь, что у вас есть страница в Facebook*, к которой привязан профессиональный аккаунт Instagram*. Это необходимо для привязки Instagram* к аккаунту Facebook* [[1]](https://deephack.atlassian.net/wiki/spaces/AKB/pages/4423155718/Instagram+Direct).
               2. Зайдите на вашу страницу в Facebook* и откройте настройки вашего аккаунта. В левом меню выберите пункт "Аккаунты" и с помощью кнопки "Добавить аккаунты" добавьте нужный Instagram*, следуя подсказкам на экране [[1]](https://deephack.atlassian.net/wiki/spaces/AKB/pages/4423155718/Instagram+Direct).

**Установка полноты**
   - 100% если ответ содержит всю возможную информацию для понимания ответа
      Вопрос "что нужно сделать, чтобы подключить только один VK канал через API?"
      Ответ: Чтобы подключить только один VK канал через API, выполните следующие шаги:
         1. В настройках группы ВК в разделе "Управление -> Работа с API -> Callback API" выберите версию API 5.103 и укажите адрес:
         - Для подключения только API (комментарии): https://chat.autofaq.ai/vk-adapter/api/vk/webhooks?api=&code=<строка_которую_должен_вернуть_сервер>
         - Для подключения только Direct (личные сообщения): https://chat.autofaq.ai/vk-adapter/api/vk/webhooks?direct=&code=<строка_которую_должен_вернуть_сервер>
         где и — ID каналов из AutoFAQ, а <строка_которую_должен_вернуть_сервер> — строка из раздела группы ВК "Управление" -> "Работа с API" -> "Callback API".
         2. Выберите типы событий в зависимости от типа подключаемого канала:
         - Для API (комментарии): "Комментарии на стене: Добавление, Редактирование, Удаление, Восстановление"
         - Для Direct (личные сообщения): "Сообщения: Входящее сообщение, Исходящее сообщение, Редактирование сообщения"
   - 70% если в ответе есть большая часть важной информации для понимания ответа
      Пример:
         Пояснение: нету информации про Onpremise
         Вопрос "Чем отличаются тарифы Enterprise+, Enterprise и Pro?"
         Ответ: Тарифы Enterprise+, Enterprise и Pro отличаются следующим образом:
                Pro: 3 890 000 ₽ / год, до 40 тыс диалогов / мес, API, рассылки 200 000 / мес.
                Enterprise: 6 590 000 ₽ / год, без ограничения по диалогам (до 100 тыс / мес), API, рассылки 500 000 / мес.
                Enterprise+: Индивидуально, без ограничений по диалогам и рассылкам, API.
   - 40% если ответ содержит только маленькую часть важной информации по которой можно понять ответ
    Пояснение: перечислены не все открытые вакансии
      Пример:
         Вопрос "Есть ли открытые вакансии?"
         Ответ: Да, на сайте AutoFAQ открыты вакансии. Они ищут специалистов, готовых работать в IT-компании над сложными проектами. В частности, сейчас открыта вакансия инженера технической поддержки с зарплатой от 150,000 до 220,000 рублей в зависимости от опыта и навыков. Требования включают опыт работы от 3 лет, знание Linux, Docker, Bash, JavaScript и другие технические навыки. Их команда распределенная, поэтому возможна удаленная работа. Подробнее можно ознакомиться на странице вакансий AutoFAQ. Есть ли еще что-то, о чём вы хотите узнать?
   - 10% если в ответе практически отсутствует обрамляющая информация для ответа
      Пояснение: нету ссылок, нету вебинаров
      Пример:
         Вопрос "Проводятся ли вебинары?"
         Ответ: Да, проводятся вебинары.

На вход тебе поступит вопрос, ответ, ответ от человека.
Ответ от человека необязательно самый полный и точный, но в целом дает неплохое представление о том как примерно
должен выглядеть правильный ответ.
Нужно по входу оценить точность и полноту ответа от модели по инструкции сверху и вернуть ее в виде json, precision и recall должны быть в диапазоне 10-100:
```
{
  "precision": xx,
  "recall": yy
}
```
"""


# Template for user content
PROMPT_TEMPLATE = (
    "Вопрос: {question}\n"
    "Ответ от человека: {human_answer}\n"
    "Ответ модели: {model_answer}\n"
)


model = "o4-mini"
temperature = 1.0  # o4-mini works only with default 1.0 temperature
if model == 'gpt-4.1':
    temperature = 0


def evaluate_row(question, model_answer, human_answer):
    """Send a single evaluation request to OpenAI API and return precision, recall."""
    content = PROMPT_TEMPLATE.format(
        question=question,
        model_answer=model_answer,
        human_answer=human_answer,
    )
    # print(f'Rules: {RULES}; Content: {content}')
    if not model_answer:
        print("No model answer provided.")
        return None, None
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RULES},
                {"role": "user", "content": content},
            ],
            temperature=temperature,
            response_format={"type": "json_schema", "json_schema": {'name': 'precision_recall',
                                                                    'schema': schema_precision_recall}},
        )
        text = response.choices[0].message.content.strip()
        print(text)
        data = json.loads(text)
        return data.get("precision"), data.get("recall")
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate precision and recall of model answers against human answers via OpenAI.")
    parser.add_argument("input_excel", help="Path to input Excel file.")
    parser.add_argument("output_excel", help="Path to output Excel file.")
    args = parser.parse_args()

    df = pd.read_excel(args.input_excel)
    human_col = "Ответ Support Team"
    question_col = "Вопрос"
    # Identify model answer columns
    model_cols = [c for c in df.columns if c.startswith("Ответ ") and c not in {human_col}]

    # Prepare new columns
    for col in model_cols:
        df[f"Точность {col}"] = None
        df[f"Полнота {col}"] = None

    # Iterate and evaluate
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        question = row.get(question_col, "")
        human_answer = row.get(human_col, "")
        for col in model_cols:
            model_answer = row.get(col, "")
            if not isinstance(model_answer, str):
                print(f"Invalid model answer for {col}: {model_answer}")
                model_answer = ""
            model_answer = model_answer.strip()
            precision, recall = evaluate_row(question, model_answer, human_answer)
            print(col, precision, recall)
            df.at[idx, f"Точность {col}"] = precision
            df.at[idx, f"Полнота {col}"] = recall

    # Save results to new Excel
    df.to_excel(args.output_excel, index=False)
    print(f"Results saved to {args.output_excel}")


if __name__ == "__main__":
    main()

# flake8: noqa
# Don't lint this file to avoid line length and unresolved import errors
