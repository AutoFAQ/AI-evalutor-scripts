# AI-Evaluator

## Данные для тестирования:

- Docx - ВКонтакте+API,+Direct-docx.docx (в репозитории)
- Pdf - AKB-ВКонтакте API, Direct-180725-145454.pdf (в репозитории)
- Confluence: https://deephack.atlassian.net/wiki/spaces/AKB (открытый)
- Website: https://autofaq.ai/ (открытый)

Вопросы и ответы по каждому типу данных находятся в questions_and_answers.xlsx.

## Инструкция по установке и запуску

1. `pip install -r requirements.txt`
2. Нужно получить ответы LLM и ответы человека и положить их input.xlsx. В input.xlsx есть уже готовые ответы.
В поле Ответ Support Team - ответ человека. Далее нужно оценить Точность и Полноту людьми согласно RULES из evaluate_precision_recall.py.
3. Для оценки точности и полноты с LLM нужно запустить: `python evaluate_precision_recall.py`. В коде можно поменять `model = "o4-mini"` на gpt-4.1. Также нужно выполнить `export OPENAI_API_KEY`.
4. После выполнения evaluate_precision_recall.py будет файл output.xlsx. В нем будут оценки от LLM. Эти оценки нужно скопировать в input.xlsx, после ответа каждой модели в конце после оценок людьми. См. пример в Сравнение ответов разных AI систем -v2.xlsx.
5. После получения итогового файла с оценками нужно запустить скрипты для подсчета статистик:
- `python analysis_code.py` - в папке analysis_images появятся графики и .csv файлы со статистиками
- `python inter_rater_analysis.py` - подсчет Cronbach's alpha
```
     system  alpha_accuracy  alpha_recall  human_acc_corr  human_acc_p  human_rec_corr  human_rec_p  mean_abs_diff_accuracy  mean_abs_diff_recall
      Witsy        0.843318      0.853178        0.429450     0.110150        0.668667     0.006422                    20.0                  14.0
     Xplain        0.748342      0.779026        0.619586     0.003573        0.460092     0.041230                    16.5                  22.5
     Yandex        0.825635      0.777502        0.421848     0.117296        0.021148     0.940368                    24.0                  14.0
AnythingLLM        0.614379      0.874797        0.109632     0.697318        0.490589     0.063348                    18.0                  14.0
       Onyx        0.837665      0.823884        0.581580     0.022957        0.778569     0.000627                    20.0                  12.0
```

- `python rater_metrics_summary.py` - подсчет точности и полноты по системе и оценщику
```
     system     rater mean_accuracy mean_recall harmonic_mean_f1 arithmetic_mean
      Witsy Человек 1         68.00       64.00            65.94           66.00
      Witsy Человек 2         72.00       62.00            66.63           67.00
      Witsy   gpt-4.1         78.00       78.00            78.00           78.00
      Witsy   o4-mini         66.00       60.00            62.86           63.00
     Xplain Человек 1         97.00       95.50            96.24           96.25
     Xplain Человек 2         80.50       73.00            76.57           76.75
     Xplain   gpt-4.1         97.00       97.00            97.00           97.00
     Xplain   o4-mini         88.00       91.00            89.47           89.50
     Yandex Человек 1         64.00       56.00            59.73           60.00
     Yandex Человек 2         84.00       66.00            73.92           75.00
     Yandex   gpt-4.1         92.00       90.00            90.99           91.00
     Yandex   o4-mini         81.20       82.00            81.60           81.60
AnythingLLM Человек 1         78.00       56.00            65.19           67.00
AnythingLLM Человек 2         80.00       54.00            64.48           67.00
AnythingLLM   gpt-4.1         86.00       84.00            84.99           85.00
AnythingLLM   o4-mini         90.00       76.00            82.41           83.00
       Onyx Человек 1         52.00       42.00            46.47           47.00
       Onyx Человек 2         64.00       50.00            56.14           57.00
       Onyx   gpt-4.1         70.00       68.00            68.99           69.00
       Onyx   o4-mini         66.53       59.80            62.99           63.17
```

## Прочее

- Скриншоты с тестирования находятся в папке screenshots.
