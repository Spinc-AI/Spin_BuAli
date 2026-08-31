# BuAli Controller

سرویس <span dir="ltr">HTTP</span> سبک و تک‌منظوره — تبدیل گزارش رادیولوژیِ گفتاری به متن اصلاح‌شده، در سه پایپ‌لاین. منطق بوعلی مستقیم در پایتون نوشته شده (`pipelines.py`)، بدون لایه‌ی انتزاعیِ عمومی.

<span dir="ltr">STT</span> و <span dir="ltr">Core_LLM</span> بخشی از همین ریپو هستند (`../stt/`، `../core_llm/`)؛ این کنترلر فقط یک کلاینتِ <span dir="ltr">HTTP</span> برای آن‌هاست.

## ساختار فایل‌ها
| فایل | مسئولیت |
|---|---|
| `main.py` | مسیرهای <span dir="ltr">HTTP</span>، اعتبارسنجی، و جلسه‌ی فعال |
| `pipelines.py` | سه پایپ‌لاین و ترتیب اجرای آن‌ها |
| `prompts.py` | system prompt ها و قالب <span dir="ltr">JSON</span> خروجی |
| `providers.py` | مسیریابیِ مدل (محلی/<span dir="ltr">OpenAI</span>/<span dir="ltr">Gemini</span>)، اعتبارنامه‌ها، فرمت‌های صوتیِ مجاز |
| `stt_client.py` | کلاینت <span dir="ltr">HTTP</span> سرویس <span dir="ltr">STT</span> |
| `llm_client.py` | کلاینت <span dir="ltr">HTTP</span> مدل زبانی (هر سه ارائه‌دهنده) |
| `evaluation_client.py` | کلاینت <span dir="ltr">HTTP</span> سرویس ارزیابی |
| `schemas.py` | شکل درخواست/پاسخ‌ها |

## پایپ‌لاین‌ها
| پایپ‌لاین | رفتار |
|---|---|
| `separate` (پیش‌فرض) | تا ۳ موتور <span dir="ltr">STT</span> مستقل صوت را رونویسی می‌کنند؛ یک <span dir="ltr">LLM</span> رونویسی‌ها را با هم تطبیق می‌دهد. |
| `multimodal` | <span dir="ltr">STT</span> حذف می‌شود؛ صوت مستقیم به یک <span dir="ltr">LLM</span> صوت‌پذیر داده می‌شود. |
| `hybrid` | هر دو همزمان: اسلات(های) <span dir="ltr">STT</span> اجرا می‌شوند **و** <span dir="ltr">LLM</span> خودش صوت را می‌شنود؛ رونویسی‌ها به‌عنوان مرجع (نه منبع اصلی حقیقت) داده می‌شوند. |

`multimodal` در عمل همان `hybrid` بدون اسلات <span dir="ltr">STT</span> است و هر دو یک مسیر کد مشترک دارند.

## انتخاب مدل
با پیشوند مدل مشخص می‌شود (`providers.py`): بدون پیشوند → محلی؛ `openai:<model>` → یک <span dir="ltr">API</span> سازگار با <span dir="ltr">OpenAI</span>؛ `gemini:<model>` → <span dir="ltr">API</span> بومیِ <span dir="ltr">Gemini</span>. هر سه ارائه‌دهنده در هر سه پایپ‌لاین کار می‌کنند (برای `multimodal`/`hybrid` مدل باید صوت‌پذیر باشد).

## اجرا
```bash
pip install -r requirements.txt
python main.py          # یا: run.bat (Windows) / ./run.sh (Linux)
```
روی `0.0.0.0:9002` بالا می‌آید (مستندات تعاملی در `/docs`).

## API
| متد و مسیر | کاربرد |
|---|---|
| `GET /` | سلامت سرویس + وضعیت <span dir="ltr">STT</span>/<span dir="ltr">LLM</span> |
| `GET /models` | پروکسیِ مدل‌های محلیِ <span dir="ltr">STT</span> |
| `GET /llm/models` | پروکسیِ مدل‌های محلیِ <span dir="ltr">LLM</span> + زیرمجموعه‌ی صوت‌پذیر |
| `GET /languages` | پروکسیِ زبان‌های پشتیبانی‌شده |
| `GET /status` | جلسه‌ی فعال فعلی (کلیدهای <span dir="ltr">API</span> هرگز برگردانده نمی‌شوند) |
| `POST /session` | بدنه: `{llm_model, pipeline?, language?, stt_slots?, llm_api_key?, llm_base_url?}` |
| `POST /run` | multipart: `file` (صوت، الزامی) + بازنویسی‌های اختیاری: `language`, `llm_api_key`, `llm_base_url`, `stt_slots_json` |
| `POST /session/unload` | آزادسازی مدل‌ها، پایان جلسه |
| `POST /evaluate` | امتیازدهی یک رونویسی در برابر مرجع — پاس‌ترو به سرویس <span dir="ltr">evaluation</span> |

کنترلر تنها درِ ورودیِ سیستم است؛ ماژول‌های `stt/`، `core_llm/` و `evaluation/` مستقیماً صدا زده نمی‌شوند. `POST /evaluate` بدنه را بدون تفسیر به سرویس <span dir="ltr">evaluation</span> می‌فرستد و پاسخ را همان‌طور برمی‌گرداند — قرارداد معیارها متعلق به همان ماژولی می‌ماند که پیاده‌اش کرده.

اعتبارنامه‌ی <span dir="ltr">STT</span> در خودِ هر اسلات تعریف می‌شود (`stt_slots[].api_key` / `.base_url`)، نه در سطح جلسه.

## تست
```bash
pip install pytest
python -m pytest tests/
```
تست‌ها سرویس‌های <span dir="ltr">STT</span>/<span dir="ltr">LLM</span> را شبیه‌سازی می‌کنند — هیچ مدلی بارگذاری نمی‌شود و هیچ درخواست شبکه‌ای ارسال نمی‌شود.

## نمونه
```bash
curl -X POST http://localhost:9002/session -H "Content-Type: application/json" \
  -d '{"pipeline": "separate", "llm_model": "aya-expanse-8b",
       "stt_slots": [{"model": "whisper"}, {"model": "openai:whisper-1"}]}'

curl -X POST http://localhost:9002/run -F "file=@report.wav"
```
