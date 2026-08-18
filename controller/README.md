# BuAli Controller

سرویس <span dir="ltr">HTTP</span> سبک و تک‌منظوره — تبدیل گزارش رادیولوژیِ گفتاری به متن اصلاح‌شده، در سه پایپ‌لاین. این پروژه از <span dir="ltr">Spin_Medical_Assistant_Project</span> جدا شده: به‌جای موتور عمومیِ <span dir="ltr">Orchestrator</span> که یک <span dir="ltr">JSON</span> instruction را تفسیر می‌کند، منطق بوعلی مستقیم در پایتون نوشته شده (`pipelines.py`) — همان رفتار و همان system prompt ها، بدون لایه‌ی انتزاعیِ عمومی.

<span dir="ltr">STT</span> و <span dir="ltr">Core_LLM</span> اینجا کپیِ محلی‌اند (`../stt/`، `../core_llm/` — بخشی از همین ریپو، نه وابسته به پروژه‌ی اصلی)؛ این کنترلر فقط یک کلاینتِ <span dir="ltr">HTTP</span>ِ سبک برای آن‌هاست (`stt_llm_client.py`) — دقیقاً همان الگویی که <span dir="ltr">Orchestrator</span> در پروژه‌ی اصلی استفاده می‌کند.

## پایپ‌لاین‌ها
| پایپ‌لاین | رفتار |
|---|---|
| `separate` (پیش‌فرض) | تا ۳ موتور <span dir="ltr">STT</span> مستقل (هرکدام محلی یا ابری) صوت را رونویسی می‌کنند؛ یک <span dir="ltr">LLM</span> رونویسی‌های موجود را با هم تطبیق می‌دهد. |
| `multimodal` | <span dir="ltr">STT</span> کاملاً حذف می‌شود؛ صوت مستقیم به یک <span dir="ltr">LLM</span> صوت‌پذیر داده می‌شود (محلی یا ابری). |
| `hybrid` | هر دو همزمان: اسلات(های) <span dir="ltr">STT</span> اجرا می‌شوند **و** <span dir="ltr">LLM</span> خودش صوت را می‌شنود؛ رونویسی‌ها به‌عنوان مرجع (نه منبع اصلی حقیقت) به <span dir="ltr">LLM</span> داده می‌شوند. |

انتخاب مدل محلی/ابری با همان قرارداد پیشوند پروژه‌ی اصلی است: بدون پیشوند → محلی (<span dir="ltr">Core_LLM</span>)؛ `openai:<model>` → یک <span dir="ltr">API</span> سازگار با <span dir="ltr">OpenAI</span>؛ `gemini:<model>` → مستقیماً <span dir="ltr">API</span> بومیِ <span dir="ltr">Gemini</span> (فقط برای مدل‌های صوت‌پذیر — پایپ‌لاین‌های `multimodal`/`hybrid`).

## اجرا
```bash
pip install -r requirements.txt
python main.py          # یا: run.bat (Windows) / ./run.sh (Linux)
```
روی `0.0.0.0:9002` بالا می‌آید (مستندات تعاملی در `/docs`). پیش‌فرض این است که `../stt/` و `../core_llm/` روی همان سرور، پورت‌های `8000`/`8001` در حال اجرا باشند (آدرس‌ها در `.env` قابل تغییرند).

## API
| متد و مسیر | کاربرد |
|---|---|
| `GET /` | سلامت سرویس + وضعیت <span dir="ltr">STT</span>/<span dir="ltr">LLM</span> |
| `GET /models` | پروکسیِ مدل‌های محلیِ <span dir="ltr">STT</span> |
| `GET /languages` | پروکسیِ زبان‌های پشتیبانی‌شده |
| `GET /status` | جلسه‌ی فعال فعلی |
| `POST /session` | بدنه: `{llm_model, pipeline, language?, stt_slots?, stt_api_key?, stt_base_url?, llm_api_key?, llm_base_url?}` |
| `POST /run` | multipart: `file` (صوت، الزامی) + بازنویسی‌های اختیاری اعتبارنامه/`stt_slots_json` |
| `POST /session/unload` | آزادسازی مدل‌ها، پایان جلسه |

## نمونه
```bash
curl -X POST http://localhost:9002/session -H "Content-Type: application/json" \
  -d '{"pipeline": "separate", "llm_model": "aya-expanse-8b",
       "stt_slots": [{"model": "whisper-large"}, {"model": "openai:whisper-1"}]}'

curl -X POST http://localhost:9002/run -F "file=@report.wav"
```
