# Spin BuAli

پروژه‌ای کاملاً مستقل و سبک برای تبدیل گزارش رادیولوژیِ گفتاری به متن اصلاح‌شده. چهار سرویس در همین ریپو زندگی می‌کنند و از طریق <span dir="ltr">HTTP</span> با هم صحبت می‌کنند — کل پروژه با یک `git clone` قابل اجراست و به هیچ ریپوی دیگری وابسته نیست.

## ساختار
| پوشه | نقش | پورت پیش‌فرض |
|---|---|---|
| [`stt/`](stt/README.md) | سرویس تشخیص گفتار (۱۰ مدل فارسی) | `8000` |
| [`core_llm/`](core_llm/README.md) | سرویس مدل زبانی، شامل مسیر صوت‌پذیر | `8001` |
| [`controller/`](controller/README.md) | مغز بوعلی: سه پایپ‌لاین، system prompt ها، جلسه | `9002` |
| [`demo_app/`](demo_app/README.md) | کلاینت دسکتاپیِ <span dir="ltr">Tkinter</span> برای تست دستی | — |
| [`evaluation/`](evaluation/README.md) | امتیازدهی رونویسی در برابر مرجعِ تأییدشده — معیارهای عمومی و بالینی | `8002` |
| [`docs/`](docs/README.md) | نقشه راه و مستندات مرجع | — |

هر پوشه یک سرویس مستقل و جداگانه‌قابل‌استقرار است: `requirements.txt` خودش، `config.py` خودش، و بدون import مستقیم از بقیه.

## شروع سریع
هر سرویس در ترمینال جداگانه (به همین ترتیب بالا بیایند — کنترلر در `/session` سلامتِ `stt`/`core_llm` را چک می‌کند):
```bash
# ترمینال ۱ — STT
cd stt && pip install -r requirements.txt && python -m app.main

# ترمینال ۲ — Core_LLM
cd core_llm && pip install -r requirements.txt && python main.py

# ترمینال ۳ — کنترلر بوعلی
cd controller && pip install -r requirements.txt && python main.py

# ترمینال ۴ — دمو
cd demo_app && pip install -r requirements.txt && python app.py
```
همه‌چیز روی `localhost` و پورت‌های پیش‌فرض بالا می‌آید. پایپ‌لاین‌های <span dir="ltr">Multimodal</span>/<span dir="ltr">Hybrid</span> به یک مدل صوت‌پذیر (`gemma-4-e4b`/`gemma-4-12b`/`qwen3-omni-30b`) در `core_llm/` نیاز دارند که VRAM قابل توجهی می‌طلبد — یا به‌جایش از یک مدل ابری (`openai:`/`gemini:`) استفاده کنید. اولین اجرای هر مدل، وزن‌هایش را از <span dir="ltr">Hugging Face</span> دانلود می‌کند (کش می‌شود، فقط بار اول کند است).

> برای فقط تست کنترلر با مدل ابری، `stt/` و `core_llm/` لازم نیستند: پایپ‌لاین `multimodal` با یک مدل `gemini:`/`openai:` هیچ سرویس محلی‌ای نمی‌خواهد.

## پایپ‌لاین‌ها
| پایپ‌لاین | صوت به <span dir="ltr">LLM</span> | رونویسی <span dir="ltr">STT</span> |
|---|---|---|
| `separate` | ❌ | ✅ (تا ۳ موتور، سپس تطبیق توسط <span dir="ltr">LLM</span>) |
| `multimodal` | ✅ | ❌ |
| `hybrid` | ✅ | ✅ (به‌عنوان مرجع، نه منبع اصلی حقیقت) |

جزئیات و انتخاب مدل محلی/ابری در [controller/README.md](controller/README.md).

## تست
```bash
cd controller && python -m pytest tests/     # کنترلر (بدون شبکه، بدون مدل)
cd evaluation && python -m pytest tests/     # نرمال‌سازی و معیارهای بالینی
cd demo_app   && python -m pytest tests/     # خروجی Word و نام فایل‌ها
cd stt        && python -m pytest tests/     # نیازمند نصب torch
```
