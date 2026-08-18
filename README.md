# Spin BuAli

پروژه‌ای کاملاً مستقل و سبک برای تبدیل گزارش رادیولوژیِ گفتاری به متن اصلاح‌شده — جداشده از <span dir="ltr">Spin_Medical_Assistant_Project</span> و بدون هیچ وابستگی‌ای به آن. به‌جای موتور عمومیِ <span dir="ltr">Orchestrator</span> (که یک instruction عمومیِ <span dir="ltr">JSON</span> را تفسیر می‌کند)، این پروژه یک کنترلر ساده و مستقیم دارد — همان سه پایپ‌لاین (<span dir="ltr">Separate</span>/<span dir="ltr">Multimodal</span>/<span dir="ltr">Hybrid</span>) و همان system prompt ها، بدون لایه‌ی انتزاعی.

<span dir="ltr">STT</span> و <span dir="ltr">Core_LLM</span> (تشخیص گفتار و مدل زبانی) اینجا کپی و بخشی از همین پروژه‌اند (`stt/`، `core_llm/`) — نه یک وابستگی به پروژه‌ی اصلی. کنترلر همچنان فقط از طریق <span dir="ltr">HTTP</span> با آن‌ها صحبت می‌کند (همان معماری قبلی)، اما هر چهار سرویس داخل همین ریپو زندگی می‌کنند و کل پروژه با یک `git clone` قابل اجراست.

## ساختار
- `stt/` — سرویس <span dir="ltr">FastAPI</span> تشخیص گفتار (پیش‌فرض پورت `8000`). جزئیات در [stt/README.md](stt/README.md).
- `core_llm/` — سرویس <span dir="ltr">FastAPI</span> مدل زبانی، شامل مسیر صوت‌پذیر برای پایپ‌لاین‌های <span dir="ltr">Multimodal</span>/<span dir="ltr">Hybrid</span> (پیش‌فرض پورت `8001`). جزئیات در [core_llm/README.md](core_llm/README.md).
- `controller/` — سرویس <span dir="ltr">FastAPI</span>ِ بوعلی (پیش‌فرض پورت `9002`). جزئیات کامل در [controller/README.md](controller/README.md).
- `demo_app/` — کلاینت دسکتاپیِ <span dir="ltr">Tkinter</span> برای تست دستی. جزئیات در [demo_app/README.md](demo_app/README.md).

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
همه‌چیز روی `localhost` و پورت‌های پیش‌فرض بالا می‌آید — نیازی به هیچ سرویس یا ریپوی دیگری نیست. پایپ‌لاین‌های <span dir="ltr">Multimodal</span>/<span dir="ltr">Hybrid</span> به یک مدل صوت‌پذیر (`gemma-4-e4b`/`gemma-4-12b`/`qwen3-omni-30b`) در `core_llm/` نیاز دارند که VRAM قابل توجهی می‌طلبد — یا به‌جایش از یک مدل ابری (`openai:`/`gemini:`) در `controller` استفاده کنید. اولین اجرای هر مدل، وزن‌هایش را از Hugging Face دانلود می‌کند (کش می‌شود، فقط بار اول کند است).
