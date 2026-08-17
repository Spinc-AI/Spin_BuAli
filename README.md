# Spin BuAli

پروژه‌ای مستقل و سبک برای تبدیل گزارش رادیولوژیِ گفتاری به متن اصلاح‌شده، جداشده از <span dir="ltr">Spin_Medical_Assistant_Project</span>. به‌جای موتور عمومیِ <span dir="ltr">Orchestrator</span> (که یک instruction عمومیِ <span dir="ltr">JSON</span> را تفسیر می‌کند)، این پروژه یک کنترلر ساده و مستقیم دارد — همان سه پایپ‌لاین (<span dir="ltr">Separate</span>/<span dir="ltr">Multimodal</span>/<span dir="ltr">Hybrid</span>) و همان system prompt ها، بدون لایه‌ی انتزاعی.

<span dir="ltr">STT</span> و <span dir="ltr">Core_LLM</span> (ماژول‌های اصلیِ تشخیص گفتار و مدل زبانی) همچنان در پروژه‌ی اصلی هستند و باید جداگانه اجرا شوند؛ این پروژه فقط از طریق <span dir="ltr">HTTP</span> با آن‌ها صحبت می‌کند.

## ساختار
- `controller/` — سرویس <span dir="ltr">FastAPI</span> (پیش‌فرض پورت `9002`). جزئیات کامل در [controller/README.md](controller/README.md).
- `demo_app/` — کلاینت دسکتاپیِ <span dir="ltr">Tkinter</span> برای تست دستی. جزئیات در [demo_app/README.md](demo_app/README.md).

## شروع سریع
```bash
# ترمینال ۱ — کنترلر
cd controller && pip install -r requirements.txt && python main.py

# ترمینال ۲ — دمو
cd demo_app && pip install -r requirements.txt && python app.py
```
پیش‌نیاز: سرویس‌های <span dir="ltr">STT</span> (پورت `8000`) و <span dir="ltr">Core_LLM</span> (پورت `8001`) از پروژه‌ی اصلی باید در حال اجرا باشند.
