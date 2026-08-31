# BuAli Demo App

کلاینت دسکتاپیِ <span dir="ltr">Tkinter</span> برای کنترلر <span dir="ltr">BuAli</span> — یک پنجره‌ی قابل اسکرول: انتخاب پایپ‌لاین (<span dir="ltr">Separate</span>/<span dir="ltr">Multimodal</span>/<span dir="ltr">Hybrid</span>)، تا ۳ اسلات <span dir="ltr">STT</span> مستقل، انتخاب مدل <span dir="ltr">LLM</span> (محلی یا ابری)، ورودی صوت (فایل یا ضبط میکروفون)، اجرا، خروجی، و ذخیره‌ی متن نهایی به‌صورت <span dir="ltr">Word (.docx)</span>.

## ساختار فایل‌ها
| فایل | مسئولیت |
|---|---|
| `app.py` | پنجره‌ی اصلی و اتصال اجزا (نقطه‌ی شروع) |
| `api.py` | تمام تماس‌های <span dir="ltr">HTTP</span> با کنترلر |
| `widgets.py` | ویجت‌های قابل استفاده‌ی مجدد (بدون <span dir="ltr">HTTP</span> و بدون thread) |
| `audio.py` | ضبط میکروفون |
| `export.py` | خروجی <span dir="ltr">Word</span> و تشخیص راست‌به‌چپ |
| `config.py` | ثابت‌ها |

## اجرا
```bash
pip install -r requirements.txt
python app.py          # یا: run.bat (Windows) / ./run.sh (Linux)
```
پیش از اجرا، کنترلر <span dir="ltr">BuAli</span> باید روی `localhost:9002` (یا آدرس دلخواه، در کادر Host/Port) در حال اجرا باشد.

فهرست مدل‌های محلی (هم <span dir="ltr">STT</span> و هم <span dir="ltr">LLM</span>) از خود کنترلر گرفته می‌شود، نه از یک لیست ثابت در کد — پس با تغییر رجیستریِ سرویس‌ها، این برنامه خود‌به‌خود به‌روز می‌ماند. اگر کنترلر در دسترس نباشد، فهرست‌ها خالی می‌مانند؛ دکمه‌ی `Refresh` را بعد از برقراری اتصال بزنید.

ضبط میکروفون به <span dir="ltr">PortAudio</span> نیاز دارد؛ اگر نصب نباشد بقیه‌ی برنامه (اجرا از روی فایل) بدون مشکل کار می‌کند.

## استفاده
۱. آدرس کنترلر را بررسی کنید (`Check connection`).
۲. پایپ‌لاین را انتخاب کنید:
   - **Separate**: تا ۳ اسلات <span dir="ltr">STT</span> را فعال و پیکربندی کنید (هرکدام محلی یا ابری).
   - **Multimodal**: فقط مدل <span dir="ltr">LLM</span> صوت‌پذیر را انتخاب کنید؛ اسلات‌های <span dir="ltr">STT</span> پنهان می‌شوند.
   - **Hybrid**: هم اسلات(های) <span dir="ltr">STT</span> و هم مدل <span dir="ltr">LLM</span> صوت‌پذیر را پیکربندی کنید.
۳. منبع <span dir="ltr">LLM</span> را انتخاب کنید (محلی یا ابری؛ برای صوت ابری با <span dir="ltr">Gemini</span>، مدل را با پیشوند <span dir="ltr">`gemini:`</span> بنویسید).
۴. `Start session` را بزنید.
۵. فایل صوتی را انتخاب یا ضبط کنید.
۶. `Run` را بزنید؛ نتیجه (شامل `raw_transcript`/`corrected_transcript`/`final_text`/`discrepancies_found`/`notes`) در کادر خروجی نمایش داده می‌شود.
۷. در صورت نیاز، با `Save Transcript (.docx)` متن نهایی را در یک فایل <span dir="ltr">Word</span> ذخیره کنید (راست‌به‌چپ خودکار برای متن فارسی).

## تست
```bash
pip install pytest
python -m pytest tests/
```
