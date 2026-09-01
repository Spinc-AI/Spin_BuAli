# Audio Preprocessing

پیش‌پردازش صوت پیش از <span dir="ltr">STT</span>: هر فایل ورودی اعتبارسنجی می‌شود، یک نسخه‌ی مشتق‌شده‌ی استاندارد (<span dir="ltr">WAV / PCM 16-bit / Mono / 16 kHz</span>) از آن ساخته می‌شود، مرزهای گفتار با <span dir="ltr">VAD</span> علامت‌گذاری می‌شوند و صوت به قطعات هم‌پوشان تقسیم می‌شود — همراه با <span dir="ltr">Timestamp</span> کامل.

**فایل اصلی هرگز نوشته یا بازنویسی نمی‌شود؛ فقط خوانده و <span dir="ltr">hash</span> می‌شود.** خروجی <span dir="ltr">VAD</span> جایگزین چیزی نمی‌شود، سکوت حذف نمی‌شود، نویز‌گیری انجام نمی‌شود و <span dir="ltr">Gain</span> دست نمی‌خورد. هر تغییری که روی نسخه‌ی مشتق‌شده اعمال می‌شود (روش <span dir="ltr">resample</span>، تعداد سمپل‌های <span dir="ltr">clip</span>‌شده) در `segments.json` ثبت می‌شود.

## اجرا
```bash
pip install -r requirements.txt
python audio_preprocessing.py recordings/ -o output
```

| سوییچ | کار |
|---|---|
| `-o, --output-dir` | ریشه‌ی خروجی (پیش‌فرض از <span dir="ltr">config</span>) |
| `-c, --config` | مسیر `preprocessing_config.json` |
| `-m, --metadata` | <span dir="ltr">JSON</span>؛ یا یک رکورد مسطح برای همه، یا کلیدخورده به نام فایل |
| `--only` | فقط این مرحله‌ها اجرا شوند (با کاما) |
| `--skip` | همه‌ی مرحله‌ها جز این‌ها |
| `--strategy` | بازنویسی `chunking.strategy` فقط برای همین اجرا |
| `--list-stages` | چاپ فهرست مرحله‌ها و خروج |
| `--report` | مسیر `preprocessing_report.csv` |
| `--no-append` | بازنویسی گزارش به‌جای افزودن |
| `--stop-on-error` | توقف کل دسته در اولین خطا (پیش‌فرض: خطا فقط در گزارش ثبت می‌شود) |

ورودی می‌تواند فایل، چند فایل یا پوشه باشد؛ پوشه بازگشتی پیمایش می‌شود.

## شش مرحله، شش فایل
هر مرحله یک ماژول مستقل است و جداگانه هم قابل اجراست؛ [`audio_preprocessing.py`](audio_preprocessing.py) فقط آن‌ها را کنار هم می‌گذارد.

| # | فایل | کار |
|---|---|---|
| ۱ | [`validation.py`](validation.py) | قابل <span dir="ltr">decode</span> بودن، خالی نبودن، مدت معتبر |
| ۲ | [`standardization.py`](standardization.py) | <span dir="ltr">WAV / PCM 16-bit / Mono / 16 kHz</span> |
| ۳ | [`vad.py`](vad.py) | مرزهای گفتار با <span dir="ltr">padding</span> — آزمایشی، بدون هیچ برشی |
| ۴ | [`chunking.py`](chunking.py) | قطعه‌بندی هم‌پوشان، تطبیقی با خودِ فایل |
| ۵ | [`timestamps.py`](timestamps.py) | ساخت `segments.json` و بررسی‌های پذیرش |
| ۶ | [`report.py`](report.py) | `preprocessing_report.csv` |

[`common.py`](common.py) مرحله نیست؛ چیزهای مشترک است (<span dir="ltr">config</span>، خواندن/نوشتن صوت، <span dir="ltr">hash</span>). هر ماژولِ مرحله فقط از `common` <span dir="ltr">import</span> می‌کند و از هیچ مرحله‌ی دیگری — برای همین می‌شود تک‌تکشان را جدا تست یا اجرا کرد:

```bash
python vad.py recording.wav          # فقط گزارش VAD
python chunking.py recording.wav     # فقط نقشه‌ی قطعه‌بندی
```

### انتخاب مرحله‌ها
```bash
python audio_preprocessing.py rec.wav -o out --skip vad
python audio_preprocessing.py rec.wav -o out --only standardization,timestamps
python audio_preprocessing.py --list-stages
```

**رد کردن یک مرحله یعنی نبودنش در <span dir="ltr">metadata</span>، نه بودنِ خالی‌اش.** با `--skip vad` هیچ کلید `vad` در `segments.json` نیست. `stages_run` هم در همان فایل ثبت می‌شود. در <span dir="ltr">CSV</span> برعکس: ستون‌ها ثابت می‌مانند و ستون‌های مرحله‌ی اجرانشده خالی‌اند، تا گزارش بین اجراها قابل مقایسه بماند.

`validation` قابل رد کردن نیست (بدون آن صوتی وجود ندارد). وابستگی‌ها بررسی می‌شوند و در سکوت برآورده **نمی‌شوند**: `--only chunking` خطا می‌دهد، نه اینکه خودش `standardization` را هم اجرا کند.

## ساختار خروجی
نام پوشه‌ی هر فایل، ۱۶ رقم اول <span dir="ltr">SHA-256</span>ِ خودِ فایل است — یعنی پردازش دوباره‌ی یک فایل همان پوشه را می‌دهد و فایل تکراری دو بار پردازش نمی‌شود.

```
output/
├── preprocessing_report.csv
└── 8af1c23d19ab70e4/
    ├── standardized.wav
    ├── chunks/
    │   ├── chunk_000.wav
    │   └── chunk_001.wav
    └── segments.json
```

`segments.json` شامل: مشخصات فایل اصلی (<span dir="ltr">hash</span>، فرمت، نرخ نمونه‌برداری، کانال، مدت، <span dir="ltr">peak/rms</span>)، مشخصات نسخه‌ی استاندارد، <span dir="ltr">metadata</span>ی ورودی، گزارش <span dir="ltr">VAD</span>، <span dir="ltr">Timestamp</span> همه‌ی <span dir="ltr">chunk</span>ها (ثانیه و شماره‌ی سمپل، به‌همراه هم‌پوشانی با قطعه‌ی قبل و بعد)، و نتیجه‌ی بررسی‌های پذیرش.

## جزئیات مرحله‌ها

### استانداردسازی
<span dir="ltr">torchaudio</span> مسیر اصلی است، چون دقیقاً همان چیزی است که سرویس <span dir="ltr">STT</span> روی همین صوت اجرا می‌کند؛ اگر نصب نباشد به‌ترتیب <span dir="ltr">scipy</span> و سپس یک <span dir="ltr">resampler</span>ِ فرکانسی با <span dir="ltr">numpy</span> جایگزین می‌شوند. روشی که واقعاً استفاده شده در `segments.json` و گزارش <span dir="ltr">CSV</span> ثبت می‌شود.

### <span dir="ltr">VAD</span>
عمداً **آزمایشی** است: نتیجه‌اش فقط در `segments.json` ثبت می‌شود تا بتوان بررسی کرد کلمه‌ای از ابتدا یا انتهای عبارت‌ها بریده می‌شود یا نه. `speech_pad_ms` همان <span dir="ltr">padding</span> اطراف گفتار است. اگر وزن‌های <span dir="ltr">Silero</span> در دسترس نباشند، پایپ‌لاین متوقف نمی‌شود؛ به یک <span dir="ltr">VAD</span>ِ انرژی‌محورِ ساده برمی‌گردد و `vad.backend` و `vad.error` این را ثبت می‌کنند. برای اتکا به <span dir="ltr">VAD</span> در تصمیم‌های بعدی، اول `vad.backend` را بررسی کنید.

برای اینکه <span dir="ltr">chunk</span>ها از روی بخش‌های گفتار بریده شوند (نه از روی کل فایل)، `chunking.source` را روی `vad_segments` بگذارید. پیش‌فرض `standardized` است تا هیچ کلمه‌ای قربانی تصمیم <span dir="ltr">VAD</span> نشود.

### <span dir="ltr">Chunking</span>
هر بخشی که از `max_duration_sec` کوتاه‌تر باشد دست‌نخورده یک <span dir="ltr">chunk</span> می‌شود. برای بخش‌های بلندتر سه راهبرد وجود دارد (`chunking.strategy`):

**`adaptive` — پیش‌فرض.** هم *تعداد* قطعه‌ها و هم *محل برش* برای هر فایل جداگانه انتخاب می‌شود:

1. همه‌ی تعدادهایی که قطعه‌هایشان داخل بازه‌ی `min`–`max` می‌افتند فهرست می‌شوند.
2. برای هر تعداد، مرزها از تقسیم مساوی شروع می‌شوند و بعد در بازه‌ی `snap_window_sec` به ساکت‌ترین لحظه‌ی اطرافشان جابه‌جا می‌شوند.
3. تعدادی برنده می‌شود که مرزهایش از کم‌ترین گفتار عبور کرده باشند.

منطق ساده است: برشی که وسط سکوت بیفتد کلمه‌ای را نصف نمی‌کند. یعنی یک گزارشِ با مکث‌های منظمِ ۲۴ ثانیه‌ای و یک گزارشِ پشت‌سرهم با طول یکسان، دو چیدمان متفاوت می‌گیرند.

جابه‌جایی مرزها هرگز نمی‌تواند قطعه‌ی غیرمجاز بسازد: هر مرز علاوه بر `snap_window_sec` به بازه‌ای محدود است که هم قطعه‌ی قبلش و هم همه‌ی قطعه‌های باقی‌مانده داخل `min`–`max` بمانند. هم‌پوشانی در هر حالت دقیقاً `overlap_sec` است.

تصمیم کامل در `segments.json` زیر `chunking.decision` ثبت می‌شود — چه تعدادهایی بررسی شدند، کدام انتخاب شد، امتیاز سکوتِ مرزها چقدر بود و گزینه‌های رد شده چه امتیازی داشتند. اگر خروجی عجیب بود، دلیلش همان‌جاست.

**`uniform`** — همان تقسیم مساوی، بدون گوش دادن به صوت. تعداد فقط از روی مدت فایل.

**`fixed`** — همه‌ی قطعات دقیقاً `target_duration_sec`، قطعه‌ی آخر هرچه باقی مانده.

وقتی `plan_chunks` بدون خودِ صوت صدا زده شود (مثلاً برای برنامه‌ریزی از روی مدت)، `adaptive` به `uniform` برمی‌گردد و `decision.listened_to_audio` این را ثبت می‌کند.

## تنظیمات
همه‌چیز در [`preprocessing_config.json`](preprocessing_config.json) است و در `segments.json` هر فایل هم اسنپ‌شات می‌شود، پس از روی خروجی می‌توان فهمید با چه تنظیماتی ساخته شده.

| بخش | نکته |
|---|---|
| `audio` | هدف نهایی فرمت: نرخ، کانال، `subtype` |
| `validation` | حداقل/حداکثر مدت، آستانه‌ی «سکوت مطلق» |
| `vad` | آستانه، حداقل طول گفتار/سکوت، `speech_pad_ms`، و `fallback` |
| `chunking` | `target/min/max_duration_sec`، `overlap_sec`، `source`، `strategy`، و زیرشاخه‌ی `adaptive` (`snap_window_sec`، `max_extra_chunks`، `frame_ms`، `extra_chunk_penalty`) |
| `output` | نام‌ها، `id_strategy`، `overwrite` |
| `policy` | ثبت صریح اینکه چه کارهایی *انجام نمی‌شود* — در `segments.json` بازتاب داده می‌شود |

## گزارش
`preprocessing_report.csv` برای هر فایل یک ردیف دارد (پیش‌فرض: افزودنی)، شامل مشخصات ورودی و خروجی، روش <span dir="ltr">resample</span>، آمار <span dir="ltr">VAD</span>، تعداد و طول <span dir="ltr">chunk</span>ها، و ستون `status`/`error` برای فایل‌هایی که رد شده‌اند. یک فایل خراب کل دسته را متوقف نمی‌کند.

## تست
```bash
pip install -r requirements-dev.txt
pytest -q                    # همه
pytest tests/test_chunking.py -q   # فقط یک مرحله
```
تست‌ها هم به همان شکل تفکیک شده‌اند — `test_validation.py`، `test_standardization.py`، `test_vad.py`، `test_chunking.py`، `test_timestamps.py`، و `test_pipeline.py` برای خودِ اجراکننده (انتخاب مرحله‌ها، گزارش، <span dir="ltr">CLI</span>). داده‌ی مشترک در `tests/conftest.py` است.

تست‌ها با صوت مصنوعی کار می‌کنند و هیچ وزن مدلی لازم ندارند. هر تست به یکی از معیارهای پذیرش گره خورده است: قالب خروجی، سقف طول <span dir="ltr">chunk</span>، هم‌پوشانی، سازگاری <span dir="ltr">Timestamp</span> با صوتِ نوشته‌شده، دست‌نخورده ماندن فایل اصلی، و <span dir="ltr">padding</span> اطراف گفتار.

یک نکته درباره‌ی <span dir="ltr">VAD</span>: <span dir="ltr">Silero</span> یک مدل واقعیِ گفتار است و روی تُنِ مصنوعی — به‌درستی — چیزی پیدا نمی‌کند. پس تست‌ها *تشخیص* <span dir="ltr">Silero</span> را نمی‌سنجند؛ آنچه سنجیده می‌شود این است که آداپتر بدون خطا اجرا شود و `vad.backend` صادقانه ثبت شود، اینکه `speech_pad_ms` واقعاً به کتابخانه برسد، و اینکه خودِ منطق <span dir="ltr">padding</span> درست کار کند. برای سنجش کیفیت واقعی <span dir="ltr">Silero</span> روی گفتار فارسی، به فایل صوتی واقعی نیاز است.
