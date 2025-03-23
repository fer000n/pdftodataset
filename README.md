# pdftodataset
تولید دیتاست فارسی از فایل‌های PDF

معرفی

این پروژه ابزاری برای استخراج متن از فایل‌های PDF و تبدیل آن به یک دیتاست آموزشی فارسی برای مدل‌های زبانی است. این ابزار با استفاده از مدل‌های زبانی از طریق Ollama، داده‌های استخراج‌شده را پردازش کرده و نمونه‌های متنوعی برای آموزش مدل‌های یادگیری ماشینی تولید می‌کند.

ویژگی‌ها

استخراج متن از فایل‌های PDF

پردازش متن و ایجاد نمونه‌های آموزشی در قالب JSON

امکان اتصال به مدل‌های زبانی از طریق Ollama

فیلتر و اصلاح داده‌های استخراج‌شده

پیش‌نیازها

قبل از استفاده از این ابزار، اطمینان حاصل کنید که موارد زیر نصب شده‌اند:

Python 3.8 یا بالاتر

کتابخانه‌های مورد نیاز که در requirements.txt مشخص شده‌اند

سرویس Ollama در حال اجرا باشد و مدل مناسب نصب شده باشد

نصب

ابتدا مخزن را کلون کنید:

git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME

سپس کتابخانه‌های مورد نیاز را نصب کنید:

pip install -r requirements.txt

نحوه استفاده

برای اجرای ابزار و استخراج دیتاست از یک فایل PDF، دستور زیر را اجرا کنید:

python dataset.py مسیر/به/فایل.pdf --output خروجی.json --model نام_مدل --host آدرس_سرور

مثال

python dataset.py example.pdf --output dataset.json --model aya-expanse-8b-IQ2_M --host http://localhost:11434

تنظیمات اختیاری

--output: مسیر فایل خروجی JSON (پیش‌فرض: dataset.json)

--model: نام مدل مورد استفاده در Ollama (پیش‌فرض: aya-expanse-8b-IQ2_M)

--host: آدرس سرور Ollama (پیش‌فرض: http://localhost:11434)

مشکلات و پیشنهادات

در صورت بروز مشکل یا داشتن پیشنهاد، لطفاً یک Issue در مخزن GitHub ثبت کنید.

لایسنس

این پروژه تحت مجوز MIT منتشر شده است.


