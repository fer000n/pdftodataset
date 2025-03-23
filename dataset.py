import os
import json
import argparse
from pypdf import PdfReader
from langchain_community.llms import Ollama  # استفاده از Ollama به جای CTransformers
from tqdm import tqdm
import logging

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """استخراج متن از فایل PDF"""
    try:
        reader = PdfReader(pdf_path)
        pages = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():  # اطمینان از خالی نبودن صفحه
                pages.append({"page_num": page_num + 1, "content": text})
        
        logger.info(f"استخراج {len(pages)} صفحه از {pdf_path} با موفقیت انجام شد")
        return pages
    except Exception as e:
        logger.error(f"خطا در استخراج متن از PDF: {str(e)}")
        return []

def create_prompt_for_page(page_content):
    """ایجاد پرامپت برای مدل هوش مصنوعی"""
    prompt = f"""
از متن صفحه زیر، حداقل 10 نمونه با کیفیت بالا برای دیتاست آموزشی فارسی در حوزه رمز ارز استخراج کن.
هر نمونه باید در قالب JSON با ساختار مشخص شده باشد.

متن صفحه:
{page_content}

لطفاً برای هر نمونه، ساختار زیر را رعایت کن:
{{
  "instruction": "[دستورالعمل یا پرسش به زبان فارسی]",
  "input": "[متن ورودی اختیاری – در صورت عدم نیاز، خالی بماند]",
  "output": "[پاسخ کامل و دقیق به دستورالعمل]",
  "category": "رمز ارز"
}}

دستورالعمل‌ها باید متنوع، واقع‌گرایانه و کاربردی باشند و شامل انواع درخواست‌ها مانند پاسخ‌دهی به سؤالات، خلاصه‌سازی، تحلیل، طبقه‌بندی و تولید محتوا باشند.
پاسخ‌ها باید جامع، دقیق و کاملاً از نظر نگارشی و دستوری صحیح باشند.
تمام محتوا باید مرتبط با حوزه رمز ارز باشد و فاقد هرگونه اطلاعات حساس، شخصی، سیاسی یا نامناسب باشد.

فقط نمونه‌های JSON را برگردان، بدون هیچ توضیح اضافی.
"""
    return prompt

def generate_dataset_for_page(llm, page_content):
    """تولید دیتاست برای یک صفحه با استفاده از مدل زبانی"""
    try:
        prompt = create_prompt_for_page(page_content)
        response = llm.invoke(prompt)  # استفاده از invoke به جای فراخوانی مستقیم
        
        # پردازش پاسخ برای استخراج داده‌های JSON
        json_data = extract_json_from_response(response)
        return json_data
    except Exception as e:
        logger.error(f"خطا در تولید دیتاست: {str(e)}")
        return []

def extract_json_from_response(response):
    """استخراج داده‌های JSON از پاسخ مدل"""
    try:
        # جستجوی بلوک‌های JSON در پاسخ
        json_objects = []
        lines = response.split('\n')
        current_json = ""
        capturing = False
        brace_count = 0
        
        for line in lines:
            line = line.strip()
            
            # تشخیص شروع یک آبجکت JSON
            if line.startswith('{') and not capturing:
                capturing = True
                current_json = line
                brace_count = line.count('{') - line.count('}')
            
            # ادامه جمع‌آوری آبجکت JSON
            elif capturing:
                current_json += line
                brace_count += line.count('{') - line.count('}')
                
                # پایان یک آبجکت JSON
                if brace_count == 0:
                    try:
                        # پاکسازی کاراکترهای اضافی که ممکن است باعث خطای پارس شوند
                        clean_json = current_json.strip()
                        # حذف کاما در آخر اگر وجود داشته باشد
                        if clean_json.endswith(','):
                            clean_json = clean_json[:-1]
                            
                        json_obj = json.loads(clean_json)
                        # اعتبارسنجی ساختار JSON
                        if all(key in json_obj for key in ["instruction", "input", "output", "category"]):
                            json_objects.append(json_obj)
                    except json.JSONDecodeError as e:
                        logger.debug(f"خطا در پارس JSON: {str(e)}, متن: {current_json[:50]}...")
                    
                    capturing = False
                    current_json = ""
        
        # اگر تعداد JSON معتبر کمتر از انتظار است، سعی کنیم کل متن را به عنوان یک آرایه JSON پارس کنیم
        if len(json_objects) < 5:
            try:
                # حذف متن‌های اضافی قبل و بعد از براکت‌های آرایه
                response = response.strip()
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    json_array = json.loads(response[start:end])
                    valid_objects = [obj for obj in json_array if isinstance(obj, dict) and 
                                  all(key in obj for key in ["instruction", "input", "output", "category"])]
                    if valid_objects:
                        json_objects = valid_objects
            except json.JSONDecodeError:
                pass
        
        # آخرین تلاش: جستجوی آبجکت‌های JSON با regex
        if len(json_objects) < 5:
            import re
            pattern = r'{\s*"instruction"\s*:\s*"[^"]*"\s*,\s*"input"\s*:\s*"[^"]*"\s*,\s*"output"\s*:\s*"[^"]*"\s*,\s*"category"\s*:\s*"[^"]*"\s*}'
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    json_obj = json.loads(match)
                    if all(key in json_obj for key in ["instruction", "input", "output", "category"]):
                        # بررسی تکراری نبودن
                        if json_obj not in json_objects:
                            json_objects.append(json_obj)
                except json.JSONDecodeError:
                    pass
        
        logger.info(f"استخراج {len(json_objects)} رکورد JSON از پاسخ مدل")
        return json_objects
    except Exception as e:
        logger.error(f"خطا در استخراج JSON: {str(e)}")
        return []

def save_dataset_to_file(dataset, output_path):
    """ذخیره دیتاست در فایل JSON"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"دیتاست با موفقیت در {output_path} ذخیره شد")
    except Exception as e:
        logger.error(f"خطا در ذخیره دیتاست: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='استخراج دیتاست از فایل PDF برای آموزش مدل‌های زبانی فارسی')
    parser.add_argument('pdf_path', help='مسیر فایل PDF ورودی')
    parser.add_argument('--output', default='dataset.json', help='مسیر فایل خروجی JSON (پیش‌فرض: dataset.json)')
    parser.add_argument('--model', default='aya-expanse-8b-IQ2_M', help='نام مدل در Ollama (پیش‌فرض: aya-expanse-8b-IQ2_M.gguf:latest)')
    parser.add_argument('--host', default='http://localhost:11434', help='آدرس سرور Ollama (پیش‌فرض: http://localhost:11434)')
    args = parser.parse_args()
    
    # بررسی وجود فایل PDF
    if not os.path.exists(args.pdf_path):
        logger.error(f"فایل PDF یافت نشد: {args.pdf_path}")
        return
    
    # بارگزاری مدل از Ollama
    logger.info(f"اتصال به Ollama و بارگزاری مدل {args.model}...")
    try:
        llm = Ollama(
            model=args.model,
            base_url=args.host,
            temperature=0.7
        )
    except Exception as e:
        logger.error(f"خطا در اتصال به Ollama: {str(e)}")
        logger.info("لطفاً مطمئن شوید که Ollama در حال اجراست و مدل مورد نظر نصب شده است.")
        logger.info("برای نصب مدل، از دستور زیر استفاده کنید:")
        logger.info(f"ollama pull {args.model}")
        return
    
    # استخراج متن از PDF
    logger.info(f"استخراج متن از {args.pdf_path}...")
    pages = extract_text_from_pdf(args.pdf_path)
    
    if not pages:
        logger.error("هیچ متنی از PDF استخراج نشد")
        return
    
    # تولید دیتاست برای هر صفحه
    all_data = []
    logger.info(f"شروع تولید دیتاست برای {len(pages)} صفحه...")
    
    for page in tqdm(pages):
        logger.info(f"پردازش صفحه {page['page_num']}...")
        page_data = generate_dataset_for_page(llm, page['content'])
        if page_data:
            all_data.extend(page_data)
            logger.info(f"تعداد {len(page_data)} رکورد از صفحه {page['page_num']} استخراج شد")
    
    # ذخیره نتایج
    logger.info(f"تعداد کل رکوردهای استخراج شده: {len(all_data)}")
    save_dataset_to_file(all_data, args.output)

if __name__ == "__main__":
    main()