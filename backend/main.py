"""
FastAPI Backend for Digital Inspector
API для детекции печатей, подписей и QR-кодов на документах
"""

import os
import io
import base64
import asyncio
from pathlib import Path
from typing import List, Optional
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# PDF обработка
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    try:
        import fitz  # PyMuPDF
        PDF_SUPPORT = True
        PDF_LIB = "pymupdf"
    except ImportError:
        PDF_SUPPORT = False
        PDF_LIB = None

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки для управления нагрузкой (определяем до использования)
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))  # Максимум параллельных запросов
REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    try:
        load_model()
        logger.info("Backend started successfully")
        logger.info(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
        logger.info(f"Device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    # Shutdown
    logger.info("Shutting down backend...")
    # Очистка ресурсов при необходимости


# Инициализация FastAPI
app = FastAPI(
    title="Digital Inspector API",
    description="API для детекции печатей, подписей и QR-кодов на документах",
    version="1.0.0",
    lifespan=lifespan
)

# Подключение rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
# ВАЖНО: Используется лучшая модель run_default_15ep2 (best.pt)
# Эта модель показала наилучшие результаты и используется в продакшене
# Для изменения модели используйте переменную окружения MODEL_PATH
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).parent.parent / "training_results" / "run_default_15ep2" / "weights" / "best.pt")
)
model = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Классы для детекции (должны соответствовать dataset.yaml)
CLASS_NAMES = {
    0: "qr_code",
    1: "signature",
    2: "stamp"
}

CLASS_COLORS = {
    0: (255, 0, 0),      # Синий для QR-кодов
    1: (0, 255, 0),      # Зеленый для подписей
    2: (0, 0, 255),      # Красный для печатей
}


class Detection(BaseModel):
    """Модель детекции"""
    class_name: str
    class_id: int
    confidence: float
    bbox: dict  # {x1, y1, x2, y2, width, height}


class PageResult(BaseModel):
    """Результаты для одной страницы PDF"""
    page_number: int
    detections: List[Detection]
    num_detections: int
    image_size: dict
    annotated_image: str  # Base64 изображение с bounding boxes


class InferenceResponse(BaseModel):
    """Ответ API с результатами детекции"""
    success: bool
    detections: List[Detection]  # Для обратной совместимости (первая страница или изображение)
    num_detections: int
    image_size: dict  # {width, height}
    inference_time: float
    model_info: dict
    annotated_image: Optional[str] = None  # Base64 изображение с bounding boxes (для изображений или первой страницы)
    pdf_info: Optional[dict] = None  # Информация о PDF (если это PDF)
    pages: Optional[List[PageResult]] = None  # Результаты для всех страниц PDF


def load_model():
    """Загрузка модели YOLO"""
    global model
    if model is None:
        logger.info(f"Loading model from: {MODEL_PATH}")
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        model.to(device)
        logger.info(f"Model loaded successfully on {device}")
        # Показываем информацию о модели
        model_file = Path(MODEL_PATH)
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            logger.info(f"Model size: {size_mb:.1f} MB")
            logger.info(f"Model file: {model_file.name}")
    return model




@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Digital Inspector API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "device": device,
        "classes": CLASS_NAMES
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }


def parse_pdf_pages(pages_str: str, total_pages: int) -> List[int]:
    """
    Парсинг строки с номерами страниц PDF
    
    Поддерживаемые форматы:
    - "-1" или пустая строка - все страницы
    - "0" - одна страница (0-indexed)
    - "0,1,3" - конкретные страницы через запятую
    - "0-2" - диапазон страниц (включительно)
    - "0,2-4,6" - комбинация конкретных страниц и диапазонов
    
    Args:
        pages_str: Строка с номерами страниц
        total_pages: Общее количество страниц в PDF
        
    Returns:
        Список индексов страниц для обработки (0-indexed)
    """
    if not pages_str or pages_str.strip() == "" or pages_str.strip() == "-1":
        # Все страницы
        return list(range(total_pages))
    
    pages_str = pages_str.strip()
    pages_set = set()
    
    # Разбиваем по запятым
    parts = pages_str.split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if '-' in part:
            # Диапазон страниц
            try:
                start, end = part.split('-', 1)
                start = int(start.strip())
                end = int(end.strip())
                # Добавляем все страницы в диапазоне
                for page in range(start, end + 1):
                    if 0 <= page < total_pages:
                        pages_set.add(page)
            except ValueError:
                logger.warning(f"Invalid page range format: {part}")
                continue
        else:
            # Одна страница
            try:
                page = int(part)
                if 0 <= page < total_pages:
                    pages_set.add(page)
            except ValueError:
                logger.warning(f"Invalid page number: {part}")
                continue
    
    # Возвращаем отсортированный список
    return sorted(list(pages_set)) if pages_set else list(range(total_pages))


def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """
    Конвертация PDF в список изображений
    
    Args:
        pdf_bytes: Байты PDF файла
        
    Returns:
        Список PIL Image объектов
        
    Raises:
        ValueError: Если PDF support не доступен или ошибка конвертации
    """
    if not PDF_SUPPORT:
        raise ValueError(
            "PDF support not available. Install pdf2image or PyMuPDF: pip install pdf2image or pip install PyMuPDF"
        )
    
    try:
        if PDF_LIB == "pymupdf":
            # Используем PyMuPDF
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Рендерим страницу в изображение (300 DPI)
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            doc.close()
            return images
        else:
            # Используем pdf2image
            images = convert_from_bytes(pdf_bytes, dpi=300)
            return images
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        raise ValueError(f"Error converting PDF: {str(e)}")


def process_image(
    image_bytes: bytes,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    agnostic_nms: bool = True,
    max_det: int = 300
) -> dict:
    """
    Обработка изображения и детекция объектов
    
    Args:
        image_bytes: Байты изображения
        conf_threshold: Порог уверенности
        iou_threshold: Порог IoU для NMS
        agnostic_nms: Использовать agnostic NMS
        max_det: Максимальное количество детекций
        
    Returns:
        Словарь с результатами детекции
    """
    import time
    start_time = time.time()
    
    # Загрузка модели
    model = load_model()
    
    # Конвертация байтов в изображение
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)
    
    # Конвертация RGB в BGR для OpenCV (если нужно)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Сохранение временного файла
    temp_path = "/tmp/temp_inference.jpg"
    cv2.imwrite(temp_path, image_np)
    
    # Инференс
    results = model(
        temp_path,
        conf=conf_threshold,
        iou=iou_threshold,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        verbose=False
    )
    
    # Обработка результатов
    detections = []
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
            conf_score = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            cls_name = CLASS_NAMES.get(cls, f"class_{cls}")
            
            detections.append({
                "class_name": cls_name,
                "class_id": cls,
                "confidence": conf_score,
                "bbox": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "width": float(box[2] - box[0]),
                    "height": float(box[3] - box[1])
                }
            })
    
    # Удаление временного файла
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    inference_time = time.time() - start_time
    
    return {
        "detections": detections,
        "num_detections": len(detections),
        "image_size": {
            "width": image.width,
            "height": image.height
        },
        "inference_time": inference_time
    }


def draw_boxes(image_bytes: bytes, detections: List[dict]) -> str:
    """
    Рисование bounding boxes на изображении и возврат base64
    
    Args:
        image_bytes: Байты исходного изображения
        detections: Список детекций
        
    Returns:
        Base64 строка изображения с bounding boxes
    """
    # Загрузка изображения
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)
    
    # Конвертация в BGR для OpenCV
    if len(image_np.shape) == 3:
        if image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif image_np.shape[2] == 3:  # RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Рисование bounding boxes
    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        class_id = det["class_id"]
        
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        
        # Цвет для класса
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        # Рисование прямоугольника
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        
        # Текст с классом и уверенностью
        label = f"{class_name}: {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Фон для текста
        cv2.rectangle(
            image_np,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Текст
        cv2.putText(
            image_np,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    # Конвертация обратно в RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    # Уменьшаем размер изображения если оно слишком большое (макс 1920px по ширине)
    max_width = 1920
    if image_pil.width > max_width:
        ratio = max_width / image_pil.width
        new_height = int(image_pil.height * ratio)
        image_pil = image_pil.resize((max_width, new_height), Image.Resampling.LANCZOS)
    
    # Конвертация в base64 с JPEG сжатием (меньше размер чем PNG)
    buffer = io.BytesIO()
    # Конвертируем RGBA в RGB если нужно (для JPEG)
    if image_pil.mode == 'RGBA':
        rgb_image = Image.new('RGB', image_pil.size, (255, 255, 255))
        rgb_image.paste(image_pil, mask=image_pil.split()[3])  # Используем альфа-канал как маску
        image_pil = rgb_image
    elif image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    # Сохраняем как JPEG с качеством 85% (хороший баланс между качеством и размером)
    image_pil.save(buffer, format="JPEG", quality=85, optimize=True)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_base64}"


@app.post("/api/inference", response_model=InferenceResponse)
@limiter.limit("30/minute")  # Rate limit: 30 запросов в минуту
async def inference(
    request: Request,
    file: UploadFile = File(...),
    pdf_page: str = Form("-1")  # "-1" = все страницы, "0,1,3" = конкретные страницы, "0-2" = диапазон
):
    """
    Инференс на загруженном изображении или PDF
    
    Args:
        request: Request объект (для rate limiting)
        file: Загруженное изображение или PDF
        pdf_page: Для PDF: "-1" = все страницы, "0,1,3" = конкретные страницы, "0-2" = диапазон (0-indexed)
        
    Returns:
        Результаты детекции
    """
    # Статические значения параметров детекции
    STATIC_CONF = 0.20
    STATIC_IOU = 0.45
    STATIC_AGNOSTIC_NMS = False
    STATIC_MAX_DET = 300
    STATIC_DRAW_BOXES = True  # Всегда рисуем bounding boxes
    
    # Ограничение параллельных запросов
    async with REQUEST_SEMAPHORE:
        try:
            file_bytes = await file.read()
            
            # Проверка на пустой файл
            if not file_bytes or len(file_bytes) == 0:
                logger.error(f"Empty file received: {file.filename}")
                raise HTTPException(status_code=400, detail="File is empty or could not be read")
            
            # Проверка максимального размера файла (100 MB)
            MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
            if len(file_bytes) > MAX_FILE_SIZE:
                logger.error(f"File too large: {len(file_bytes)} bytes (max: {MAX_FILE_SIZE})")
                raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f} MB")
            
            content_type = file.content_type or ""
            
            logger.info(f"Received file: {file.filename}, size: {len(file_bytes)} bytes, content_type: {content_type}")
            
            # Обработка PDF
            if content_type == "application/pdf" or file.filename.lower().endswith('.pdf'):
                if not PDF_SUPPORT:
                    raise HTTPException(
                        status_code=400,
                        detail="PDF support not available. Install pdf2image or PyMuPDF"
                    )
                
                # Конвертация PDF в изображения
                try:
                    logger.info(f"Converting PDF to images, size: {len(file_bytes)} bytes")
                    pdf_images = pdf_to_images(file_bytes)
                    logger.info(f"PDF converted to {len(pdf_images)} images")
                except ValueError as e:
                    logger.error(f"PDF conversion error: {e}")
                    raise HTTPException(status_code=400, detail=str(e))
                except Exception as e:
                    logger.error(f"Unexpected PDF conversion error: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise HTTPException(status_code=400, detail=f"Error converting PDF: {str(e)}")
                
                if len(pdf_images) == 0:
                    raise HTTPException(status_code=400, detail="PDF is empty or could not be converted")
                
                total_pages = len(pdf_images)
                total_inference_time = 0.0
                all_detections = []
                pages_results = []
                
                # Парсинг номеров страниц для обработки
                pages_to_process_indices = parse_pdf_pages(pdf_page, total_pages)
                pages_to_process = [(i, pdf_images[i]) for i in pages_to_process_indices]
                
                # Обработка каждой страницы
                for page_num, pil_image in pages_to_process:
                    # Конвертация PIL Image в bytes
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    image_bytes = img_byte_arr.getvalue()
                    
                    # Обработка страницы
                    page_result = process_image(
                        image_bytes,
                        conf_threshold=STATIC_CONF,
                        iou_threshold=STATIC_IOU,
                        agnostic_nms=STATIC_AGNOSTIC_NMS,
                        max_det=STATIC_MAX_DET
                    )
                    
                    # Рисование boxes
                    try:
                        annotated_img = draw_boxes(image_bytes, page_result["detections"])
                    except Exception as e:
                        logger.warning(f"Error drawing boxes for page {page_num}: {e}")
                        # Создаем пустое изображение или используем оригинал
                        annotated_img = None
                    
                    # Сохранение результатов страницы
                    pages_results.append({
                        "page_number": page_num,
                        "detections": page_result["detections"],
                        "num_detections": page_result["num_detections"],
                        "image_size": page_result["image_size"],
                        "annotated_image": annotated_img
                    })
                    
                    all_detections.extend(page_result["detections"])
                    total_inference_time += page_result["inference_time"]
                
                # Результат для первой страницы (для обратной совместимости)
                first_page_result = pages_results[0] if pages_results else None
                
                result = {
                    "detections": first_page_result["detections"] if first_page_result else [],
                    "num_detections": sum(p["num_detections"] for p in pages_results),
                    "image_size": first_page_result["image_size"] if first_page_result else {"width": 0, "height": 0},
                    "inference_time": total_inference_time,
                    "pdf_info": {
                        "total_pages": total_pages,
                        "processed_pages": len(pages_to_process),
                        "is_pdf": True
                    },
                    "pages": pages_results,
                    "annotated_image": first_page_result["annotated_image"] if first_page_result else None
                }
                
            # Обработка изображения
            elif content_type.startswith("image/"):
                image_bytes = file_bytes
                
                # Обработка
                result = process_image(
                    image_bytes,
                    conf_threshold=STATIC_CONF,
                    iou_threshold=STATIC_IOU,
                    agnostic_nms=STATIC_AGNOSTIC_NMS,
                    max_det=STATIC_MAX_DET
                )
                result["pdf_info"] = {"is_pdf": False}
            else:
                raise HTTPException(
                    status_code=400,
                    detail="File must be an image (jpg, png, etc.) or PDF"
                )
            
            # Рисование boxes
            annotated_image = None
            if STATIC_DRAW_BOXES:
                try:
                    annotated_image = draw_boxes(image_bytes, result["detections"])
                except Exception as e:
                    logger.warning(f"Error drawing boxes: {e}")
                    # Продолжаем без аннотированного изображения
                    annotated_image = None
            
            # Подготовка ответа
            try:
                response_data = {
                    "success": True,
                    "detections": [Detection(**d) for d in result["detections"]],
                    "num_detections": result["num_detections"],
                    "image_size": result["image_size"],
                    "inference_time": result["inference_time"],
                    "model_info": {
                        "model_path": MODEL_PATH,
                        "device": device,
                        "classes": CLASS_NAMES
                    },
                    "annotated_image": result.get("annotated_image") or annotated_image,
                    "pdf_info": result.get("pdf_info", {"is_pdf": False}),
                    "pages": None
                }
                
                # Добавляем результаты всех страниц для PDF
                if result.get("pdf_info", {}).get("is_pdf") and result.get("pages"):
                    response_data["pages"] = [
                        PageResult(
                            page_number=p["page_number"],
                            detections=[Detection(**d) for d in p["detections"]],
                            num_detections=p["num_detections"],
                            image_size=p["image_size"],
                            annotated_image=p["annotated_image"]
                        )
                        for p in result["pages"]
                    ]
                
                logger.info(f"Response prepared: {result['num_detections']} detections, {len(response_data.get('pages', []) or [])} pages")
                return InferenceResponse(**response_data)
            except Exception as e:
                import traceback
                logger.error(f"Error preparing response: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Error preparing response: {str(e)}")
        
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.error(f"Inference error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/batch-inference")
@limiter.limit("10/minute")  # Rate limit: 10 батч-запросов в минуту
async def batch_inference(
    request: Request,
    files: List[UploadFile] = File(...),
    pdf_page: str = Form("-1")  # "-1" = все страницы, "0,1,3" = конкретные страницы, "0-2" = диапазон
):
    """
    Батч-инференс на нескольких изображениях или PDF файлах
    
    Поддерживает:
    - Изображения (JPG, PNG, BMP и т.д.)
    - PDF файлы (все страницы или конкретная страница)
    
    Args:
        request: Request объект (для rate limiting)
        files: Список загруженных изображений или PDF файлов
        pdf_page: Для PDF: "-1" = все страницы, "0,1,3" = конкретные страницы, "0-2" = диапазон (0-indexed)
        
    Returns:
        Результаты обработки для каждого файла
    """
    # Статические значения параметров детекции
    STATIC_CONF = 0.20
    STATIC_IOU = 0.45
    STATIC_AGNOSTIC_NMS = False
    STATIC_MAX_DET = 300
    STATIC_DRAW_BOXES = True  # Всегда рисуем bounding boxes
    
    # Ограничение параллельных запросов
    async with REQUEST_SEMAPHORE:
        results = []
        
        # Ограничение количества файлов в батче
        MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "20"))
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum batch size: {MAX_BATCH_SIZE}"
            )
        
        # Лимит на общее количество страниц PDF (защита от перегрузки памяти)
        MAX_TOTAL_PDF_PAGES = int(os.getenv("MAX_TOTAL_PDF_PAGES", "100"))
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
        
        # Сначала читаем все файлы и проверяем лимиты
        files_data = []
        total_pdf_pages = 0
        
        for file in files:
            try:
                file_bytes = await file.read()
                
                if not file_bytes or len(file_bytes) == 0:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "File is empty or could not be read"
                    })
                    continue
                
                if len(file_bytes) > MAX_FILE_SIZE:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f} MB"
                    })
                    continue
                
                content_type = file.content_type or ""
                
                # Проверяем, является ли файл PDF и подсчитываем страницы
                if content_type == "application/pdf" or (file.filename and file.filename.lower().endswith('.pdf')):
                    if PDF_SUPPORT:
                        try:
                            # Подсчитываем количество страниц
                            pdf_images = pdf_to_images(file_bytes)
                            pdf_pages_count = len(pdf_images)
                            
                            # Определяем сколько страниц будем обрабатывать
                            pages_to_process_indices = parse_pdf_pages(pdf_page, pdf_pages_count)
                            pages_to_process = len(pages_to_process_indices)
                            
                            total_pdf_pages += pages_to_process
                            
                            if total_pdf_pages > MAX_TOTAL_PDF_PAGES:
                                results.append({
                                    "filename": file.filename,
                                    "success": False,
                                    "error": f"Total PDF pages limit exceeded. Maximum: {MAX_TOTAL_PDF_PAGES} pages across all PDFs"
                                })
                                continue
                        except Exception as e:
                            logger.warning(f"Error checking PDF pages for {file.filename}: {e}")
                            # Продолжаем обработку, но с осторожностью
                
                # Сохраняем данные файла для обработки
                files_data.append({
                    "filename": file.filename,
                    "content_type": content_type,
                    "bytes": file_bytes
                })
            
            except Exception as e:
                logger.error(f"Error reading file {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"Error reading file: {str(e)}"
                })
        
        # Теперь обрабатываем файлы
        for file_data in files_data:
            try:
                file_bytes = file_data["bytes"]
                filename = file_data["filename"]
                content_type = file_data["content_type"]
                
                # Обработка PDF
                if content_type == "application/pdf" or (filename and filename.lower().endswith('.pdf')):
                    if not PDF_SUPPORT:
                        results.append({
                            "filename": filename,
                            "success": False,
                            "error": "PDF support not available. Install pdf2image or PyMuPDF"
                        })
                        continue
                    
                    try:
                        # Конвертация PDF в изображения
                        pdf_images = pdf_to_images(file_bytes)
                        
                        if len(pdf_images) == 0:
                            results.append({
                                "filename": filename,
                                "success": False,
                                "error": "PDF is empty or could not be converted"
                            })
                            continue
                        
                        total_pages = len(pdf_images)
                        # Парсинг номеров страниц для обработки
                        pages_to_process_indices = parse_pdf_pages(pdf_page, total_pages)
                        pages_to_process = [(i, pdf_images[i]) for i in pages_to_process_indices]
                        
                        # Обработка страниц PDF
                        all_detections = []
                        pages_results = []
                        total_inference_time = 0.0
                        
                        for page_num, pil_image in pages_to_process:
                            # Конвертация PIL Image в bytes
                            img_byte_arr = io.BytesIO()
                            pil_image.save(img_byte_arr, format='PNG')
                            image_bytes = img_byte_arr.getvalue()
                            
                            # Обработка страницы
                            page_result = process_image(
                                image_bytes,
                                conf_threshold=STATIC_CONF,
                                iou_threshold=STATIC_IOU,
                                agnostic_nms=STATIC_AGNOSTIC_NMS,
                                max_det=STATIC_MAX_DET
                            )
                            
                            # Рисование boxes если нужно
                            annotated_img = None
                            if STATIC_DRAW_BOXES:
                                try:
                                    annotated_img = draw_boxes(image_bytes, page_result["detections"])
                                except Exception as e:
                                    logger.warning(f"Error drawing boxes for {filename} page {page_num}: {e}")
                            
                            pages_results.append({
                                "page_number": page_num,
                                "detections": page_result["detections"],
                                "num_detections": page_result["num_detections"],
                                "image_size": page_result["image_size"],
                                "annotated_image": annotated_img
                            })
                            
                            all_detections.extend(page_result["detections"])
                            total_inference_time += page_result["inference_time"]
                        
                        # Результат для файла
                        first_page_result = pages_results[0] if pages_results else None
                        results.append({
                            "filename": filename,
                            "success": True,
                            "detections": first_page_result["detections"] if first_page_result else [],
                            "num_detections": sum(p["num_detections"] for p in pages_results),
                            "image_size": first_page_result["image_size"] if first_page_result else {"width": 0, "height": 0},
                            "inference_time": total_inference_time,
                            "pdf_info": {
                                "total_pages": total_pages,
                                "processed_pages": len(pages_to_process),
                                "is_pdf": True
                            },
                            "pages": pages_results,
                            "annotated_image": first_page_result["annotated_image"] if first_page_result and STATIC_DRAW_BOXES else None
                        })
                    
                    except Exception as e:
                        logger.error(f"Error processing PDF {filename}: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        results.append({
                            "filename": filename,
                            "success": False,
                            "error": f"Error processing PDF: {str(e)}"
                        })
                
                # Обработка изображения
                elif content_type.startswith("image/"):
                    image_bytes = file_bytes
                    result = process_image(
                        image_bytes,
                        conf_threshold=STATIC_CONF,
                        iou_threshold=STATIC_IOU,
                        agnostic_nms=STATIC_AGNOSTIC_NMS,
                        max_det=STATIC_MAX_DET
                    )
                    
                    # Рисование boxes если нужно
                    annotated_image = None
                    if STATIC_DRAW_BOXES:
                        try:
                            annotated_image = draw_boxes(image_bytes, result["detections"])
                        except Exception as e:
                            logger.warning(f"Error drawing boxes for {filename}: {e}")
                    
                    results.append({
                        "filename": filename,
                        "success": True,
                        **result,
                        "pdf_info": {"is_pdf": False},
                        "annotated_image": annotated_image
                    })
                else:
                    results.append({
                        "filename": filename,
                        "success": False,
                        "error": "File must be an image (jpg, png, etc.) or PDF"
                    })
            
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e)
                })
        
        return {"results": results}


if __name__ == "__main__":
    import uvicorn
    
    # Получение настроек из переменных окружения
    workers = int(os.getenv("UVICORN_WORKERS", "4"))  # Количество воркеров
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Запуск с несколькими воркерами для лучшей производительности
    if workers > 1:
        # Используем uvicorn с workers (требует запуск через командную строку)
        logger.info(f"Starting uvicorn with {workers} workers")
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
    else:
        # Одиночный воркер (для разработки)
        logger.info("Starting uvicorn with single worker")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )

