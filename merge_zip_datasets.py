#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для объединения нескольких YOLO датасетов из ZIP-архивов.
"""

import os
import shutil
import zipfile
import glob
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set


class DatasetMerger:
    def __init__(self, zip_files: List[str], output_dir: str = "merged_dataset"):
        self.zip_files = zip_files
        self.output_dir = output_dir
        self.extracted_dir = "extracted"
        self.stats = {
            "archives_processed": 0,
            "images_added": 0,
            "annotations_added": 0,
            "files_renamed": 0,
            "files_skipped": 0,
            "datasets": []
        }
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        self.skip_files = {'.DS_Store', 'Thumbs.db', '__MACOSX'}
        self.class_names = {}
        self.max_class_index = -1
        
    def log(self, message: str, level: str = "INFO"):
        """Вывод лога с форматированием."""
        prefix = {
            "INFO": "[INFO]",
            "WARN": "[WARN]",
            "ERROR": "[ERROR]",
            "SUCCESS": "[✓]"
        }.get(level, "[INFO]")
        print(f"{prefix} {message}")
    
    def extract_archives(self):
        """Распаковка всех ZIP-архивов."""
        self.log("Начало распаковки архивов...")
        
        if os.path.exists(self.extracted_dir):
            shutil.rmtree(self.extracted_dir)
        os.makedirs(self.extracted_dir, exist_ok=True)
        
        for idx, zip_file in enumerate(self.zip_files, 1):
            if not os.path.exists(zip_file):
                self.log(f"Архив не найден: {zip_file}", "WARN")
                continue
            
            dataset_name = os.path.splitext(os.path.basename(zip_file))[0]
            extract_path = os.path.join(self.extracted_dir, f"dataset{idx}")
            
            try:
                self.log(f"Распаковка {zip_file} -> {extract_path}")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                self.stats["archives_processed"] += 1
                self.stats["datasets"].append({
                    "name": dataset_name,
                    "path": extract_path,
                    "index": idx
                })
            except Exception as e:
                self.log(f"Ошибка при распаковке {zip_file}: {e}", "ERROR")
    
    def find_yolo_structure(self, dataset_path: str) -> Dict[str, Dict[str, str]]:
        """Рекурсивный поиск структуры YOLO в датасете."""
        structure = {
            "train": {"images": None, "labels": None},
            "val": {"images": None, "labels": None}
        }
        
        # Рекурсивный поиск папок
        for root, dirs, files in os.walk(dataset_path):
            root_lower = root.lower()
            
            # Поиск train/val папок
            if 'train' in root_lower or 'training' in root_lower:
                if 'image' in root_lower or any(f.lower().endswith(tuple(self.image_extensions)) for f in files):
                    if structure["train"]["images"] is None:
                        structure["train"]["images"] = root
                elif 'label' in root_lower or any(f.lower().endswith('.txt') for f in files):
                    if structure["train"]["labels"] is None:
                        structure["train"]["labels"] = root
            
            if 'val' in root_lower or 'valid' in root_lower or 'validation' in root_lower or 'test' in root_lower:
                if 'image' in root_lower or any(f.lower().endswith(tuple(self.image_extensions)) for f in files):
                    if structure["val"]["images"] is None:
                        structure["val"]["images"] = root
                elif 'label' in root_lower or any(f.lower().endswith('.txt') for f in files):
                    if structure["val"]["labels"] is None:
                        structure["val"]["labels"] = root
        
        # Если не найдено, ищем любые папки с изображениями и labels
        if structure["train"]["images"] is None or structure["train"]["labels"] is None:
            images_dirs = []
            labels_dirs = []
            
            for root, dirs, files in os.walk(dataset_path):
                has_images = any(f.lower().endswith(tuple(self.image_extensions)) for f in files)
                has_labels = any(f.lower().endswith('.txt') for f in files)
                
                if has_images and not has_labels:
                    images_dirs.append(root)
                elif has_labels and not has_images:
                    labels_dirs.append(root)
                elif has_images and has_labels:
                    images_dirs.append(root)
                    labels_dirs.append(root)
            
            if images_dirs and labels_dirs:
                structure["train"]["images"] = images_dirs[0]
                structure["train"]["labels"] = labels_dirs[0]
        
        return structure
    
    def is_valid_image(self, image_path: str) -> bool:
        """Проверка валидности изображения."""
        try:
            # Простая проверка по расширению и размеру файла
            if not os.path.exists(image_path):
                return False
            if os.path.getsize(image_path) == 0:
                return False
            return True
        except:
            return False
    
    def is_valid_label(self, label_path: str) -> bool:
        """Проверка валидности label-файла."""
        try:
            if not os.path.exists(label_path):
                return False
            if os.path.getsize(label_path) == 0:
                return False
            
            # Проверка формата YOLO (поддерживает обычный YOLO и OBB)
            with open(label_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return False
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # YOLO: минимум 5 значений (class + 4 координаты)
                    # YOLO OBB: 9 значений (class + 8 координат)
                    if len(parts) < 5:
                        return False
                    try:
                        class_id = int(parts[0])
                        if class_id < 0:
                            return False
                    except ValueError:
                        return False
            
            return True
        except:
            return False
    
    def read_data_yaml(self, dataset_path: str) -> Dict[int, str]:
        """Чтение data.yaml и извлечение имен классов."""
        yaml_files = ['data.yaml', 'dataset.yaml']
        names = {}
        
        for yaml_file in yaml_files:
            yaml_path = os.path.join(dataset_path, yaml_file)
            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Простой парсинг YAML для names
                        in_names = False
                        for line in content.split('\n'):
                            line = line.strip()
                            if 'names:' in line.lower():
                                in_names = True
                                continue
                            if in_names:
                                if line and not line.startswith('#') and ':' in line:
                                    # Формат: "  0: class_name" или "0: class_name"
                                    parts = line.split(':', 1)
                                    if len(parts) == 2:
                                        try:
                                            class_id = int(parts[0].strip())
                                            class_name = parts[1].strip()
                                            if class_name:
                                                names[class_id] = class_name
                                        except ValueError:
                                            pass
                                elif line and not line.startswith(' ') and not line.startswith('\t'):
                                    # Конец секции names
                                    break
                    if names:
                        break
                except Exception as e:
                    self.log(f"Ошибка при чтении {yaml_path}: {e}", "WARN")
        
        return names
    
    def collect_classes_from_labels(self, dataset_path: str):
        """Сбор информации о классах из всех label-файлов."""
        # Сначала проверяем наличие data.yaml
        yaml_names = self.read_data_yaml(dataset_path)
        if yaml_names:
            for class_id, class_name in yaml_names.items():
                self.class_names[class_id] = class_name
                self.max_class_index = max(self.max_class_index, class_id)
        
        # Проверяем наличие names.txt
        names_file = None
        for root, dirs, files in os.walk(dataset_path):
            if 'names.txt' in files:
                names_file = os.path.join(root, 'names.txt')
                break
        
        if names_file:
            try:
                with open(names_file, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        class_name = line.strip()
                        if class_name:
                            self.class_names[idx] = class_name
                            self.max_class_index = max(self.max_class_index, idx)
            except Exception as e:
                self.log(f"Ошибка при чтении names.txt: {e}", "WARN")
        
        # Сканируем все txt файлы для определения максимального индекса
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith('.txt') and file != 'names.txt':
                    label_path = os.path.join(root, file)
                    if self.is_valid_label(label_path):
                        try:
                            with open(label_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        parts = line.split()
                                        if parts:
                                            try:
                                                class_id = int(parts[0])
                                                if class_id >= 0:
                                                    self.max_class_index = max(self.max_class_index, class_id)
                                            except ValueError:
                                                pass
                        except Exception as e:
                            pass
    
    def get_image_base_name(self, image_path: str) -> str:
        """Получение базового имени изображения без расширения."""
        base = os.path.splitext(os.path.basename(image_path))[0]
        return base
    
    def copy_file_with_rename(self, src: str, dst_dir: str, 
                              dataset_index: int, existing_files: Set[str]) -> str:
        """Копирование файла с автоматическим переименованием при конфликтах."""
        base_name = os.path.basename(src)
        name, ext = os.path.splitext(base_name)
        
        # Проверяем, существует ли файл с таким именем
        if base_name in existing_files:
            new_name = f"d{dataset_index}_{base_name}"
            self.stats["files_renamed"] += 1
        else:
            new_name = base_name
        
        dst_path = os.path.join(dst_dir, new_name)
        shutil.copy2(src, dst_path)
        existing_files.add(new_name)
        
        return new_name
    
    def copy_label_with_class_remap(self, src: str, dst_dir: str, 
                                    dataset_index: int, existing_files: Set[str],
                                    class_mapping: Dict[int, int]) -> str:
        """Копирование label-файла с переназначением классов."""
        base_name = os.path.basename(src)
        name, ext = os.path.splitext(base_name)
        
        # Проверяем, существует ли файл с таким именем
        if base_name in existing_files:
            new_name = f"d{dataset_index}_{base_name}"
            self.stats["files_renamed"] += 1
        else:
            new_name = base_name
        
        dst_path = os.path.join(dst_dir, new_name)
        
        # Читаем исходный файл и переназначаем классы
        try:
            with open(src, 'r', encoding='utf-8') as f_in:
                with open(dst_path, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            f_out.write('\n')
                            continue
                        
                        parts = line.split()
                        if parts:
                            try:
                                old_class_id = int(parts[0])
                                # Переназначаем класс
                                new_class_id = class_mapping.get(old_class_id, old_class_id)
                                parts[0] = str(new_class_id)
                                f_out.write(' '.join(parts) + '\n')
                            except ValueError:
                                f_out.write(line + '\n')
                        else:
                            f_out.write(line + '\n')
        except Exception as e:
            self.log(f"Ошибка при копировании {src}: {e}", "WARN")
            # Fallback: простое копирование
            shutil.copy2(src, dst_path)
        
        existing_files.add(new_name)
        return new_name
    
    def merge_dataset(self, dataset_info: Dict, existing_files: Dict[str, Set[str]]):
        """Объединение одного датасета в общий."""
        dataset_path = dataset_info["path"]
        dataset_index = dataset_info["index"]
        dataset_name = dataset_info["name"]
        
        self.log(f"Обработка датасета {dataset_index}: {dataset_name}")
        
        # Читаем data.yaml для получения имен классов
        dataset_classes = self.read_data_yaml(dataset_path)
        
        # Создаем маппинг классов: все классы из датасета переназначаем на новый индекс
        # Используем следующий доступный индекс класса (max_class_index + 1)
        new_class_id = self.max_class_index + 1
        class_mapping = {}
        if dataset_classes:
            # Если есть несколько классов в датасете, переназначаем их последовательно
            for old_class_id in sorted(dataset_classes.keys()):
                class_mapping[old_class_id] = new_class_id
                # Сохраняем имя класса
                class_name = dataset_classes[old_class_id]
                self.class_names[new_class_id] = class_name
                self.max_class_index = max(self.max_class_index, new_class_id)
                new_class_id += 1
        else:
            # Если нет data.yaml, используем следующий доступный индекс
            class_mapping[0] = new_class_id
            self.class_names[new_class_id] = f"dataset{dataset_index}"
            self.max_class_index = max(self.max_class_index, new_class_id)
        
        # Находим структуру
        structure = self.find_yolo_structure(dataset_path)
        
        # Обрабатываем train и val
        for split in ["train", "val"]:
            images_dir = structure[split]["images"]
            labels_dir = structure[split]["labels"]
            
            if images_dir is None:
                continue
            
            output_images_dir = os.path.join(self.output_dir, "images", split)
            output_labels_dir = os.path.join(self.output_dir, "labels", split)
            
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_labels_dir, exist_ok=True)
            
            # Находим все изображения
            image_files = []
            for ext in self.image_extensions:
                image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
                image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
            
            # Обрабатываем каждое изображение
            for image_path in image_files:
                # Пропускаем мусорные файлы
                if any(skip in image_path for skip in self.skip_files):
                    self.stats["files_skipped"] += 1
                    continue
                
                if not self.is_valid_image(image_path):
                    self.stats["files_skipped"] += 1
                    continue
                
                # Копируем изображение
                image_name = self.copy_file_with_rename(
                    image_path, output_images_dir, dataset_index, 
                    existing_files[f"images_{split}"]
                )
                
                # Ищем соответствующий label
                base_name = self.get_image_base_name(image_path)
                label_found = False
                
                # Ищем label в labels_dir
                if labels_dir and os.path.exists(labels_dir):
                    for ext in ['.txt']:
                        label_path = os.path.join(labels_dir, base_name + ext)
                        if os.path.exists(label_path):
                            if self.is_valid_label(label_path):
                                self.copy_label_with_class_remap(
                                    label_path, output_labels_dir, dataset_index,
                                    existing_files[f"labels_{split}"], class_mapping
                                )
                                self.stats["annotations_added"] += 1
                                label_found = True
                            break
                
                # Если не нашли в labels_dir, ищем рядом с изображением
                if not label_found:
                    for ext in ['.txt']:
                        label_path = os.path.join(images_dir, base_name + ext)
                        if os.path.exists(label_path):
                            if self.is_valid_label(label_path):
                                self.copy_label_with_class_remap(
                                    label_path, output_labels_dir, dataset_index,
                                    existing_files[f"labels_{split}"], class_mapping
                                )
                                self.stats["annotations_added"] += 1
                                break
                
                self.stats["images_added"] += 1
    
    def generate_yaml(self):
        """Генерация dataset.yaml файла."""
        yaml_path = os.path.join(self.output_dir, "dataset.yaml")
        
        # Формируем names
        names_dict = {}
        for i in range(self.max_class_index + 1):
            if i in self.class_names:
                names_dict[i] = self.class_names[i]
            else:
                names_dict[i] = f"class{i}"
        
        # Записываем YAML
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {os.path.abspath(self.output_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("names:\n")
            for idx, name in sorted(names_dict.items()):
                f.write(f"  {idx}: {name}\n")
        
        self.log(f"Создан файл: {yaml_path}")
    
    def print_stats(self):
        """Вывод статистики."""
        print("\n" + "="*60)
        print("СТАТИСТИКА ОБЪЕДИНЕНИЯ ДАТАСЕТОВ")
        print("="*60)
        print(f"Обработано архивов:     {self.stats['archives_processed']}")
        print(f"Добавлено изображений:  {self.stats['images_added']}")
        print(f"Добавлено аннотаций:    {self.stats['annotations_added']}")
        print(f"Переименовано файлов:   {self.stats['files_renamed']}")
        print(f"Пропущено файлов:       {self.stats['files_skipped']}")
        print(f"Максимальный класс:     {self.max_class_index}")
        print(f"Всего классов:          {self.max_class_index + 1}")
        print("="*60)
    
    def run(self):
        """Основной метод запуска."""
        self.log("Запуск объединения датасетов...")
        
        # Шаг 1: Распаковка архивов
        self.extract_archives()
        
        if self.stats["archives_processed"] == 0:
            self.log("Не найдено архивов для обработки!", "ERROR")
            return
        
        # Шаг 2: Создание выходной директории
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels", "val"), exist_ok=True)
        
        # Шаг 3: Объединение датасетов
        existing_files = {
            "images_train": set(),
            "images_val": set(),
            "labels_train": set(),
            "labels_val": set()
        }
        
        for dataset_info in self.stats["datasets"]:
            self.merge_dataset(dataset_info, existing_files)
        
        # Шаг 4: Генерация YAML (классы уже собраны при merge_dataset)
        self.generate_yaml()
        
        # Шаг 6: Вывод статистики
        self.print_stats()
        
        self.log("Объединение завершено!", "SUCCESS")


def main():
    """Главная функция."""
    # Поиск архивов yolov8-obb в текущей директории
    zip_files = []
    
    # Ищем архивы yolov8-obb
    for zip_file in glob.glob("*yolov8-obb.zip"):
        if os.path.exists(zip_file):
            zip_files.append(zip_file)
    
    # Сортируем для предсказуемого порядка (QR code первым, затем IDP)
    # Используем кастомную сортировку: QR code первым
    def sort_key(filename):
        if 'qr' in filename.lower() or 'qr_code' in filename.lower():
            return (0, filename)
        else:
            return (1, filename)
    
    zip_files.sort(key=sort_key)
    
    if not zip_files:
        print("[ERROR] Не найдено ZIP-архивов yolov8-obb в текущей директории!")
        print("Ожидаются файлы с расширением *yolov8-obb.zip")
        return
    
    print(f"[INFO] Найдено архивов: {len(zip_files)}")
    for zip_file in zip_files:
        print(f"  - {zip_file}")
    print()
    
    # Запуск объединения
    merger = DatasetMerger(zip_files)
    merger.run()


if __name__ == "__main__":
    main()

