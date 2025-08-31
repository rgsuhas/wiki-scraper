import json

# Full list of cells (markdown and code) from the user's script
full_cells = [
 ("markdown", [
  "# Complete Animal Detection Training Script for Google Colab\n",
  "\n",
  "This notebook trains a custom YOLOv8 model for animal detection.\n",
  "\n",
  "**Classes:** bison, cow, deer, donkey, elephant, fox, goat, hippopotamus, hyena, kangaroo, leopard, lion, ox, panda, rhinoceros, tiger, wolf\n",
  "\n",
  "**Training:** 300 epochs with YOLOv8n model\n"
 ]),
 ("code", [
  "# Install dependencies\n",
  "!pip install ultralytics opencv-python scikit-learn pyyaml\n",
  "!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n",
  "print(\"✅ Dependencies installed successfully!\")"
 ]),
 ("code", [
  "import torch, os\n",
  "from pathlib import Path\n",
  "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
  "if torch.cuda.is_available():\n",
  "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
  "    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")\n",
  "else:\n",
  "    print(\"Using CPU\")\n",
  "!mkdir -p animals_dataset yolo_dataset\n",
  "print(\"✅ Setup completed!\")"
 ]),
 ("code", [
  "# Download sample dataset (optional)\n",
  "import urllib.request, os\n",
  "animal_classes = ['bison','cow','deer','donkey','elephant','fox','goat','hippopotamus','hyena','kangaroo','leopard','lion','ox','panda','rhinoceros','tiger','wolf']\n",
  "for animal in animal_classes:\n",
  "    os.makedirs(f'animals_dataset/{animal}', exist_ok=True)\n",
  "sample_urls = {\n",
  "    'bison':['https://images.unsplash.com/photo-1552084117-56a987666449?w=400'],\n",
  "    'cow':['https://images.unsplash.com/photo-1546445317-29d45416c916?w=400'],\n",
  "    'deer':['https://images.unsplash.com/photo-1552083375-1447ce886485?w=400'],\n",
  "    'lion':['https://images.unsplash.com/photo-1549366021-9f761d450615?w=400'],\n",
  "    'tiger':['https://images.unsplash.com/photo-1561731216-c3a4d99437d5?w=400'],\n",
  "    'elephant':['https://images.unsplash.com/photo-1552083375-1447ce886485?w=400'],\n",
  "    'panda':['https://images.unsplash.com/photo-1552083375-1447ce886485?w=400'],\n",
  "    'wolf':['https://images.unsplash.com/photo-1552083375-1447ce886485?w=400']}\n",
  "for animal,urls in sample_urls.items():\n",
  "    for i,url in enumerate(urls):\n",
  "        try:\n",
  "            filename=f'animals_dataset/{animal}/{animal}_{i+1}.jpg'\n",
  "            urllib.request.urlretrieve(url, filename)\n",
  "            print(f\"Downloaded {filename}\")\n",
  "        except Exception as e:\n",
  "            print(f\"Failed {url}: {e}\")\n",
  "print(\"✅ Sample dataset downloaded!\")"
 ]),
 ("code", [
  "# Upload your own dataset (alternative)\n",
  "from google.colab import files\n",
  "import zipfile, os\n",
  "print(\"Upload your dataset zip file:\")\n",
  "uploaded = files.upload()\n",
  "for filename in uploaded.keys():\n",
  "    if filename.endswith('.zip'):\n",
  "        with zipfile.ZipFile(filename,'r') as zip_ref:\n",
  "            zip_ref.extractall('animals_dataset')\n",
  "        print(f\"Extracted {filename} to animals_dataset/\")\n",
  "        break\n",
  "!find animals_dataset -type d | head -20"
 ]),
 ("code", [
  "# Prepare dataset\n",
  "import os, shutil, random, yaml, cv2, numpy as np\n",
  "from pathlib import Path\n",
  "from sklearn.model_selection import train_test_split\n",
  "class DatasetPreparator:\n",
  "    def __init__(self,dataset_path=\"animals_dataset\",output_path=\"yolo_dataset\"):\n",
  "        self.dataset_path=Path(dataset_path)\n",
  "        self.output_path=Path(output_path)\n",
  "        self.output_path.mkdir(exist_ok=True)\n",
  "        self.animal_classes={'bison':0,'cow':1,'deer':2,'donkey':3,'elephant':4,'fox':5,'goat':6,'hippopotamus':7,'hyena':8,'kangaroo':9,'leopard':10,'lion':11,'ox':12,'panda':13,'rhinoceros':14,'tiger':15,'wolf':16}\n",
  "        self.class_names={v:k for k,v in self.animal_classes.items()}\n",
  "    def prepare_dataset(self):\n",
  "        yolo_dataset=self.output_path\n",
  "        for s in [\"train\",\"val\",\"test\"]:\n",
  "            (yolo_dataset/\"images\"/s).mkdir(parents=True,exist_ok=True)\n",
  "            (yolo_dataset/\"labels\"/s).mkdir(parents=True,exist_ok=True)\n",
  "        image_data=[]\n",
  "        for animal in self.animal_classes:\n",
  "            animal_path=self.dataset_path/animal\n",
  "            if animal_path.exists():\n",
  "                cid=self.animal_classes[animal]\n",
  "                for f in animal_path.glob(\"*.jpg\"):\n",
  "                    image_data.append({'image_path':f,'class_id':cid,'class_name':animal})\n",
  "        if not image_data:\n",
  "            print(\"No images found\")\n",
  "            return None\n",
  "        train_data,temp=train_test_split(image_data,train_size=0.8,random_state=42,stratify=[d['class_id'] for d in image_data])\n",
  "        val_data,test_data=train_test_split(temp,train_size=0.5,random_state=42,stratify=[d['class_id'] for d in temp])\n",
  "        for split,data in [(\"train\",train_data),(\"val\",val_data),(\"test\",test_data)]:\n",
  "            for item in data:\n",
  "                dest_image=yolo_dataset/\"images\"/split/item['image_path'].name\n",
  "                shutil.copy2(item['image_path'],dest_image)\n",
  "                label_file=yolo_dataset/\"labels\"/split/(item['image_path'].stem+\".txt\")\n",
  "                with open(label_file,'w') as f:\n",
  "                    f.write(f\"{item['class_id']} 0.5 0.5 1.0 1.0\\n\")\n",
  "        config={'path':str(yolo_dataset.absolute()),'train':'images/train','val':'images/val','test':'images/test','nc':len(self.animal_classes),'names':list(self.animal_classes.keys())}\n",
  "        with open(yolo_dataset/\"dataset.yaml\",'w') as f: yaml.dump(config,f)\n",
  "        return yolo_dataset\n",
  "preparator=DatasetPreparator()\n",
  "yolo_dataset=preparator.prepare_dataset()"
 ]),
 ("code", [
  "# Train model\n",
  "from ultralytics import YOLO\n",
  "import json\n",
  "from pathlib import Path\n",
  "def train_model(model_size=\"n\",epochs=300,batch_size=16):\n",
  "    config_file=Path(\"yolo_dataset/dataset.yaml\")\n",
  "    if not config_file.exists():\n",
  "        print(\"Dataset not found\")\n",
  "        return None\n",
  "    model=YOLO(f'yolov8{model_size}.pt')\n",
  "    train_args={'data':str(config_file),'epochs':epochs,'batch':batch_size,'imgsz':640,'patience':50,'save':True,'save_period':25,'cache':True,'device':'auto','workers':4,'project':'animal_detection_training','name':f'yolov8{model_size}_animals_300epochs','exist_ok':True}\n",
  "    results=model.train(**train_args)\n",
  "    model_info={'model_size':model_size,'epochs':epochs,'batch_size':batch_size,'classes':preparator.animal_classes,'class_names':preparator.class_names,'training_results':str(results)}\n",
  "    with open(f\"model_info_{model_size}_300epochs.json\",'w') as f: json.dump(model_info,f,indent=2)\n",
  "    return model\n",
  "trained_model=train_model()"
 ]),
 ("code", [
  "# Evaluate model\n",
  "if 'trained_model' in locals() and trained_model:\n",
  "    results=trained_model.val()\n",
  "    print(f\"mAP50: {results.box.map50:.3f}, mAP50-95: {results.box.map:.3f}\")\n",
  "else:\n",
  "    print(\"No trained model\")"
 ]),
 ("code", [
  "# Test on sample image\n",
  "import cv2\n",
  "from pathlib import Path\n",
  "if 'trained_model' in locals() and trained_model:\n",
  "    test_images=list(Path(\"yolo_dataset/images/test\").glob(\"*.jpg\"))\n",
  "    if test_images:\n",
  "        sample=str(test_images[0])\n",
  "        results=trained_model(sample)\n",
  "        for r in results:\n",
  "            for box in r.boxes:\n",
  "                cid=int(box.cls[0]);conf=float(box.conf[0])\n",
  "                print(preparator.class_names.get(cid),conf)\n",
  "        cv2.imwrite(\"test_result.jpg\",results[0].plot())"
 ]),
 ("code", [
  "# Download model\n",
  "import json\n",
  "if 'trained_model' in locals() and trained_model:\n",
  "    !zip -r animal_detection_model.zip animal_detection_training/\n",
  "    with open(\"model_info_n_300epochs.json\") as f: info=json.load(f)\n",
  "    print(info)"
 ]),
 ("code", [
  "# Test uploaded image\n",
  "from google.colab import files\n",
  "if 'trained_model' in locals() and trained_model:\n",
  "    uploaded=files.upload()\n",
  "    image_path=list(uploaded.keys())[0]\n",
  "    results=trained_model(image_path)\n",
  "    cv2.imwrite(\"uploaded_test_result.jpg\",results[0].plot())"
 ]),
 ("code", [
  "# Export model\n",
  "if 'trained_model' in locals() and trained_model:\n",
  "    trained_model.export(format='onnx')\n",
  "    try:\n",
  "        trained_model.export(format='engine')\n",
  "    except Exception as e:\n",
  "        print(e)\n",
  "    try:\n",
  "        trained_model.export(format='coreml')\n",
  "    except Exception as e:\n",
  "        print(e)"
 ]),
 ("markdown", [
  "## Usage Instructions\n",
  "1. Run cells in order.\n",
  "2. Enable GPU in Colab (Runtime → Change runtime type → GPU).\n",
  "3. Train will take 2–4 hours.\n",
  "4. Troubleshooting:\n",
  "   - Reduce batch size if OOM.\n",
  "   - Ensure dataset present.\n",
  "   - Check GPU enabled.\n"
 ])
]

# Build notebook
nb = {
 "cells": [],
 "metadata": {
  "kernelspec": {"display_name": "Python 3","language": "python","name": "python3"},
  "language_info": {"name": "python","version": "3.10"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

for ctype, src in full_cells:
    nb["cells"].append({
        "cell_type": ctype,
        "metadata": {},
        "source": src,
        "execution_count": None if ctype == "code" else None,
        "outputs": [] if ctype == "code" else []
    })

# Save notebook to file
final_path = "animal_detection_training_full.ipynb"
with open(final_path, "w") as f:
    json.dump(nb, f, indent=2)

print(f"Notebook saved to {final_path}")
