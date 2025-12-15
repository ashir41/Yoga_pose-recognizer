# Yoga-pose-Recognizer
An image classification model from data collection, cleaning, model training, deployment and API integration. <br/>
The model can classify 11 different types of Yoga pose <br/>
The types are following: <br/>
1. Boat Pose yoga
2. Bridge Pose yoga
3. Chair Pose yoga
4. Child Pose yoga
5. Cobra Pose yoga
6. Downward Dog Pose yoga
7. Mountain Pose yoga
8. Tree Pose yoga
9. Triangle Pose yoga
10. Warrior 1 Pose yoga
11. Warrior 2 Pose yoga


# Dataset Preparation
**Data Collection:** Downloaded from DuckDuckGo using term name <br/>
**DataLoader:** Used fastai DataBlock API to set up the DataLoader. <br/>
**Data Augmentation:** fastai provides default data augmentation which operates in GPU. <br/>
Details can be found in `notebooks/data_prep.ipynb`

# Training and Data Cleaning
**Training:** Fine-tuned a resnet50 model for 5 epochs (3 times) and got upto ~91% accuracy. <br/>
**Data Cleaning:** This part took the highest time. Since I collected data from browser, there were many noises. Also, there were images that contained. I cleaned and updated data using fastai ImageClassifierCleaner. I cleaned the data each time after training or finetuning, except for the last time which was the final iteration of the model. <br/>

# Model Deployment
I deployed the model to a Gradio app (HuggingFace Spaces or a hosted Gradio share). The deployment implementation lives in the `deployment` folder (see `deployment/app.py`). The app expects a trained, exported FastAI learner in the `models/` directory. By default the code looks for:

```
models/yoga-pose-recognizer-v2.pkl
```

If you exported a model with a different filename, update the `model_path` in `deployment/app.py` accordingly.

<img src = "deployment/gradio_app.png" width="700" height="350">

# Running the Gradio app locally
1. Install dependencies (example):
   - pip install -U fastai gradio

2. Ensure the model file is present:
   - Place your exported learner at `models/yoga-pose-recognizer-v2.pkl` (or change `model_path` inside `deployment/app.py`).

3. Ensure there is a `test_images/` folder containing example .jpg files (the app builds the examples list from that folder).

4. Run the app:
   - python deployment/app.py

The script will launch a Gradio interface. The script also contains code to enable unpickling on Windows (it aliases pathlib.PosixPath to WindowsPath when running on Windows) to avoid UnsupportedOperation errors when loading pickled learners exported on Linux/macOS.

# Inference notebook
If you prefer notebook-based inference or to test the exported learner in Colab, see `notebooks/inference.ipynb`. The notebook uses the same model filename pattern (e.g. `models/yoga-pose-recognizer-v2.pkl`).

# API integration with GitHub Pages
The deployed model API is integrated [here](https://ashir41.github.io/Yoga_pose-recognizer/) in a GitHub Pages Website. Implementation and other details can be found in the `docs` folder.
