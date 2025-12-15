from fastai.vision.all import load_learner, PILImage
import gradio as gr
import os
import pathlib

# If a model was exported on Linux/macOS it may reference pathlib.PosixPath inside
# the pickle. On Windows that can raise UnsupportedOperation when unpickling.
# Alias PosixPath to WindowsPath only on Windows so unpickling succeeds.
if os.name == 'nt':
  pathlib.PosixPath = pathlib.WindowsPath

# Use an existing model file from the repo (update if you exported a different one)
model_path = 'models/yoga-pose-recognizer-v2.pkl'
if not os.path.exists(model_path):
  raise FileNotFoundError(f"Model not found: {model_path}. Available files: {os.listdir('models')}")
model = load_learner(model_path)

pose_labels =['Boat Pose yoga',
              'Bridge Pose yoga',
              'Chair Pose yoga',
              'Child Pose yoga',
              'Cobra Pose yoga',
              'Downward Dog Pose yoga',
              'Mountain Pose yoga',
              'Tree Pose yoga',
              'Triangle Pose yoga',
              'Warrior 1 Pose yoga',
              'Warrior 2 Pose yoga']

def safe_predict(img):
    model.eval()
    try:
        pred, idx, probs = model.predict(img)
        try:
            probs_list = list(map(float, probs))
            if len(probs_list) == len(pose_labels):
                return dict(zip(pose_labels, probs_list))
            return {'result': str(pred), 'probs': probs_list}
        except Exception:
            return {'result': str(pred)}
    except TypeError as e:
        # fallback for learners trained with structured targets (dicts/masks)
        try:
            dl = model.dls.test_dl([img], with_labels=False)
            inp, preds, *rest = model.get_preds(dl=dl, with_input=True, with_decoded=False)
            raw = preds[0]
            try:
                import torch
                if isinstance(raw, torch.Tensor):
                    raw = raw.detach().cpu().numpy().tolist()
            except Exception:
                pass
            return {'pred_raw': raw}
        except Exception as e2:
            return {'error': 'predict and fallback both failed', 'orig': str(e), 'fallback': str(e2)}

def recognize_image(image):
  res = safe_predict(image)
  # If we have a full label->prob map we're done
  if isinstance(res, dict) and all(lbl in res for lbl in pose_labels[:3]):
    return res
  if 'pred_raw' in res:
    return {'result': str(res['pred_raw'])}
  if 'result' in res:
    return {'result': str(res['result'])}
  return {'error': str(res)}


image = gr.Image(width=192, height=192)
label = gr.Label()
examples = ['test_images/' + f for f in os.listdir('test_images') if f.endswith('.jpg')]
iface = gr.Interface(
    fn=recognize_image,
    inputs=image,
    outputs=label,
    examples=examples
)
iface.launch(inline=False, share = True)


