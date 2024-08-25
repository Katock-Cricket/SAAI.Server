import asyncio
import json
import os
import queue
import threading
import time

import soundfile

import torch

from fairseq import checkpoint_utils

import utils
from svc.inference.infer_tool import Svc
from svc.modules.F0Predictor.fcpe.model import FCPEInfer
from AudioPathCalculator import AudioPathCalculator

svc_models = {}
sampling_rate = 44100
vec_path = "./SAAI.Server/svc/pretrain/checkpoint_best_legacy_500.pt"
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [vec_path],
    suffix="",
)
print("load model(s) from {}".format(vec_path))
hubert_dict = {
    "vec768l12": utils.get_speech_encoder("vec768l12", "cpu", models, saved_cfg, task),
    "vec256l9": utils.get_speech_encoder("vec256l9", "cpu", models, saved_cfg, task)
}
static_fcpe = FCPEInfer(model_path="./SAAI.Server/svc/pretrain/fcpe.pt", device="cpu", dtype=torch.float32)
print("preload fcpe")
audioPathCalculator = AudioPathCalculator()

with open('./SAAI.Server/presets.json', 'r') as f:
    presets = json.load(f)
    vc_transform_cn_map = presets['vc_transform_cn_map']
    vc_transform_en_map = presets['vc_transform_en_map']


def preload_svc_model():
    for f in os.listdir("./SAAI.Server/svc/models"):
        model = Svc(fr"./SAAI.Server/svc/models/{f}/{f}.pth", f"./SAAI.Server/svc/models/{f}/config_{f}.json", "cpu")
        model.hubert_model = hubert_dict[model.speech_encoder]
        svc_models[f] = model


preload_svc_model()


def svc_fn(raw_audio_path, cn, speaker):
    if raw_audio_path is None:
        return None

    vc_transform = vc_transform_cn_map[speaker] if cn else vc_transform_en_map[speaker]
    model = svc_models[speaker]

    out_audio = model.slice_inference(raw_audio_path=raw_audio_path,
                                      spk=speaker,
                                      slice_db=-40,
                                      cluster_infer_ratio=0,
                                      noice_scale=0.4,
                                      clip_seconds=10,
                                      tran=vc_transform,
                                      f0_predictor=static_fcpe,
                                      auto_predict_f0=False)
    os.remove(raw_audio_path)
    return out_audio


async def svc_worker(request_queue):
    while True:
        try:
            raw_audio_path, cn, speaker, conn = request_queue.get()
            start = time.time()
            svc_audio = svc_fn(raw_audio_path, cn, speaker)
            save_path, wavNumber = audioPathCalculator.calc_save_path()
            soundfile.write(save_path, svc_audio, 44100, format="wav")
            utils.normalize(save_path)
            end = time.time()
            print("[SVC] cost ", f"{(end - start):.3f}", " seconds", "\n")
            gen_audio_path = audioPathCalculator.default_pakName + ";" + audioPathCalculator.default_bankNumber + ";" + wavNumber
            conn.send(gen_audio_path.encode())
        except RuntimeError as e:
            print(e)
        request_queue.task_done()


svc_queue = queue.Queue()


def launch_svc():
    processing_thread = threading.Thread(target=asyncio.run, args=(svc_worker(svc_queue),), daemon=True)
    processing_thread.start()
