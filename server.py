import asyncio
import os
import re
import shutil
import socket
import tempfile
import time
import threading
import queue

import librosa
import soundfile
import edge_tts
from pydub import AudioSegment, silence

import utils
from inference.infer_tool import Svc
from fairseq import checkpoint_utils

print("This is the tts-svc backend server for GTASA")
print("AI audio generation program")
print("==========Don't close me===========")

svc_models = {}
sampling_rate = 44100
vec_path = "./SAAI.Server/pretrain/checkpoint_best_legacy_500.pt"
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [vec_path],
    suffix="",
)
print("load model(s) from {}".format(vec_path))
hubert_dict = {
    "vec768l12": utils.get_speech_encoder("vec768l12", "cpu", models, saved_cfg, task),
    "vec256l9": utils.get_speech_encoder("vec256l9", "cpu", models, saved_cfg, task)
}

vc_transform_map = {
    "carl": 3,
    "ryder": 3,
    "sweet": 4,
    "smoke": 5
}
f0p = "fcpe"
tts_voice_en = "km-KH-PisethNeural"
tts_voice_cn = "zh-CN-YunxiNeural"
tts_rate = 0


class AudioPathCalculator:
    def __init__(self):
        self.prefix = "./modloader/SAAI/audio/SFX/"
        self.bak_prefix = "./modloader/SAAI/audio_bak/"
        self.audio_iter = 1
        self.default_pakName = "SCRIPT"
        self.default_bankNumber = "22"

    def next_iter(self):
        if self.audio_iter < 6:
            return self.audio_iter + 1
        else:
            return 1

    def next_iter_and_add(self):
        ret = self.audio_iter
        self.audio_iter = self.next_iter()
        return ret

    # def recover_next_from_bak(self):
    #     it_bak = str(self.audio_iter)
    #     bak_path = self.bak_prefix + "sound_00" + it_bak + "_bak.wav"
    #     rec_path = self.prefix + self.default_pakName + "/Bank_0" + self.default_bankNumber + "/sound_00" + it_bak + ".wav"
    #     if os.path.exists(rec_path):
    #         os.remove(rec_path)
    #     shutil.copy2(bak_path, rec_path)

    def calc_save_path(self):
        # self.recover_next_from_bak()
        it = str(self.next_iter_and_add())
        save_path = self.prefix + self.default_pakName + "/Bank_0" + self.default_bankNumber + "/sound_00" + it + ".wav"
        return save_path, it


audioPathCalculator = AudioPathCalculator()


def svc_fn(raw_audio_path, speaker):
    if raw_audio_path is None:
        return None

    vc_transform = vc_transform_map[speaker]
    model = svc_models[speaker]

    model.hubert_model = hubert_dict[model.speech_encoder]
    out_audio = model.slice_inference(raw_audio_path=raw_audio_path,
                                      spk=speaker,
                                      slice_db=-40,
                                      cluster_infer_ratio=0,
                                      noice_scale=0.4,
                                      clip_seconds=10,
                                      tran=vc_transform,
                                      f0_predictor=f0p,
                                      auto_predict_f0=False)
    model.clear_empty()
    os.remove(raw_audio_path)
    return out_audio


async def tts_fn(input_text, cn):
    input_text = re.sub(r"[\n\,\(\) ]", "", input_text)
    voice = tts_voice_cn if cn else tts_voice_en
    ratestr = "+{:.0%}".format(tts_rate) if tts_rate >= 0 else "{:.0%}".format(tts_rate)
    communicate = edge_tts.Communicate(text=input_text, voice=voice, rate=ratestr)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        temp_path = tmp_file.name
    await communicate.save(temp_path)
    return temp_path


async def generate_audio_main(content, speaker, cn) -> str:
    print("content: ", content, "speaker: ", speaker)
    tts_audio_path = await tts_fn(content, cn)
    utils.cut_silence(tts_audio_path)
    svc_audio = svc_fn(tts_audio_path, speaker)
    save_path, wavNumber = audioPathCalculator.calc_save_path()
    soundfile.write(save_path, svc_audio, sampling_rate, format="wav")
    utils.normalize(save_path)
    return audioPathCalculator.default_pakName + ";" + audioPathCalculator.default_bankNumber + ";" + wavNumber


def preload_svc_model():
    for f in os.listdir("./SAAI.Server/models"):
        model = Svc(fr"./SAAI.Server/models/{f}/{f}.pth", f"./SAAI.Server/models/{f}/config_{f}.json", "cpu")
        svc_models[f] = model


async def process_request(request_queue):
    while True:
        message, conn = request_queue.get()
        if message:
            try:
                content, speaker, cn = message.split(';', 2)
                start = time.time()
                cn = True if cn == "cn" else False
                gen_audio_path = await generate_audio_main(content, speaker, cn)
                end = time.time()
                print("cost ", f"{(end - start):.3f}", " seconds", "\n")
                conn.send(gen_audio_path.encode())
            except RuntimeError as e:
                print(e, conn)
        request_queue.task_done()


def handle_client(conn, address, request_queue):
    while True:
        try:
            message = conn.recv(1024).decode()
        except Exception as e:
            print(e, conn)
            break
        if message:
            request_queue.put((message, conn))
            print("====================================================\n"
                  + "Connection from GTASA asi process:" + str(address))
        else:
            break
    conn.close()


def work():
    host = '127.0.0.1'
    port = 65432

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(6)
    print("Waiting for connection...")

    request_queue = queue.Queue()
    processing_thread = threading.Thread(target=asyncio.run, args=(process_request(request_queue),))
    processing_thread.daemon = True
    processing_thread.start()

    while True:
        conn, address = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(conn, address, request_queue))
        client_thread.daemon = True
        client_thread.start()


if __name__ == '__main__':
    preload_svc_model()
    asyncio.run(work())
