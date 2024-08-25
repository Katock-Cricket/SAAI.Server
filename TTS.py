import asyncio
import json
import re
import tempfile
import threading
import time
from queue import Queue

import edge_tts

import utils

with open('./SAAI.Server/presets.json', 'r') as f:
    presets = json.load(f)
    tts_voice_cn_map = presets['tts_voice_cn_map']
    tts_voice_en_map = presets['tts_voice_en_map']
    tts_speed_cn_map = presets['tts_speed_cn_map']
    tts_speed_en_map = presets['tts_speed_en_map']


def get_accent():
    try:
        with open('./scripts/SAAI.ini', 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#'):
                    continue
                if 'accent =' in line:
                    parts = line.split('=')
                    if len(parts) == 2:
                        value = parts[1].strip()
                        if value == '0' or value == ' 0':
                            return True
                        elif value == '1' or value == ' 1':
                            return False
        return True
    except RuntimeError as e:
        return False


accent = get_accent()


async def tts_fn(input_text, cn, spk):
    input_text = re.sub(r"[\n,() ]", "", input_text)
    if accent:
        voice = tts_voice_cn_map[spk] if cn else tts_voice_en_map[spk]
        tts_rate = tts_speed_cn_map[spk] if cn else tts_speed_en_map[spk]
    else:
        voice = tts_voice_cn_map['carl'] if cn else tts_voice_en_map['carl']
        tts_rate = tts_speed_cn_map['carl'] if cn else tts_speed_en_map['carl']

    rate = f"{tts_rate:+d}%"
    communicate = edge_tts.Communicate(text=input_text, voice=voice, rate=rate)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        temp_path = tmp_file.name
    await communicate.save(temp_path)
    return temp_path


from SVC import svc_queue


async def tts_worker(request_queue):
    while True:
        content, cn, speaker, conn = request_queue.get()
        try:
            speaker = speaker.lower()
            print("content: ", content, "speaker: ", speaker)
            start = time.time()
            tts_audio_path = await tts_fn(content, cn, speaker)
            if cn:
                utils.cut_silence(tts_audio_path)
            end = time.time()
            print("[TTS] cost ", f"{(end - start):.3f}", " seconds", "\n")
            svc_queue.put((tts_audio_path, cn, speaker, conn))
        except RuntimeError as e:
            print(e)
        request_queue.task_done()

tts_queue = Queue()


def launch_tts():
    processing_thread = threading.Thread(target=asyncio.run, args=(tts_worker(tts_queue),), daemon=True)
    processing_thread.start()