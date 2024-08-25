

class AudioPathCalculator:
    def __init__(self):
        self.prefix = "./modloader/SAAI/audio/SFX/"
        self.audio_iter = 1
        self.default_pakName = "SCRIPT"
        self.default_bankNumber = "42"
        self.default_max_wavNumber = 15

    def next_iter(self):
        if self.audio_iter < self.default_max_wavNumber:
            return self.audio_iter + 1
        else:
            return 1

    def next_iter_and_add(self):
        ret = self.audio_iter
        self.audio_iter = self.next_iter()
        return ret

    def calc_save_path(self):
        it = self.next_iter_and_add()
        if it < 10:
            wav_postfix = "/sound_00" + str(it) + ".wav"
        else:
            wav_postfix = "/sound_0" + str(it) + ".wav"

        save_path = self.prefix + self.default_pakName + "/Bank_0" + self.default_bankNumber + wav_postfix
        return save_path, str(it)