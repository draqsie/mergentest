import datetime
import random

class HomeostaticLIFNeuron:
    def __init__(self, beta=0.95, base_threshold=1.2, target_activity=0.15):
        self.beta = beta
        self.base_threshold = base_threshold
        self.threshold = base_threshold
        self.mem = 0.0 
        self.activity_trace = 0.0 
        self.target_activity = target_activity 
        self.homeostasis_rate = 0.05 

    def step(self, current_in):
        self.mem = (self.mem * self.beta) + current_in
        spike = 0
        if self.mem >= self.threshold:
            self.mem -= self.threshold
            spike = 1
            
        self.activity_trace = (self.activity_trace * 0.9) + spike
        error = self.activity_trace - self.target_activity
        self.threshold += error * self.homeostasis_rate
        self.threshold = max(0.5, min(self.threshold, self.base_threshold * 3.0))
        return spike

class MergenPureBiologicalEngine:
    def __init__(self):
        self.num_steps = 100
        self.anlam_neuron = HomeostaticLIFNeuron(beta=0.95, base_threshold=1.2, target_activity=0.2)
        self.celiski_neuron = HomeostaticLIFNeuron(beta=0.90, base_threshold=1.5, target_activity=0.1)

    def _text_to_sensory_stimulus(self, text):
        anlam_akimi = 0.0
        celiski_akimi = 0.0
        words = text.lower().split()
        for w in words:
            density = sum(ord(c) for c in w) / (len(w) * 255.0)
            anlam_akimi += density * 0.8
            if len(w) > 5:
                celiski_akimi += density * 0.6
        return anlam_akimi, celiski_akimi

    def _generate_response(self, anlam_spikes, celiski_spikes, durum):
        """Biyolojik duruma göre saf, algoritmik bir metin sentezler (LLM olmadan)."""
        if "Stresli" in durum:
            return "Kortikal aşırı yüklenme. Eşik değerleri maksimumda. Veri akışı reddedildi, sistem dinlenmeye geçiyor."
        elif "Meraklı" in durum:
            return "Hücresel sessizlik. Eşikler düşürüldü. Ağ, daha yoğun frekanslı bağlamlar bekliyor."
        
        if celiski_spikes > anlam_spikes:
            return "Girdideki mantıksal çelişki oranı yüksek. Sentetik örüntü kırılamadı, anlam bütünlüğü kurulamıyor."
        elif anlam_spikes > 15:
            return "Yüksek anlam yoğunluğu tespit edildi. Kelimeler parçalandı, kalıcı öz nöral ağa işlendi."
        else:
            return "Sıradan veri. Düşük kortikal aktivite. Zaman adımı stabil şekilde tamamlandı."

    def process(self, query, user_name):
        start_time = datetime.datetime.now()
        anlam_current, celiski_current = self._text_to_sensory_stimulus(query)
        
        anlam_spikes = 0
        celiski_spikes = 0
        prev_anlam_thresh = self.anlam_neuron.threshold
        
        for step in range(self.num_steps):
            noise_a = (step % 3) * 0.05
            noise_c = (step % 2) * 0.08
            
            if self.anlam_neuron.step(anlam_current + noise_a) == 1:
                anlam_spikes += 1
            if self.celiski_neuron.step(celiski_current + noise_c) == 1:
                celiski_spikes += 1

        kavrayis = min((anlam_spikes / self.num_steps) * 100 * 2.0, 100.0)
        celiski = min((celiski_spikes / self.num_steps) * 100 * 2.0, 100.0)
        
        anlam_durum = "Stabil"
        if self.anlam_neuron.threshold > prev_anlam_thresh * 1.1:
            anlam_durum = "Stresli/Yorgun (Sakinleşiyor)"
        elif self.anlam_neuron.threshold < prev_anlam_thresh * 0.9:
            anlam_durum = "Aç/Meraklı (Hassaslaştı)"

        response_text = self._generate_response(anlam_spikes, celiski_spikes, anlam_durum)
        latency = (datetime.datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "spikes_fired": {"anlam": anlam_spikes, "celiski": celiski_spikes},
            "kavrayis": kavrayis,
            "celiski": celiski,
            "homeostazi": {
                "anlam_esik": self.anlam_neuron.threshold,
                "anlam_durum": anlam_durum
            },
            "response": response_text,
            "latency": latency,
            "steps": self.num_steps
        }
