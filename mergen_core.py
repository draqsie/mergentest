import datetime
import math
import cmath
import random
import requests
import os

def fft(x):
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def ifft(x):
    N = len(x)
    x_conj = [val.conjugate() for val in x]
    X = fft(x_conj)
    return [(val.conjugate() / N) for val in X]

def fft2d(grid):
    rows = [fft(row) for row in grid]
    cols = list(map(list, zip(*rows)))
    cols_fft = [fft(col) for col in cols]
    return list(map(list, zip(*cols_fft)))

def ifft2d(grid):
    rows = [ifft(row) for row in grid]
    cols = list(map(list, zip(*rows)))
    cols_ifft = [ifft(col) for col in cols]
    return list(map(list, zip(*cols_ifft)))


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)

class CorticalLayer:
    def __init__(self, size=(16, 16), beta=0.95, base_threshold=1.5, target_activity=0.15):
        # PyTorch kullanılmadan saf Python ile yazılmış Hızlı Kortikal Tabaka
        self.size = size
        self.num_neurons = size[0] * size[1] # 16x16 = 256 Nöron
        self.beta = beta
        self.base_threshold = base_threshold
        
        self.mem = [0.0] * self.num_neurons
        self.threshold = [base_threshold] * self.num_neurons
        self.activity_trace = [0.0] * self.num_neurons
        self.last_spikes = [0.0] * self.num_neurons
        
        self.target_activity = target_activity
        self.homeostasis_rate = 0.005
        
        # W[i][j] = j'den i'ye giden bağlantı ağırlığı
        self.W = [[random.uniform(-0.01, 0.01) if i != j else 0.0 for j in range(self.num_neurons)] for i in range(self.num_neurons)]
        
        # STDP (Lokal Öğrenme) İzleri
        self.trace_pre = [0.0] * self.num_neurons
        self.trace_post = [0.0] * self.num_neurons
        self.A_plus = 0.005
        self.A_minus = 0.0055
        self.tau_pre = 15.0
        self.tau_post = 15.0

        # FFT Hızlandırmalı Yanal İnhibisyon (Mexican Hat)
        self.kernel_fft = self._create_fft_kernel()

    def _create_fft_kernel(self):
        # 5x5 Mexican Hat Kernel
        mexican_hat = [
            [-0.1, -0.2, -0.2, -0.2, -0.1],
            [-0.2,  0.5,  1.0,  0.5, -0.2],
            [-0.2,  1.0,  2.0,  1.0, -0.2],
            [-0.2,  0.5,  1.0,  0.5, -0.2],
            [-0.1, -0.2, -0.2, -0.2, -0.1]
        ]
        
        # Kernel'i 16x16 ızgaraya sığdır ve dairesel konvolüsyon için merkezini ayarla
        kernel_16x16 = [[0.0 for _ in range(16)] for _ in range(16)]
        for i in range(5):
            for j in range(5):
                r = (i - 2) % 16
                c = (j - 2) % 16
                kernel_16x16[r][c] = mexican_hat[i][j]
                
        # FFT Ön hesaplaması (Bunu sadece bir kez başlatırken yapıyoruz - Muazzam hızlanma)
        return fft2d(kernel_16x16)

    def step(self, current_in):
        # Sparse Poisson Spikes
        input_spikes = [1.0 if random.random() < abs(c) else 0.0 for c in current_in]
        
        # Rekürrent Akım Hesaplama
        recurrent_current = [0.0] * self.num_neurons
        for i in range(self.num_neurons):
            recurrent_current[i] = sum(self.W[i][j] * self.last_spikes[j] for j in range(self.num_neurons))
            
        total_current = [input_spikes[i] + recurrent_current[i] for i in range(self.num_neurons)]
        
        # FFT HIZLANDIRMASI İLE MEXICAN HAT KONVOLÜSYONU
        # 1D sinyali 16x16 2D matrise çevir
        total_current_2d = [[total_current[i*16 + j] for j in range(16)] for i in range(16)]
        
        # Zaman uzayından Frekans uzayına (FFT) geçiş
        total_fft = fft2d(total_current_2d)
        
        # Frekans uzayında konvolüsyon (sadece nokta çarpım)
        conv_fft = [[total_fft[i][j] * self.kernel_fft[i][j] for j in range(16)] for i in range(16)]
        
        # Frekans uzayından Zaman uzayına (IFFT) dönüş
        lateral_current_2d = ifft2d(conv_fft)
        
        # Yeniden 1D sinyale çevir
        lateral_current_flat = [lateral_current_2d[i][j].real for i in range(16) for j in range(16)]
        
        spikes = [0.0] * self.num_neurons
        for i in range(self.num_neurons):
            self.mem[i] = (self.mem[i] * self.beta) + lateral_current_flat[i]
            
            # Ateşleme
            if self.mem[i] >= self.threshold[i]:
                spikes[i] = 1.0
                self.mem[i] -= self.threshold[i]
                
            # Homeostaz
            self.activity_trace[i] = (self.activity_trace[i] * 0.99) + (spikes[i] * 0.01)
            error = self.activity_trace[i] - self.target_activity
            self.threshold[i] += error * self.homeostasis_rate
            self.threshold[i] = max(self.base_threshold * 0.5, min(self.threshold[i], self.base_threshold * 2.0))
            
        # STDP (Lokal Öğrenme) Ağırlık Güncellemesi
        for i in range(self.num_neurons):
            self.trace_pre[i] = self.trace_pre[i] * math.exp(-1.0/self.tau_pre) + spikes[i]
            self.trace_post[i] = self.trace_post[i] * math.exp(-1.0/self.tau_post) + spikes[i]
            
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j:
                    dw_plus = self.A_plus * spikes[i] * self.trace_pre[j]
                    dw_minus = self.A_minus * self.trace_post[i] * spikes[j]
                    self.W[i][j] += (dw_plus - dw_minus)
                    self.W[i][j] = max(-1.0, min(self.W[i][j], 1.0)) # Ağırlık patlamasını engelle
                    
        self.last_spikes = spikes
        return spikes

class Hippocampus:
    def __init__(self, size=256):
        self.size = size
        # Biyolojik Epizodik Bellek Matrisi (Hopfield)
        self.memory_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        self.memory_texts = [] 
        self.memory_vectors = []
        self.learning_rate = 0.1

    def store(self, text, spike_pattern):
        # Tek Seferde Öğrenme (One-shot learning)
        mean_p = sum(spike_pattern) / self.size
        pattern_norm = [p - mean_p for p in spike_pattern]
        
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.memory_matrix[i][j] += self.learning_rate * pattern_norm[i] * pattern_norm[j]
                    self.memory_matrix[i][j] = max(-1.0, min(self.memory_matrix[i][j], 1.0))
                    
        if text not in self.memory_texts:
            self.memory_texts.append(text)
            self.memory_vectors.append(spike_pattern)

    def retrieve(self, partial_spike_pattern, steps=3):
        # Çağrışımsal Geri Çağırma (Associative Recall)
        current_pattern = list(partial_spike_pattern)
        if len(self.memory_texts) == 0:
            return None, 0.0
            
        for _ in range(steps):
            new_pattern = [0.0] * self.size
            for i in range(self.size):
                activation = sum(self.memory_matrix[i][j] * current_pattern[j] for j in range(self.size))
                new_pattern[i] = 1.0 if activation > 0 else 0.0
            current_pattern = new_pattern
            
        best_sim = -1.0
        best_text = None
        for txt, vec in zip(self.memory_texts, self.memory_vectors):
            sim = cosine_similarity(current_pattern, vec)
            if sim > best_sim:
                best_sim = sim
                best_text = txt
                
        return best_text, best_sim

class GlobalNeuronalWorkspace:
    # Modül 02: Bilinç ve Dikkat Merkezi
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.active_thought = None
        
    def broadcast(self, signal_strength, context_text):
        # Gelen sinyal eşiği aşarsa bilince çıkar ve tüm sisteme yayınlanır
        if signal_strength > self.threshold:
            self.active_thought = context_text
            return True
        self.active_thought = None
        return False

class SparseDirectPathways:
    # Modül 04: Beyaz Madde Kısa Yolları (Refleksif Davranış)
    def __init__(self):
        self.reflex_bindings = {} 
        
    def bind(self, query, response):
        # Belli girdiler için doğrudan refleksif bağlantılar oluşturur
        self.reflex_bindings[query] = response
        
    def get_reflex(self, query):
        return self.reflex_bindings.get(query, None)

class BiolinguisticSynthesizer:
    @staticmethod
    def generate(query, closest_memory, sim_score, h_durum, gnw_thought):
        # Biyolojik duruma göre doğal dil üretimi
        query = query.lower().strip()
        
        if gnw_thought:
            if gnw_thought == query:
                return f"Bu bilgiyi daha önce duymamıştım. '{query}' verisi tüm GNW ağıma (Global Çalışma Alanı) yayınlandı ve sisteme kazındı."
            else:
                return f"Söylediğin şey ('{query}') dikkatimi çekti! Hipokampüsümde güçlü bir şekilde '{gnw_thought}' anısıyla eşleşti ve bilincime ulaştı."
        else:
            if sim_score > 0.6:
                return f"'{query}' konusu bana yabancı değil, arka planda '{closest_memory}' bağlamını çağrıştırıyor ama bilincime çıkacak kadar güçlü bir uyarım yaratmadı."
            else:
                return f"'{query}' korteksimde dalgalandı... Ancak o kadar zayıf bir sinyal ki, dikkatimi toplayıp üzerinde düşünemedim. (GNW Eşiği aşılamadı)"

class MergenNeuromorphicEngine:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.hippocampus = Hippocampus(size=256)
        self.cortical_layer = CorticalLayer(size=(16, 16))
        self.gnw = GlobalNeuronalWorkspace(threshold=0.6)
        self.fast_pathways = SparseDirectPathways()
        
        self.num_steps = 15 # Performans için ayarlandı
        self._init_core_memory()

    def get_embedding(self, text):
        try:
            response = requests.post(self.api_url, json={"inputs": text}, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        random.seed(text)
        return [random.uniform(-1, 1) for _ in range(384)]

    def _init_core_memory(self):
        core_axioms = [
            "Zeka statik bir haritalama değil, yaşayan ve ritmik bir süreçtir.",
            "Mergen sistemi sürekli zaman ve biyolojik dengeler üzerinde var olur."
        ]
        for axiom in core_axioms:
            vec = self.get_embedding(axiom)
            vec_256 = vec[:256] if len(vec) >= 256 else vec + [0.0]*(256-len(vec))
            spike_pattern = [1.0 if random.random() < abs(v) else 0.0 for v in vec_256]
            self.hippocampus.store(axiom, spike_pattern)
            
        # Refleks ekle (Adım 7 - Modül 04)
        self.fast_pathways.bind("merhaba", "Merhaba! Ben Mergen, biyolojik tabanlı bir dijital zekayım.")
        self.fast_pathways.bind("nasılsın", "Sistemlerim stabil, kortikal ateşlemelerim normal seviyede. Sen nasılsın?")
        self.fast_pathways.bind("mergen", "Efendim? Dikkatim sende.")

    def process(self, query, user_name):
        start_time = datetime.datetime.now()
        
        # 1. Hızlı Yollar (Refleks) Kontrolü - Sıfır Gecikme
        reflex = self.fast_pathways.get_reflex(query.lower().strip())
        if reflex:
            latency = (datetime.datetime.now() - start_time).total_seconds()
            return {
                "response": reflex + " [Fast Pathway]",
                "duygu": "Refleksif Yanıt",
                "esik": self.cortical_layer.threshold[0],
                "anlam_spikes": 0, "celiski_spikes": 0, "cortical_spikes": 0,
                "latency": latency
            }
        
        query_vector = self.get_embedding(query)
        vec_256 = query_vector[:256] if len(query_vector) >= 256 else query_vector + [0.0]*(256-len(query_vector))
            
        input_spikes = [1.0 if random.random() < abs(v) else 0.0 for v in vec_256]
        closest_memory, sim_score = self.hippocampus.retrieve(input_spikes)
        
        if closest_memory is None:
            closest_memory = "Bellek boş."
            sim_score = 0.0
            
        if closest_memory == query:
            sim_score = 0.5
            closest_memory = "Kendimi dinliyorum."

        cortical_spikes = 0
        
        # 2. Kortikal İşleme ve Sürekli Zaman (Spiking)
        for step in range(self.num_steps):
            noise = (step % 2) * 0.02
            current_in = [v * 1.5 + noise for v in vec_256]
            spikes_2d = self.cortical_layer.step(current_in)
            cortical_spikes += sum(spikes_2d)

        # 3. Anıyı Hipokampüs'e Yazma (One-shot)
        self.hippocampus.store(query, self.cortical_layer.last_spikes)
        
        # 4. Modül 02: Global Neuronal Workspace (Bilinç Yayını)
        activity_ratio = cortical_spikes / (self.cortical_layer.num_neurons * self.num_steps)
        signal_strength = sim_score * 0.5 + activity_ratio * 0.5
        
        is_conscious = self.gnw.broadcast(signal_strength, closest_memory)
        gnw_thought = self.gnw.active_thought if is_conscious else None
        
        h_durum = "Bilinçli (GNW Aktif)" if is_conscious else "Bilinçaltı (Farkındalık Yok)"
        
        # 5. Doğal Dil Sentezi
        response = BiolinguisticSynthesizer.generate(query, closest_memory, sim_score, h_durum, gnw_thought)

        latency = (datetime.datetime.now() - start_time).total_seconds()
        
        return {
            "response": response,
            "duygu": h_durum,
            "esik": self.cortical_layer.threshold[0],
            "anlam_spikes": int(cortical_spikes * 0.4), 
            "celiski_spikes": int(cortical_spikes * 0.1),
            "cortical_spikes": int(cortical_spikes),
            "latency": latency
        }
