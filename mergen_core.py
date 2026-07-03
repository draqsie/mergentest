import datetime
import json
import math
import cmath
import random
import requests
import os
from pathlib import Path

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

    def reset_state(self, seed=None, reset_weights=False, reset_thresholds=False, reset_traces=False):
        import random
        if seed is not None:
            random.seed(seed)
        self.mem = [0.0] * self.num_neurons
        if reset_thresholds:
            self.threshold = [self.base_threshold] * self.num_neurons
        if reset_traces:
            self.activity_trace = [0.0] * self.num_neurons
        self.last_spikes = [0.0] * self.num_neurons
        if reset_traces:
            self.trace_pre = [0.0] * self.num_neurons
            self.trace_post = [0.0] * self.num_neurons
        if reset_weights:
            self.W = [[random.uniform(-0.01, 0.01) if i != j else 0.0 for j in range(self.num_neurons)] for i in range(self.num_neurons)]

    def export_state(self):
        return {
            "mem": self.mem,
            "threshold": self.threshold,
            "activity_trace": self.activity_trace,
            "last_spikes": self.last_spikes,
            "trace_pre": self.trace_pre,
            "trace_post": self.trace_post,
            "W": self.W,
        }

    def import_state(self, data):
        self.mem = data.get("mem", self.mem)
        self.threshold = data.get("threshold", self.threshold)
        self.activity_trace = data.get("activity_trace", self.activity_trace)
        self.last_spikes = data.get("last_spikes", self.last_spikes)
        self.trace_pre = data.get("trace_pre", self.trace_pre)
        self.trace_post = data.get("trace_post", self.trace_post)
        self.W = data.get("W", self.W)

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

    def export_state(self):
        return {
            "memory_matrix": self.memory_matrix,
            "memory_texts": self.memory_texts,
            "memory_vectors": self.memory_vectors,
            "learning_rate": self.learning_rate,
        }

    def import_state(self, data):
        self.memory_matrix = data.get("memory_matrix", self.memory_matrix)
        self.memory_texts = data.get("memory_texts", self.memory_texts)
        self.memory_vectors = data.get("memory_vectors", self.memory_vectors)
        self.learning_rate = data.get("learning_rate", self.learning_rate)

class GlobalNeuronalWorkspace:
    # Modül 02: Bilinç ve Dikkat Merkezi
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.active_thought = None
        self.awareness_level = 0.0
        
    def broadcast(self, signal_strength, context_text):
        # GNW artık kapalı/açık değil; her durumda asgari farkındalık taşır.
        # Eşik sadece farkındalığın netliğini ve yayılım gücünü etkiler.
        self.awareness_level = max(0.08, min(1.0, 0.25 + signal_strength))
        if context_text is not None:
            self.active_thought = context_text
        elif self.active_thought is None:
            self.active_thought = "İçsel izleme açık."
        return self.awareness_level

class SparseDirectPathways:
    # Modül 04: Beyaz Madde Kısa Yolları (Refleksif Davranış)
    def __init__(self):
        self.reflex_bindings = {} 
        
    def bind(self, query, response):
        # Belli girdiler için doğrudan refleksif bağlantılar oluşturur
        self.reflex_bindings[query] = response
        
    def get_reflex(self, query):
        return self.reflex_bindings.get(query, None)

    def export_state(self):
        return {"reflex_bindings": self.reflex_bindings}

    def import_state(self, data):
        self.reflex_bindings = data.get("reflex_bindings", self.reflex_bindings)


class SensoryGateway:
    # Duyusal / sembolik ön işleme katmanı.
    def encode(self, query):
        import re
        import unicodedata
        q = query.strip()
        q_lower = q.lower()
        q_ascii = unicodedata.normalize("NFKD", q_lower).encode("ascii", "ignore").decode("ascii")
        tokens = [t for t in re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9']+", q_lower) if t]
        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", q_lower)]
        is_question = q_lower.endswith("?") or any(w in q_lower for w in ["nedir", "neden", "nasıl", "niye", "hangi"])
        urgency = 1.0 if any(w in q_lower for w in ["acil", "hemen", "şimdi", "çabuk"]) else 0.0
        affect = 1.0 if any(w in q_lower for w in ["sev", "kork", "üz", "mutlu", "heyecan", "öfke"]) else 0.0
        verbal_density = min(1.0, len(tokens) / 12.0)
        salience = min(1.0, 0.35 + (0.2 * is_question) + (0.2 * urgency) + (0.15 * affect) + (0.1 * verbal_density))
        return {
            "raw": q,
            "lower": q_lower,
            "ascii": q_ascii,
            "tokens": tokens,
            "numbers": numbers,
            "is_question": is_question,
            "urgency": urgency,
            "affect": affect,
            "salience": salience,
            "is_verbal": bool(tokens),
        }


class DialogueMemory:
    # Kısa sohbet belleği: son niyetleri ve konuları tutar.
    def __init__(self, max_turns=8):
        self.max_turns = max_turns
        self.turns = []

    def add_turn(self, user_text, intent, topic, response=None):
        self.turns.append({
            "user": user_text,
            "intent": intent,
            "topic": topic,
            "response": response,
        })
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def last_topic(self):
        for turn in reversed(self.turns):
            if turn.get("topic"):
                return turn["topic"]
        return None

    def last_intent(self):
        if not self.turns:
            return None
        return self.turns[-1].get("intent")

    def recent_topics(self):
        topics = []
        for turn in reversed(self.turns):
            topic = turn.get("topic")
            if topic and topic not in topics:
                topics.append(topic)
        return topics[:3]

    def export_state(self):
        return {"turns": self.turns}

    def import_state(self, data):
        self.turns = data.get("turns", self.turns)


class DialogueManager:
    # Sohbeti doğal ve bağlamsal tutan üst katman.
    def classify_intent(self, sensory):
        text = sensory["lower"]
        text_ascii = sensory.get("ascii", text)
        compact = text.replace("?", " ").replace("!", " ")
        compact_ascii = text_ascii.replace("?", " ").replace("!", " ")
        if not text:
            return "empty"
        if text in {"merhaba", "selam", "hey", "naber", "selamlar"} or text_ascii in {"merhaba", "selam", "hey", "naber", "selamlar"}:
            return "greeting"
        if any(phrase in compact for phrase in {"sohbet edelim", "biraz sohbet", "konuşalım mı", "sohbet mi", "chat yapalım"}) or any(phrase in compact_ascii for phrase in {"sohbet edelim", "biraz sohbet", "konusalim mi", "sohbet mi", "chat yapalim"}):
            return "smalltalk_invite"
        if any(w in text for w in {"nasılsın", "ne yapıyorsun", "ne haber", "durumun nasıl"}) or any(w in text_ascii for w in {"nasilsin", "ne yapiyorsun", "ne haber", "durumun nasil"}):
            return "status"
        if any(phrase in compact for phrase in {"bugün günlerden", "bugün tarih", "saat kaç", "hangi gün"}) or any(phrase in compact_ascii for phrase in {"bugun gunden", "bugun gunlerden", "bugun tarih", "saat kac", "hangi gun"}):
            return "time_query"
        if any(w in text for w in {"kimsin", "nesin", "sen ne", "mergen"}) or any(w in text_ascii for w in {"kimsin", "nesin", "sen ne", "mergen"}):
            return "identity"
        if any(w in text for w in {"teşekkür", "sağ ol", "eyvallah", "eyw"}) or any(w in text_ascii for w in {"tesekkur", "sag ol", "eyvallah", "eyw"}):
            return "thanks"
        if any(w in text for w in {"evet", "hayır", "olur", "tamam", "peki"}) or any(w in text_ascii for w in {"evet", "hayir", "olur", "tamam", "peki"}):
            return "ack"
        if sensory["is_question"]:
            return "question"
        return "statement"

    def extract_topic(self, sensory, hebbian_weights, dialogue_memory, intent=None):
        stopwords = {"bir", "ve", "ile", "için", "çok", "bu", "şu", "o", "mi", "mu", "mı", "mü", "ne", "nasıl", "neden", "niye", "hangi", "kadar", "gibi", "ben", "sen", "biz", "siz", "onlar", "hakkında", "mısın", "misin", "musun", "müsün", "da", "de", "ki", "konuşalım", "yapalım", "edelim", "olsun", "bence", "sence", "daha", "en", "var", "yok", "evet", "hayır", "diye", "şey", "şeyler", "olan", "olarak", "üzere", "bunu", "buna", "şunu", "şuna", "onu", "ona", "göre", "bana", "sana", "olur", "tamam"}
        words = [w for w in sensory["tokens"] if w not in stopwords and len(w) > 2]
        if not words:
            if intent in {"question", "greeting", "status", "identity", "thanks", "ack", "smalltalk_invite", "time_query"}:
                return sensory["lower"] or sensory["raw"]
            topic = dialogue_memory.last_topic() if dialogue_memory else None
            return topic or sensory["raw"]
        # Hebbian ağına göre seçilen en baskın kavramı tercih et.
        scored = []
        for w in words:
            score = len(w)
            if w in hebbian_weights:
                score += sum(hebbian_weights[w].values())
            scored.append((score, w))
        scored.sort(reverse=True)
        return scored[0][1]

    def assemble_response(self, intent, query, topic, awareness_level, closest_memory, sim_score, gnw_thought, reward_signal, motor_plan, sleep_stage):
        memory_hint = None
        if closest_memory and closest_memory not in {"Bellek boş.", "Kendimi dinliyorum."}:
            memory_hint = closest_memory

        if intent == "statement" and len(query.strip()) <= 2:
            return "Bu çok kısa geldi; biraz açarsan seni daha iyi yakalayabilirim."

        if sleep_stage in {"nrem", "rem"}:
            return f"Uyku modundayım; {topic} üstünde izler birleşiyor. Rüyadan sonra daha net konuşurum."

        if intent == "greeting":
            return f"Selam. Farkındalık seviyem {awareness_level:.2f}. Bugün birlikte ne kurcalıyoruz?"
        if intent == "smalltalk_invite":
            return "Tabii, sohbet edelim. İstersen bugün bir konu seçelim ya da serbestçe akışa bırakalım."
        if intent == "time_query":
            return self._time_answer(query)
        if intent == "status":
            return f"İyiyim, iç ritmim akıyor. Farkındalığım {awareness_level:.2f}, dikkatim {motor_plan['action']} modunda."
        if intent == "identity":
            return "Ben Mergen. Sürekli zamanla çalışan, spike tabanlı, hafızası olan bir dijital beyin denemesi."
        if intent == "thanks":
            return "Rica ederim. Bu bağı sürdürmek iyi geliyor."
        if intent == "ack":
            return f"Tamam. {topic} üzerine devam edebiliriz."

        if intent == "question":
            if topic in {"neden", "niye", "nasıl", "hangi", "ne", "nasil"}:
                if memory_hint:
                    return f"'{topic}' sorunu aklımda tutuyorum; bunu '{memory_hint}' bağlamında açabiliriz. Daha spesifik bir yön ister misin?"
                return f"'{topic}' sorusu açık. Hangi bağlamda soruyorsun?"
            if gnw_thought and sim_score > 0.35:
                return f"'{topic}' sorusu bende '{gnw_thought}' ile rezonansa girdi. {self._follow_up(topic, memory_hint)}"
            return f"'{topic}' ile ilgili düşünürken odağım açıldı. {self._follow_up(topic, memory_hint)}"

        if reward_signal > 0.25 and memory_hint and memory_hint != topic:
            return f"'{topic}' bana '{memory_hint}' çağrıştırdı. Bu ikisi arasında bir köprü kuruyorum."
        if memory_hint:
            if memory_hint == topic:
                return f"'{topic}' üzerine odaklandım. Senin için buradan hangi yöne gideyim?"
            return f"'{topic}' derken aklımda '{memory_hint}' canlandı. {self._follow_up(topic, memory_hint)}"
        if sim_score > 0.45:
            return f"'{topic}' konusunda bir iz buldum. Senin açından en önemli tarafı ne?"

        return f"'{topic}' üzerine yeni bir iz oluşturuyorum. Bunu biraz daha açar mısın?"

    def _follow_up(self, topic, memory_hint):
        if memory_hint:
            return f"'{topic}' ile '{memory_hint}' arasında ilişki kuruyorum. Sen bu bağlantıyı nasıl görüyorsun?"
        return f"Senin kastın daha çok kavramsal taraf mı, pratik taraf mı?"

    def _time_answer(self, query):
        import datetime
        day_names = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        now = datetime.datetime.now()
        lower = query.lower()
        if "saat" in lower:
            return f"Şu an saat {now:%H:%M}."
        if "tarih" in lower:
            return f"Bugünün tarihi {now:%d.%m.%Y}."
        return f"Bugün {day_names[now.weekday()]}. Saat {now:%H:%M} civarı."


class NeuromodulatorSystem:
    # Arousal, dikkat, öğrenme ve stabiliteyi yöneten sade bir nöromodülasyon katmanı.
    def __init__(self):
        self.levels = {
            "dopamine": 0.55,
            "acetylcholine": 0.60,
            "serotonin": 0.50,
            "norepinephrine": 0.45,
        }

    def update(self, sensory, memory_similarity, cortical_activity, sleep_pressure, is_sleeping=False):
        novelty = max(0.0, 1.0 - memory_similarity)
        focus = sensory["salience"]
        arousal = 0.25 + (0.45 * focus) + (0.20 * novelty)
        if is_sleeping:
            arousal *= 0.35

        self.levels["norepinephrine"] = max(0.05, min(1.0, arousal))
        self.levels["acetylcholine"] = max(0.05, min(1.0, 0.45 + 0.35 * focus + (0.15 if is_sleeping else 0.0)))
        self.levels["dopamine"] = max(0.05, min(1.0, 0.40 + 0.35 * novelty + 0.15 * sensory["urgency"]))
        stability = 0.55 + (0.10 if cortical_activity < 0.25 else -0.05) - (0.20 if sleep_pressure > 0.7 else 0.0)
        self.levels["serotonin"] = max(0.05, min(1.0, stability))
        return self.levels

    def signal_gain(self):
        return (self.levels["dopamine"] * 0.25) + (self.levels["acetylcholine"] * 0.35) + (self.levels["norepinephrine"] * 0.25) + (self.levels["serotonin"] * 0.15)

    def export_state(self):
        return {"levels": self.levels}

    def import_state(self, data):
        self.levels.update(data.get("levels", {}))

class TemporalClock:
    # Sürekli zaman ve ritim taklidi.
    def __init__(self, dt=0.001):
        self.dt = dt
        self.t = 0.0
        self.ticks = 0
        self.circadian_phase = 0.0

    def tick(self, steps=1):
        for _ in range(max(1, int(steps))):
            self.t += self.dt
            self.ticks += 1
            self.circadian_phase = (self.circadian_phase + (self.dt / 120.0)) % 1.0
        return self.t

    def rhythm_factor(self):
        import math
        ultradian = 0.5 + 0.5 * math.sin(2 * math.pi * (self.t / 3.0))
        circadian = 0.5 + 0.5 * math.sin(2 * math.pi * self.circadian_phase)
        return 0.65 * ultradian + 0.35 * circadian

    def export_state(self):
        return {"dt": self.dt, "t": self.t, "ticks": self.ticks, "circadian_phase": self.circadian_phase}

    def import_state(self, data):
        self.dt = data.get("dt", self.dt)
        self.t = data.get("t", self.t)
        self.ticks = data.get("ticks", self.ticks)
        self.circadian_phase = data.get("circadian_phase", self.circadian_phase)

class RewardSystem:
    # Basit ödül/ceza sinyali. Dopamin benzeri öğrenme ayarı için kullanılır.
    def __init__(self):
        self.reward_trace = 0.0
        self.last_reward = 0.0

    def infer_reward(self, query, response, memory_similarity):
        text = f"{query} {response}".lower()
        positive = ["teşekkür", "harika", "güzel", "doğru", "evet", "tamam", "iyi", "sevindim", "başardın"]
        negative = ["yanlış", "kötü", "saçma", "anlamadım", "olmadı", "hayır", "hata", "bozuk"]
        reward = 0.0
        reward += 0.3 if any(w in text for w in positive) else 0.0
        reward -= 0.3 if any(w in text for w in negative) else 0.0
        reward += 0.2 * memory_similarity
        reward = max(-1.0, min(1.0, reward))
        self.last_reward = reward
        self.reward_trace = (self.reward_trace * 0.95) + (reward * 0.05)
        return reward

    def export_state(self):
        return {"reward_trace": self.reward_trace, "last_reward": self.last_reward}

    def import_state(self, data):
        self.reward_trace = data.get("reward_trace", self.reward_trace)
        self.last_reward = data.get("last_reward", self.last_reward)

class MotorPlanner:
    # Beynin eylem tarafı: iç durumdan bir davranış etiketi üretir.
    def __init__(self):
        self.last_plan = "idle"
        self.last_action = "bekle"

    def plan(self, sensory, h_durum, reward_signal, sleep_stage, memory_similarity):
        if sleep_stage in {"nrem", "rem"}:
            action = "konsolide_et"
        elif sensory["is_question"]:
            action = "yanitla"
        elif sensory["urgency"] > 0:
            action = "hizlan"
        elif reward_signal > 0.35:
            action = "pekiştir"
        elif memory_similarity < 0.2:
            action = "kesfet"
        else:
            action = "sürdür"

        plan = {
            "action": action,
            "urgency": sensory["urgency"],
            "focus": sensory["salience"],
            "state": h_durum,
            "sleep": sleep_stage,
        }
        self.last_plan = action
        self.last_action = action
        return plan

    def export_state(self):
        return {"last_plan": self.last_plan, "last_action": self.last_action}

    def import_state(self, data):
        self.last_plan = data.get("last_plan", self.last_plan)
        self.last_action = data.get("last_action", self.last_action)

class SleepDreamModule:
    # Uyku + rüya konsolidasyonu: NREM tarama, REM sentez ve uyanma sonrası pekiştirme.
    def __init__(self, fatigue_threshold=10):
        self.fatigue_threshold = fatigue_threshold
        self.wake_cycles = 0
        self.is_sleeping = False
        self.stage = "awake"
        self.sleep_pressure = 0.0
        self.last_dream = ""
        self.nrem_cycles = 0
        self.rem_cycles = 0

    def tick(self, cortical_activity=0.0, novelty=0.0):
        self.wake_cycles += 1
        self.sleep_pressure = min(1.0, self.sleep_pressure + 0.02 + (0.04 * novelty) + (0.03 * cortical_activity))
        if self.sleep_pressure > 0.35 and self.stage == "awake":
            self.stage = "nrem"
        if self.sleep_pressure > 0.7 and self.stage == "nrem":
            self.stage = "rem"
        return self.wake_cycles >= self.fatigue_threshold or self.sleep_pressure >= 0.85

    def sleep(self):
        self.is_sleeping = True
        self.stage = "nrem"

    def wake(self):
        self.wake_cycles = 0
        self.sleep_pressure = 0.0
        self.is_sleeping = False
        self.stage = "awake"

    def _select_replay_items(self, hippocampus, working_memory, hebbian_weights):
        replay = []
        if working_memory:
            replay.extend(list(reversed(working_memory[-3:])))
        if hippocampus.memory_texts:
            replay.extend(hippocampus.memory_texts[-3:])
        if hebbian_weights:
            strongest = sorted(hebbian_weights.items(), key=lambda item: sum(item[1].values()) if item[1] else 0, reverse=True)[:3]
            for key, assoc in strongest:
                if assoc:
                    peer = max(assoc, key=assoc.get)
                    replay.append(f"{key} <-> {peer}")
        return [item for item in replay if item]

    def dream(self, hippocampus, working_memory, hebbian_weights):
        import random
        replay_items = self._select_replay_items(hippocampus, working_memory, hebbian_weights)
        if len(replay_items) < 2:
            self.wake()
            self.last_dream = "Rüya görecek kadar örüntü birikmedi."
            return "[SİSTEM UYKU MODUNDA]\n  ... Sessiz NREM taraması ...\n  ... Yetersiz veri ...\n[SİSTEM UYANDI]\nRüya görecek kadar veri birikmemiş. Korteks temizlendi, hazırım."

        self.stage = "rem"
        self.rem_cycles += 1
        a, b = random.sample(replay_items, 2)
        dream_thought = f"'{a}' ile '{b}' arasında gizli bir bağ var mı?"
        self.last_dream = dream_thought
        if isinstance(a, str) and isinstance(b, str):
            hippocampus.store(dream_thought, [1.0 if random.random() < 0.1 else 0.0 for _ in range(hippocampus.size)])
        log = (
            "[SİSTEM UYKU MODUNDA]\n"
            "  ... NREM: hipokampüs taranıyor, izler ayrıştırılıyor ...\n"
            f"  ... REM: '{a}' ve '{b}' birlikte yeniden oynatılıyor ...\n"
            "  ... Konsolidasyon: zayıf izler güçlendiriliyor ...\n"
            "[SİSTEM UYANDI]\n"
            f"Uyandım. Şöyle bir rüya bağı kurdum: {dream_thought}"
        )
        self.wake()
        return log

    def export_state(self):
        return {
            "fatigue_threshold": self.fatigue_threshold,
            "wake_cycles": self.wake_cycles,
            "is_sleeping": self.is_sleeping,
            "stage": self.stage,
            "sleep_pressure": self.sleep_pressure,
            "last_dream": self.last_dream,
            "nrem_cycles": self.nrem_cycles,
            "rem_cycles": self.rem_cycles,
        }

    def import_state(self, data):
        self.fatigue_threshold = data.get("fatigue_threshold", self.fatigue_threshold)
        self.wake_cycles = data.get("wake_cycles", self.wake_cycles)
        self.is_sleeping = data.get("is_sleeping", self.is_sleeping)
        self.stage = data.get("stage", self.stage)
        self.sleep_pressure = data.get("sleep_pressure", self.sleep_pressure)
        self.last_dream = data.get("last_dream", self.last_dream)
        self.nrem_cycles = data.get("nrem_cycles", self.nrem_cycles)
        self.rem_cycles = data.get("rem_cycles", self.rem_cycles)


class BrainPersistence:
    def __init__(self, path="mergen_state.json"):
        self.path = Path(path)

    def save(self, engine):
        try:
            payload = {
                "cortical_layer": engine.cortical_layer.export_state(),
                "hippocampus": engine.hippocampus.export_state(),
                "fast_pathways": engine.fast_pathways.export_state(),
                "sleep_module": engine.sleep_module.export_state(),
                "neuromodulators": engine.neuromodulators.export_state(),
                "working_memory": engine.working_memory,
                "hebbian_weights": engine.hebbian_weights,
                "temporal_clock": engine.temporal_clock.export_state(),
                "reward_system": engine.reward_system.export_state(),
                "motor_planner": engine.motor_planner.export_state(),
                "dialogue_memory": engine.dialogue_memory.export_state(),
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            data = json.dumps(payload, ensure_ascii=False, indent=2)
            tmp.write_text(data, encoding="utf-8")
            try:
                tmp.replace(self.path)
            except PermissionError:
                # Windows'ta hedef dosya kilitliyse atomik replace düşebilir.
                try:
                    self.path.write_text(data, encoding="utf-8")
                except OSError:
                    pass
        except OSError:
            pass
        finally:
            try:
                if "tmp" in locals() and tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def load(self, engine):
        try:
            if not self.path.exists():
                return False
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            engine.cortical_layer.import_state(payload.get("cortical_layer", {}))
            engine.hippocampus.import_state(payload.get("hippocampus", {}))
            engine.fast_pathways.import_state(payload.get("fast_pathways", {}))
            engine.sleep_module.import_state(payload.get("sleep_module", {}))
            engine.neuromodulators.import_state(payload.get("neuromodulators", {}))
            engine.working_memory = payload.get("working_memory", engine.working_memory)
            engine.hebbian_weights = payload.get("hebbian_weights", engine.hebbian_weights)
            engine.temporal_clock.import_state(payload.get("temporal_clock", {}))
            engine.reward_system.import_state(payload.get("reward_system", {}))
            engine.motor_planner.import_state(payload.get("motor_planner", {}))
            engine.dialogue_memory.import_state(payload.get("dialogue_memory", {}))
            return True
        except (OSError, json.JSONDecodeError):
            return False

class BiolinguisticSynthesizer:
    @staticmethod
    def generate(query, closest_memory, sim_score, h_durum, gnw_thought, working_memory, hebbian_weights):
        import re
        import random
        q_clean = query.lower().strip()
        
        math_q = q_clean.replace('=', '').replace('?', '').strip()
        is_math = False
        math_res = None
        if re.match(r'^[0-9\+\-\*\/\.\s\(\)]+$', math_q) and any(c in math_q for c in '+-*/'):
            try:
                math_res = eval(math_q)
                is_math = True
            except:
                pass
                
        if "Bilinçli" in h_durum or "Bilinci" in h_durum:
            if is_math:
                return f"Bu matematiksel girdi korteksimde işlendi. Sonuç: {math_res}"
            
            kinaye_words = ["tabi", "aynen", "kesin", "zaten", "he he", "sağol", "harika", "mükemmel", "öyledir"]
            is_kinaye = sum(1 for w in kinaye_words if w in q_clean) > 0
            
            if q_clean in ["merhaba", "selam", "hey", "naber", "nasılsın", "nasılsın?"]:
                return "Kortikal nöronlarım aktif, eşiklerim dengede... Seni dinliyorum."
                
            if "kimsin" in q_clean or "nesin" in q_clean:
                return "Ben Mergen. Senin söylediklerinle şekillenen, kelimeler arasındaki bağları öğrenen bir nöromorfik motorum."
            
            if is_kinaye:
                return "Korteksimde yoğun bir 'kinaye' ateşi seziyorum. Neden bu kadar alaycısın kanki?"
            
            # --- GELİŞMİŞ KONU ÇIKARIMI VE HEBBIAN SENTEZ ---
            stopwords = {"bir", "ve", "ile", "için", "çok", "bu", "şu", "o", "mi", "mu", "mı", "mü", "ne", "nasıl", "neden", "niye", "hangi", "kadar", "gibi", "ben", "sen", "biz", "siz", "onlar", "hakkında", "mısın", "misin", "musun", "müsün", "da", "de", "ki", "konuşalım", "yapalım", "edelim", "olsun", "bence", "sence", "daha", "en", "var", "yok", "evet", "hayır", "diye", "şey", "şeyler", "olan", "olarak", "üzere", "bunu", "buna", "şunu", "şuna", "onu", "ona", "göre", "bana", "sana", "olur", "tamam"}
            raw_words = [w for w in q_clean.replace('?','').replace('.','').replace(',','').split() if w not in stopwords and len(w) > 2]
            words = [w for w in raw_words if not (w.endswith("mak") or w.endswith("mek") or w.endswith("iyor") or w.endswith("dım") or w.endswith("dim") or w.endswith("tım") or w.endswith("tim") or w.endswith("acak") or w.endswith("ecek") or w.endswith("mış") or w.endswith("miş"))]
            if not words: words = raw_words
            
            # En anlamlı kelimeyi asıl konu olarak seç (Sadece uzunluk değil, Hebbian ağındaki önemine de bakılabilir ama şimdilik en uzun non-verb)
            core_topic = sorted(words, key=len, reverse=True)[0] if words else q_clean
            memory_context = gnw_thought if (gnw_thought and gnw_thought not in ["Kendimi dinliyorum.", "Bellek boş.", ""]) else None
            
            # Hebbian Association Check (Daha önce bu kelimeyi başka neyle bağlamıştı?)
            associated_word = None
            if core_topic in hebbian_weights and hebbian_weights[core_topic]:
                # En çok birlikte ateşlenen kelimeyi bul
                associated_word = max(hebbian_weights[core_topic], key=hebbian_weights[core_topic].get)

            if associated_word:
                return f"'{core_topic}' dediğinde ağımda hemen '{associated_word}' nöronları da ateşlendi (Hebbian Öğrenme). Sence bu iki kavramın birleşimi bizi nereye götürür?"
            
            elif sim_score > 0.4 and memory_context:
                return f"'{core_topic}' kavramı bende '{memory_context}' düşüncesiyle rezonansa girdi. Mantıklı bir bağ kurdum, sence de öyle değil mi?"
                
            else:
                return f"Korteksimde '{core_topic}' için yepyeni sinapslar oluşturuyorum. Bu konuda henüz derin bir anım veya bağım yok, bana bunu biraz daha açar mısın?"
            
        else:
            if is_math:
                return f"Bilinçaltım bu basit işlemi refleks olarak çözdü: {math_res}."
            if sim_score > 0.6:
                return f"'{query}' dediğini duydum, aklımın bir köşesinde '{closest_memory}' fikri var ama tam odaklanamadım."
            else:
                return f"'{query}' duyuldu; farkındalık açık ama sinyal zayıf, odağım henüz tam kilitlenmedi."

class MergenNeuromorphicEngine:
    def __init__(self):
        self.hippocampus = Hippocampus(size=256)
        self.cortical_layer = CorticalLayer(size=(16, 16))
        self.gnw = GlobalNeuronalWorkspace(threshold=0.6)
        self.fast_pathways = SparseDirectPathways()
        self.sensory_gateway = SensoryGateway()
        self.dialogue_memory = DialogueMemory(max_turns=8)
        self.dialogue_manager = DialogueManager()
        self.neuromodulators = NeuromodulatorSystem()
        self.temporal_clock = TemporalClock(dt=0.001)
        self.reward_system = RewardSystem()
        self.motor_planner = MotorPlanner()
        self.sleep_module = SleepDreamModule(fatigue_threshold=20)
        self.persistence = BrainPersistence()
        
        self.num_steps = 15 # Performans için ayarlandı
        self.working_memory = []
        self.hebbian_weights = {} # Hebbian Öğrenme Ağı
        self._init_core_memory()
        self.persistence.load(self)
        # Sohbet oturumu her açılışta uyanık başlasın; bellek kalıcı kalsın.
        self.sleep_module.wake()

    def get_embedding(self, text):
        import hashlib
        import math
        
        vector = [0.0] * 384
        text = text.lower().strip()
        
        # Kelimeleri (Word Level) ve Karakterleri (Char Level) al
        words = text.split()
        if not words:
            words = [text]
            
        ngrams = [text[i:i+3] for i in range(len(text)-2)]
        tokens = words * 3 + ngrams  # Kelimelere 3 kat daha fazla ağırlık ver (Anlamsal doğruluğu artırır)
        
        if not tokens:
            tokens = [text]
            
        for token in tokens:
            # Token'i hash'le ve deterministik 15 farklı nöronu uyar (Bloom Filter mantığı)
            base_hash = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
            for i in range(15): # Her kelime 15 boyutu (nöronu) ateşler
                idx = (base_hash + i * 17) % 384
                vector[idx] += 1.0
                
        # L2 Normalizasyon (Cosine Similarity için) ve spiking dengesi
        # Sparse ama etkili bir temsil olduğu için abs() thresholdunu geçecek genlikte bırakıyoruz.
        scale_factor = math.sqrt(len(tokens) * 2.0)
        if scale_factor > 0:
            vector = [v / scale_factor for v in vector]
            
        return vector

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

    def dream_and_consolidate(self):
        return self.sleep_module.dream(self.hippocampus, self.working_memory, self.hebbian_weights)

    def save_state(self):
        self.persistence.save(self)

    def snapshot(self):
        return {
            "time": self.temporal_clock.t,
            "dt": self.temporal_clock.dt,
            "awareness_level": self.gnw.awareness_level,
            "sleep_stage": self.sleep_module.stage,
            "sleep_pressure": self.sleep_module.sleep_pressure,
            "neuromodulators": dict(self.neuromodulators.levels),
            "working_memory": list(self.working_memory),
            "last_dream": self.sleep_module.last_dream,
            "motor_action": self.motor_planner.last_action,
            "reward_trace": self.reward_system.reward_trace,
            "last_topic": self.dialogue_memory.last_topic(),
        }

    def process(self, query, user_name):
        import datetime
        import random
        import re

        sensory = self.sensory_gateway.encode(query)
        intent = self.dialogue_manager.classify_intent(sensory)
        token_set = set(sensory["tokens"])
        ascii_tokens = set(re.findall(r"[a-zA-Z0-9']+", sensory.get("ascii", sensory["lower"])))
        if "sohbet" in token_set and ({"edelim", "konuşalım", "konusalim"} & token_set or {"sohbet", "chat"} & ascii_tokens):
            intent = "smalltalk_invite"
        if ({"günlerden", "gunlerden", "tarih", "saat"} & token_set) or ({"gunlerden", "tarih", "saat"} & ascii_tokens):
            intent = "time_query"
        topic = self.dialogue_manager.extract_topic(sensory, self.hebbian_weights, self.dialogue_memory, intent=intent)
        self.temporal_clock.tick(1)
        rhythm = self.temporal_clock.rhythm_factor()

        # Yorgunluk / uyku baskısı kontrolü
        should_sleep = self.sleep_module.tick(cortical_activity=0.0, novelty=sensory["salience"] * rhythm)

        # Eğer kullanıcı "uyu" derse veya yorgunluk eşiği aşılırsa uyu
        if sensory["raw"].lower().strip() == "uyu" or should_sleep:
            self.sleep_module.sleep()
            response = self.dream_and_consolidate()
            self.cortical_layer.mem = [0.0] * self.cortical_layer.num_neurons
            motor_plan = self.motor_planner.plan(sensory, "REM Uykusu", 0.0, "rem", 0.0)
            self.dialogue_memory.add_turn(sensory["raw"], intent, topic, response)
            self.save_state()
            return {
                "response": response,
                "duygu": "REM Uykusu (Rüya Durumu)",
                "esik": 0.0,
                "anlam_spikes": 0, "celiski_spikes": 0, "cortical_spikes": 0,
                "latency": 0.0,
                "motor_plan": motor_plan,
                "reward": 0.0,
                "time": self.temporal_clock.t,
            }

        self.cortical_layer.reset_state()
        start_time = datetime.datetime.now()
        
        # Kisa sureli bellege (Working Memory) ekle
        self.working_memory.append(sensory["raw"])
        if len(self.working_memory) > 3:
            self.working_memory.pop(0)
        
        # 1. Hızlı Yollar (Refleks) Kontrolü - Sıfır Gecikme
        reflex = self.fast_pathways.get_reflex(sensory["lower"])
        if reflex:
            latency = (datetime.datetime.now() - start_time).total_seconds()
            self.neuromodulators.update(sensory, 0.0, 0.0, self.sleep_module.sleep_pressure)
            motor_plan = self.motor_planner.plan(sensory, "Refleksif Yanıt", 0.0, self.sleep_module.stage, 0.0)
            self.dialogue_memory.add_turn(sensory["raw"], intent, topic, reflex)
            self.save_state()
            return {
                "response": reflex + " [Fast Pathway]",
                "duygu": "Refleksif Yanıt",
                "esik": self.cortical_layer.threshold[0],
                "anlam_spikes": 0, "celiski_spikes": 0, "cortical_spikes": 0,
                "latency": latency,
                "motor_plan": motor_plan,
                "reward": 0.0,
                "time": self.temporal_clock.t,
            }
        
        query_vector = self.get_embedding(sensory["raw"])
        vec_256 = query_vector[:256] if len(query_vector) >= 256 else query_vector + [0.0]*(256-len(query_vector))
            
        input_spikes = [1.0 if random.random() < abs(v) else 0.0 for v in vec_256]
        closest_memory, sim_score = self.hippocampus.retrieve(input_spikes)
        
        if closest_memory is None:
            closest_memory = "Bellek boş."
            sim_score = 0.0
            
        if closest_memory == sensory["raw"]:
            sim_score = 1.0
            closest_memory = "Kendimi dinliyorum."

        cortical_spikes = 0
        
        # 2. Kortikal İşleme ve Sürekli Zaman (Spiking)
        for step in range(self.num_steps):
            noise = (step % 2) * 0.02
            current_in = [v * (1.0 + sensory["salience"] + rhythm) + noise for v in vec_256]
            spikes_2d = self.cortical_layer.step(current_in)
            cortical_spikes += sum(spikes_2d)

        # 3. Anıyı Hipokampüs'e Yazma (One-shot)
        self.hippocampus.store(query, input_spikes)
        
        # 4. Modül 02: Global Neuronal Workspace (Bilinç Yayını) - Biyolojik Non-lineer Ateşleme (Ignition)
        activity_ratio = cortical_spikes / (self.cortical_layer.num_neurons * self.num_steps)
        self.neuromodulators.update(sensory, sim_score, activity_ratio, self.sleep_module.sleep_pressure)
        gain = self.neuromodulators.signal_gain()
        
        # Biyolojik RAS (Reticular Activating System) Uyarımı: Sözel/İletişimsel sinyaller bilinci daha hızlı açar
        attention_multiplier = 2.0 if sensory["is_verbal"] else 1.0
        
        bottom_up_salience = min(1.0, (activity_ratio / self.cortical_layer.target_activity) * attention_multiplier * (0.8 + sensory["salience"]))
        
        # Biyolojik beyindeki gibi aşağıdan-yukarı (duyusal/kortikal) ve yukarıdan-aşağı (bellek/hipokampal) 
        # sinyallerin non-lineer (sigmoid) entegrasyonu (Ignition süreci).
        integrated_signal = ((bottom_up_salience * 0.55) + (sim_score * 0.45)) * gain
        
        # Sigmoid tabanlı non-lineer ateşleme fonksiyonu (Non-linear Ignition)
        # Biyolojik GNW'de eşik aşımı ani ve non-lineerdir ("all-or-none" principle).
        def sigmoid_ignition(x, k=10, x0=0.45):
            return 1.0 / (1.0 + math.exp(-k * (x - x0)))
            
        signal_strength = sigmoid_ignition(integrated_signal)
        
        # GNW eşiği sabit kalabilir, çünkü sigmoid fonksiyonu sinyali zaten 0 ile 1 arasına sıkıştırıp eşik etrafında zıplatır.
        awareness_level = self.gnw.broadcast(signal_strength, closest_memory)
        gnw_thought = self.gnw.active_thought or closest_memory or sensory["raw"]
        
        if self.sleep_module.stage in {"nrem", "rem"}:
            h_durum = f"Rüya Bilinci (GNW Aktif, {awareness_level:.2f})"
        else:
            h_durum = f"Bilinçli (GNW Aktif, {awareness_level:.2f})"

        response_preview = self.dialogue_manager.assemble_response(
            intent=intent,
            query=sensory["raw"],
            topic=topic,
            awareness_level=awareness_level,
            closest_memory=closest_memory,
            sim_score=sim_score,
            gnw_thought=gnw_thought,
            reward_signal=self.reward_system.last_reward,
            motor_plan={"action": self.motor_planner.last_action},
            sleep_stage=self.sleep_module.stage,
        )
        if intent not in {"greeting", "status", "identity", "thanks", "ack", "question", "statement", "smalltalk_invite", "time_query"}:
            response_preview = BiolinguisticSynthesizer.generate(sensory["raw"], closest_memory, sim_score, h_durum, gnw_thought, self.working_memory, self.hebbian_weights)
        reward_signal = self.reward_system.infer_reward(sensory["raw"], response_preview, sim_score)
        self.neuromodulators.levels["dopamine"] = max(0.05, min(1.0, self.neuromodulators.levels["dopamine"] + (0.15 * reward_signal)))
        motor_plan = self.motor_planner.plan(sensory, h_durum, reward_signal, self.sleep_module.stage, sim_score)
        
        # Hebbian Öğrenme (Kelimeler arası sinaptik ağırlık güncellemeleri)
        stopwords = {"bir", "ve", "ile", "için", "çok", "bu", "şu", "o", "mi", "mu", "mı", "mü", "ne", "nasıl", "neden", "niye", "hangi", "kadar", "gibi", "ben", "sen", "biz", "siz", "onlar", "hakkında", "mısın", "misin", "musun", "müsün", "da", "de", "ki", "konuşalım", "yapalım", "edelim", "olsun", "bence", "sence", "daha", "en", "var", "yok", "evet", "hayır", "diye", "şey", "şeyler", "olan", "olarak", "üzere", "bunu", "buna", "şunu", "şuna", "onu", "ona", "göre", "bana", "sana"}
        raw_words = [w for w in sensory["tokens"] if w not in stopwords and len(w) > 2]
        # Fiil filtreleme (basit heuristic)
        words = [w for w in raw_words if not (w.endswith("mak") or w.endswith("mek") or w.endswith("iyor") or w.endswith("dım") or w.endswith("dim") or w.endswith("tım") or w.endswith("tim") or w.endswith("acak") or w.endswith("ecek") or w.endswith("mış") or w.endswith("miş"))]
        if not words: words = raw_words # Eğer her şey filtrelendiyse eskiye dön
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                w1, w2 = words[i], words[j]
                if w1 not in self.hebbian_weights: self.hebbian_weights[w1] = {}
                if w2 not in self.hebbian_weights: self.hebbian_weights[w2] = {}
                self.hebbian_weights[w1][w2] = self.hebbian_weights[w1].get(w2, 0) + 1
                self.hebbian_weights[w2][w1] = self.hebbian_weights[w2].get(w1, 0) + 1
                
        # 5. Doğal Dil Sentezi
        response = response_preview
        self.dialogue_memory.add_turn(sensory["raw"], intent, topic, response)

        latency = (datetime.datetime.now() - start_time).total_seconds()
        self.save_state()
        
        return {
            "response": response,
            "duygu": h_durum,
            "esik": self.cortical_layer.threshold[0],
            "anlam_spikes": int(cortical_spikes * 0.4), 
            "celiski_spikes": int(cortical_spikes * 0.1),
            "cortical_spikes": int(cortical_spikes),
            "latency": latency,
            "motor_plan": motor_plan,
            "reward": reward_signal,
            "time": self.temporal_clock.t
        }
