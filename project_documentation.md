# Traffic Benchmark — Project Documentation

**Author:** Adam Amrani  
**Stage:** Data Science — Benchmark LLM  
**Date:** March 2026

---

## 1. Descrizione del Progetto

### Obiettivo generale

Il progetto ha come scopo la creazione di un **benchmark originale** per valutare le capacità di ragionamento sequenziale e strutturato dei Large Language Models (LLM). Il benchmark è costruito attorno a uno scenario realistico di traffico stradale, in cui più veicoli si muovono e interagiscono seguendo regole precise.

### Contesto e motivazione

I benchmark esistenti come BIG-Bench, BIG-Bench Hard (BBH) e BIG-Bench Extra Hard (BBEH) hanno dimostrato che i modelli moderni raggiungono performance molto elevate su task di ragionamento generale, ma faticano quando:

- le sequenze di aggiornamenti sono lunghe
- il contesto contiene informazioni irrilevanti
- il ragionamento richiede il mantenimento di uno stato interno coerente nel tempo

In particolare, studi sul **state tracking** mostrano che anche i modelli più potenti degradano rapidamente all'aumentare del numero di passi da simulare. Il position tracking di oggetti in movimento è uno dei contesti più concreti in cui questo limite emerge.

### Perché il traffico stradale

Il dominio del traffico è stato scelto perché combina naturalmente tre tipi di ragionamento che si vogliono valutare separatamente:

- **Ragionamento sequenziale e spaziale** — seguire la posizione di più veicoli in un ambiente discreto
- **Ragionamento basato su regole** — applicare correttamente le norme del codice della strada (precedenza, segnali, rotatorie)
- **Ragionamento sotto vincoli e ambiguità** — situazioni di overlap spaziale, violazioni, inferenze su stati parzialmente noti

A differenza dei benchmark esistenti che usano oggetti astratti, questo contesto è intuitivo, realistico e permette di valutare quanto i modelli sappiano trasferire conoscenza sul mondo reale a un problema strutturato.

### Struttura del benchmark

Il benchmark è composto da due parti:

**Core Dataset** — 1200 esempi in un contesto controllato e pulito, senza rumore aggiuntivo. Ogni task è rappresentata da 300 esempi con difficoltà base (3 veicoli, sequenze di 2–4 passi).

**Extended Dataset** *(da sviluppare in seguito)* — versione aumentata del Core Dataset con: più veicoli (4–5), sequenze più lunghe, no-op steps (azioni che si annullano), informazioni irrilevanti. Valuta la robustezza e la generalizzazione dei modelli.

### Formato degli esempi

Ogni esempio è un **quesito a scelta multipla** con 5 opzioni:

- 1 risposta **corretta**
- 2 risposte **quasi vere** (plausibili, richiedono ragionamento per escluderle)
- 2 risposte **chiaramente false** (impossibili nel contesto dato)

Ogni esempio è salvato in formato **JSONL** (un oggetto per riga), con la seguente struttura:

```json
{
  "id": "task1_0042",
  "task": "position_tracking",
  "prompt": "Three vehicles are at a multi-lane road.\nVehicle A is in the left lane...",
  "scenario": {
    "vehicles": [
      {"id": "A", "position": "left_lane", "direction": "north"},
      {"id": "B", "position": "center_lane", "direction": "north"},
      {"id": "C", "position": "right_lane", "direction": "north"}
    ],
    "environment": "multi_lane_road"
  },
  "events": [
    "Vehicle B changes to the left lane.",
    "Vehicle A moves forward.",
    "Vehicle C changes to the right lane."
  ],
  "question": "Where is Vehicle A at the end of the sequence?",
  "choices": {
    "A": "left lane",
    "B": "center lane",
    "C": "right lane",
    "D": "inside intersection",
    "E": "roundabout lane"
  },
  "answer": "B",
  "distractor_type": {
    "A": "near_true",
    "C": "near_true",
    "D": "highly_false",
    "E": "highly_false"
  },
  "metadata": {
    "num_vehicles": 3,
    "num_events": 3,
    "queried_vehicle": "A",
    "environment": "multi_lane_road",
    "difficulty": "base"
  }
}
```

### Ambienti

Gli scenari possono svolgersi in tre ambienti:

- `intersection` — incrocio a quattro vie senza semaforo
- `multi_lane_road` — strada a tre corsie (left, center, right)
- `roundabout` — rotatoria con corsie di approccio e uscita

### Modelli da valutare

I modelli sono divisi in tre fasce:

**Small** — LLaMA 1B, DeepSeek 1.5B, Qwen 1.5B  
**Medium** — Phi-3.5 Mini, LLaMA 3B, Mixtral-8x7B, LLaMA 3-8B  
**Large/Frontier** — LLaMA 3-70B, GPT-4o, Gemini 2.0

---

## 2. Lavoro Svolto

### 2.1 Ambiente di sviluppo

**Strumento principale:** Claude Code — interfaccia da terminale che permette all'AI di leggere, scrivere e modificare file direttamente nel filesystem del progetto, eseguire codice Python e lavorare in modo agentivo su task complesse.

**Editor:** VS Code — usato per ispezionare i file, navigare la struttura del progetto ed eseguire gli script Python dal terminale integrato.

**Flusso di lavoro:**
- Claude Code crea e modifica i file in locale
- VS Code permette di leggere e verificare il codice
- Git sincronizza il lavoro su GitHub

### 2.2 Repository GitHub

La repository è stata creata su GitHub con nome `traffic-benchmark`, visibilità privata, con `.gitignore` Python e `README.md` iniziale.

**Struttura delle cartelle definita:**

```
traffic-benchmark/
│
├── domain/                    # Modulo condiviso — regole, entità, simulazione
│   ├── __init__.py
│   ├── entities.py            # Classi base: Vehicle, ScenarioState, enum
│   ├── rules.py               # Regole del traffico per tutti gli ambienti
│   ├── scenario.py            # Builder degli scenari + apply_action()
│   └── renderer.py            # Conversione stato → testo in linguaggio naturale
│
├── generators/                # Un file per task
│   ├── __init__.py
│   ├── task1_position.py
│   ├── task2_rightofway.py
│   ├── task3_violation.py
│   └── task4_overlap.py
│
├── dataset/
│   ├── core/
│   │   ├── task1_position.jsonl
│   │   ├── task2_rightofway.jsonl
│   │   ├── task3_violation.jsonl
│   │   ├── task4_overlap.jsonl
│   │   └── core_dataset.jsonl      # merged finale
│   └── stats/
│       └── distribution_report.json
│
├── scripts/
│   ├── generate_all.py
│   └── validate_dataset.py
│
├── tests/
│   └── test_generators.py
│
└── README.md
```

**Comandi git usati per sincronizzare:**

```bash
git add .
git commit -m "feat: descrizione del commit"
git push origin main
```

### 2.3 Cartella `domain/` — Codice scritto

La cartella `domain/` costituisce il **contratto condiviso** tra tutti i generatori. Nessun Task Agent può inventare regole o posizioni: tutto passa da questi 4 file.

---

#### `entities.py`

Definisce tutti gli oggetti base del mondo simulato tramite `dataclass` e `Enum`.

**Classi principali:**

`Direction` — enum dei 4 punti cardinali: `NORTH`, `SOUTH`, `EAST`, `WEST`

`Action` — enum delle 8 azioni possibili per un veicolo:
- `MOVE_FORWARD`, `STOP`, `TURN_LEFT`, `TURN_RIGHT`
- `CHANGE_LEFT`, `CHANGE_RIGHT`
- `ENTER_ROUNDABOUT`, `EXIT_ROUNDABOUT`

`Environment` — enum dei 3 ambienti: `INTERSECTION`, `MULTI_LANE`, `ROUNDABOUT`

`Lane` — enum delle posizioni su strada: `LEFT`, `CENTER`, `RIGHT`, `ROUNDABOUT_LANE`

`IntentDirection` — enum dell'intenzione dichiarata di un veicolo, usata **solo nel Task 2**. Contiene esattamente i 3 valori documentati nel PDF: `GO_STRAIGHT`, `TURN_LEFT`, `TURN_RIGHT`. I valori `EXIT_NORTH/SOUTH/EAST/WEST` sono stati rimossi perché non documentati.

`Vehicle` — dataclass con campi: `id`, `position`, `direction`, `intent`, `inside_intersection`, `stopped`

`ScenarioState` — dataclass che rappresenta lo stato globale: lista di veicoli, ambiente, step corrente, log degli eventi

---

#### `rules.py`

Implementa le regole del traffico per tutti e tre gli ambienti.

**Funzioni principali:**

`right_of_way_intersection(v1, v2)` — applica la regola della precedenza a destra agli incroci senza segnali. Usa una tabella `APPROACH_PRIORITY` che mappa coppie di direzioni di approccio all'ID del veicolo con priorità.

`right_of_way_roundabout(v_inside, v_entering)` — implementa la regola della rotatoria: chi è già dentro ha **sempre** priorità su chi vuole entrare. Questa funzione era mancante nella versione precedente.

`roundabout_can_enter(entering, vehicles_inside)` — restituisce `True` se nessun veicolo sta circolando dentro la rotatoria.

`right_of_way(v1, v2, env)` — dispatcher generico che sceglie automaticamente la logica corretta in base all'ambiente.

`is_violation_stop_sign(entered_without_stopping)` — veicolo entra senza fermarsi con stop sign presente

`is_violation_right_of_way(vehicle_id, priority_vehicle_id)` — veicolo entra nonostante non abbia la precedenza

`is_violation_roundabout_entry(entering, vehicles_inside)` — veicolo entra in rotatoria senza cedere il passo

`get_valid_actions(env)` — restituisce la lista di azioni valide per ogni ambiente

`vehicles_overlap(v1, v2)` — True se entrambi i veicoli sono contemporaneamente dentro l'incrocio

---

#### `scenario.py`

Contiene i **builder** degli scenari iniziali e il motore di simulazione `apply_action()`.

**Builder:**

`build_intersection_scenario(num_vehicles, with_intent)` — costruisce uno scenario valido all'incrocio, campionando direzioni casuali distinte. Se `with_intent=True`, assegna un'intenzione casuale a ogni veicolo (usato nel Task 2).

`build_multi_lane_scenario(num_vehicles)` — costruisce uno scenario su strada a corsie, assegnando corsie distinte a ogni veicolo.

`build_roundabout_scenario(num_vehicles)` — costruisce uno scenario alla rotatoria, con il primo veicolo già dentro e gli altri in approccio.

`build_scenario(env, **kwargs)` — dispatcher che chiama il builder corretto in base all'ambiente.

**Motore di simulazione:**

`apply_action(state, vehicle_id, action)` — applica un'azione a un veicolo e ne muta lo stato. Per ogni tipo di azione:
- `MOVE_FORWARD` — sposta il veicolo dentro l'incrocio se in approccio
- `STOP` — imposta `stopped = True`
- `TURN_LEFT / TURN_RIGHT` — aggiorna `direction` con `_rotate_direction()` e imposta la posizione di uscita
- `CHANGE_LEFT / CHANGE_RIGHT` — aggiorna la corsia scorrendo l'array `LEFT → CENTER → RIGHT`
- `ENTER/EXIT_ROUNDABOUT` — aggiorna `inside_intersection` e `position`

Restituisce sempre una stringa in linguaggio naturale dell'evento (es. `"Vehicle A turns left."`).

`_rotate_direction(direction, turn)` — ruota la direzione in senso orario o antiorario nell'ordine `N → E → S → W`.

`_lane_index(position)` — restituisce l'indice della corsia corrente, con fallback al centro.

---

#### `renderer.py`

Converte gli oggetti Python in **testo leggibile da un LLM**. È puramente un formattatore — non contiene logica.

`describe_vehicle(v)` — descrive lo stato iniziale di un singolo veicolo in una frase in inglese.

`describe_scenario(state)` — descrive l'intera configurazione iniziale. Esempio di output:
```
Three vehicles are at a multi-lane road.
Vehicle A is in the left lane.
Vehicle B is in the center lane.
Vehicle C is in the right lane.
```

`describe_events(events)` — formatta la lista di eventi come sequenza numerata:
```
Sequence of events:
1. Vehicle A moves forward.
2. Vehicle B changes to the left lane.
```

`render_prompt(scenario_text, events, question, choices)` — assembla il prompt completo pronto per l'LLM, unendo scenario, sequenza, domanda e opzioni A–E.

---

## 3. Task 1 — Position Tracking

### 3.1 Descrizione

Il Task 1 valuta la capacità del modello di **seguire la posizione di più veicoli** attraverso una sequenza di azioni e rispondere correttamente su dove si trova un veicolo specifico alla fine della sequenza.

Il modello deve applicare mentalmente ogni passo in ordine, aggiornare la propria rappresentazione interna dello stato, e non perdere traccia di nessun veicolo. È il task di state tracking più puro del benchmark.

**Ambienti usati:** `multi_lane_road` e `intersection`. La rotatoria è esclusa dal Task 1 perché le posizioni interne sono meno distinguibili — è più adatta al Task 4 (Overlap).

**Struttura di ogni esempio:**
- Stato iniziale: 3 veicoli in posizioni distinte
- Sequenza: 2–4 azioni in ordine sequenziale
- Domanda: posizione finale di un veicolo specifico
- 5 opzioni: 1 corretta, 2 quasi vere, 2 chiaramente false

**Esempio completo — multi_lane_road:**

```
Three vehicles are at a multi-lane road.
Vehicle A is in the left lane.
Vehicle B is in the center lane.
Vehicle C is in the right lane.

Sequence of events:
1. Vehicle B changes to the left lane.
2. Vehicle A moves forward.
3. Vehicle C changes to the right lane.

Question: Where is Vehicle A at the end of the sequence?

A) left lane          ← near_true  (posizione iniziale di A)
B) center lane        ← CORRETTA   (A era in left, B si è spostato in left → A avanza in center)
C) right lane         ← near_true  (destinazione intuitivamente errata ma plausibile)
D) inside intersection ← highly_false (impossibile su multi_lane_road)
E) roundabout lane    ← highly_false (ambiente sbagliato)
```

**Esempio completo — intersection:**

```
Three vehicles are at an intersection.
Vehicle A is in the northern approach.
Vehicle B is in the eastern approach.
Vehicle C is in the southern approach.

Sequence of events:
1. Vehicle A moves forward.
2. Vehicle C moves forward.
3. Vehicle A turns left.

Question: Where is Vehicle A at the end of the sequence?

A) northern approach    ← near_true  (posizione iniziale)
B) inside intersection  ← near_true  (stato intermedio dopo step 1)
C) west exit            ← CORRETTA   (svolta sinistra da nord → esce a ovest)
D) east exit            ← highly_false
E) right lane           ← highly_false (ambiente sbagliato)
```

---

### 3.2 Paletti e soluzioni

**Problema 1 — Collisioni di corsia**

Su `multi_lane_road` due veicoli non possono occupare la stessa corsia simultaneamente. Se B è già in `left_lane` e A prova `CHANGE_LEFT` dalla `center_lane`, il codice deve rilevare l'occupazione e bloccare l'azione.

*Soluzione:* `safe_apply_action()` controlla prima di ogni `CHANGE_LEFT/RIGHT` che la corsia di destinazione sia libera. Se è occupata, restituisce `None` e il generatore scarta l'azione e riprova.

---

**Problema 2 — Azioni ai bordi**

`CHANGE_LEFT` dalla `left_lane` è fisicamente impossibile. Stesso per `CHANGE_RIGHT` dalla `right_lane`. Un generatore che non lo controlla produce scenari invalidi.

*Soluzione:* `safe_apply_action()` verifica l'indice della corsia corrente prima di eseguire il cambio. Se il veicolo è già al bordo, l'azione viene ignorata e si riprova.

---

**Problema 3 — Veicolo interrogato mai mosso**

Se la domanda riguarda Vehicle A ma la sequenza muove solo B e C, il modello può rispondere correttamente ignorando tutti gli eventi — è un shortcut che abbassa la qualità del dataset.

*Soluzione:* `generate_sequence()` forza l'ultimo step a coinvolgere il veicolo interrogato se non è ancora stato mosso nella sequenza.

---

**Problema 4 — Posizione finale uguale a quella iniziale**

Se la sequenza produce per il veicolo interrogato una posizione finale identica a quella iniziale, la domanda è banale o irrisolvibile senza informazioni aggiuntive.

*Soluzione:* Al termine della simulazione, se `final_position == start_position`, `generate_sequence()` restituisce `None`. Il chiamante scarta l'esempio e ne genera uno nuovo.

---

**Problema 5 — Distrattori non plausibili**

Le 2 opzioni "quasi vere" devono essere posizioni che il veicolo avrebbe potuto ragionevolmente raggiungere. Le 2 "chiaramente false" devono essere posizioni impossibili dato l'ambiente corrente.

*Soluzione:* `build_choices()` costruisce:
- near_true 1: posizione iniziale del veicolo interrogato
- near_true 2: posizione intermedia della sequenza (o una corsia adiacente)
- highly_false 1 e 2: posizioni valide in un ambiente diverso (es. `roundabout_lane` su `multi_lane_road`, `left_lane` su `intersection`)

---

**Problema 6 — Bilanciamento delle risposte**

Senza controllo esplicito il generatore tende a mettere la risposta corretta sempre nella stessa posizione, creando un bias facilmente sfruttabile dal modello.

*Soluzione:* `key_schedule` costruisce esattamente 20 occorrenze di ciascuna lettera A–E e le mescola con `random.shuffle()` prima di iniziare la generazione. Ogni esempio riceve la sua lettera target dal schedule, e `assign_letters()` piazza la risposta corretta esattamente in quella posizione.

---

### 3.3 Codice — `generators/task1_position.py`

Il file genera 100 esempi (espandibili a 300 modificando `N_EXAMPLES`).

**Flusso di generazione di un singolo esempio:**

1. Scegliere casualmente l'ambiente (`multi_lane_road` o `intersection`) e il numero di step (2–4)
2. Costruire lo stato iniziale con `build_multi_lane_scenario()` o `build_intersection_scenario()`
3. Scegliere casualmente il veicolo interrogato
4. Fare uno snapshot dello stato iniziale per il testo del prompt (prima che le mutazioni lo alterino)
5. Eseguire `generate_sequence()` su una copia profonda dello stato
6. Se la sequenza non è valida (veicolo non mosso, posizione invariata), scartare e riprovare
7. Registrare la posizione intermedia del veicolo interrogato (per usarla come near_true)
8. Costruire le 5 opzioni con `build_choices()`
9. Assegnare le lettere A–E con `assign_letters()`, piazzando la risposta corretta alla lettera del key_schedule
10. Assemblare il prompt con `render_prompt()` e restituire il dizionario completo

**Funzioni principali:**

`safe_apply_action(state, vid, action)` — wrapper di `apply_action()` con controlli di validità (bordi e collisioni). Restituisce `None` se l'azione non è eseguibile.

`generate_sequence(state, queried_vid, env, n_steps)` — genera una sequenza valida di n_steps azioni con le garanzie descritte nei paletti. Restituisce `None` se non riesce a trovare una sequenza valida entro 50 tentativi.

`build_choices(correct_pos, start_pos, intermediate_pos, env)` — costruisce il dizionario grezzo con le 5 opzioni classificate per tipo.

`assign_letters(choices_dict, correct_key)` — mescola le 5 opzioni e piazza la risposta corretta alla lettera richiesta. Restituisce `choices` (dict A–E → testo) e `distractor_type` (dict A–E → tipo).

`generate_example(example_id, correct_key)` — genera un singolo esempio completo. Restituisce `None` se nessuna sequenza valida è stata trovata.

`generate_task1(n, output_path)` — loop principale: genera n esempi rispettando il key_schedule, salva in JSONL e stampa la distribuzione delle risposte.

**Come eseguire:**

```bash
# dalla root del progetto, con virtualenv attivo
source venv/bin/activate
python generators/task1_position.py
```

Output atteso:
```
Saved 100 examples to dataset/core/task1_position.jsonl

Answer distribution:
  A:  20  ████████████████████
  B:  20  ████████████████████
  C:  20  ████████████████████
  D:  20  ████████████████████
  E:  20  ████████████████████

Environment distribution:
  multi_lane_road: 53
  intersection: 47
```

---
