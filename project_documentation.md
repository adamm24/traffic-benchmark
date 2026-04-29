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

Il benchmark è organizzato in tre livelli progressivi:

**Core Dataset** — 400 esempi in un contesto controllato e pulito, senza rumore aggiuntivo. Ogni task è rappresentata da 100 esempi con difficoltà base (3 veicoli, sequenze di 2–4 passi). È il livello su cui vengono eseguite le baseline evaluation.

**Extended Dataset** *(da sviluppare)* — versione aumentata del Core Dataset con: più veicoli (4–5), sequenze più lunghe, no-op steps (azioni che si annullano), informazioni irrilevanti. Stesse operazioni di ragionamento del core, ma con carico cognitivo maggiore. Valuta la robustezza e la generalizzazione dei modelli.

**Higher-Order Reasoning Tasks** *(da sviluppare)* — task families che introducono direzioni di ragionamento qualitativamente diverse dal core: **ragionamento controfattuale** ("cosa sarebbe successo se X non fosse avvenuto?") e **ragionamento a ritroso** ("dato lo stato finale, quale sequenza di eventi lo ha prodotto?"). A differenza dell'extended dataset — che è una versione più difficile delle stesse operazioni — questi task richiedono simulazione condizionale e inferenza inversa. Rappresentano la superficie di valutazione primaria per i modelli specializzati nel ragionamento (Group 6).

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

I modelli da valutare coprono un intervallo che va da modelli sub-2B a modelli frontier proprietari, organizzati in 6 gruppi basati su ipotesi di ricerca specifiche. Per la lista completa dei modelli, i gruppi e la strategia di valutazione, vedi **Sezione 7**.

---

## 2. Lavoro Svolto

### 2.1 Ambiente di sviluppo

**Strumento principale:** terminale locale con script Python per generazione, validazione e debug.

**Editor:** VS Code — usato per ispezionare i file, navigare la struttura del progetto ed eseguire gli script Python dal terminale integrato.

**Flusso di lavoro:**
- VS Code permette di leggere e verificare il codice
- Git sincronizza il lavoro su GitHub

### 2.2 Repository GitHub

La repository è stata creata su GitHub con nome `traffic-benchmark`, visibilità privata, con `.gitignore` Python e `README.md` iniziale.

**Struttura principale della repository (stato corrente):**

```
traffic-benchmark/
│
├── domain/                    # Modulo condiviso — regole, entità, simulazione
│   ├── __init__.py
│   ├── entities.py            # Classi base: Vehicle, ScenarioState, enum
│   ├── rules.py               # Regole del traffico per tutti gli ambienti
│   ├── scenario.py            # Builder degli scenari + apply_action()
│   ├── render.py              # Conversione stato → testo in linguaggio naturale
│   ├── fsm.py                 # FSM esplicita delle transizioni
│   ├── trajectory.py          # Traiettorie in incrocio (Task 2)
│   └── vocabulary.py          # Vocabolario controllato delle label
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
│   └── core_simulation/
│       ├── task1_position.jsonl
│       ├── task2_rightofway.jsonl
│       ├── task3_violation.jsonl
│       └── task4_overlap.jsonl
│
├── documentation/
│   ├── task1_documentation.md
│   ├── task2_documentation.md
│   ├── task3_documentation.md
│   └── task4_documentation.md
│
├── scripts/
│   ├── build_core_simulation.py
│   ├── validate_task2.py
│   ├── validate_task3.py
│   ├── validate_task4.py
│   └── validate_task4_distribution.py
│
├── tests/
│   └── test_domain.py
│
└── README.md
```

### 2.3 Cartella `domain/` — Codice scritto

La cartella `domain/` costituisce il **contratto condiviso** tra tutti i generatori. Nessun generatore può inventare regole o posizioni: tutto passa dal modulo di dominio.

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

`IntentDirection` — enum dell'intenzione dichiarata di un veicolo, usata **solo nel Task 2**. Contiene i tre valori usati dal generatore corrente: `GO_STRAIGHT`, `TURN_LEFT`, `TURN_RIGHT`.

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

#### `render.py`

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

### 3.4 Problemi Tecnici Risolti

**Problema 1 — Falsi positivi dal confronto stringa (vocabulary split)**

Un audit esterno che confrontava i testi delle opzioni con le tracce degli eventi trovava 87/100 record con apparenti discrepanze: la traccia interna registrava `east_exit` mentre l'opzione mostrava `the eastern exit`. Sembrava un errore critico di integrità.

*Causa radice:* La discrepanza è by design. `vocabulary.py` mappa `east_exit → the eastern exit` via `label_of()`. Il generatore usa questa funzione correttamente; l'invariante `near_true_grounded_in_visited` passa perché usa anch'essa `label_of()` internamente. I falsi positivi erano prodotti da un confronto stringa grezzo che non applicava la mappatura.

*Risoluzione:* Nessuna modifica al codice. I check di qualità esterni devono usare `label_of()` per il confronto, non le chiavi interne snake_case.

---

**Problema 2 — Alto tasso di retry (36% → 11%)**

Il piano builder era interamente reattivo: costruiva un piano completo e poi eseguiva tutti e 13 gli invarianti post-hoc. Per esempi hard con molti vincoli simultanei (palindrome, interleaving, actor diversity, action streak), questo causava frequenti ripartenze complete. Un record richiedeva fino a 22 tentativi.

*Causa radice:* Nessun pruning anticipato durante la costruzione del piano. I rami invalidi venivano rilevati solo dopo essere stati completamente costruiti.

*Correzione:* Tre invariant check spostati in-loop come guard anticipate: action streak guard (pruna se gli ultimi 3 step hanno la stessa azione), ABAB alternation guard (pruna se gli ultimi 4 step alternano veicolo A-B-A-B), action diversity lookahead (rifiuta il piano se completo ma con meno di `MIN_DISTINCT_ACTIONS`).

*Risultato:* Tasso di retry sceso dal 36% all'11%, massimo tentativi da 22 a 2. La chiamata post-hoc a `_validate_example()` è preservata come check di correttezza autoritativo.

---

**Problema 3 — STOP sottorappresentato (4% → 7.8%)**

L'azione STOP appariva solo in 17/~420 azioni totali (4%). Era strutturalmente deprioritizzata nel pool di azioni per evitare sequenze banali; per i veicoli non interrogati agli incroci veniva sempre rimossa dal pool.

*Correzione:* Alzata la probabilità di front-prioritization di STOP dal 20% al 35% per il veicolo interrogato.

*Nota:* Dopo l'aggiunta del guard in-loop dello streak (problema 2), l'aumento è stato parzialmente compensato. Il risultato finale è 34 occorrenze (7.8%), considerato accettabile.

---

**Problema 4 — Sbilanciamento ambiente nel tier easy (intersection 20 vs multi_lane_road 14)**

I due schedule (difficulty e environment) venivano generati e mescolati indipendentemente, lasciando la loro distribuzione congiunta non controllata. Il tier easy risultava con il 43% di record in più per intersection rispetto a multi_lane_road.

*Correzione:* Sostituiti i due schedule indipendenti con un singolo schedule congiunto difficulty × environment. Ogni cella riceve esattamente `n // (n_difficulties × n_environments)` slot, con il resto distribuito round-robin. La lista congiunta viene mescolata e poi separata nuovamente, così il codice a valle rimane invariato.

*Risultato:* Ogni cella ha esattamente 16 o 17 record (target: 100 / 6 ≈ 16.7).

---

### 3.5 Limitazioni Accettate

**TURN_RIGHT (11%) vs TURN_LEFT (13.6%):** Asimmetria strutturale — le svolte a destra hanno meno contesti di applicazione validi negli scenari di incrocio per design del dominio. Non è un errore di ingegneria.

**STOP (7.8%):** Leggermente al di sotto delle altre azioni (~22%). Dopo i due fix (priorità aumentata + guard in-loop), è il miglior risultato raggiungibile senza rilassare i vincoli di non-trivialità.

**Overlap tra livelli di difficoltà:** Easy e medium condividono alcuni conteggi di step. La difficoltà è definita comportamentalmente (interleaving obbligatorio, actor diversity) e non solo dal numero di eventi.

---

### 3.6 Stato Finale (v14)

| Metrica | Valore |
|---|---|
| Record | 100 |
| Versione generator | task1_position.v14 |
| Invarianti | 13/13 passing, 0 failures |
| Distribuzione risposte | 20 per lettera (A–E) |
| Distribuzione difficulty × environment | 16–17 per cella (6 celle) |
| Record con >1 tentativo | 11% (max 2 tentativi) |
| Quota azione STOP | 7.8% |
| Quota CHANGE_LEFT/RIGHT | ~22–23% ciascuno |
| Prompt unici | sì |
| ID unici | sì |

```bash
python generators/task1_position.py --n 100 --seed 13 --out dataset/core/task1_position.jsonl
```

---

## 4. Task 2 — Right-of-Way Reasoning

### 4.1 Descrizione

Il Task 2 valuta la capacità del modello di **applicare correttamente le regole di precedenza** a uno snapshot statico di uno scenario con 3 veicoli.

A differenza del Task 1 (che richiede simulazione sequenziale), il Task 2 non richiede simulazione: il modello riceve una configurazione istantanea di veicoli con posizioni e intenzioni dichiarate, e deve determinare quale veicolo ha la precedenza. Questa capacità — recuperare e applicare conoscenza normativa a una situazione strutturata concreta — è distinta dal puro state tracking.

**Ambienti usati:** `intersection` (70 record) e `roundabout` (30 record).

- **Intersection:** regola di precedenza a destra. Il veicolo con priorità dipende dalla direzione di approccio e dall'intenzione dichiarata (go straight, turn left, turn right). Il modello deve usare entrambe le informazioni — la sola direzione non è sufficiente.
- **Roundabout:** chi è già in circolazione all'interno ha sempre priorità assoluta su chi vuole entrare. Regola semplice e assoluta.

**Struttura di ogni esempio:**
- Scenario statico: 3 veicoli con posizioni, direzioni e intenzioni dichiarate
- Domanda: quale veicolo ha la precedenza?
- 5 opzioni: 1 risposta corretta (veicolo prioritario), 2 quasi vere (veicolo che cede + "Both can pass at the same time"), 2 chiaramente false (affermazioni di politica errate)

---

### 4.2 Architettura del Generatore

**Pairwise assessment e dominanza globale:**

La priorità viene derivata tramite valutazione a coppie: per ogni coppia di veicoli viene creato un `PairAssessment` che registra se esiste un conflitto, chi vince, e se l'esito cambia con o senza le intenzioni dichiarate. Un veicolo è il dominante globale se vince tutti i conflitti pairwise. Scenari senza un unico dominante globale vengono scartati e rigenerati.

**Sistema di relabeling:**

Un sistema di relabeling mappa gli ID interni ai label finali A/B/C tramite una ricerca di permutazioni che soddisfa più vincoli simultaneamente:
- Vehicle A è escluso dall'essere il veicolo prioritario negli incroci (blocca l'euristica alfabetica)
- Il veicolo interno alla rotatoria è sempre etichettato A (compensa la sottorappresentazione di A negli scenari di incrocio)

**Conflict-pair schedule:**

Prima del loop di generazione viene costruito un `pair_schedule` che pre-assegna la coppia di conflitto desiderata per ogni slot (~33 record per coppia: A-B, A-C, B-C). Candidati con coppia non corrispondente allo slot vengono scartati.

**Key schedule:** Esattamente 20 record per lettera di risposta (A–E).

---

### 4.3 Struttura dei Distrattori

- **near_true (2):** Il veicolo che cede la precedenza (la risposta sbagliata più plausibile — è attivamente coinvolto nel conflitto) e "Both can pass at the same time" (testa se il modello capisce che in un conflitto di precedenza c'è sempre un vincitore chiaro). "Both can pass" è una opzione fissa in ogni record (100/100).
- **highly_false (2):** Affermazioni di politica errate costruite dai dati reali dello scenario — es. "The vehicle turning left always has priority", "Vehicles entering the roundabout have priority". Usano direzioni/intenzioni effettive per non cadere in pool generici.

---

### 4.4 Blocco degli Shortcut

Tre classi di shortcut vengono bloccate sistematicamente:

1. **Shortcut per direzione:** La risposta corretta non può essere determinata dalla sola direzione dei veicoli senza leggere le intenzioni. Invarianti: `direction_only_does_not_match_priority` + `intent_sensitive_priority_pair`.
2. **Shortcut alfabetico:** Il primo veicolo non-left-turning in ordine alfabetico non coincide con il veicolo prioritario. Invariante: `alphabetical_non_left_heuristic_fails`.
3. **Shortcut rotatoria:** Le rotatorie sono strutturalmente più semplici; la loro proporzione (30%) è controllata per evitare che indovinare sempre il veicolo interno produca un'accuratezza artificialmente alta.

---

### 4.5 Invarianti (7)

| Invariante | Cosa verifica |
|---|---|
| `five_distinct_options` | Tutte le 5 scelte sono testualmente distinte |
| `priority_conflicts_with_all_others` | Il veicolo prioritario è in conflitto con entrambi gli altri |
| `pair_conflict_count_at_least_2` | Almeno 2 conflitti pairwise distinti nello scenario |
| `intent_sensitive_priority_pair` | Per incroci: direction-only diverge dal risultato intent-aware. Per rotatorie: True per default |
| `direction_only_does_not_match_priority` | L'euristica direction-only non predice la risposta corretta |
| `answer_text_matches_priority` | Il testo della risposta corretta corrisponde al label del veicolo prioritario |
| `alphabetical_non_left_heuristic_fails` | Il primo veicolo non-left-turning in ordine alfabetico ≠ veicolo prioritario |

---

### 4.6 Problemi Risolti

**Problema 1 — Logica pairwise invece di dominanza globale:** La risposta corretta derivava dal vincitore di un singolo confronto pairwise, non dal dominante globale. Scenari a 3 veicoli potevano avere un "vincitore locale" che perdeva contro il terzo veicolo. *Correzione:* Dominanza globale stretta — il veicolo prioritario deve vincere ogni conflitto pairwise. Scenari senza dominante globale unico vengono scartati.

**Problema 2 — Shortcut direction-only possibile:** Alcuni scenari di incrocio erano risolvibili con una semplice euristica basata sulla direzione, senza leggere le intenzioni. *Correzione:* Aggiunto l'invariante `direction_only_does_not_match_priority` come condizione di rigetto; `intent_sensitive_priority_pair` garantisce che almeno un conflitto pairwise cambi esito quando si usano le intenzioni.

**Problema 3 — `intent_sensitive_priority_pair` sempre False per le 30 rotatorie:** L'invariante era calcolato solo nel blocco `if state.environment == INTERSECTION`. Per le rotatorie rimaneva `False` — 30 falsi negativi nel dataset. *Correzione:* Aggiunto `else: intent_sensitive_with_priority = True` (regola posizionale, intent non applicabile per design). *Risultato:* 0 failures su 100 record.

**Problema 4 — Squilibrio coppia di conflitto (A-B: 65, A-C: 29, B-C: 6):** Nessun controllo sulla distribuzione delle coppie. *Correzione:* `pair_schedule` pre-assegna la coppia desiderata per ogni slot. *Risultato:* A-B: 33, A-C: 33, B-C: 34.

**Problema 5 — "Both can pass" presente in solo 24% dei record:** Il distractor veniva incluso condizionalmente. *Correzione:* Reso membro fisso del pool near_true per ogni record. *Risultato:* 100/100 record.

---

### 4.7 Limitazioni Accettate

**Vehicle B è prioritario in soli 3/100 record.** Il relabeling esclude A dalla priorità agli incroci e assegna sempre A come interno nelle rotatorie. Combinato con il vincolo B-C balance, B risulta raramente prioritario. Accettato come tradeoff per mantenere gli invarianti anti-shortcut.

**`turn right` sottorappresentato (17 occorrenze vs 70 per `turn left`).** I veicoli con intenzione di svoltare a destra generano meno conflitti strutturali. Proprietà emergente del dominio.

**Campo `inside_intersection` sempre False per gli incroci, True solo per le rotatorie.** Il campo codifica "il veicolo sta circolando dentro la rotatoria". È un problema di naming, non di correttezza del dataset. Un futuro rename opzionale a `in_roundabout_lane` migliorerebbe la chiarezza senza impattare i dati.

---

### 4.8 Stato Finale (v3)

| Metrica | Valore |
|---|---|
| Record | 100 |
| Versione generator | task2_rightofway_v3 |
| Invarianti | 7/7 passing, 0 failures |
| Distribuzione risposte | 20 per lettera (A–E) |
| Ambienti | intersection 70, roundabout 30 |
| Coppie di conflitto | A-B: 33, A-C: 33, B-C: 34 |
| "Both can pass" presente | 100/100 |
| Euristica direction-only su incroci | 0/70 corretti |
| Euristica alfabetica su incroci | 0/70 corretti |
| Prompt unici | sì |
| ID unici | sì |

```bash
python generators/task2_rightofway.py --n 100 --seed 42 --out dataset/core/task2_rightofway.jsonl
```

---

## 5. Task 3 — Violation Detection

### 5.1 Descrizione

Il Task 3 valuta la capacità del modello di **identificare il primo veicolo che commette una violazione del codice della strada** in una breve sequenza di eventi, oppure di determinare correttamente che nessuna violazione è avvenuta.

Questo richiede due sotto-capacità distinte: applicare le regole di dominio alle singole azioni (come nel Task 2) e attribuire correttamente la prima violazione all'attore giusto lungo una traccia multi-evento (come nel Task 1). È il task che combina il maggior numero di capacità contemporaneamente.

**Ambienti usati:** tutti e tre — `intersection` (34 record), `roundabout` (33 record), `multi_lane_road` (33 record).

**Struttura di ogni esempio:**
- Scenario iniziale: 3 veicoli con posizioni e direzioni
- Sequenza: 2–3 eventi
- Domanda: quale veicolo ha commesso la prima violazione?
- 5 opzioni con struttura semantica fissa in ogni record: `Vehicle A`, `Vehicle B`, `Vehicle C`, `No vehicle can be determined`, `Another vehicle (not A, B, or C)`

La quinta opzione ("Another vehicle") è sempre sbagliata per design — testa se il modello confina la risposta agli attori osservabili invece di speculare su veicoli non presenti nello scenario.

---

### 5.2 Classi di Violazione

Sette classi di violazione definite e gestite tramite la whitelist `ALLOWED_VIOLATION_TYPES`:

| Classe | Ambiente | Regola |
|---|---|---|
| `turn_without_entering` | intersection | Il veicolo svolta senza essere entrato nell'incrocio |
| `forward_from_exit` | intersection | Il veicolo avanza da una corsia di uscita |
| `intersection_right_of_way` | intersection | Il veicolo entra quando un altro ha la precedenza |
| `lane_change_out_of_bounds_left` | multi_lane_road | CHANGE_LEFT dalla left_lane |
| `lane_change_out_of_bounds_right` | multi_lane_road | CHANGE_RIGHT dalla right_lane |
| `roundabout_entry_no_yield` | roundabout | Il veicolo entra mentre un altro sta circolando dentro |
| `no_violation` | tutti | Nessuna azione illegale nella sequenza |

---

### 5.3 Livelli di Difficoltà

Difficoltà derivata da due proprietà strutturali: numero totale di eventi e posizione della violazione nella sequenza.

- **easy:** 2 eventi; violazione all'ultimo step. Il modello non ha bisogno di lookahead — la violazione è sempre l'azione finale.
- **medium:** 3 eventi; violazione allo step 1 o 2. La violazione è precoce, con rumore legale dopo di essa.
- **hard:** 3 eventi; violazione all'ultimo step, oppure nessuna violazione con 3 eventi. Il modello deve processare l'intera traccia prima di determinare l'esito.

---

### 5.4 Controllo della Diversità

Due vincoli paralleli limitano la ripetizione strutturale:
- `ACTION_PATTERN_REUSE_CAP = 2`: ogni sequenza di azioni può apparire al massimo 2 volte
- `MAX_ACTION_PATTERN_REUSE_TOTAL = 20`: il numero totale di record con qualsiasi pattern ripetuto non supera 20

Soft fallback per la difficoltà: dopo `SLOT_EARLY_FALLBACK_ATTEMPTS = 180` tentativi falliti su un slot, il generatore accetta il tier di difficoltà più vicino disponibile invece di crashare.

---

### 5.5 Invarianti (9)

| Invariante | Cosa verifica |
|---|---|
| `no_duplicate_options` | Tutte le 5 scelte sono testualmente distinte |
| `fixed_option_set` | Le 5 scelte sono esattamente i label semantici attesi |
| `answer_in_choices` | La lettera di risposta corretta è presente nelle scelte |
| `correct_vehicle_not_missing_from_choices` | Il label del veicolo violatore appare nelle scelte |
| `undetermined_correct_only_for_no_violation` | "No vehicle can be determined" è corretto solo per record no_violation |
| `violation_step_none_only_for_no_violation` | `violation_step` è None solo per record no_violation |
| `all_events_valid_format` | Tutti gli eventi hanno i campi richiesti |
| `first_illegal_event_matches_metadata` | `metadata.violation_step` corrisponde al primo evento illegale del replay |
| `target_matches_replay` | La lettera di risposta corretta corrisponde all'attribuzione della violazione nel replay |

---

### 5.6 Problemi Risolti

**Problema 1 — `invalid_fsm_transition` come tipo di violazione:** Quando `apply_action()` incontrava una transizione FSM impossibile, il fallimento interno veniva restituito come tipo di violazione. Due record avevano questo come ground truth. *Causa radice:* `_replay_first_violation()` non distingueva tra fallimenti FSM interni e violazioni di dominio genuine. *Correzione:* Funzione patchata per rilevare i fallimenti FSM e restituire un sentinel no-violation. Aggiunta whitelist `ALLOWED_VIOLATION_TYPES` per validazione batch. *Risultato:* 0 record con `invalid_fsm_transition`.

**Problema 2 — Saturazione del cap di riuso (94/100 record affetti):** Con `ACTION_PATTERN_REUSE_CAP = 2`, il cap era saturato quasi universalmente: 43 pattern distinti apparivano esattamente 2 volte. *Correzione finale:* Sistema dual-constraint — cap rimane a 2 ma aggiunto `MAX_ACTION_PATTERN_REUSE_TOTAL = 20` come tetto globale. *Risultato:* Massimo ripetizioni = 2; record affetti ≤ 20.

**Problema 3 — Crash del generatore:** Abbassare il cap a 1 (primo tentativo di correzione del problema 2) causava: `RuntimeError: Unable to satisfy strict difficulty tier 'easy' at slot 98`. *Causa radice:* Lo spazio dei piani del tier easy si esauriva con cap=1 — sequenze da 2 eventi con violazione finale hanno un vocabolario intrinsecamente ristretto. Il meccanismo di slot-swap non trovava un partner dello stesso tier e crashava con errore hard. *Correzione:* Cap rialzato a 2, aggiunto `MAX_ACTION_PATTERN_REUSE_TOTAL`, convertita la gestione della difficoltà da hard-crash a soft-fallback dopo 180 tentativi per slot.

**Problema 4 — Instabilità della distribuzione difficoltà:** Multiple strategie tentate senza raggiungere bilanciamento stabile: enforcement stretto (crash), slot-swap con partner dello stesso tier (funziona ma non risolve l'esaurimento del plan space), cap tier-specifici (inconsistenza tra tier), retry budget più alto (riduce varianza ma non elimina lo sbilanciamento strutturale). La distribuzione finale easy: 30, medium: 37, hard: 33 è il miglior risultato raggiungibile — accettato come policy.

---

### 5.7 Limitazioni Accettate

**`roundabout_entry_no_yield` al ~31% delle violazioni.** La rotatoria ha esattamente una classe di violazione nel dominio corrente. I ~25 record di rotatoria con violazione sono necessariamente tutti `roundabout_entry_no_yield`. Proprietà del dominio, non errore di ingegneria — analogo alla sottorappresentazione di `turn right` nel Task 2.

**Distribuzione difficoltà ±7 dal target (easy 30, medium 37, hard 33).** Il tier medium assorbe naturalmente l'overflow dai tier easy e hard quando il loro plan space si esaurisce. Bias strutturale documentato e accettato.

**"Another vehicle (not A, B, or C)" è sempre sbagliato.** By design — testa la corretta delimitazione dello scope di risposta agli attori osservabili.

---

### 5.8 Stato Finale (v9)

| Metrica | Valore |
|---|---|
| Record | 100 |
| Versione generator | task3_violation_v9 |
| Invarianti | 9/9 passing, 0 failures |
| Distribuzione risposte | 20 per lettera (A–E) |
| Ambienti | intersection 34, roundabout 33, multi_lane_road 33 |
| Difficoltà | easy 30, medium 37, hard 33 |
| Tipi di violazione | forward_from_exit 10, intersection_right_of_way 8, lane_change_left 15, lane_change_right 13, roundabout_entry_no_yield 25, turn_without_entering 9, no_violation 20 |
| Record `invalid_fsm_transition` | 0 |
| `slot_attempts` massimo | 290 (2 slot medium); 85/100 risolti in 1 tentativo |
| Prompt unici | sì |
| ID unici | sì |

```bash
python generators/task3_violation.py --n 100 --seed 42 --out dataset/core/task3_violation.jsonl
```

---

## 6. Task 4 — Overlap Reasoning

### 6.1 Descrizione

Il Task 4 valuta la **certezza epistemica sotto ambiguità spaziale parziale**: quando più veicoli si sovrappongono all'interno di un incrocio o di una rotatoria, o quando l'ordine relativo su una strada a più corsie non è specificato, il modello deve separare ciò che è derivabile con certezza da ciò che è solo plausibile.

A differenza dei Task 1–3, le opzioni sono affermazioni complete (non semplici etichette di posizione o nomi di veicoli). Esattamente una affermazione è certamente vera; due sono quasi vere (incerte); due sono chiaramente false (contraddette dallo stato ricostruito dal replay).

**Ambienti usati:** `intersection` (50 record), `roundabout` (30 record), `multi_lane_road` (20 record).

**Domanda in ogni esempio:** "Which of the following statements is certainly true at the end of the sequence?"

---

### 6.2 Tipi di Scenario (6)

| Tipo | Record | Descrizione |
|---|---|---|
| `two_overlap_one_outside` | 20 | Due veicoli sovrapposti dentro l'incrocio, uno è fuori |
| `two_overlap_third_exited` | 10 | Due sovrapposti, il terzo è già uscito |
| `one_inside_one_exited_one_approach` | 20 | Uno dentro, uno uscito, uno in approccio |
| `roundabout_overlap` | 20 | Due veicoli sovrapposti nella rotatoria |
| `roundabout_non_entry` | 10 | Veicolo che non ha ancora tentato di entrare nella rotatoria |
| `multi_lane_positioning` | 20 | Posizionamento su strada a più corsie con ordine relativo non specificato |

---

### 6.3 Tassonomia degli Statement Quasi Veri (5 tipi epistemici)

| Tipo | Esempio |
|---|---|
| `spatial_present` | `"Vehicle X is ahead of Vehicle Y."` — stato di overlap corrente non verificabile |
| `moved_past` | `"Vehicle X has already moved past Vehicle Y."` — posizione relativa durante presenza simultanea |
| `past_overlap` | `"Vehicle X was ahead of Vehicle Y inside the intersection."` — incertezza overlap passata; verificato tramite replay della coppia referenziata |
| `will_future` | `"Vehicle X will exit before Vehicle Y enters the intersection."` — predizione futura non derivabile dallo stato finale |
| `lane_order_unknown` | `"Vehicle X is ahead of Vehicle Y on the road."` — ordine di corsia non specificato su multi_lane |

---

### 6.4 Validatore Indipendente

`scripts/validate_task4.py` è separato dal generatore e verifica autonomamente:
1. Ricostruzione dello scenario dal JSON
2. Replay degli eventi via `apply_action()`
3. Classificazione indipendente di ogni affermazione (`true/uncertain/false`) — i handler past-tense richiedono che la coppia di veicoli referenziata abbia effettivamente prodotto overlap durante il replay
4. Ricalcolo dell'unica opzione certamente vera
5. Confronto con la risposta dichiarata
6. Contatori di qualità: prompt duplicati, leak di certezza nei near_true, contaminazione cross-environment

`scripts/validate_task4_distribution.py` aggiunge controlli di distribuzione e pattern per il Task 4: bilanciamento lettere/ambienti/difficoltà, cap di riuso e riconoscimento delle famiglie di statement.

---

### 6.5 Salvaguardie Anti-Collapse

I cap di riuso scalano con la dimensione del dataset (parametro `n`):
- `CORRECT_TEXT_CAP = max(20, ceil(n/15))`: limita il riuso del testo della risposta corretta
- `EVENT_SIG_CAP = max(20, ceil(n/10))`: limita il riuso della firma della sequenza di eventi
- `statement_signature_cap = max(6, ceil(n/25))`: limita il riuso per ogni combinazione `(environment, scenario_type, certainly_true_category, normalized_choices)`

Il bilanciamento delle menzioni di veicoli nelle risposte corrette è controllato tramite soft gap cap per risposte single-vehicle e hard gap cap di 10 per le menzioni di veicoli.

---

### 6.6 Gap di Vocabolario del Dominio

**Problema noto:** `domain/scenario.py` usa `left_lane/center_lane/right_lane` per `multi_lane_road`, ma `domain/vocabulary.py` non espone label per questi tre stati.

**Workaround locale (read-only):** Il Task 4 usa una mappatura interna al generatore:
- `left_lane → the left lane`
- `center_lane → the center lane`
- `right_lane → the right lane`

Nessuna modifica è stata apportata a `domain/`. Reviewers esterni devono essere consapevoli di questo gap quando leggono `domain/vocabulary.py`.

---

### 6.7 Invarianti (13)

| # | Cosa verifica |
|---|---|
| 1 | L'affermazione corretta è replay-true |
| 2 | Entrambi i `near_true` sono replay-uncertain |
| 3 | Entrambi i `highly_false` sono replay-false |
| 4 | Cinque affermazioni distinte |
| 5 | Nessun label di posizione cross-environment |
| 6 | Il replay corrisponde a `audit.final_state` |
| 7 | Overlap rilevato (eccetto categorie valide senza overlap: `containment_non_entry`, `lane_position`) |
| 8 | Almeno due veicoli attivi |
| 9 | Nessun action streak di lunghezza 3 |
| 10 | Nessun pattern ABAB degli attori |
| 11 | Solo ID canonici `Vehicle A/B/C` |
| 12 | Lettera di risposta segue il key schedule |
| 13 | Copertura completa di `audit.option_rationale` |

---

### 6.8 Stato Finale (v6)

| Metrica | Valore |
|---|---|
| Record | 100 |
| Versione generator | task4_overlap_v6 |
| `wrong` | 0 |
| `invalid` | 0 |
| Distribuzione risposte | A/B/C/D/E = 20 ciascuno |
| Ambienti | intersection 50, roundabout 30, multi_lane_road 20 |
| Difficoltà | easy 33, medium 33, hard 34 |
| Menzioni veicoli nelle risposte corrette | A=34, B=33, C=35 |
| `event_sig_max` | 4 (cap=20) |
| `correct_text_max` | 4 (cap=20) |
| Past-overlap pair mismatch | 0 |
| Prompt unici | sì |
| ID unici | sì |

```bash
python generators/task4_overlap.py --seed 42 --out dataset/core/task4_overlap.jsonl
# Validazione indipendente:
python scripts/validate_task4.py --input dataset/core/task4_overlap.jsonl
python scripts/validate_task4_distribution.py --input dataset/core/task4_overlap.jsonl
```

---

## 7. Evaluation Design

### 7.1 Modelli e Gruppi di Valutazione

I modelli da valutare sono organizzati in **6 gruppi basati su ipotesi** — ogni gruppo testa una domanda di ricerca specifica, non solo una classe di dimensione.

| Gruppo | Modelli | Ipotesi testata |
|---|---|---|
| G1 — Small / sub-4B | Qwen2.5-1.5B-Instruct, Phi-3.5-mini-instruct (3.8B) | I modelli piccoli possono gestire Task 1–2 ma degradano su Task 3–4? |
| G2 — Mid-range 7–8B | Mistral-7B-Instruct-v0.3, LLaMA-3.1-8B-Instruct | Il salto da sub-4B a 7–8B produce miglioramenti misurabili su tutte le task? |
| G3 — MoE / sparse | Mixtral-8x7B-Instruct | Un'architettura sparse con capacità equivalente a un modello denso più grande produce profili di errore diversi? |
| G4 — Frontier closed | GPT-4o, altro modello frontier proprietario | Qual è il soffitto di performance? Il benchmark rimane impegnativo anche per i modelli frontier? |
| G5 — Multilingual | (uno dei modelli G1/G2 con training multilingue) | Il training multilingue aiuta o ostacola il ragionamento strutturale in inglese? |
| G6 — Reasoning-specialized | QwQ-32B (o equivalente) | I modelli con chain-of-thought esplicito outperformano i modelli instruction-tuned standard a parità di task? |

---

### 7.2 Allineamento Task–Gruppo

| Task | G1 small | G2 mid | G3 MoE | G4 frontier | G5 multilingual | G6 reasoning |
|---|---|---|---|---|---|---|
| T1 Position Tracking | Baseline | Confronto | Confronto | Soffitto | Confronto | Principale |
| T2 Right-of-Way | Baseline | Confronto | Confronto | Soffitto | Rule retrieval | Principale |
| T3 Violation Detection | Stress test | Confronto | Confronto | Soffitto | Confronto | Principale |
| T4 Overlap Reasoning | Stress test | Confronto | Confronto | Soffitto | Confronto | Focus primario |

---

### 7.3 Domande di Ricerca

**RQ1 — State tracking scalability:** L'accuratezza degrada all'aumentare della lunghezza della sequenza nel Task 1? La degradazione è uniforme tra i livelli easy/medium/hard o è discontinua?

**RQ2 — Rule retrieval vs tracking:** Esiste un profilo di errore distinto tra Task 1 (tracking puro) e Task 2 (regole statiche)? I modelli che eccellono nel tracking sono anche buoni nel rule retrieval?

**RQ3 — Temporal attribution:** Il Task 3 introduce un costo misurabile rispetto ai Task 1 e 2 presi separatamente? L'errore è concentrato nell'attribuzione temporale (sbagliare il veicolo) o nel rilevamento della violazione?

**RQ4 — Epistemic calibration:** Nel Task 4, i modelli selezionano distrattori `near_true` più spesso di quanto atteso per caso? Questo segnalerebbe una difficoltà specifica nel distinguere certezza da plausibilità.

**RQ5 — Size vs architecture:** L'accuratezza correla monotonicamente con la dimensione del modello, o ci sono architetture (MoE, reasoning-specialized) che spezzano questa correlazione?

**RQ6 — Frontier ceiling:** I modelli frontier (G4) risolvono il benchmark con accuratezza vicina al 100%, o rimane spazio significativo di miglioramento?

---

### 7.4 Piano di Valutazione Baseline

**Fase 1 — Modelli prioritari:**
1. LLaMA-3.1-8B-Instruct (G2 — reference point mid-range)
2. GPT-4o (G4 — soffitto di performance)
3. Qwen2.5-1.5B-Instruct (G1 — lower bound)
4. Mistral-7B-Instruct-v0.3 (G2 — confronto mid-range)

**Trigger per espandere la valutazione:**
- Se G1 ha accuratezza >70% su Task 1 → includere modelli ancora più piccoli
- Se G4 supera il 95% su tutte le task → riesaminare la difficoltà del benchmark per il frontier
- Se la differenza G2–G4 è <10pp su Task 4 → riesaminare la difficoltà del task
- Se G2 performa meglio di G3 → aggiungere Mixtral per analisi dell'architettura
- Se i profili di errore di G1 e G2 convergono su Task 3–4 → la struttura degli errori è più informativa del gap di performance

**Protocollo di valutazione:**
- Prompt: `{testo benchmark}\n\nAnswer with only the letter of your choice (A, B, C, D, or E).`
- Parsing: normalizzazione alla prima lettera valida (`"B"`, `"B."`, `"(B)"`, `"The answer is B"` → `"B"`)
- Decoding: greedy (`do_sample=False`)
- `max_new_tokens`: 16
- Ogni run salvato riga per riga con flag `--resume` per ripresa sicura in caso di interruzione

---

## 8. Dataset Validation

### 8.1 Framework a Tre Livelli

La validazione del dataset è organizzata in tre livelli progressivi:

**Layer 1 — Validazione automatica** completata

Ogni dataset è verificato da un validator indipendente che replica la simulazione degli eventi da zero (senza accesso al codice del generatore) e confronta il ground truth con la risposta dichiarata.

Controlli comuni a tutti i task:
- Replay deterministico degli eventi dall'initial state
- Verifica che la risposta corretta sia l'unica derivabile dallo stato finale
- `wrong = 0`, `invalid = 0` su tutti e 4 i task
- 0 prompt duplicati, distribuzione risposte uniforme (20 per lettera)
- 0 violazioni degli invarianti

**Layer 2 — Audit di leggibilità umana** *(da eseguire)*

Campione: 10 esempi per task (40 totali), stratificati per difficoltà e ambiente.

Checklist per ogni esempio:
1. Il testo del prompt è grammaticalmente corretto e non ambiguo?
2. La descrizione dello scenario corrisponde alle posizioni specificate?
3. La sequenza di eventi è narrativamente coerente?
4. La risposta corretta è verificabile manualmente dalla sequenza?
5. I distrattori near_true richiedono effettivamente ragionamento per essere esclusi?
6. I distrattori highly_false sono chiaramente impossibili nel contesto dato?
7. Nessuna opzione contiene formulazioni che possano suggerire la risposta corretta?

**Layer 3 — Shortcut audit a livello di dataset** *(da eseguire)*

Sei euristiche di shortcut vengono testate computazionalmente sull'intero dataset:

| Euristica | Task | Test |
|---|---|---|
| H1 — Answer-letter frequency | Tutti | Deviazione massima da 20/100 per lettera |
| H2 — First-vehicle shortcut | T1, T3 | % risposte corrette corrispondenti a Vehicle A |
| H3 — Direction-only | T2 | Accuratezza euristica direction-only su incroci |
| H4 — Alphabetical priority | T2 | Accuratezza euristica alphabetical-first-non-left su incroci |
| H5 — Last-event attribution | T3 | % violazioni all'ultimo evento della sequenza |
| H6 — Dominant statement type | T4 | Proporzione massima di un singolo tipo di affermazione |

Soglie accettabili: H1 ≤ ±2 dal target, H3 = 0%, H4 = 0%, H5 ≤ 60%, H2 ≤ 40%, H6 ≤ 35%.
