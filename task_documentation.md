# Task Documentation — Traffic Benchmark

**Autore:** Adam Amrani  
**Ultimo aggiornamento:** Aprile 2026

Questo documento raccoglie, per ogni task del benchmark, i vincoli progettuali (paletti), i bug riscontrati durante la validazione, e le soluzioni adottate. La struttura è pensata per essere estesa man mano che vengono sviluppate le task successive.

---

## Indice

- [1. Task 1 — Position Tracking](#1-task-1--position-tracking)
  - [1.1 Descrizione della task](#11-descrizione-della-task)
  - [1.2 Parametri di generazione](#12-parametri-di-generazione)
  - [1.3 Paletti di design](#13-paletti-di-design)
  - [1.4 Bug riscontrati in fase di validazione](#14-bug-riscontrati-in-fase-di-validazione)
  - [1.5 Metriche prima/dopo i fix](#15-metriche-primadopo-i-fix)
  - [1.6 Validazione finale](#16-validazione-finale)
  - [1.7 Note e osservazioni](#17-note-e-osservazioni)

---

# 1. Task 1 — Position Tracking

## 1.1 Descrizione della task

Il Task 1 valuta la capacita del modello di seguire la posizione di piu veicoli attraverso una sequenza di azioni e rispondere correttamente alla domanda "Where is Vehicle X at the end of the sequence?".

Il modello deve applicare mentalmente ogni passo in ordine, aggiornare la propria rappresentazione interna dello stato, e non perdere traccia di nessun veicolo. E il task di state tracking piu puro del benchmark.

**File generatore:** `generators/task1_position.py`  
**Output:** `dataset/core/task1_position.jsonl`

---

## 1.2 Parametri di generazione

| Parametro | Valore |
|---|---|
| Ambienti | `multi_lane_road`, `intersection` (roundabout esclusa) |
| Numero veicoli | 3 |
| Numero step | 2–4 |
| Opzioni per domanda | 5 (A–E) |
| Distribuzione opzioni | 1 corretta, 2 near_true, 2 highly_false |
| Esempi target (core) | 300 (attualmente 100 per test) |
| Bilanciamento risposte | key_schedule con esattamente N/5 per lettera |

**Azioni valide per ambiente (dopo i fix):**

- `multi_lane_road`: `CHANGE_LEFT`, `CHANGE_RIGHT`
- `intersection`: `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`

---

## 1.3 Paletti di design

Questi sono i vincoli progettuali individuati durante la fase di progettazione del generatore, prima della scrittura del codice. Ogni paletto rappresenta un requisito che il generatore deve soddisfare per produrre esempi validi e di qualita.

### 1.3.1 Collisioni di corsia (multi_lane)

**Problema:** su `multi_lane_road` due veicoli non dovrebbero occupare la stessa corsia contemporaneamente. Se B e gia in `left_lane` e A prova `CHANGE_LEFT` dalla `center_lane`, il codice deve gestire il conflitto.

**Soluzione iniziale:** `safe_apply_action()` controllava che la corsia di destinazione fosse libera prima di un `CHANGE_LEFT`/`CHANGE_RIGHT`. Se occupata, restituiva `None` e il generatore scartava l'azione.

**Nota:** questo vincolo e stato successivamente rilassato — vedi Bug 1 nella sezione 1.4.

### 1.3.2 Azioni ai bordi (multi_lane)

**Problema:** `CHANGE_LEFT` dalla `left_lane` e fisicamente impossibile. Stesso per `CHANGE_RIGHT` dalla `right_lane`. Un generatore che non lo controlla produce scenari invalidi.

**Soluzione:** `safe_apply_action()` verifica l'indice della corsia corrente prima di eseguire il cambio. Se il veicolo e gia al bordo, l'azione viene rifiutata (`return None`) e il generatore riprova con un'altra azione.

**Codice rilevante:**
```python
if action == Action.CHANGE_LEFT and idx <= 0:
    return None   # already leftmost
if action == Action.CHANGE_RIGHT and idx >= len(LANE_ORDER) - 1:
    return None   # already rightmost
```

### 1.3.3 Veicolo interrogato mai mosso

**Problema:** se la domanda riguarda Vehicle A ma la sequenza muove solo B e C, il modello puo rispondere correttamente ignorando tutti gli eventi — e uno shortcut che abbassa la qualita del dataset.

**Soluzione:** `generate_sequence()` forza il queried vehicle ad agire almeno una volta. Se all'ultimo step il veicolo non si e ancora mosso, viene selezionato obbligatoriamente.

**Raffinamento successivo:** per l'ambiente `intersection` il minimo e stato portato a 2 azioni (vedi Bug 2 nella sezione 1.4).

### 1.3.4 Posizione finale uguale a quella iniziale

**Problema:** se la sequenza produce per il veicolo interrogato una posizione finale identica a quella iniziale, la domanda e banale o irrisolvibile. Il modello potrebbe rispondere correttamente senza seguire nessun evento.

**Soluzione:** al termine della simulazione, se `final_position == start_position`, la sequenza viene scartata e il generatore riprova. Questo controllo e implementato nel loop principale di `generate_sequence()`:

```python
final_pos = trial_state.get_vehicle(queried_vid).position
if final_pos == start_pos:
    continue
```

### 1.3.5 Distrattori intelligenti

**Problema:** le opzioni errate devono essere credibili e diversificate. Due opzioni "quasi vere" devono essere posizioni raggiungibili, due "chiaramente false" devono essere impossibili nel contesto.

**Soluzione:** `build_choices()` costruisce le 5 opzioni con questa logica:

| Tipo | Strategia |
|---|---|
| **correct** | posizione finale calcolata dalla simulazione |
| **near_true 1** | posizione iniziale del veicolo interrogato |
| **near_true 2** | posizione intermedia della sequenza, oppure corsia adiacente |
| **highly_false 1** | posizione di un ambiente diverso (es. "roundabout lane" su multi_lane) |
| **highly_false 2** | altra posizione di un ambiente diverso (es. "northern exit" su multi_lane) |

Se il near_true 2 non puo essere determinato (nessuna posizione intermedia, nessuna corsia adiacente), il codice usa un fallback: sceglie una posizione casuale dello stesso ambiente che non sia gia stata usata.

### 1.3.6 Bilanciamento delle risposte

**Problema:** senza controllo esplicito il generatore tenderebbe a piazzare la risposta corretta sempre nella stessa posizione (es. sempre "A"), creando un bias facilmente sfruttabile.

**Soluzione:** `key_schedule` costruisce esattamente N/5 occorrenze di ciascuna lettera A–E e le mescola con `random.shuffle()` prima di iniziare la generazione. Ogni esempio riceve la sua lettera target, e `assign_letters()` piazza la risposta corretta esattamente in quella posizione.

```python
assert n % 5 == 0, "N_EXAMPLES must be a multiple of 5 for balanced keys"
key_schedule = []
per_key = n // 5
for letter in LETTERS:
    key_schedule.extend([letter] * per_key)
random.shuffle(key_schedule)
```

### 1.3.7 Svolta solo dall'interno dell'incrocio

**Problema:** un veicolo in posizione di approach (es. `north_approach`) non puo svoltare direttamente — deve prima entrare nell'incrocio con `MOVE_FORWARD`.

**Soluzione:** `safe_apply_action()` blocca `TURN_LEFT` e `TURN_RIGHT` se `inside_intersection` e `False`:

```python
if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
    if not v.inside_intersection:
        return None   # must MOVE_FORWARD first
```

---

## 1.4 Bug riscontrati in fase di validazione

Dopo la scrittura della prima versione del generatore, lo script e stato eseguito e il dataset generato e stato analizzato. Sono emersi 4 bug, tutti risolti.

### 1.4.1 BUG 1 — Deadlock multi_lane (CRITICO)

**Sintomo:** il 100% degli esempi generati era di tipo `intersection`, con 0% `multi_lane_road`. Il generatore provava a creare scenari multi_lane ma falliva sistematicamente, ripiegando sempre su intersection.

**Causa:** `build_multi_lane_scenario()` piazza 3 veicoli su 3 corsie (`left`, `center`, `right`). Con tutte e tre le corsie occupate, il collision check in `safe_apply_action()` bloccava ogni tentativo di `CHANGE_LEFT`/`CHANGE_RIGHT` perche la corsia di destinazione risultava sempre occupata. Inoltre, `MOVE_FORWARD` non cambia la posizione su `multi_lane_road` (il veicolo resta nella stessa corsia), quindi era un no-op. Risultato: nessun veicolo poteva mai cambiare posizione, ogni sequenza veniva scartata perche `final_pos == start_pos`, e il budget di retry si esauriva.

**Analisi dettagliata del deadlock:**

```
Stato iniziale:  A=left_lane, B=center_lane, C=right_lane

A vuole CHANGE_RIGHT → center_lane occupata da B → BLOCCATO
B vuole CHANGE_LEFT  → left_lane occupata da A   → BLOCCATO
B vuole CHANGE_RIGHT → right_lane occupata da C  → BLOCCATO
C vuole CHANGE_LEFT  → center_lane occupata da B → BLOCCATO
MOVE_FORWARD per chiunque → nessun cambio di posizione → NO-OP

Nessuna sequenza valida possibile.
```

**Soluzione (due interventi):**

1. **Rimosso il collision check per i cambi di corsia.** Nel modello discreto, due veicoli sulla stessa corsia si trovano a punti diversi lungo la strada. Condividere una corsia e fisicamente plausibile e non crea incoerenze nel position tracking.

2. **Rimosso `MOVE_FORWARD` dalle azioni multi_lane.** Dato che `MOVE_FORWARD` non modifica la posizione della corsia, includerlo nel pool di azioni serviva solo a sprecare passi e rendere piu difficile la generazione di sequenze valide.

**Codice rimosso (collision check):**
```python
# PRIMA — bloccava tutto:
if action in (Action.CHANGE_LEFT, Action.CHANGE_RIGHT):
    dest_idx = idx - 1 if action == Action.CHANGE_LEFT else idx + 1
    dest_lane = LANE_ORDER[dest_idx]
    for other in state.vehicles:
        if other.id != vehicle_id and other.position == dest_lane:
            return None   # lane occupied
```

**Codice modificato (azioni):**
```python
# PRIMA:
ACTIONS_BY_ENV = {
    Environment.MULTI_LANE: [Action.MOVE_FORWARD, Action.CHANGE_LEFT, Action.CHANGE_RIGHT],
    ...
}

# DOPO:
ACTIONS_BY_ENV = {
    Environment.MULTI_LANE: [Action.CHANGE_LEFT, Action.CHANGE_RIGHT],
    ...
}
```

---

### 1.4.2 BUG 2 — Monopolio "inside the intersection" (CRITICO)

**Sintomo:** il 92% delle risposte corrette era "inside the intersection". Un modello che rispondesse sempre con questa stringa avrebbe ottenuto un'accuratezza del 92%, rendendo il benchmark inutile.

**Causa:** due problemi combinati.

1. `MOVE_FORWARD` non veniva bloccato quando il veicolo era gia dentro l'incrocio. Questo significava che un veicolo poteva fare `MOVE_FORWARD` piu volte di seguito: la prima volta entrava nell'incrocio (utile), le successive erano no-op che sprecavano passi senza produrre svolte.

2. Il queried vehicle era obbligato ad agire solo una volta. Una singola azione `MOVE_FORWARD` bastava a soddisfare il vincolo "veicolo interrogato mosso almeno una volta", ma produceva invariabilmente la posizione "inside the intersection" come risposta.

**Soluzione (due interventi):**

1. **Blocco di `MOVE_FORWARD` quando gia inside.** Aggiunto un guard in `safe_apply_action()` che impedisce `MOVE_FORWARD` se il veicolo e gia dentro l'incrocio:

```python
if action == Action.MOVE_FORWARD and v.inside_intersection:
    return None
```

Questo forza il generatore a scegliere `TURN_LEFT` o `TURN_RIGHT` per un veicolo gia dentro l'incrocio, producendo posizioni di uscita diverse.

2. **Minimo 2 azioni per il queried vehicle su intersection.** Modificata `generate_sequence()` per richiedere almeno 2 mosse del veicolo interrogato nell'ambiente intersection (una per entrare, una per svoltare). Per multi_lane 1 azione e sufficiente perche ogni cambio corsia modifica la posizione:

```python
min_queried_moves = 2 if env == Environment.INTERSECTION else 1
```

La logica di scheduling riserva abbastanza step rimanenti per garantire che il veicolo interrogato raggiunga il minimo:

```python
remaining = n_steps - step_idx
queried_deficit = min_queried_moves - queried_move_count
if queried_deficit >= remaining:
    vid = queried_vid   # force queried vehicle
```

---

### 1.4.3 BUG 3 — Eventi no-op MOVE_FORWARD (MAGGIORE)

**Sintomo:** 25 eventi su circa 300 totali erano `MOVE_FORWARD` applicati a veicoli gia dentro l'incrocio. Il testo dell'evento veniva generato ("Vehicle A moves forward.") ma lo stato non cambiava.

**Causa:** `apply_action()` eseguiva `MOVE_FORWARD` impostando `inside_intersection = True` e `position = "inside_intersection"`, anche se il veicolo era gia in quello stato. L'evento veniva aggiunto al log ma era semanticamente vuoto.

**Impatto:** gli eventi no-op inquinavano la sequenza con informazioni fuorvianti. Un LLM potrebbe interpretare "Vehicle A moves forward" come un cambiamento reale, portando a errori di ragionamento.

**Soluzione:** risolto dal guard aggiunto per il Bug 2 (`MOVE_FORWARD` bloccato se `inside_intersection == True`). Non e stato necessario un intervento separato.

---

### 1.4.4 BUG 4 — Loop di ri-entrata exit-to-inside (MAGGIORE)

**Sintomo:** 4 esempi contenevano sequenze dove un veicolo svoltava (raggiungendo un'uscita), poi faceva di nuovo `MOVE_FORWARD`, rientrando nell'incrocio. Esempio:

```
1. Vehicle C moves forward.       → C entra nell'incrocio
2. Vehicle C turns left.           → C esce a south_exit
3. Vehicle C moves forward.        → C rientra nell'incrocio (!)
4. Vehicle B moves forward.        → B entra nell'incrocio
```

**Causa:** dopo una svolta, `inside_intersection` viene impostato a `False` e la posizione diventa `{direction}_exit`. Il guard del Bug 2 controllava solo `inside_intersection`, che a quel punto era `False`, quindi `MOVE_FORWARD` veniva permesso. `apply_action()` rimetteva il veicolo a `inside_intersection` — creando un ciclo assurdo di entrata-uscita-rientro.

**Soluzione:** aggiunto un guard specifico in `safe_apply_action()` che blocca `MOVE_FORWARD` quando il veicolo si trova in una posizione di uscita:

```python
if action == Action.MOVE_FORWARD and v.position.endswith("_exit"):
    return None
```

Dopo il fix, un veicolo che ha raggiunto un'uscita non puo piu rientrare nell'incrocio.

---

## 1.5 Metriche prima/dopo i fix

| Metrica | Prima dei fix | Dopo i fix |
|---|---|---|
| Distribuzione ambienti | 100% intersection | ~47% intersection, ~53% multi_lane |
| "inside the intersection" come risposta | 92% | 0% |
| Posizioni corrette distinte | 2 | 7 |
| Eventi no-op MOVE_FORWARD | 25 | 0 |
| Loop di ri-entrata | 4 | 0 |
| MOVE_FORWARD ripetuti (stesso veicolo) | 35 | 0 |
| Bilanciamento risposte A–E | 20-20-20-20-20 | 20-20-20-20-20 |
| Testi duplicati nelle opzioni | 0 | 0 |

**Distribuzione delle risposte corrette dopo i fix:**

```
the center lane:     34
the western exit:    14
the northern exit:   12
the eastern exit:    11
the left lane:       10
the right lane:      10
the southern exit:    9
```

**Distribuzione delle azioni dopo i fix:**

```
moves forward:              88
changes to the right lane:  86
changes to the left lane:   75
turns left:                 28
turns right:                26
```

---

## 1.6 Validazione finale

Tutti i 100 esempi generati dopo i fix sono stati verificati tramite un **replay indipendente della simulazione**: uno script separato ricostruisce lo stato iniziale dai dati JSON, riapplica ogni evento passo per passo, calcola la posizione finale del veicolo interrogato, e confronta il risultato con la risposta dichiarata.

**Risultato: 0 errori su 100 esempi — tutte le risposte sono logicamente corrette.**

Lo script di validazione verifica anche che non ci siano testi duplicati tra le 5 opzioni di ogni domanda.

---

## 1.7 Note e osservazioni

### Esempio della documentazione non riproducibile

L'esempio illustrativo nella sezione 3.1 di `project_documentation.md` per l'ambiente `multi_lane_road` descrive una meccanica di "push" tra veicoli: quando B si sposta nella corsia di A, A viene implicitamente spostato nella corsia adiacente. Questa meccanica non e implementata nel codice (ne sarebbe semplice farlo in modo coerente). L'esempio va considerato puramente illustrativo.

### Esclusione della roundabout

La roundabout e esclusa dal Task 1 perche le posizioni interne sono poco distinguibili — un veicolo dentro la rotatoria ha sempre posizione `roundabout_lane`, senza distinzione spaziale. Questo la rende piu adatta al Task 4 (Overlap Reasoning).

### STOP escluso dalle azioni

L'azione `STOP` non e inclusa nel pool di azioni del Task 1 perche non modifica la posizione del veicolo. Includerla aggiungerebbe solo rumore senza valore per il position tracking. Resta disponibile per altri task (es. Task 3 — Violation Detection).

### Scalabilita

Il generatore e configurato per produrre 100 esempi (`N_EXAMPLES = 100`). Per il Core Dataset finale il valore va portato a 300. Il key_schedule richiede che `N_EXAMPLES` sia un multiplo di 5 per garantire il bilanciamento perfetto delle risposte.

---

<!-- TEMPLATE PER LE TASK SUCCESSIVE

# N. Task N — Nome della Task

## N.1 Descrizione della task
## N.2 Parametri di generazione
## N.3 Paletti di design
### N.3.1 Paletto 1
### N.3.2 Paletto 2
...
## N.4 Bug riscontrati in fase di validazione
### N.4.1 BUG 1 — Titolo (GRAVITA)
### N.4.2 BUG 2 — Titolo (GRAVITA)
...
## N.5 Metriche prima/dopo i fix
## N.6 Validazione finale
## N.7 Note e osservazioni

-->
