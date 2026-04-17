# Domain Documentation — Traffic Benchmark

**Autore:** Adam Amrani  
**Data:** Aprile 2026  
**Ultima revisione:** Aprile 2026

---

## Indice

- [1. Panoramica del dominio](#1-panoramica-del-dominio)
- [2. Struttura del modulo `domain/`](#2-struttura-del-modulo-domain)
- [3. Bug e limitazioni — Task 1 (Position Tracking)](#3-bug-e-limitazioni--task-1-position-tracking)
- [4. Bug e limitazioni — Task 2 (Right-of-Way Reasoning)](#4-bug-e-limitazioni--task-2-right-of-way-reasoning)
- [5. Fix applicati al codice](#5-fix-applicati-al-codice)
- [6. Tavola di riepilogo](#6-tavola-di-riepilogo)
- [7. Note progettuali e decisioni di design](#7-note-progettuali-e-decisioni-di-design)

---

## 1. Panoramica del dominio

Il modulo `domain/` è il **contratto condiviso** tra tutti i generatori del benchmark. Definisce le entità del mondo simulato, le regole del traffico, il motore di simulazione e il renderer in linguaggio naturale. Nessun generatore può inventare regole o posizioni fuori da questo contratto.

Il dominio modella uno scenario di traffico stradale discreto con tre ambienti:

| Ambiente | Descrizione |
|---|---|
| `intersection` | Incrocio a quattro vie senza semaforo né segnali |
| `multi_lane_road` | Strada a tre corsie (left, center, right), direzione nord |
| `roundabout` | Rotatoria con corsie di approccio e un anello interno |

Le **azioni** disponibili per i veicoli sono definite in `Action` (enum in `entities.py`) e filtrate per ambiente in `ACTIONS_BY_ENV` (nei generatori) e `VALID_ACTIONS` (in `rules.py`).

---

## 2. Struttura del modulo `domain/`

```
domain/
├── entities.py     # Classi base: Vehicle, ScenarioState, Enum (Direction, Action, Lane, ...)
├── rules.py        # Regole del traffico (precedenza, violazioni, overlap)
├── scenario.py     # Builder degli scenari + apply_action() (motore di simulazione)
└── render.py       # Conversione stato → linguaggio naturale per i prompt LLM
```

### `entities.py`

Definisce tramite `dataclass` ed `Enum` tutti gli oggetti base:

- `Direction` — 4 punti cardinali: `NORTH`, `SOUTH`, `EAST`, `WEST`
- `Action` — 8 azioni: `MOVE_FORWARD`, `STOP`, `TURN_LEFT`, `TURN_RIGHT`, `CHANGE_LEFT`, `CHANGE_RIGHT`, `ENTER_ROUNDABOUT`, `EXIT_ROUNDABOUT`
- `Environment` — 3 ambienti: `INTERSECTION`, `MULTI_LANE`, `ROUNDABOUT`
- `Lane` — 4 posizioni di corsia: `LEFT`, `CENTER`, `RIGHT`, `ROUNDABOUT_LANE`
- `IntentDirection` — intenzione dichiarata (solo Task 2): `GO_STRAIGHT`, `TURN_LEFT`, `TURN_RIGHT`
- `Vehicle` — veicolo con campi `id`, `position`, `direction`, `intent`, `inside_intersection`, `stopped`
- `ScenarioState` — stato globale: lista di veicoli, ambiente, step corrente, log degli eventi

### `rules.py`

Implementa le regole per tutti e tre gli ambienti:

- `right_of_way_intersection(v1, v2)` — regola della precedenza a destra tramite lookup su `APPROACH_PRIORITY`
- `right_of_way_roundabout(v_inside, v_entering)` — chi è dentro ha sempre priorità
- `right_of_way(v1, v2, env)` — dispatcher generico per ambiente
- `is_violation_*` — funzioni di rilevamento violazioni (stop sign, precedenza, rotatoria)
- `get_valid_actions(env)` — lista azioni ammesse per ambiente (a scopo generale, non task-specifico)
- `vehicles_overlap(v1, v2)` — True se entrambi i veicoli sono dentro l'incrocio contemporaneamente

### `scenario.py`

Builder degli scenari iniziali e motore di simulazione:

- `build_intersection_scenario(num_vehicles, with_intent)` — campiona direzioni distinte
- `build_multi_lane_scenario(num_vehicles)` — assegna corsie distinte
- `build_roundabout_scenario(num_vehicles)` — primo veicolo dentro, gli altri in approccio
- `apply_action(state, vehicle_id, action)` — muta lo stato e restituisce l'evento in linguaggio naturale
- `_rotate_direction(direction, turn)` — rotazione cardinale per le svolte
- `_lane_index(position)` — indice della corsia corrente (-1 se non è una corsia)

### `render.py`

Conversione pura da oggetti Python a testo leggibile da un LLM:

- `describe_vehicle(v)` — descrive lo stato iniziale di un veicolo
- `describe_scenario(state)` — descrive l'intera configurazione iniziale
- `describe_events(events)` — formatta la lista di eventi come sequenza numerata
- `render_prompt(scenario_text, events, question, choices)` — assembla il prompt completo A–E

---

## 3. Bug e limitazioni — Task 1 (Position Tracking)

Questa sezione documenta tutti i problemi identificati nel generatore `generators/task1_position.py` e nel sottostante modello di dominio utilizzato dal Task 1. Per ogni bug è indicato lo stato: **CORRETTO** (fix applicato al codice), **NOTO** (limitazione progettuale documentata e consapevolmente accettata) o **PARZIALE** (parzialmente mitigato).

---

### T1-B01 — Semantica delle azioni non coerente tra `MOVE_FORWARD` e `TURN_*`
**Gravità:** Medio  
**Stato:** NOTO

`MOVE_FORWARD` porta un veicolo dall'approccio all'interno dell'incrocio, cambiando `position` e `inside_intersection`. Le azioni `TURN_LEFT` e `TURN_RIGHT` portano il veicolo dall'interno a un'uscita, cambiando `direction` e poi `position`. Le due azioni operano su strati semantici diversi (entra vs. esce) ma sono entrambe presentate all'LLM come azioni del veicolo senza distinguerle formalmente.

**Impatto sul benchmark:** Il modello può confondersi tra lo stato "è dentro l'incrocio" (dopo `MOVE_FORWARD`) e lo stato "è uscito" (dopo `TURN_*`), portando a errori di tracking. Questo è in parte il punto del benchmark — testare se il modello gestisce questa asimmetria.

**Decisione progettuale:** La sequenza obbligatoria `MOVE_FORWARD → TURN_*` è imposta da `safe_apply_action()` e garantita dalla generazione (`min_queried_moves = 2` per intersection). L'asimmetria rimane documentata come complessità intenzionale.

---

### T1-B02 — `TURN_*` implica un movimento implicito non formalizzato nel dominio base
**Gravità:** Medio  
**Stato:** PARZIALE

`apply_action()` in `scenario.py` non verifica che il veicolo sia `inside_intersection` prima di eseguire `TURN_LEFT` o `TURN_RIGHT`. Se chiamato direttamente (bypassando `safe_apply_action`), un veicolo in approccio potrebbe svoltare senza essere entrato.

**Fix applicato:** Il guard è in `safe_apply_action()` del generatore:
```python
if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
    if not v.inside_intersection:
        return None   # must MOVE_FORWARD first
```

**Limitazione residua:** Il domain layer (`apply_action`) non è auto-proteggente. Chiamate dirette senza `safe_apply_action` possono produrre stati invalidi. La validazione è responsabilità del chiamante.

---

### T1-B03 — Assenza di una FSM esplicita (stati e transizioni non definiti formalmente)
**Gravità:** Alto (architetturale)  
**Stato:** NOTO

Il ciclo di vita di un veicolo all'incrocio ha stati impliciti non formalizzati:

```
approach → inside_intersection → {direction}_exit
```

Questi stati derivano da combinazioni dei campi `position`, `inside_intersection` e `stopped`, ma non esiste nessuna classe `VehicleState` o tabella di transizioni che li renda espliciti. Questo significa che:
- Stati invalidi (es. `inside_intersection=True` + `position="left_lane"`) sono costruibili
- Non c'è modo di chiedere "quali azioni sono valide nello stato corrente?" senza logica esterna
- `safe_apply_action()` reimplementa la FSM implicitamente, producendo duplicazione

**Impatto:** Rende difficile estendere il dominio (Task 3, 4) senza rischi di inconsistenza.

**Decisione progettuale:** Refactoring a FSM esplicita non è stato fatto perché il Task 1 funziona correttamente con i guard attuali. È segnalato come debito tecnico per le task successive.

---

### T1-B04 — Mancanza di gestione degli stati terminali (`_exit`)
**Gravità:** Medio  
**Stato:** PARZIALE

Una volta che un veicolo raggiunge una posizione `{dir}_exit`, non è formalmente definito cosa succede. Il veicolo ha lasciato l'incrocio e non dovrebbe interagire più con gli altri. Tuttavia, il bug "loop di ri-entrata" (descritto in `task_documentation.md`, §1.4.4) mostrava che `MOVE_FORWARD` poteva riportare il veicolo dentro l'incrocio da un'uscita.

**Fix applicato in `safe_apply_action()`:**
```python
if action == Action.MOVE_FORWARD and v.position.endswith("_exit"):
    return None
```

**Limitazione residua:** L'uscita non è uno stato terminale hard nel domain: nulla impedisce ad `apply_action()` di essere chiamato su un veicolo `_exit` direttamente. L'invariante è mantenuto solo dal generatore.

---

### T1-B05 — Collisioni non definite (`multi_lane`)
**Gravità:** Medio  
**Stato:** NOTO (design consapevole)

Nel modello discreto, due veicoli possono occupare la stessa corsia contemporaneamente. Il collision check originale è stato rimosso (Bug 1 in `task_documentation.md`) perché causava deadlock totale: con 3 veicoli su 3 corsie, nessun cambio corsia era possibile.

**Decisione progettuale:** Nel modello discreto, due veicoli sulla stessa corsia sono implicitamente a posizioni diverse lungo la strada (diversi "segmenti"). Il position tracking riguarda la corsia (dimensione laterale), non la posizione longitudinale. La condivisione della corsia è fisicamente plausibile e non ambigua per il task.

**Conseguenza documentata:** Gli esempi `multi_lane` non modellano collisioni laterali reali. Il benchmark valuta il tracking delle corsie, non la gestione dei conflitti di occupazione.

---

### T1-B06 — Lane change senza vincoli di occupazione formali
**Gravità:** Basso  
**Stato:** NOTO

Strettamente correlato a T1-B05. Non esiste una regola che dica "non puoi entrare in una corsia occupata". Come documentato, questa è una scelta progettuale. Gli esempi risultanti sono comunque coerenti per il position tracking perché la domanda chiede "dove si trova il veicolo X" e la risposta è deterministica.

---

### T1-B07 — Azioni sempre valide in `apply_action()` (assenza di validazione nel domain layer)
**Gravità:** Medio  
**Stato:** NOTO + FIX PARZIALE

`apply_action()` in `scenario.py` non valida l'azione rispetto allo stato corrente. Esempi:
- `CHANGE_LEFT` su `left_lane` non solleva errore — semplicemente non muove il veicolo (dopo fix T1-B02bis)
- `TURN_LEFT` su un veicolo in approccio non è bloccato dal domain
- `MOVE_FORWARD` su `multi_lane` genera un evento senza cambiare lo stato

**Fix applicato (questo ciclo di revisione):** `apply_action()` ora genera l'evento CHANGE_LEFT/RIGHT solo se la posizione cambia effettivamente (vedi §5). I guard semantici più complessi rimangono in `safe_apply_action()`.

---

### T1-B08 — `MOVE_FORWARD` troppo generico e non contestuale
**Gravità:** Medio  
**Stato:** NOTO

L'azione `MOVE_FORWARD` ha semantica diversa per ambiente:
- `intersection` → approach entra dentro l'incrocio
- `multi_lane` → nessun cambio di posizione (no-op per il position tracking, quindi esclusa dal pool Task 1)
- `roundabout` → il domain usa `ENTER_ROUNDABOUT`/`EXIT_ROUNDABOUT` che sono più espliciti

Questa genericità crea asimmetria: per intersection si usa `MOVE_FORWARD`, per roundabout si usa `ENTER_ROUNDABOUT`. Il modello è inconsistente tra ambienti.

**Decisione progettuale:** La separazione `ENTER_ROUNDABOUT`/`EXIT_ROUNDABOUT` per la rotatoria è intenzionale perché questi due eventi hanno semantica chiaramente distinta (entrata vs. uscita). `MOVE_FORWARD` per intersection è mantenuto per semplicità del prompt in linguaggio naturale ("moves forward" è più leggibile di "enters the intersection").

---

### T1-B09 — Modellazione incoerente del roundabout rispetto all'intersection
**Gravità:** Medio  
**Stato:** NOTO

All'incrocio:
```
approach --[MOVE_FORWARD]--> inside_intersection --[TURN_*]--> {dir}_exit
```

Alla rotatoria:
```
{dir}_approach --[ENTER_ROUNDABOUT]--> roundabout_lane --[EXIT_ROUNDABOUT]--> {dir}_exit
```

Il roundabout usa azioni dedicate per entrare/uscire, mentre l'intersection usa azioni generiche (`MOVE_FORWARD`, `TURN`). Questo significa che le azioni nei prompt sembrano diverse pur descrivendo concetti analoghi (entrare in uno spazio condiviso, uscirne). Un LLM che ragiona su Task 1 potrebbe sviluppare euristiche diverse per i due ambienti.

**Decisione progettuale:** Il roundabout è escluso dal Task 1 per questa ragione (posizioni interne non distinguibili — solo `roundabout_lane`). L'inconsistenza rimane per i task futuri (Task 4 — Overlap Reasoning).

---

### T1-B10 — Generazione delle label non completamente automatizzata
**Gravità:** Basso  
**Stato:** NOTO

Nel `build_choices()` di `task1_position.py`, i fallback per i distrattori `highly_false` usano stringhe hardcoded:
```python
hf1 = false_pool[0] if len(false_pool) > 0 else "off the road"
hf2 = false_pool[1] if len(false_pool) > 1 else "unknown location"
```

Le label `"off the road"` e `"unknown location"` non sono posizioni del dominio, ma stringhe di fallback inventate. Se il pool di false options si esaurisce (es. se `nt2` coincide con una delle 4 posizioni cross-env), il distrattore non è più derivato dal dominio.

**Stato attuale:** Il pool `_CROSS_ENV_FALSE` ha 4 elementi per ambiente. In condizioni normali entrambi i `highly_false` sono sempre presenti. Il fallback è teoricamente raggiungibile ma non osservato sui 100 esempi generati.

---

### T1-B11 — Eventi ridondanti (MOVE_FORWARD ripetuti, no-op)
**Gravità:** Alto  
**Stato:** CORRETTO

Prima dei fix documentati in `task_documentation.md` (§1.4.2 e §1.4.3), `MOVE_FORWARD` poteva essere applicato più volte sullo stesso veicolo già dentro l'incrocio, generando eventi semanticamente vuoti ("Vehicle A moves forward." quando A era già `inside_intersection`).

**Fix applicato:**
```python
if action == Action.MOVE_FORWARD and v.inside_intersection:
    return None
```

Risultato: 0 eventi no-op su 100 esempi dopo il fix (da 25 nel baseline).

---

### T1-B12 — Distrattori non sempre coerenti
**Gravità:** Medio  
**Stato:** PARZIALE

Due classi di problemi:

**a) `near_true_2` come fallback generico:** Se non c'è una posizione intermedia e non c'è una corsia adiacente disponibile, `build_choices()` pesca a caso dalle posizioni dello stesso ambiente. Il distrattore risultante potrebbe non essere "quasi vero" in modo significativo.

**b) Posizioni `highly_false` troppo ovvie:** Per `multi_lane`, `"inside the intersection"` e `"the roundabout lane"` sono immediatamente riconoscibili come ambienti sbagliati. Un LLM potrebbe eliminarle per ragionamento ambientale piuttosto che per tracking della posizione.

**Stato:** Il problema (a) è mitigato dal fallback a `all_labels`. Il problema (b) è una limitazione del design del Core Dataset — il dataset Extended potrà usare distrattori più sofisticati.

---

### T1-B13 — Distribuzione sbilanciata delle risposte
**Gravità:** Alto  
**Stato:** CORRETTO

Senza controllo esplicito, il generatore tendeva a piazzare la risposta corretta sempre nella stessa posizione. Risolto con `key_schedule`:
```python
key_schedule = []
per_key = n // 5
for letter in LETTERS:
    key_schedule.extend([letter] * per_key)
random.shuffle(key_schedule)
```
Distribuzione finale: 20-20-20-20-20 garantita.

---

### T1-B14 — Uso ambiguo della variabile `position`
**Gravità:** Alto (architetturale)  
**Stato:** NOTO

Il campo `position` di `Vehicle` viene usato per rappresentare concetti eterogenei:

| Valore | Tipo semantico |
|---|---|
| `"left_lane"` | Posizione su corsia (dimensione laterale) |
| `"north_approach"` | Punto di accesso a un incrocio (direzionale) |
| `"inside_intersection"` | Stato logico (non una posizione spaziale) |
| `"north_exit"` | Punto di uscita dall'incrocio (direzionale + terminale) |
| `"roundabout_lane"` | Corsia circolare (interna) |

Il problema principale è che `"inside_intersection"` non è una posizione nello spazio ma uno stato logico che dovrebbe stare in `inside_intersection: bool` (che esiste già). I due campi sono ridondanti e sincronizzati manualmente in `apply_action()`.

**Impatto:** Un veicolo `inside_intersection=True` ha sempre `position="inside_intersection"`. Se per qualsiasi motivo i due si desincronizzano, il comportamento è indefinito. Nessuna inconsistenza è stata osservata nei test attuali.

**Decisione progettuale:** Il refactoring di `position` in un tipo più ricco (es. una sealed class o dataclass per tipo di posizione) è segnalato come debito tecnico. Non è stato fatto perché il dataset è generato automaticamente e la sincronizzazione è garantita dall'unico punto di mutazione (`apply_action()`).

---

## 4. Bug e limitazioni — Task 2 (Right-of-Way Reasoning)

---

### T2-B01 — Regole di precedenza non formalizzate completamente
**Gravità:** Alto  
**Stato:** NOTO

`APPROACH_PRIORITY` copre solo 8 coppie di direzioni (conflitti laterali). Le coppie opposte (NORTH-SOUTH, EAST-WEST) restituiscono `None` — assunto: nessun conflitto. Ma questo è vero **solo** se entrambi i veicoli vanno dritto. Se uno svolta a sinistra attraversando la traiettoria del veicolo opposto, c'è un conflitto reale non modellato.

**Esempio non gestito:**
```
A: NORTH → intende TURN_LEFT (svolta verso ovest, attraversa la corsia di B)
B: SOUTH → intende GO_STRAIGHT (procede verso nord)
```
Questi due veicoli hanno un conflitto reale, ma `right_of_way_intersection(A, B)` restituisce `None` (assunzione "opposti = nessun conflitto").

**Impatto attuale:** Il generatore di Task 2 filtra esplicitamente le coppie opposte con `_has_lateral_conflict()`, quindi non genera scenari con questo tipo di conflitto. La limitazione è consistente internamente ma esclude una classe di scenari realistica.

---

### T2-B02 — Mancato uso dell'intenzione (`intent`) nella decisione di precedenza
**Gravità:** Alto  
**Stato:** NOTO

`Vehicle.intent` (GO_STRAIGHT, TURN_LEFT, TURN_RIGHT) è assegnato a ogni veicolo all'incrocio, appare nel testo del prompt, ma non viene usato dalla funzione `right_of_way_intersection()`. La precedenza è calcolata solo sulla direzione di approccio.

**Conseguenza:** Il benchmark chiede all'LLM di ragionare sulla precedenza, ma il ground truth ignora l'intenzione. Uno scenario in cui A (da nord, intende girare a destra) e B (da est) — dove la svolta a destra di A non interseca la traiettoria di B — verrebbe comunque risolto con la stessa precedenza di uno dove A va dritto.

**Stato:** Per il Core Dataset questa limitazione è accettata — la precedenza a destra si applica all'approccio, non alla traiettoria. Il campo `intent` è visibile all'LLM per renderlo più realistico ma non influenza il ground truth nel dataset corrente.

---

### T2-B03 — Gestione errata dei veicoli opposti (assunti senza conflitto)
**Gravità:** Medio  
**Stato:** NOTO + parzialmente mitigato da filtro

Come descritto in T2-B01, due veicoli che arrivano da direzioni opposte sono trattati come "nessun conflitto" (`right_of_way_intersection` restituisce `None`). Il generatore evita questo caso con `_has_lateral_conflict()`, ma la funzione di dominio stessa non lo gestisce esplicitamente.

**Fix parziale:** `_build_intersection_with_conflict()` usa `_has_lateral_conflict()` per garantire che solo coppie con conflitto laterale vengano selezionate. Gli scenari con veicoli opposti vengono esclusi silenziosamente.

---

### T2-B04 — Assenza di modellazione esplicita dei conflitti tra traiettorie
**Gravità:** Alto (architetturale)  
**Stato:** NOTO

Il domain modella la precedenza basandosi sulla **direzione di approccio** al momento dello scenario iniziale, non sulle **traiettorie effettive** dei veicoli. Un sistema corretto dovrebbe:

1. Determinare la traiettoria di ogni veicolo in base a `direction` + `intent`
2. Calcolare se le traiettorie si intersecano
3. Applicare la regola di precedenza solo sulle coppie che si intersecano

Questo richiederebbe una mappa topologica dell'incrocio (quali corsie di uscita corrispondono a quali combinazioni direction+intent) che il domain non implementa.

**Impatto:** Il benchmark funziona come "test della regola della precedenza a destra data la direzione di approccio", non come "test della precedenza basata su traiettorie". Per il livello del Core Dataset questa semplificazione è accettabile.

---

### T2-B05 — Mancanza di gestione della simultaneità (più veicoli potrebbero passare insieme)
**Gravità:** Medio  
**Stato:** NOTO

In certi scenari con 3 veicoli, due coppie di veicoli potrebbero non avere conflitto tra loro (es. A da Nord e C da Sud non si ostacolano se vanno dritti). Il generatore sceglie una coppia con conflitto e assegna un'unica risposta corretta, ignorando che altri veicoli potrebbero già muoversi liberamente.

**Conseguenza:** La risposta "Both can pass at the same time" è usata come distractor `near_true`, ma in alcuni scenari potrebbe essere tecnicamente corretta per coppie non in conflitto.

**Stato attuale:** Il prompt descrive 3 veicoli e chiede quale ha la precedenza — la domanda è implicita sulla coppia in conflitto. Un LLM attento potrebbe notare l'ambiguità. Per il Core Dataset la semplificazione è accettata.

---

### T2-B06 — Risposta forzata unica anche quando non necessaria
**Gravità:** Medio  
**Stato:** NOTO

Il formato MCQ con 5 opzioni include sempre "Both can pass at the same time" come `near_true`. In scenari dove effettivamente non c'è conflitto, questa opzione sarebbe la risposta corretta — ma il generatore garantisce sempre uno scenario con conflitto, quindi non si verifica. L'opzione è però potenzialmente ambigua se l'LLM ragiona su veicoli diversi dalla coppia target.

---

### T2-B07 — Incoerenza delle label rispetto alla regola di precedenza non esplicitata nel prompt
**Gravità:** Medio  
**Stato:** NOTO

Il prompt include frasi come "There are no traffic lights or signs." che implica che si applica la regola della precedenza a destra, ma non la enuncia esplicitamente. Il benchmark si aspetta che l'LLM conosca questa regola dal proprio training.

**Conseguenza:** Il benchmark non misura "saper applicare la regola della precedenza a destra data la regola" ma "sapere che la regola della precedenza a destra si applica in assenza di segnali". Questa è una scelta intenzionale di progettazione — si vuole testare la conoscenza del mondo reale.

---

### T2-B08 — Mancanza di definizione dell'ordine temporale (arrivo simultaneo)
**Gravità:** Basso  
**Stato:** NOTO

La regola della precedenza a destra si applica quando due veicoli arrivano **contemporaneamente** all'incrocio. Se un veicolo arriva prima, ha la precedenza per ordine di arrivo. Il benchmark assume sempre arrivo simultaneo, che è un caso speciale non esplicitato nel prompt.

**Impatto:** Potenziale confusione per LLM che ragionano sull'ordine di arrivo. Il testo del scenario non include informazioni sull'ordine di arrivo, quindi l'LLM dovrebbe assumere simultaneità. Da chiarire nella documentazione dei prompt.

---

### T2-B09 — Uso parziale della precedenza a destra (solo direzione di approccio, non traiettoria di uscita)
**Gravità:** Medio  
**Stato:** NOTO

La "precedenza a destra" dovrebbe applicarsi al veicolo che proviene dalla destra rispetto alla direzione di marcia del veicolo che deve cedere. La tabella `APPROACH_PRIORITY` implementa questa logica correttamente per i conflitti laterali semplici, ma non considera:
- La direzione di uscita intesa (turn vs. straight)
- Situazioni a 3 veicoli dove più regole si sovrappongono

La tabella copre il caso base in modo corretto. Le situazioni più complesse sono fuori scope del Core Dataset.

---

### T2-B10 — Logica del roundabout non completamente separata da quella dell'intersection
**Gravità:** Medio  
**Stato:** CORRETTO (parzialmente)

Il dispatcher `right_of_way(v1, v2, env)` ha un fallback nel ramo `ROUNDABOUT`:
```python
elif env == Environment.ROUNDABOUT:
    if v1.inside_intersection:
        return right_of_way_roundabout(v1, v2)
    elif v2.inside_intersection:
        return right_of_way_roundabout(v2, v1)
    return right_of_way_intersection(v1, v2)   # ← fallback su logica incrocio
```

Se nessun veicolo è dentro la rotatoria, il dispatcher usa la logica dell'incrocio. Alla rotatoria, però, le regole di approccio non sono identiche a quelle dell'incrocio — tipicamente, è il primo ad arrivare o l'accordo locale. Usare `right_of_way_intersection` qui è tecnicamente sbagliato.

**Mitigazione attuale:** `_build_roundabout_with_conflict()` garantisce sempre che almeno un veicolo sia già dentro la rotatoria prima di generare uno scenario. Il fallback a `right_of_way_intersection` non viene quindi mai raggiunto dai generatori attuali.

**Stato:** Bug latente nel dispatcher generico, non raggiunto dai generatori correnti.

---

### T2-B11 — Metadata (`priority_vehicle`, `conflict_pair`) non sempre coerenti
**Gravità:** Basso  
**Stato:** NOTO

`conflict_pair` è sempre ordinato alfabeticamente (`sorted([priority_vid, yielding_vid])`), non in ordine di priorità. Questo significa che da `conflict_pair` non si può inferire chi cede a chi senza leggere `priority_vehicle`. L'informazione è ridondante ma non auto-esplicativa.

**Esempio:**
```json
"priority_vehicle": "C",
"conflict_pair": ["A", "C"]
```
Si capisce che C ha la precedenza, ma non è immediato dal solo `conflict_pair`. Potenziale fonte di confusione per analisi downstream.

---

### T2-B12 — Distrattori talvolta ambigui o troppo forti
**Gravità:** Medio  
**Stato:** NOTO

**a) Highly_false troppo forte:** "No vehicle can pass" e "All vehicles must stop" sono risposta chiaramente assurde in uno scenario senza semafori rossi o incidenti. Un LLM le elimina immediatamente senza ragionare sulle regole.

**b) Highly_false "terzo veicolo" potenzialmente debolmente falso:** `Vehicle {third_vid}` è la risposta `highly_false` perché è il veicolo non coinvolto nel conflitto principale. Ma se il terzo veicolo è effettivamente in conflitto con la risposta corretta (in uno scenario a 3 vie), non è così "chiaramente falso".

**Stato:** Il problema (a) è una limitazione del Core Dataset. Il problema (b) è mitigato dal fatto che `_build_intersection_with_conflict()` sceglie casualmente la coppia di conflitto tra tutte le coppie possibili — il "terzo" è semplicemente escluso dal conflitto selezionato.

---

### T2-B13 — Dipendenza implicita da convenzioni del traffico non dichiarate
**Gravità:** Medio  
**Stato:** NOTO (design intenzionale)

Il benchmark si aspetta che l'LLM conosca la regola della precedenza a destra senza che venga enunciata nel testo. Questo è intenzionale: si vuole testare la conoscenza del mondo reale del modello, non la sua capacità di seguire istruzioni.

Il rischio è che modelli addestrati su diverse convenzioni nazionali (dove la precedenza a destra non è universale) possano avere performance artificialmente basse non per mancanza di capacità di ragionamento ma per mancanza di conoscenza specifica.

---

### T2-B14 — Assenza di validazione automatica delle label
**Gravità:** Medio  
**Stato:** NOTO

Non esiste uno script che verifichi automaticamente, per ogni esempio di Task 2:
- Che le 5 label siano tutte distinte
- Che la risposta corretta sia effettivamente derivabile dalla logica implementata
- Che il veicolo indicato come `priority_vehicle` nei metadata corrisponda alla risposta corretta nel JSON

Questa validazione esiste implicitamente (il generatore costruisce la risposta dallo stesso calcolo usato per il ground truth) ma non c'è un replay indipendente come quello fatto per Task 1 (§1.6 di `task_documentation.md`).

---

### T2-B15 — `right_of_way_roundabout` aveva condizione errata
**Gravità:** Alto  
**Stato:** CORRETTO

**Problema originale:** La funzione controllava `if v_inside.inside_intersection:` prima di ritornare `v_inside.id`. Se la precondizione veniva violata (veicolo non effettivamente dentro la rotatoria), la funzione restituiva `v_entering.id` — esattamente l'opposto della regola corretta.

**Fix applicato:**
```python
# Prima (ERRATO):
def right_of_way_roundabout(v_inside, v_entering):
    if v_inside.inside_intersection:
        return v_inside.id
    return v_entering.id   # ← BUG: concede precedenza al veicolo entrante

# Dopo (CORRETTO):
def right_of_way_roundabout(v_inside, v_entering):
    return v_inside.id     # il veicolo dentro ha SEMPRE precedenza
```

---

## 5. Fix applicati al codice

Questo ciclo di revisione ha prodotto le seguenti modifiche al codice sorgente:

### Fix 1 — `domain/scenario.py`: `_lane_index` ritornava valore errato per posizioni non-corsia

**File:** `domain/scenario.py`  
**Funzione:** `_lane_index()`

**Prima:**
```python
def _lane_index(position: str) -> int:
    try:
        return LANE_ORDER.index(position)
    except ValueError:
        return 1  # default to center if position is not a lane
```

**Dopo:**
```python
def _lane_index(position: str) -> int:
    """Returns index of lane in LEFT-CENTER-RIGHT order, or -1 if not a lane."""
    try:
        return LANE_ORDER.index(position)
    except ValueError:
        return -1  # sentinel: position is not a lane (consistent with task1_position.py)
```

**Motivo:** Il vecchio default `1` (center) faceva sì che un veicolo su `inside_intersection` o `north_approach` venisse teletrasportato silenziosamente su `left_lane` quando `apply_action` riceveva `CHANGE_LEFT`. Con `-1` la condizione `if current_idx > 0` è False per posizioni non-corsia, bloccando correttamente l'azione.

---

### Fix 2 — `domain/scenario.py`: `apply_action` generava evento CHANGE_LEFT/RIGHT anche senza cambiare stato

**File:** `domain/scenario.py`  
**Funzione:** `apply_action()`

**Prima:**
```python
elif action == Action.CHANGE_LEFT:
    current_idx = _lane_index(v.position)
    if current_idx > 0:
        v.position = LANE_ORDER[current_idx - 1]
    event = f"Vehicle {v.id} changes to the left lane."   # ← sempre generato
```

**Dopo:**
```python
elif action == Action.CHANGE_LEFT:
    current_idx = _lane_index(v.position)
    if current_idx > 0:                          # valid lane AND not leftmost
        v.position = LANE_ORDER[current_idx - 1]
        event = f"Vehicle {v.id} changes to the left lane."
    # else: no state change, no event

elif action == Action.CHANGE_RIGHT:
    current_idx = _lane_index(v.position)
    if 0 <= current_idx < len(LANE_ORDER) - 1:  # valid lane AND not rightmost
        v.position = LANE_ORDER[current_idx + 1]
        event = f"Vehicle {v.id} changes to the right lane."
    # else: no state change, no event

# ...

if event:                    # only log and advance step if action had an effect
    state.event_log.append(event)
    state.step += 1
return event
```

**Motivo:** L'evento veniva aggiunto all'`event_log` e `state.step` veniva incrementato anche quando il veicolo non si era mosso. Questo poteva generare eventi falsi ("Vehicle A changes to the left lane." senza cambiamento reale) se `apply_action` veniva chiamato direttamente bypassando `safe_apply_action`.

---

### Fix 3 — `domain/rules.py`: `right_of_way_roundabout` aveva logica pericolosa

**File:** `domain/rules.py`  
**Funzione:** `right_of_way_roundabout()`

Vedi T2-B15 sopra. La condizione `if v_inside.inside_intersection:` è stata rimossa. La funzione ora ritorna sempre `v_inside.id`.

---

### Fix 4 — `generators/task2_rightofway.py`: import inutilizzato

**File:** `generators/task2_rightofway.py`

**Prima:**
```python
from domain.render import describe_scenario, render_prompt
```

**Dopo:**
```python
from domain.render import describe_scenario
# render_prompt intentionally not imported: Task 2 has no event sequence,
# so the prompt is assembled inline without the events block.
```

**Motivo:** `render_prompt` includeva un blocco "Sequence of events:" non appropriato per Task 2 (scenario statico). Il prompt viene costruito inline. L'import inutilizzato è stato rimosso per chiarezza.

---

### Fix 5 — `domain/entities.py`: `Vehicle.describe()` usava la stringa grezza della posizione

**File:** `domain/entities.py`  
**Metodo:** `Vehicle.describe()`

**Prima:**
```python
def describe(self) -> str:
    base = f"Vehicle {self.id} is in the {self.position}"  # ← "in the north_approach"
    if self.direction:
        base += f", approaching from the {self.direction.value}"
    if self.intent:
        base += f", intending to {self.intent.value}"
    return base + "."
```

**Dopo:**
```python
def describe(self) -> str:
    # NOTE: prefer domain.render.describe_vehicle() for prompt generation.
    pos_label = self.position.replace("_", " ")
    base = f"Vehicle {self.id} is in {pos_label}"
    if self.intent:
        base += f", intending to {self.intent.value}"
    return base + "."
```

**Motivo:** Il metodo produceva output illeggibile ("north_approach" con underscore). Corretto per coerenza con `render.py`, rimuovendo anche il campo `direction` (non usato in `render.describe_vehicle`). Il metodo è codice di debug — i generatori usano `render.describe_vehicle()`.

---

## 6. Tavola di riepilogo

### Task 1

| ID | Titolo | Gravità | Stato |
|---|---|---|---|
| T1-B01 | Semantica MOVE_FORWARD vs TURN incoerente | Medio | NOTO |
| T1-B02 | TURN implica MOVE_FORWARD non formalizzato nel domain | Medio | PARZIALE |
| T1-B03 | Assenza FSM esplicita | Alto | NOTO (debito tecnico) |
| T1-B04 | Stato terminale `_exit` non formalizzato | Medio | PARZIALE |
| T1-B05 | Collisioni multi-lane non modellate | Medio | NOTO (design) |
| T1-B06 | Lane change senza vincoli di occupazione | Basso | NOTO (design) |
| T1-B07 | Assenza di validazione nel domain layer | Medio | PARZIALE (Fix 2) |
| T1-B08 | MOVE_FORWARD troppo generico | Medio | NOTO (design) |
| T1-B09 | Roundabout e intersection modellate diversamente | Medio | NOTO (design) |
| T1-B10 | Label fallback hardcoded (`"off the road"`) | Basso | NOTO |
| T1-B11 | Eventi ridondanti (MOVE_FORWARD ripetuto) | Alto | **CORRETTO** |
| T1-B12 | Distrattori non sempre coerenti | Medio | PARZIALE |
| T1-B13 | Distribuzione risposte sbilanciata | Alto | **CORRETTO** |
| T1-B14 | Variabile `position` semanticamente ambigua | Alto | NOTO (debito tecnico) |

### Task 2

| ID | Titolo | Gravità | Stato |
|---|---|---|---|
| T2-B01 | Precedenza non formalizzata per veicoli opposti con svolte | Alto | NOTO |
| T2-B02 | `intent` non usato nel calcolo della precedenza | Alto | NOTO (design) |
| T2-B03 | Veicoli opposti assunti senza conflitto | Medio | PARZIALE (filtro) |
| T2-B04 | Nessuna modellazione di conflitti tra traiettorie | Alto | NOTO (debito tecnico) |
| T2-B05 | Simultaneità non gestita (più veicoli potrebbero passare) | Medio | NOTO |
| T2-B06 | Risposta unica forzata anche quando non necessaria | Medio | NOTO |
| T2-B07 | Regola di precedenza non esplicitata nel prompt | Medio | NOTO (design) |
| T2-B08 | Arrivo simultaneo assunto ma non dichiarato | Basso | NOTO |
| T2-B09 | Precedenza a destra solo su direzione, non traiettoria | Medio | NOTO |
| T2-B10 | Logica roundabout non separata da intersection nel dispatcher | Medio | PARZIALE |
| T2-B11 | `conflict_pair` non ordinato per priorità | Basso | NOTO |
| T2-B12 | Distrattori talvolta ambigui o troppo forti | Medio | NOTO |
| T2-B13 | Dipendenza implicita da convenzioni del traffico | Medio | NOTO (design) |
| T2-B14 | Assenza di validazione automatica delle label Task 2 | Medio | NOTO |
| T2-B15 | `right_of_way_roundabout` con condizione errata | Alto | **CORRETTO** (Fix 3) |

### Fix al domain layer (indipendenti dalla task)

| Fix | File | Descrizione |
|---|---|---|
| Fix 1 | `domain/scenario.py` | `_lane_index` ritornava 1 per posizioni non-corsia → teleport silenzioso |
| Fix 2 | `domain/scenario.py` | `apply_action` CHANGE_LEFT/RIGHT generava evento anche senza muovere il veicolo |
| Fix 3 | `domain/rules.py` | `right_of_way_roundabout` logica errata con condizione ridondante/pericolosa |
| Fix 4 | `generators/task2_rightofway.py` | Import inutilizzato `render_prompt` rimosso |
| Fix 5 | `domain/entities.py` | `Vehicle.describe()` usava posizione raw con underscore |

---

## 7. Note progettuali e decisioni di design

### Perché il collision check è stato rimosso dal multi-lane

Il collision check originale impediva che due veicoli occupassero la stessa corsia. Con 3 veicoli su 3 corsie, ogni cambio corsia era impossibile → deadlock totale → 0% esempi multi_lane. La scelta di rimuoverlo è motivata dal fatto che il modello è **discreto per corsia, non per segmento**: due veicoli sulla stessa corsia si trovano a posizioni diverse lungo la strada.

### Perché `intent` non influenza il ground truth in Task 2

Il Task 2 testa la "regola della precedenza a destra" come regola semplice basata sulla direzione di approccio. Includere l'intent nel calcolo richiederebbe la modellazione completa delle traiettorie (T2-B04), che è fuori scope del Core Dataset. L'intent è incluso nel testo per rendere lo scenario più realistico e aumentare la difficoltà per l'LLM, non per modificare il ground truth.

### Separazione `VALID_ACTIONS` (rules.py) vs `ACTIONS_BY_ENV` (generatori)

`VALID_ACTIONS` in `rules.py` rappresenta le azioni **fisicamente possibili** per ambiente (per uso in Task 3 — Violation Detection, dove STOP e MOVE_FORWARD su multi_lane sono rilevanti). `ACTIONS_BY_ENV` nei generatori è una restrizione **task-specifica** che esclude azioni valide ma non informative per quel task (es. MOVE_FORWARD escluso da Task 1 su multi_lane perché è un no-op per il position tracking). I due dizionari coesistono per design.

### Debito tecnico segnalato per le task successive

- **FSM esplicita per i veicoli** (T1-B03): necessaria per Task 3 e Task 4 dove gli stati terminali e le violazioni devono essere rilevabili formalmente.
- **Modellazione delle traiettorie** (T2-B04): necessaria per una precedenza basata su traiettorie reali anziché solo direzione di approccio.
- **Tipo semantico per `position`** (T1-B14): separare posizione-corsia, posizione-incrocio e stato-logico ridurrebbe la superficie di bug per Task 3 e Task 4.
- **Script di validazione per Task 2** (T2-B14): replay indipendente sul JSONL generato, analogo a quello esistente per Task 1.
