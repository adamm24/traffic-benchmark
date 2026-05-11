# Task 2 Documentation — Right-of-Way Reasoning (v4.1)

**Generator:** `/Users/adamamrani/Desktop/STAGE-DATA-SCIENCE/traffic-benchmark/generators/task2_rightofway.py` (`task2_rightofway_v4`)
**Dataset:** `/Users/adamamrani/Desktop/STAGE-DATA-SCIENCE/traffic-benchmark/dataset/core/task2_rightofway.jsonl`

## Objective
Valutare ragionamento right-of-way (non memorizzazione pattern):
- priorità univoca quando esiste,
- riconoscimento esplicito dei casi senza vincitore univoco.

## Problemi trovati (ultima audit)
1. **Trailer copy-paste errato** in alcuni `intersection_no_clear_turnmix`: frase “All vehicles are proceeding straight...” usata con intenti misti.
2. **Ambiguità tipo task2_0079**: presenza di straight opposti simultanei nei turnmix (poteva ridurre chiarezza del “ONE vehicle”).
3. **Sotto-specifica dei misti**: mancava una nota interpretativa esplicita nei casi con turni misti.
4. **Shortcut potenziale**: “No vehicle has an unambiguous right of way” correlato solo a casi 4-veicoli.

## Fix implementati
1. **Trailer condizionato per subtype**
- `intersection_no_clear_straight`: può usare la frase “All vehicles are proceeding straight...”.
- `intersection_no_clear_turnmix`: usa solo context neutri + nota regole.

2. **Filtro anti-ambiguità nei turnmix**
- Nei `turnmix` è obbligatorio almeno un conflitto reale `right-turn vs left-turn`.
- Esclusi i casi con coppia `straight` opposta simultanea (`opposite straight pair`).

3. **Clausola interpretativa aggiunta nei turnmix**
- Context line include nota:
  - left turn cede a oncoming straight/right quando c’è conflitto;
  - altrimenti vale priority-to-the-right.

4. **De-correlazione “No unambiguous” da 4-veicoli**
- Aggiunto subtype `intersection_no_clear_threeway` (3 veicoli, tutti straight, no vincitore univoco nel formalismo).
- Quindi i no-clear non sono più esclusivi dei 4-veicoli.

5. **Validator hardening (`scripts/validate_task2.py`)**
- Supporto subtype no-clear: `straight`, `turnmix`, `threeway`.
- Gate qualità aggiunti:
  - imbalance answer-key,
  - collapse winner-label intersection,
  - role-pattern overuse,
  - overuse literal correct text,
  - correlazione no-clear con soli 4-veicoli,
  - bias lettera del distractor ripetuto.

6. **Compatibilità pipeline**
- `scripts/build_core_simulation.py` aggiornato: gestisce correttamente no-clear come gold valido quando non esiste priority dominante.

## Risultato finale (seed=42, n=100)
Comandi:
- `python generators/task2_rightofway.py --n 100 --seed 42 --out dataset/core/task2_rightofway.jsonl`
- `python scripts/validate_task2.py --input dataset/core/task2_rightofway.jsonl`

Esito:
- `Validated 100 examples; 0 failed`.
- Answer letters: `A/B/C/D/E = 20` ciascuna.
- Environment: `intersection 60`, `roundabout 40`.
- Resolution: `unique_priority 78`, `intersection_no_clear 22`.
- No-clear vehicle-count split: `4-veicoli 16`, `3-veicoli 6`.
- No-clear subtypes: `straight 8`, `turnmix 8`, `threeway 6`.
- Recomputed priority totals (solo unique): `A 26, B 26, C 26`.
- Direction-only shortcut accuracy (intersection unique): `0/38`.

## Tradeoff residui
- I roundabout restano più semplici degli intersection (quota controllata al 40%).
- `Both can pass...` resta frequente nei casi a priorità univoca (by design), ma non compare nei no-clear.
- Campo `inside_intersection` mantenuto per compatibilità schema.
