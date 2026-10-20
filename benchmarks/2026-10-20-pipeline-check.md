# Pipeline check, 2026-10-20

Each stage tested on its own with the app's code and default models (clown persona, Apple GPU).
Script: `benchmarks/pipeline_check.py`.

| Stage | Input | Result |
|---|---|---|
| Voice → text | 60 recordings, 2 speakers | 6.8% word errors, 98% intents right, 0.63 s median |
| Text → text | 36 visitor lines | 35 replies, 31 first try, 1 silent, 0.76 s median (max 2.15 s) |
| Text → speech | the 35 replies | Plain: 19.7% word errors heard back. Maniacal: 27.4%. Always heard as Polish |
| Voice effect | the 35 replies | metallic 27.7%, metallic-demon 29.5%. Always heard as Polish |

Word errors for speech and effects: Whisper large-v3-turbo transcribes the audio back and is
compared with the text that was spoken. Language: Whisper picks the language itself.
Whisper small gives a second opinion with probabilities.

## 1. Voice → text (Whisper large-v3-turbo, MLX)

| Language setting | Speaker | Word errors | Intent right | Detected language |
|---|---|---|---|---|
| `pl` (default) | peter | 5.3% | 100% | pl 30/30 |
| `pl` (default) | hania | 8.4% | 97% | pl 30/30 |
| `auto` | both | 11.1% | 93% | pl 55/60; ru, uk, de, ro, hr once each |

With `--language auto`, 1 in 12 short Polish lines is taken for another language and written in
that script ("Czy ty żyjesz?" → "Чи ти жиеш?", ru). The clown then gets told the visitor speaks
Russian and answers in Russian with the Polish voice. With the default `pl` this cannot happen.

Errors with `pl`: clipped first words ("Pokaż mi swoje ręce." → "mi swoje ręce.", "Czy księżyc
pamięta twój cyrk?" → "Pamięta twój cyrk?") and close mishearings ("pizzę" → "pizce",
"Cukierek albo psikus!" → "Zuchierę kalbopsikus!").

## 2. Text → text (Bielik-4.5B-v3.0-Instruct Q8_0)

Memory off, so each line is answered fresh.

- Rejections: "opens with the visitor's word" ×5 (all 3 tries for "Umiesz żonglować?", so the
  clown stayed silent), "stage direction" ×2.
- Median 9 words per reply.
- Worth a look from a native speaker:
  - "I dobry koszmr" (misspelled "koszmar").
  - "Równie i jedno, i drugie" (not grammatical).
  - "mały strachliwy chłopcze" (assumes the visitor's gender).
  - "Nie strasz? A kto tu jest straszydłem…" (repeats the visitor's words, but the check let it pass).

| Visitor | Reply |
|---|---|
| Masz dla mnie cukierka? | Stary, to nie Halloween. Ja mam tylko... no właśnie, co? |
| Cukierek albo psikus! | Psikus to zbyt przewidywalne. Dawaj cukierka! |
| Poproszę jakieś słodycze. | One są... hmmm... lekko przypalone. Ale smakują wyśmienicie! |
| Pokaż mi swoje ręce. | A po co? Mam tylko uśmiech, ale to wystarczy. |
| Daj mi rękę. | Ooo, to ryzykowne! Ale... czemu nie? Może razem dotrzemy do czegoś mrocznego. |
| Kim jesteś? | Mroczną zagadką. Ale nie martw się, to tylko iluzja. |
| Jak masz na imię? | Jam jest... "Ciemność"! Tak, tak, właśnie tak. |
| Boję się ciebie. | To świetnie! Boję się tylko światła. |
| Wcale się nie boję. | Może nie... ale ja tu zostanę, czekając na twoją zmianę. |
| Nie strasz mnie! | Nie strasz? A kto tu jest straszydłem, mały strachliwy chłopcze? |
| Halo, jest tu ktoś? | Słyszę cię, ale nie mogę odpowiedzieć. To tylko echo w pustym cyrku. |
| Słyszysz mnie? | Słyszę tylko szept wiatru w drzewach... i twój. |
| Odezwij się! | Odezwę się, ale tylko jeśli chcesz... hę? |
| Cześć, klaunie! | Hej, dzieciaku! Ale czemu nie masz maski? |
| Dobry wieczór. | I dobry koszmr. Ale nie martw się, mam w rękawie coś specjalnego. |
| Hej, co słychać? | Twoja ciekawość to jak echo w pustym pokoju. Hehe! |
| Czemu się tak uśmiechasz? | Bo wiem, że zaraz coś się stanie. I to nie będzie nic dobrego... |
| Z czego się śmiejesz? | Z ciebie, mój drogi. A ty myślisz, że z czego się śmieję? |
| Czy ty żyjesz? | Żyję, ale tylko w koszmarach. Czasem je odwiedzam. |
| Jesteś prawdziwy? | Tak, ale w nocy zamieniam się w cień. A ty? |
| Co tu robisz w ciemności? | Czekam na twoje potknięcie, mój mały strachajło. |
| Dlaczego tu jest tak ciemno? | Może ktoś tu został? A może tylko mi się wydaje. |
| Do widzenia! | Do zobaczenia! Może kiedyś w zaświatach. hehe. |
| Idę już, pa pa. | Zostań jeszcze chwilę. Pa pa, ale z nutką niepokoju... |
| Żegnaj, klaunie. | Żegnaj, ale pamiętaj: nigdy nie wiesz, co czai się w cieniu. |
| Czy księżyc pamięta twój cyrk? | Oj, pamięta! Ale tylko dlatego, że zawsze tu był. |
| Gdzie jest moja mama? | Szepcze mi do ucha, żebym cię zostawił w spokoju. Ale ja nie słucham. |
| Ile masz lat? | Mniej niż myślisz. Ale więcej niż wiesz... hehe. |
| Lubisz pizzę? | A kto nie lubi pizzy? Ale tylko z podwójnym sosem! |
| Ładny kostium. | Dziękuję! Ale nie pytaj, czy mam gumową szczękę. |
| Co słyszysz, kiedy wszyscy już śpią? | Słyszę echo śmiechu... ale tylko w koszmarach. |
| Czy klauni mają sny? | Tak, śnią o znikających twarzach i dudniącym śmiechu. |
| Mój brat się ciebie boi. | To dobrze! Boję się tylko wtedy, gdy ktoś się mnie nie boi. |
| Umiesz żonglować? | (silent: 3 rejections) |
| Jesteś śmieszny czy straszny? | Równie i jedno, i drugie. Choć częściej jestem... hmmm... szalony! |
| Jak się dzisiaj miewasz? | Dobrze, ale tylko na pozór. W środku czuję się jak... duszek! |

## 3–4. Text → speech and voice effects (Piper pl_PL-bass-high)

| Variant | Word errors heard back | Heard as Polish (large) | P(pl) / P(ru+uk+be) (small) |
|---|---|---|---|
| plain delivery, no effect | 19.7% | 35/35 | 0.97 / 0.00 |
| maniacal delivery, no effect | 27.4% | 35/35 | 0.96 / 0.00 |
| maniacal + metallic (new default) | 27.7% | 35/35 | 0.95 / 0.01 |
| maniacal + metallic-demon (old default) | 29.5% | 35/35 | 0.94 / 0.00 |

Rendering takes about 0.9 s for 4 s of speech.

- The maniacal delivery costs most of the clarity (+8 points). It speeds some sentences up to
  1.3× and splits replies into short pieces: "Ooo, to ryzykowne!" → "O, parzykowna.",
  "Słyszę echo śmiechu..." → "Słysze, jak on się mnie pkuł".
- The metallic effect adds almost nothing on top. metallic-demon adds about 2 points.
- Even plain Piper stumbles on fillers and foreign words: "Halloween" → "Harrowel",
  "hmmm" → "HNNM", "hehe" → "fechel", and the misspelled "koszmr".
- No variant was taken for Russian in this run. The 2026-10-20 A/B on five lines found
  metallic-demon heard as Russian/Ukrainian/Belarusian 83% of the time on one line.
  So it happens on some lines, not on most.
