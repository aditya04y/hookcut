# Same episodes, different analysis models

Episodes compared: 11 (run A: `outputs/run_pass1_mixed/result.json`, run B: `outputs/run_full/result.json`)

- Strongest hook beat agrees (same beat or within 3 s): **6/11**
- Strongest cliffhanger beat agrees: **1/11**
- Overlap of strong beats (score >= 4), mean Jaccard: **0.31**

| Ep | Model A | Model B | top hook A / B | agree | top cliff A / B | agree | strong A/B | Jaccard | opening A/B | ending A/B |
|---|---|---|---|---|---|---|---|---|---|---|
| E01 | gemini:gemini-4-hydrogen | gemini:gemini-3.5-flash-lite | E01.B027 / E01.B027 | True | E01.B057 / E01.B089 | False | 8/5 | 0.3 | 2/2 | 4/5 |
| E02 | gemini:gemini-3.8-flash | gemini:gemini-3.5-flash-lite | E02.B002 / E02.B067 | False | E02.B026 / E02.B064 | False | 15/7 | 0.29 | 5/4 | 4/3 |
| E03 | gemini:gemini-4-hydrogen | gemini:gemini-3.5-flash-lite | E03.B071 / E03.B071 | True | E03.B116 / E03.B112 | False | 8/5 | 0.44 | 3/2 | 5/5 |
| E05 | gemini:gemini-3.8-flash | gemini:gemini-3.5-flash-lite | E05.B032 / E05.B032 | True | E05.B015 / E05.B056 | False | 5/3 | 0.33 | 2/2 | 2/4 |
| E06 | gemini:gemini-3.8-flash | gemini:gemini-3.5-flash-lite | E06.B002 / E06.B031 | False | E06.B052 / E06.B089 | False | 4/3 | 0.0 | 4/2 | 3/3 |
| E07 | gemini:gemini-4-hydrogen | gemini:gemini-3.5-flash-lite | E07.B011 / E07.B042 | False | E07.B032 / E07.B041 | False | 5/5 | 0.11 | 1/1 | 1/4 |
| E08 | gemini:gemini-3.8-flash | gemini:gemini-3.5-flash-lite | E08.B027 / E08.B027 | True | E08.B026 / E08.B057 | False | 3/2 | 0.25 | 1/2 | 3/3 |
| E09 | gemini:gemini-4-hydrogen | gemini:gemini-3.5-flash-lite | E09.B009 / E09.B079 | False | E09.B042 / E09.B088 | False | 4/5 | 0.5 | 3/3 | 4/5 |
| E10 | gemini:gemini-4-hydrogen | gemini:gemini-3.5-flash-lite | E10.B005 / E10.B005 | True | E10.B055 / E10.B059 | False | 6/6 | 0.5 | 4/4 | 4/5 |
| E11 | gemini:gemini-4-hydrogen | gemini:gemini-3.5-flash-lite | E11.B078 / E11.B078 | True | E11.B085 / E11.B082 | False | 8/5 | 0.18 | 2/2 | 5/5 |
| E23 | gemini:gemini-3.8-flash | gemini:gemini-3.5-flash-lite | E23.B040 / E23.B045 | False | E23.B067 / E23.B066 | True | 4/5 | 0.5 | 2/2 | 5/5 |
