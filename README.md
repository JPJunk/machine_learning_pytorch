All scripts work with Windows 11

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        11/02/2026     09:05                Py311
d-----        11/02/2026     10:14                Py313


    Directory: \Py311


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        11/02/2026     10:45                OCR_pipeline
d-----        11/02/2026     09:06                Photo_pipeline


    Directory: \Py311\OCR_pipeline

OCR pipeline using PaddleOCR for detection and EasyOCR for recognition.
Includes installation guide that must be followed for all dependencies to work.

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        14/01/2026     15:22        1332881 frequency_dictionary_en_82_765.txt
-a----        11/02/2026     10:37         112574 ocr_pipeline.py
-a----        11/02/2026     08:27           7444 OCR_pipeline.txt


    Directory: \Py311\Photo_pipeline

Photo organization and repair pipeline with:
# ## **Stage 1 — FaceNet + HDBSCAN**
# Identity clusters.

# ## **Stage 2 — CLIP + UMAP + HDBSCAN**
# Appearance clusters for Stage‑1 noise.

# ## **Stage 3 — CLIP + KMeans**
# Forced grouping for Stage‑2 noise.

# This produces a global identity ID for every face.

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        11/02/2026     10:36          16850 organize_photos.py
-a----        11/02/2026     10:40           5592 photo_repair_pipeline.py


    Directory: \Py313


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        11/02/2026     09:09                FiveInARow
d-----        11/02/2026     10:15                MobileNetV2


    Directory: \Py313\FiveInARow

Five in a row boardgame using AlphaZero like architecture:
- Monte Carlo Tree Search
- Policy and Value Nets
- doesn't include pretrained model

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        26/01/2026     15:29           5221 agent_persistence.py
-a----        26/01/2026     15:02          17941 drl.py
-a----        11/02/2026     08:56         146720 Gomoku,tuning.pdf
-a----        26/01/2026     13:41           7143 gomoku.py
-a----        26/01/2026     14:32           9327 gui.py
-a----        26/01/2026     14:31           5581 main.py
-a----        09/02/2026     05:50          18647 mcts.py
-a----        03/02/2026     07:42          12807 play_game.py
-a----        26/01/2026     13:44           3269 policy_and_value_nets.py
-a----        26/01/2026     16:08           2538 pruning_mcts_test_harness.py
-a----        26/01/2026     14:18            671 replay_buffer.py
-a----        26/01/2026     14:29           3238 symmetry_utils.py
-a----        26/01/2026     16:16          16917 utils.py


    Directory: \Py313\MobileNetV2

Photo classification etc. using MobileNetV2

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        11/02/2026     10:42           1706 classify.py
-a----        11/02/2026     10:40           1946 embed.py
-a----        11/02/2026     10:41           4558 photo_clusters.py
-a----        11/02/2026     10:42           3314 predict.py
-a----        09/01/2026     11:02           5189 retrain.py
-a----        11/02/2026     10:42            858 similarity.py
