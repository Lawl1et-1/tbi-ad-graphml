cd /d/rayanportfolio/tbi-ad-graphml
PY=".venv/Scripts/python -W ignore"
L=results/phase5/log_models.txt
$PY src/phase5_models.py gat --graph string --seeds 42 43 >> $L 2>&1
$PY src/phase5_models.py gat --graph random --seeds 42 >> $L 2>&1
$PY src/phase5_models.py gat --graph none --seeds 42 >> $L 2>&1
$PY src/phase5_models.py gat --graph coexpr --seeds 42 >> $L 2>&1
$PY src/phase5_models.py lodo --models gat >> $L 2>&1
$PY src/phase5_models.py evaluate >> $L 2>&1
$PY src/phase6_transfer.py fit >> results/phase6_log.txt 2>&1
$PY src/phase6_transfer.py score >> results/phase6_log.txt 2>&1
echo ALL_DONE >> $L
