arpa2fst --write-symbol-table=en-70k-0.2-pruned.sym en-70k-0.2-pruned.lm en-70k-0.2-pruned.fst
fstprint en-70k-0.2-pruned.fst | python eps2backoff.py en-70k-0.2-pruned.sym | fstcompile > en-70k-0.2-pruned_disamb.fst
python make_lexicon_fst.py en-70k-0.2-pruned.sym | fstcompile > L.fst
fsttablecompose L.fst en-70k-0.2-pruned_disamb.fst | fstdeterminizelog | fstminimizeencoded | fstpushspecial | fstpushspecial > LG.fst
