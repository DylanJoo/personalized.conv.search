cp -r models/ckpt/start-contriever-ms-B160-A0.1/checkpoint-20000/
cp -r data/ikat/ikat.test.t5ntr_history_3-3.jsonl \
	    --device cuda:2 \
	    --concat_ptkb \
	    --batch_size 16
	# [NOTE] ikat-subtask3 required.
