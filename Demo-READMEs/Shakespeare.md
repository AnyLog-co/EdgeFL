# Shakespeare next-character demo

This demo trains a federated character model on Tiny Shakespeare. Speaking roles are split across AnyLog nodes, so each node sees a different set of voices. The publisher labels every sample with the next character and keeps inserting those rows for as long as it runs. Each training round reads only the timestamp window of rows inserted for that round.

## Publisher

The publisher takes the REST address of every operator that should store data. Edit `edgefl/data/shakespeare/nodes.example.json` or pass the nodes on the command line. `num_rounds` of `0` means the stream continues until you stop it. When a role's lines are exhausted, that role starts again at the beginning.

```bash
cd edgefl/data/shakespeare
python3 publish_shakespeare.py --config nodes.example.json
```

Or without a config file:

```bash
python3 publish_shakespeare.py \
  --nodes 127.0.0.1:32149,127.0.0.1:32249,127.0.0.1:32349 \
  --db-name shakespeare_fl \
  --epoch-start 2026-10-01T17:00:00Z \
  --round-duration 60 \
  --samples-per-round 64 \
  --seq-len 40
```

Round 1 inserts rows timestamped in `[epoch, epoch + duration)`. Round 2 uses the next window, and so on. Inserts are spread across the window instead of landing in one burst. The process prints `DATA_EPOCH_START`, `ROUND_DURATION_SECONDS`, and `SEQ_LEN`, and writes them to `publish_state.json`. Put those three values in every training-node env file. A dry run prints the role split and the SQL window without contacting AnyLog:

```bash
python3 publish_shakespeare.py --config nodes.example.json --dry-run
```

Connect each operator to `shakespeare_fl` before the first insert, the same way the other demos connect their logical database.

## Training nodes

Env templates live in `edgefl/env_files/shakespeare/`. Point `EXTERNAL_IP` and `EXTERNAL_TCP_IP_PORT` at each operator, and set `DATA_EPOCH_START` from the publisher. `SEQ_LEN` and `ROUND_DURATION_SECONDS` have to match the publisher or the time-range query will not line up with the inserted rows.

```bash
cd edgefl
dotenv -f env_files/shakespeare/shakespeare-agg.env run -- uvicorn platform_components.aggregator.aggregator_server:app --host 0.0.0.0 --port 8080
dotenv -f env_files/shakespeare/shakespeare1.env run -- uvicorn platform_components.node.node_server:app --host 0.0.0.0 --port 8081
dotenv -f env_files/shakespeare/shakespeare2.env run -- uvicorn platform_components.node.node_server:app --host 0.0.0.0 --port 8082
dotenv -f env_files/shakespeare/shakespeare3.env run -- uvicorn platform_components.node.node_server:app --host 0.0.0.0 --port 8083
```

Initialize and start training against the aggregator, using one entry in `nodeUrls` per training node. The handler waits until a round's window has closed, then runs:

```sql
sql shakespeare_fl format=json and stat=false SELECT timestamp, sequence, label, role FROM shakespeare_train WHERE timestamp >= '<round start>' AND timestamp < '<round end>'
```

The test table is queried with the same bounds. Leave the publisher running while training is in progress so later rounds have rows in their windows.

Training and live inference use a PyTorch embedding and LSTM (`torch` in `requirements.txt`). FedAvg still averages the NumPy parameter arrays the handler returns.

## Live text prediction

The GUI can ask one training node to continue a line. Start it from `gui/edgefl-gui` with `npm start`, open the Inference step, choose **Text**, and set the training-node URL to the node you want (for example `localhost:8081`). The prompt is encoded with the Tiny Shakespeare vocabulary and posted to that node's `/infer` endpoint. The reply includes the continuation and the top next-character probabilities.

`GEN_LEN` on the training node is how many characters are generated. `INFERENCE_TEMPERATURE` controls sampling. `0` is greedy.
