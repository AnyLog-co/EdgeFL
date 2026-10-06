# MedMNIST blood-cell demo

This demo trains a federated classifier on [BloodMNIST](https://medmnist.com/), the 8-class blood-cell dataset from MedMNIST v2. Each image is a 28×28 RGB slide of one normal blood cell. The model is a PyTorch CNN. FedAvg still averages the NumPy tensors the handler returns.

Slides stay on the operator that received them. The publisher never sends one operator's shard to another.

## Publisher

The publisher's input is the REST `host:port` of every AnyLog / EdgeLake operator, separated by commas. It downloads BloodMNIST on first run, holds out the six eval images, and keeps inserting labeled rows until you stop it.

```bash
cd edgefl/data/medmnist
python3 publish_medmnist_info.py 127.0.0.1:32149,127.0.0.1:32249,127.0.0.1:32349
```

Those addresses are the operator REST ports (`EXTERNAL_IP` in each node env file), not the EdgeFL training ports 8081–8083.

With three operators the default round inserts 1536 training images and 192 test images, split evenly, so each operator gets 512 training slides and 64 test slides. Classes are balanced inside that split. The next round continues through the operator's shard, and when the shard runs out it starts over. `num_rounds` of `0` (the default) streams until the process is interrupted.

Each row is scalar image info, the same kind of record as the chest X-ray demo: `filename`, `width`, `height`, `label`, `class_name`, and `round_number`. The PNG is written to `published_images/` and is not sent to AnyLog. A round is inserted immediately. Training selects `WHERE round_number = N`, then opens the file named in that row.

`publish_medmnist.py` is the earlier publisher that inserted the pixel matrix. It is kept for reference. Use `publish_medmnist_info.py`.

A dry run downloads the dataset and prints the split without contacting AnyLog:

```bash
python3 publish_medmnist_info.py 127.0.0.1:32149,127.0.0.1:32249,127.0.0.1:32349 --dry-run
```

Connect each operator to `mydb` before the first insert:

```bash
connect dbms mydb where type = psql and user = demo and password = passwd and ip = 192.1.1.1 and port = 5432 and memory = true
```

Leave the publisher running while training is in progress so later rounds have rows when a node asks for that `round_number`.

## Training nodes

Env templates live in `edgefl/env_files/medmnist/`. Point `EXTERNAL_IP` and `EXTERNAL_TCP_IP_PORT` at each operator.

`TRAIN_EPOCHS` defaults to 20, so a node fits its local images twenty times before returning weights. `TRAIN_HISTORY_ROUNDS` defaults to 4, so after a few rounds the fit includes the slides from the recent rounds, not only the latest batch. That is the local workload behind a usable model: about 512 new images per operator each round, revisited for 20 epochs, with up to four rounds kept in the query. On one operator's four-round set (about 2,000 images) that schedule reaches the mid-70s on held-out cells, well above the 12.5% chance line for eight classes.

```bash
cd edgefl
dotenv -f env_files/medmnist/medmnist-agg.env run -- uvicorn platform_components.aggregator.aggregator_server:app --host 0.0.0.0 --port 8080
dotenv -f env_files/medmnist/medmnist1.env run -- uvicorn platform_components.node.node_server:app --host 0.0.0.0 --port 8081
dotenv -f env_files/medmnist/medmnist2.env run -- uvicorn platform_components.node.node_server:app --host 0.0.0.0 --port 8082
dotenv -f env_files/medmnist/medmnist3.env run -- uvicorn platform_components.node.node_server:app --host 0.0.0.0 --port 8083
```

Initialize and start training against the aggregator. The handler loads the rows for the current `round_number` and runs:

```sql
sql mydb format=json and stat=false SELECT timestamp, filename, width, height, label, class_name, round_number FROM medmnist_train WHERE round_number >= '<first>' AND round_number <= '<current>'
```

The test table uses the same bounds. The node opens `filename` under `IMAGE_ROOT_DIR`. AnyLog does not store the pixel array.

## Eval images

`edgefl/data/medmnist/eval_images/` holds six test slides that the publisher does not insert. The file name is the correct class:

| File | Correct prediction |
| --- | --- |
| `0_basophil.png` | basophil |
| `1_eosinophil.png` | eosinophil |
| `2_erythroblast.png` | erythroblast |
| `4_lymphocyte.png` | lymphocyte |
| `6_neutrophil.png` | neutrophil |
| `7_platelet.png` | platelet |

Regenerate them with `python3 export_eval_images.py` from `edgefl/data/medmnist`. The PNGs are nearest-neighbor scaled to 224×224 so the cell is visible. The training node averages those blocks back to the original 28×28 matrix when you upload one.

## GUI

Start the GUI from `gui/edgefl-gui` with `npm start`. Open the Inference step, choose **MedMNIST**, and set the training-node URL (for example `localhost:8081`). Upload one of the labeled PNGs, or any other PNG/JPG of a blood cell. The page sends that image to the node's `/infer` endpoint. The node converts it to a 28×28×3 matrix and returns the predicted class, which you can compare with the file name, plus the probability of every class.

`Evaluate Test Set` on that same node scores the test rows stored on its operator.

BloodMNIST is from Yang et al., MedMNIST v2, Scientific Data, 2023, licensed CC BY 4.0. Label 3 in the full dataset is "immature granulocytes"; the eval folder does not include that class or monocyte.
