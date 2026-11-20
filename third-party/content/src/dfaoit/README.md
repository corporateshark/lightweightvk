# DFAOIT networks

The neural order-independent transparency networks of `samples/ARM_002_NeuralOIT.cpp`, after "Deep and Fast Approximate
Order Independent Transparency" (Tsopouridis, Vasilakis, Fudos, Computer Graphics Forum 2024). Each one is a per-pixel
multilayer perceptron, quantized to 8 bits, that predicts the colour of the fragments behind the 4 nearest ones of a
pixel. Each file here, `dfaoit-<hidden1>x<hidden2>.bin`, holds the integer weights of one network; the sample builds its
TOSA graph at start-up, shaped for its tensor (`kTensorWidth` x `kTensorHeight`), and runs it as a `VK_ARM_data_graph`
pipeline. It picks a network with `--network <hidden1>x<hidden2>` (default `16x16`).

## How these were trained

`dfaoit-16x16` is 14 -> 16 (ReLU) -> 16 (ReLU) -> 3 (sigmoid) and `dfaoit-32x16` is 14 -> 32 (ReLU) -> 16 (ReLU) -> 3
(sigmoid), both trained with every default of `tools/train_dfaoit.py` except the topology:

```
python tools/train_dfaoit.py --hidden1 16 --hidden2 16
python tools/train_dfaoit.py --hidden1 32 --hidden2 16
```

The defaults: 20,000,000 training and 500,000 validation pixels sampled from synthetic scenes (`tools/dfaoit_scenes.py`,
seed 1234, no scene file or renderer involved), 120 float epochs at a learning rate of 1e-3 and a batch of 8192, then 30
epochs of quantization-aware fine-tuning at a tenth of that rate, with a loss power of 0.5. Each took 20-25 minutes on
a desktop CPU with Python 3.14 and numpy 2.3.5, nearly all of it sampling the scenes. For both, the integer graph matched
the training-time simulation on all but 0.00% of the outputs, by at most 3/255.

| | Validation MSE (8-bit, synthetic pixels) | MSE against the A-buffer on Bistro | Inference |
|---|---|---|---|
| `16x16` | 0.000022 | 0.0000658 | 1.01 ms |
| `32x16` | 0.000020 | 0.0000649 | 1.75 ms |

The Bistro numbers are `ARM_002_NeuralOIT --technique dfaoit --network <N>x<M> --compare --timings` at frame 240, once
the textures have loaded, at alpha 0.3 (the sample's default view), under the Arm ML Emulation Layer on an NVIDIA GPU.
`16x16` is the default: it costs about 60% of the inference time for the same image.

## How to train one

1. Training needs Python 3 with numpy and PyTorch.
2. Train: `python tools/train_dfaoit.py --hidden1 <N> --hidden2 <M>`. It writes `dfaoit-<N>x<M>.bin` into this folder.
   `--samples 8000000` is about three times quicker and measurably worse; `--cache <file>` keeps the sampled pixels for
   the next run.
3. Run `ARM_002_NeuralOIT --technique dfaoit --network <N>x<M> --compare` to see its error against the A-buffer.

`--hidden1` sets the widest intermediate tensor, which is what the inference cost tracks. The docstrings of
`tools/train_dfaoit.py` and `tools/dfaoit_scenes.py` have the details of the features, the loss and the integer graph.
