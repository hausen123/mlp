FFN Adapter Routing Specification (Mean Key Method)

1. Overview

This specification describes a routing mechanism for multiple FFN adapters using mean hidden-state vectors as routing keys.
The base transformer model remains frozen. Each task is represented byban adapter (e.g., LoRA) and a routing key.
At inference time the adapter whose key is most similar to the current hidden state is selected.

2. Model Structure

Transformer Layer

    h -> Attention -> h_attn
    h_attn -> FFN -> h_ffn

Adapter is applied as

    h_out = FFN(h_attn) + Adapter_i(h_attn)

3. Data Structures

AdapterSet

    adapters : list of Adapter
    keys     : list of Vector(d_model)

Adapter

    A : matrix (r x d_model)
    B : matrix (d_ff x r)

Vector

    float[d_model]

4. Adapter Training

function train_adapter(dataset):
   
    freeze(base_model)
    initialize(adapter)
    hidden_list = []
    for batch in dataset:
        h = forward_to_FFN_input(batch)
        hidden_list.append(h)
        y1 = forward_with_basemodel(h, basemodel)
        y2 = forward_with_adapter(h, adapter)
        loss = compute_loss(y1, y2)
        update(adapter)
    return adapter, hidden_list

5. Key Generation

    H = concat(hidden_list)
    key = mean(H)
    return key

6. Mean Function

function mean(H):

    N = number_of_vectors(H)
    sum = zero_vector(d_model)
    for v in H:
        sum = sum + v
    return sum / N

7. Adapter Registration

function register_adapter(adapter, key):

    AdapterSet.adapters.append(adapter)
    AdapterSet.keys.append(key)

8. Routing Algorithm

function route_adapter(h):

    best_score = -infinity
    best_index = 0

    for i in range(len(AdapterSet.keys)):
        k = AdapterSet.keys[i]
        score = cosine_similarity(h, k)
        if score > best_score:
            best_score = score
            best_index = i
    return AdapterSet.adapters[best_index]

9. Cosine Similarity

function cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

10. Adapter Forward

function adapter_forward(adapter, h):

    z = A * h
    delta = B * z
    return delta

11. FFN Forward With Routing

function FFN_forward(h):
    adapter = route_adapter(h)
    base_output = FFN(h)
    delta = adapter_forward(adapter, h)
    return base_output + delta

15. Possible Extensions

Top-k routing
Soft routing
Layer-wise routing
PCA-based key
