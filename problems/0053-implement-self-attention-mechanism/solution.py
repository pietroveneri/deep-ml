import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    d_k = len(K[1])
    Q = np.array(Q)
    K = np.array(K)
    V = np.array(V)

    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    rows = range(len(scores))
    attention_weights = []
    for row in rows:
        attention_weights.append(softmax(scores[row]))
    
    return np.dot(attention_weights, V)

def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exp_scores = [np.exp(s - max_score) for s in scores] # Max score to prevent overflow !! 
    total = sum(exp_scores)
    return [e / total for e in exp_scores]