import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# 生成Hadamard矩阵的递归函数
def generate_hadamard_matrix(n):
    """
    生成Hadamard矩阵
    参数:
        n (int): 矩阵的阶数（必须是2的幂）
    返回:
        np.ndarray: Hadamard矩阵
    """
    if n == 1:
        return np.array([[1]])
    else:
        H = generate_hadamard_matrix(n // 2)
        return np.block([[H, H], [H, -H]])

# 执行Hadamard变换
def hadamard_transform(x):
    """
    对输入向量执行Hadamard变换
    参数:
        x (np.ndarray): 输入向量（长度必须是2的幂）
    返回:
        np.ndarray: Hadamard变换后的向量
    """
    n = len(x)
    if not (n & (n - 1) == 0 and n != 0):
        raise ValueError("输入向量长度必须是2的幂")
    
    # 生成Hadamard矩阵
    H = generate_hadamard_matrix(n)
    # 进行矩阵乘法
    return np.dot(H, x)

def fast_hadamard_transform(x):
    """
    快速Hadamard变换（递归实现）
    参数:
        x (np.ndarray): 输入向量（长度必须是2的幂）
    返回:
        np.ndarray: 变换后的向量
    """
    n = len(x)
    if n == 1:
        return x
    elif not (n & (n - 1) == 0 and n != 0):
        raise ValueError("输入向量长度必须是2的幂")
    
    # 分为偶数部分和奇数部分
    x_even = x[:n // 2]
    x_odd = x[n // 2:]
    
    # 递归计算子问题
    y_even = fast_hadamard_transform(x_even)
    y_odd = fast_hadamard_transform(x_odd)
    
    # 合并结果
    y = np.zeros(n, dtype=x.dtype)
    y[:n // 2] = y_even + y_odd
    y[n // 2:] = y_even - y_odd
    return y

# 快速Hadamard变换（迭代实现）
def fast_hadamard_transform_iterative(x):
    """
    快速Hadamard变换（迭代实现）
    参数:
        x (np.ndarray): 输入向量（长度必须是2的幂）
    返回:
        np.ndarray: 变换后的向量
    """
    n = len(x)
    if not (n & (n - 1) == 0 and n != 0):
        raise ValueError("输入向量长度必须是2的幂")
    
    y = np.copy(x)
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(step):
                a = y[i + j]
                b = y[i + j + step]
                y[i + j] = a + b
                y[i + j + step] = a - b
        step *= 2
    return y


def fast_hadamard_transform_visualize_input_output_indices(x):
    """
    Fast Hadamard Transform with input and output indices labeled on the butterfly diagram
    Parameters:
        x (np.ndarray): Input vector (length must be a power of 2)
    Returns:
        np.ndarray: Transformed vector
    """
    n = len(x)
    if not (n & (n - 1) == 0 and n != 0):
        raise ValueError("Input length must be a power of 2")
    
    y = np.copy(x)
    step = 1
    stage = 0  # Track the current stage for plotting
    
    # Initialize graph
    G = nx.DiGraph()
    pos = {}  # Node positions
    labels = {}  # Node labels for visualization
    
    # Add input layer
    for i in range(n):
        node = f"0_{i}"
        G.add_node(node)
        pos[node] = (stage, -i)
        labels[node] = str(i)  # Show actual indices for input layer
    
    # Process each step
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(step):
                # Index changes during butterfly operation
                a_index = i + j
                b_index = i + j + step
                
                # Add nodes for the next layer
                output_node_1 = f"{stage + 1}_{a_index}"
                output_node_2 = f"{stage + 1}_{b_index}"
                
                # Add edges representing butterfly connections
                G.add_edge(f"{stage}_{a_index}", output_node_1)
                G.add_edge(f"{stage}_{b_index}", output_node_1)
                G.add_edge(f"{stage}_{a_index}", output_node_2)
                G.add_edge(f"{stage}_{b_index}", output_node_2)
                
                # Add positions for visualization
                pos[output_node_1] = (stage + 1, -a_index)
                pos[output_node_2] = (stage + 1, -b_index)
                
                # Only add labels for the final output layer
                if step * 2 == n:
                    labels[output_node_1] = str(a_index)
                    labels[output_node_2] = str(b_index)
        
        step *= 2
        stage += 1
    
    # Draw the butterfly diagram
    plt.figure(figsize=(14, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_size=2000,
        node_color="lightblue",
        font_size=10,
        font_color="black",
        edge_color="gray",
        arrowsize=10
    )
    plt.title("Butterfly Diagram with Input and Output Indices", fontsize=14)
    plt.show()
    
    return y

def fast_hadamard_transform_visualize_values(x):
    """
    Fast Hadamard Transform with values displayed on all nodes in the butterfly diagram
    Parameters:
        x (np.ndarray): Input vector (length must be a power of 2)
    Returns:
        np.ndarray: Transformed vector
    """
    n = len(x)
    if not (n & (n - 1) == 0 and n != 0):
        raise ValueError("Input length must be a power of 2")
    
    y = np.copy(x)
    step = 1
    stage = 0  # Track the current stage for plotting
    
    # Initialize graph
    G = nx.DiGraph()
    pos = {}  # Node positions
    labels = {}  # Node labels for visualization
    
    # Add input layer with initial values
    for i in range(n):
        node = f"0_{i}"
        G.add_node(node)
        pos[node] = (stage, -i)
        labels[node] = f"{x[i]:.1f}"  # Show initial values for input layer
    
    # Process each step
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(step):
                # Perform butterfly operation
                a_index = i + j
                b_index = i + j + step
                a_val = y[a_index]
                b_val = y[b_index]
                
                # Compute new values
                y[a_index] = a_val + b_val
                y[b_index] = a_val - b_val
                
                # Add nodes for the next layer
                output_node_1 = f"{stage + 1}_{a_index}"
                output_node_2 = f"{stage + 1}_{b_index}"
                
                # Add edges representing butterfly connections
                G.add_edge(f"{stage}_{a_index}", output_node_1)
                G.add_edge(f"{stage}_{b_index}", output_node_1)
                G.add_edge(f"{stage}_{a_index}", output_node_2)
                G.add_edge(f"{stage}_{b_index}", output_node_2)
                
                # Add positions and labels for visualization
                pos[output_node_1] = (stage + 1, -a_index)
                pos[output_node_2] = (stage + 1, -b_index)
                labels[output_node_1] = f"{y[a_index]:.1f}"
                labels[output_node_2] = f"{y[b_index]:.1f}"
        
        step *= 2
        stage += 1
    
    # Draw the butterfly diagram
    plt.figure(figsize=(14, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_size=2000,
        node_color="lightblue",
        font_size=10,
        font_color="black",
        edge_color="gray",
        arrowsize=10
    )
    plt.title("Butterfly Diagram with Node Values", fontsize=14)
    plt.show()
    
    return y


# 示例用法
if __name__ == "__main__":
    # 输入向量
    x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    # 执行Hadamard变换
    y = hadamard_transform(x)
    fast_y = fast_hadamard_transform(x)
    print("输入向量:", x)
    print("Hadamard变换结果:", y)
    print("快速Hadamard变换结果:", fast_y)
    fast_y_iterative = fast_hadamard_transform_visualize_values(x)

    
    print("快速Hadamard变换结果（迭代实现）:", fast_y_iterative)
