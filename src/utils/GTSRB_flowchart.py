import matplotlib.pyplot as plt
import networkx as nx

# Create directed graph
G = nx.DiGraph()

# Define nodes with attributes: label, color, shape
nodes = {
    "input": ("GTSRB Dataset\n39,209 training images\n12,630 test images\n43 classes", "lightblue"),
    
    "preprocessing": ("Data Preprocessing Pipeline", "lightgreen"),
    "roi": ("ROI Cropping\n(extract sign region)", "lightgreen"),
    "resize": ("Resize to 224x224 pixels", "lightgreen"),
    "normalize": ("Normalize (ImageNet mean/std)", "lightgreen"),

    "augmentation": ("Training Data Augmentation", "lightgreen"),
    "rotation": ("Random Rotation ±15°", "lightgreen"),
    "affine": ("Random Affine Transform", "lightgreen"),
    "color": ("Color Jitter", "lightgreen"),
    "perspective": ("Random Perspective", "lightgreen"),
    "erase": ("Random Erasing 10%", "lightgreen"),

    "model": ("ResNet50 Deep Neural Network", "orange"),
    "input_layer": ("Input: 224x224x3", "orange"),
    "conv": ("Conv Layer + Max Pool", "orange"),
    "residual1": ("Residual Block Stage 1", "orange"),
    "residual2": ("Residual Block Stage 2", "orange"),
    "residual3": ("Residual Block Stage 3", "orange"),
    "residual4": ("Residual Block Stage 4", "orange"),
    "gap": ("Global Average Pooling", "orange"),
    "dropout": ("Dropout 0.4 + Linear(2048→43)", "orange"),
    "output_layer": ("Output: 43 class probabilities", "orange"),

    "training": ("Model Training Process\nOptimizer: AdamW\nLoss: Weighted CrossEntropy\nScheduler: ReduceLROnPlateau\nEarlyStopping(patience=10)", "lightgreen"),
    
    "validation": ("Model Evaluation\nValidation Set 10%\nTest Set 12,630\nMetrics Computation", "red"),
    
    "output": ("Trained Model\nEpoch 18, 100% Val Acc, 99.11% Test Acc", "lightblue"),
    "deployment": ("Production Inference\nTraffic_Sign_Classifier.ipynb\nPredicted class & Confidence", "lightblue"),
}

# Add nodes
for node, (label, color) in nodes.items():
    G.add_node(node, label=label, color=color)

# Define edges
edges = [
    ("input","preprocessing"),
    ("preprocessing","roi"), ("roi","resize"), ("resize","normalize"),

    ("normalize","augmentation"),
    ("augmentation","rotation"), ("augmentation","affine"), ("augmentation","color"), 
    ("augmentation","perspective"), ("augmentation","erase"),
    
    ("rotation","model"), ("affine","model"), ("color","model"), 
    ("perspective","model"), ("erase","model"),

    ("model","input_layer"), ("input_layer","conv"), ("conv","residual1"),
    ("residual1","residual2"), ("residual2","residual3"), ("residual3","residual4"),
    ("residual4","gap"), ("gap","dropout"), ("dropout","output_layer"), ("output_layer","training"),
    
    ("training","validation"),
    ("validation","output"),
    ("output","deployment")
]

G.add_edges_from(edges)

# Draw the graph
pos = nx.spring_layout(G, k=2, iterations=100)  # spring layout for clarity
plt.figure(figsize=(18,12))

# Draw nodes
colors = [data['color'] for _, data in G.nodes(data=True)]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=3000, alpha=0.9)

# Draw edges
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='black')

# Draw labels
labels = {n: data['label'] for n, data in G.nodes(data=True)}
nx.draw_networkx_labels(G, pos, labels, font_size=9, font_family="Arial")

plt.axis('off')
plt.title("GTSRB ResNet50 Traffic Sign Classification Pipeline", fontsize=16)
plt.show()
