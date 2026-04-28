This is a Python-based “mathematical crochet” simulation script.

I define five type of stitches not only as geometric operations but also as tensor/graph operations.

Core Concepts

We treat the fabric as a directed acyclic graph (DAG).

    • Node: A stitch.
    
    • Edge: A physical connection (dependency) between threads.

    • Mathematical Meaning:
    
        1. sc (Short Crochet): Identity mapping (1-to-1). Transmits information while preserving dimension.
        
        2. inc (Increase): Upsampling/splitting (1-to-2). Increases manifold curvature (creates waves/hyperbolic surfaces).
        
        3. dec (Decrease): Pooling/Aggregation (2-to-1). Reduces manifold curvature (creates contraction/spherical surfaces).
        
        4. ch (Chain Stitch): Offset / Spacing. Independent of the previous layer’s input; depends only on the sequence’s predecessor (increases local extensibility).
        
        5. dc (Double Chain Stitch): Weighted feature (1-to-1 with height). Similar to the short stitch, but with greater weight (line length/height), allowing for spanning greater spatial distances.
        

This code is not just a graphics generator; it is a prototype simulator for CrochetNet:

    1. foundation_chain (Input Embedding):
    
        ◦ Initializes the latent space. Similar to the positional encoding in a Transformer or the input token sequence.
        
    2. inc (Split Layer / Upsampling):
    
        ◦ Code logic: Reads 1 prev_node and generates 2 new_nodes.
        
        ◦ Geometric meaning: The perimeter increases, and the radius expands.
        
        ◦ AI Implications: Similar to transposed convolutions in GANs, generating high-dimensional details from low-dimensional features. If inc is applied at every node, the space rapidly transforms into a hyperbolic surface (with folds resembling the edges of a mushroom), which is a method for increasing network capacity.
        
    3. dec (Pooling / Aggregation):
    
        ◦ Code logic: Reads 2 prev_nodes and generates 1 new_node.
        
        ◦ Geometric meaning: The perimeter shrinks, forming a cup-shaped or spherical contraction.
        
        ◦ AI Implications: Similar to Max Pooling in CNNs or token merging in Attention mechanisms. It forces the network to compress information from two features into one, extracting the primary features.
        
    4. sc vs dc (Activation Weights):
    
        ◦ Both are 1-to-1 mappings.
        
        ◦ sc is a standard activation function (e.g., ReLU).
        
        ◦ DC has a higher “height” attribute and can be viewed as connections with greater weights (or broader temporal spans).
        
Expected Results

When you run this script, you will see a structure resembling a spider web or radar chart:

    • The center is a gray starting node.
    
    • Orange nodes radiate outward (increasing in density).

    • Then, red subtraction nodes appear (density decreases, and the spacing between nodes widens).
    
    • The outermost ring consists of purple long needles.
    
    • Blue solid lines represent “physical pull” (information flow from the previous layer to the next).
    
    • Gray dashed lines represent the “weaving order” (the temporal sequence of the RNN).
    
This forms a physically interpretable neural network structure.


Prerequisites

You need to install the following libraries:

codeBash
pip install streamlit networkx plotly numpy

🧶 Full code (crochetnet.py)

Save the following code as crochetnet.py, then run the following command in the terminal:
streamlit run crochetnet.py


How to Use This Web Application

    1. Run the program: Type `streamlit run crochetnet.py` in the terminal. A page will open automatically in your browser.
    
    2. Set the starting point: In the left sidebar, set the Foundation Chain (e.g., 6). This corresponds to the Transformer’s <BOS> (Begin of Sentence) or initial embedding.
    
    3. Write Code: Enter “crochet code” in the text box. I’ve built in a simple parser that supports the following syntax:
    
        ◦ 6 inc: Perform 6 increases.
        
        ◦ all sc: Crochet single crochet stitches for the entire round (automatically calculated based on the number of stitches from the previous round).
        
        ◦ 1 sc, 1 inc * 6: This is the classic circular crochet formula (increase every 1 stitch, repeat 6 times).
        
        ◦ all dec: Decrease all stitches (knit 2 together).
        
    4. Click Run CrochetNet:
    
        ◦ You will see an interactive Plotly chart generated on the right.
        
        ◦ The center is the starting stitch, radiating outward.
        
        ◦ Orange points represent increases (Inc); you’ll notice a “fork” there (1 becomes 2).
        
        ◦ Red points represent decreases (Dec); you’ll notice two blue lines converging into a single point (2 becomes 1).
        
        ◦ Hover your mouse over a point to see its ID and Type.
        
💡 What can we see from this visualization?

    • ResNet residual connections: Although the code doesn’t explicitly draw cross-layer connections, if you look closely at the Inc (increments), it essentially copies an input feature twice and passes it to the next layer.
    
    • Pooling layers: Look at the “dec” (decrease). This is the physical representation of Max/Avg Pooling. Two neurons from the previous layer converge into one, compressing the information.
    
    • RNN sequences: The gray dotted lines represent time steps within the same layer. Computations occur one after another.
    
    • Manifold Geometry: If you keep applying `inc`, the diagram becomes very crowded (representing exponential growth in a hyperbolic space); if you keep applying `dec`, the diagram rapidly narrows or even breaks apart (representing a closed space).
    
You can try inputting a crazy pattern, such as three consecutive rows of `all inc`, to see how the network structure explodes!
