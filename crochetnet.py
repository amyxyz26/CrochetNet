import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import math

# ==========================================
# 1. Core Logic: CrochetManifold (Modified Version)
# ==========================================
class CrochetManifold:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.rows = []  # [[id, id], [id...]]
        self.current_row_nodes = []
        self.prev_row_ptr = 0
        self.logs = [] # Record operation logs
        
    def log(self, message):
        self.logs.append(message)

    def _add_node(self, stitch_type, height=1.0):
        node_id = self.node_counter
        self.graph.add_node(node_id, type=stitch_type, height=height, row=len(self.rows))
        self.current_row_nodes.append(node_id)
        self.node_counter += 1
        
        # Horizontal connection (Sequence / RNN connection)
        if len(self.current_row_nodes) > 1:
            prev_node = self.current_row_nodes[-2]
            self.graph.add_edge(prev_node, node_id, relationship='sequence')
        
        # If it is a head-to-tail connection of each round (spiral), a special connection 
        # can be added here, but it is usually not needed in the topological graph
        return node_id

    def _get_prev_input(self, consume_count=1):
        if not self.rows:
            return []
        
        prev_row = self.rows[-1]
        # Loop processing: if the pointer exceeds the length of the previous round 
        # (e.g., spiral crochet), take modulo to return to the beginning.
        # But for strict mathematical checking, we report an error here if it exceeds, 
        # or simulate the infinite extension of the spiral.
        if self.prev_row_ptr + consume_count > len(prev_row):
            # Simple processing: if there are not enough stitches, it cannot be executed
            raise ValueError(f"Not enough remaining stitches in the previous round! Need {consume_count}, remaining {len(prev_row) - self.prev_row_ptr}")
            
        inputs = prev_row[self.prev_row_ptr : self.prev_row_ptr + consume_count]
        self.prev_row_ptr += consume_count
        return inputs

    # --- Stitch definitions ---
    def foundation_chain(self, count):
        for _ in range(count):
            self._add_node("ch")
        self._finalize_row()
        self.log(f"Row 0: Foundation chain with {count} chains")

    def sc(self):
        try:
            inputs = self._get_prev_input(1)
            node = self._add_node("sc")
            self.graph.add_edge(inputs[0], node, relationship='structure')
            return True
        except ValueError as e:
            return False

    def inc(self):
        try:
            inputs = self._get_prev_input(1)
            # Increase generates two nodes, attached to the same input
            node_a = self._add_node("inc")
            node_b = self._add_node("inc")
            self.graph.add_edge(inputs[0], node_a, relationship='structure')
            self.graph.add_edge(inputs[0], node_b, relationship='structure')
            return True
        except ValueError:
            return False

    def dec(self):
        try:
            inputs = self._get_prev_input(2)
            node = self._add_node("dec")
            self.graph.add_edge(inputs[0], node, relationship='structure')
            self.graph.add_edge(inputs[1], node, relationship='structure')
            return True
        except ValueError:
            return False
            
    def dc(self):
        try:
            inputs = self._get_prev_input(1)
            node = self._add_node("dc", height=2.0)
            self.graph.add_edge(inputs[0], node, relationship='structure')
            return True
        except ValueError:
            return False

    def _finalize_row(self):
        if self.current_row_nodes:
            self.rows.append(self.current_row_nodes)
            count = len(self.current_row_nodes)
            prev_count = len(self.rows[-2]) if len(self.rows) > 1 else 0
            diff = count - prev_count
            sign = "+" if diff > 0 else ""
            self.log(f"Row {len(self.rows)-1} completed: {count} stitches ({sign}{diff})")
            self.current_row_nodes = []
            self.prev_row_ptr = 0
        else:
            self.log("Warning: Empty row, no stitches generated")

# ==========================================
# 2. Layout & Plotting Logic (Plotly)
# ==========================================
def generate_interactive_plot(net):
    pos = {}
    # Polar coordinate layout calculation
    for r_idx, row_nodes in enumerate(net.rows):
        num_nodes = len(row_nodes)
        if num_nodes == 0: continue
        
        radius = (r_idx + 1) * 1.5 
        
        for n_idx, node in enumerate(row_nodes):
            theta = 2 * np.pi * n_idx / num_nodes + (r_idx * 0.1)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            pos[node] = np.array([x, y])

    edge_x_struct = []
    edge_y_struct = []
    edge_x_seq = []
    edge_y_seq = []

    for edge in net.graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        rel = edge[2].get('relationship')
        
        if rel == 'structure':
            edge_x_struct.extend([x0, x1, None])
            edge_y_struct.extend([y0, y1, None])
        else:
            edge_x_seq.extend([x0, x1, None])
            edge_y_seq.extend([y0, y1, None])

    edge_trace_struct = go.Scatter(
        x=edge_x_struct, y=edge_y_struct,
        line=dict(width=2, color='#3366CC'),
        hoverinfo='none',
        mode='lines',
        name='Structure (Deep)'
    )

    edge_trace_seq = go.Scatter(
        x=edge_x_seq, y=edge_y_seq,
        line=dict(width=1, color='#888888', dash='dot'),
        hoverinfo='none',
        mode='lines',
        name='Sequence (RNN)'
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    color_map = {'ch': '#999999', 'sc': '#109618', 'inc': '#FF9900', 'dec': '#DC3912', 'dc': '#990099'}
    
    for node in net.graph.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        
        stitch_type = node[1]['type']
        row_id = node[1]['row']
        node_text.append(f"ID: {node[0]}<br>Type: {stitch_type.upper()}<br>Row: {row_id}")
        node_color.append(color_map.get(stitch_type, 'black'))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=12,
            line_width=1
        ),
        name='Neurons (Stitches)'
    )

    # Fixed Layout syntax here
    fig = go.Figure(data=[edge_trace_seq, edge_trace_struct, node_trace],
                 layout=go.Layout(
                    title={
                        'text': 'CrochetNet Topological Visualization',
                        'font': {'size': 16}
                    },
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600
                 ))
    return fig

# ==========================================
# 3. Streamlit UI Interface
# ==========================================

st.set_page_config(page_title="CrochetNet Generator", layout="wide")

st.title("🧶 CrochetNet: Neural Crochet Topology Generator")
st.markdown("""
**Crochet Is All You Need.** 
This is a dynamic generator based on the "Neural Crochet" theory. We treat neural networks as crocheted pieces:
*   **sc (Short)**: Linear transmission (Identity)
*   **inc (Increase)**: Upsampling / Entropy increase -> Generates hyperbolic crinkles
*   **dec (Decrease)**: Downsampling / Pooling -> Generates spherical contraction
""")

# --- Sidebar: Hyperparameters ---
st.sidebar.header("🛠️ Crocheting Parameters (Hyperparameters)")

start_chain = st.sidebar.number_input("Foundation Chain Count", min_value=3, max_value=20, value=6)

# Define pattern input
st.sidebar.subheader("Define your neural network layers (Pattern)")
st.sidebar.markdown("Each line represents a round. Format: `count stitch` (e.g. `6 inc`) or `all sc`")

default_pattern = """6 inc
1 sc, 1 inc * 6
all sc
1 sc, 1 dec * 6
all dc
"""
pattern_input = st.sidebar.text_area("Enter Crochet Code (Pattern Code)", value=default_pattern, height=200)

run_btn = st.sidebar.button("Start Crocheting (Run CrochetNet)")

# --- Main Logic Execution ---
if run_btn:
    net = CrochetManifold()
    
    # 1. Execute foundation chain
    net.foundation_chain(start_chain)
    
    # 2. Parse and execute code
    lines = pattern_input.strip().split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip().lower()
        if not line: continue
        
        # Get the number of stitches in previous round, used to calculate "all" or "around"
        prev_count = len(net.rows[-1]) if net.rows else 0
        
        current_ops = [] # List of operations to execute in this round
        
        # Simple parser logic
        if "all" in line or "around" in line:
            # All same stitch
            if "sc" in line: current_ops = ["sc"] * prev_count
            elif "inc" in line: current_ops = ["inc"] * prev_count
            elif "dc" in line: current_ops = ["dc"] * prev_count
            elif "dec" in line: current_ops = ["dec"] * (prev_count // 2) # dec needs 2 stitches to become 1
        
        elif "*" in line:
            # Repeat pattern: "1 sc, 1 inc * 6"
            parts = line.split('*')
            repeat_times = int(parts[1].strip())
            pattern_group = parts[0].split(',')
            
            group_ops = []
            for p in pattern_group:
                p = p.strip() # "1 sc"
                p_parts = p.split(' ')
                if len(p_parts) >= 2:
                    count = int(p_parts[0])
                    op_type = p_parts[1]
                    group_ops.extend([op_type] * count)
            
            current_ops = group_ops * repeat_times
            
        else:
            # Simple format: "6 inc"
            parts = line.split(' ')
            if len(parts) >= 2 and parts[0].isdigit():
                count = int(parts[0])
                op_type = parts[1]
                current_ops = [op_type] * count
        
        # Execute operations for this round
        success_count = 0
        for op in current_ops:
            res = False
            if op == 'sc': res = net.sc()
            elif op == 'inc': res = net.inc()
            elif op == 'dec': res = net.dec()
            elif op == 'dc': res = net.dc()
            
            if res: success_count += 1
        
        net._finalize_row()

    # --- Result Display ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🌐 Topological Graph")
        fig = generate_interactive_plot(net)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📝 Compiler Logs")
        st.code("\n".join(net.logs), language="text")
        
        st.info("Legend:\n"
                "🟢 sc (Short) - Keep\n"
                "🟠 inc (Increase) - Expand\n"
                "🔴 dec (Decrease) - Contract\n"
                "🟣 dc (Double) - Strong weight\n"
                "🔵 Blue line - Structural dependency\n"
                "⚪ Gray line - Sequential order")

else:
    st.info("Click the 'Start Crocheting' button on the left to generate the neural network structure.")
