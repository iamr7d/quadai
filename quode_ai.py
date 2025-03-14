import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import google.generativeai as genai
import json
import os
import time
import uuid
from PIL import Image
import io
import numpy as np
import pennylane as qml  # Quantum computing library

# Configure page
st.set_page_config(
    page_title="NODE.AI - Decision Tree Story Writer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .story-box {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    .path-option {
        background-color: #EFF6FF;
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid #60A5FA;
        margin-bottom: 10px;
        cursor: pointer;
    }
    .path-option:hover {
        background-color: #DBEAFE;
    }
    .quantum-path {
        background-color: #EDE9FE;
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid #8B5CF6;
        margin-bottom: 10px;
    }
    .selected-path {
        background-color: #DBEAFE;
        border-left: 3px solid #2563EB;
    }
    .story-title {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .quantum-badge {
        background-color: #8B5CF6;
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini API client
def initialize_gemini_client(api_key):
    genai.configure(api_key=api_key)
    return genai

# Quantum circuit for story path generation
def quantum_path_generator(num_paths, seed=None, entanglement=0.5):
    """
    Uses a quantum circuit to generate a superposition of story paths
    
    Parameters:
    - num_paths: Number of story paths to generate
    - seed: Random seed for reproducibility
    - entanglement: Amount of entanglement between qubits (0-1)
    
    Returns:
    - List of path weights based on quantum probabilities
    """
    # Number of qubits needed to represent the paths
    num_qubits = max(1, int(np.ceil(np.log2(num_paths))))
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create a quantum device
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev)
    def quantum_circuit():
        # Create superposition
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
            
        # Add entanglement between paths
        if entanglement > 0:
            for i in range(num_qubits-1):
                qml.CRZ(entanglement * np.pi, wires=[i, i+1])
                if entanglement > 0.5:
                    qml.CNOT(wires=[i, i+1])
            
        # Add random rotations for diversity
        for i in range(num_qubits):
            qml.RY(np.random.uniform(0, 2*np.pi), wires=i)
            qml.RZ(np.random.uniform(0, 2*np.pi), wires=i)
            
        # Measure all qubits
        return [qml.probs(wires=i) for i in range(num_qubits)]
    
    # Get the quantum probabilities
    probs = quantum_circuit()
    
    # Use quantum probabilities to influence path weights
    path_weights = []
    for i in range(min(num_paths, 2**num_qubits)):
        # Convert index to binary representation
        binary = format(i, f'0{num_qubits}b')
        weight = 1.0
        for j, bit in enumerate(binary):
            bit_val = int(bit)
            weight *= probs[j][bit_val]
        path_weights.append(weight)
    
    # Normalize weights
    total = sum(path_weights)
    if total > 0:
        path_weights = [w/total for w in path_weights]
    else:
        path_weights = [1.0/num_paths for _ in range(num_paths)]
    
    return path_weights

# Story Node class with quantum properties
class StoryNode:
    def __init__(self, content, node_id=None, parent_id=None, quantum_weight=1.0):
        self.id = node_id if node_id else str(uuid.uuid4())
        self.content = content
        self.parent_id = parent_id
        self.children = []
        self.quantum_weight = quantum_weight
        self.quantum_state = "superposition"  # Can be "superposition" or "collapsed"
    
    def add_child(self, child_node):
        self.children.append(child_node)
        return child_node.id
    
    def collapse(self):
        """Simulate quantum collapse - path becomes definite"""
        self.quantum_state = "collapsed"
    
    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "parent_id": self.parent_id,
            "quantum_weight": self.quantum_weight,
            "quantum_state": self.quantum_state,
            "children": [child.id for child in self.children]
        }

# Story Tree class with quantum features
class StoryTree:
    def __init__(self):
        self.nodes = {}
        self.root_id = None
        self.graph = nx.DiGraph()
        self.entanglement_level = 0.5  # Default entanglement level
    
    def add_root(self, content):
        node = StoryNode(content)
        self.nodes[node.id] = node
        self.root_id = node.id
        self.graph.add_node(node.id, content=content[:50] + "..." if len(content) > 50 else content)
        return node.id
    
    def add_node(self, content, parent_id, quantum_weight=1.0):
        if parent_id not in self.nodes:
            return None
        
        node = StoryNode(content, parent_id=parent_id, quantum_weight=quantum_weight)
        self.nodes[node.id] = node
        self.nodes[parent_id].add_child(node)
        
        self.graph.add_node(node.id, content=content[:50] + "..." if len(content) > 50 else content)
        self.graph.add_edge(parent_id, node.id, weight=quantum_weight)
        
        return node.id
    
    def get_node(self, node_id):
        return self.nodes.get(node_id)
    
    def get_path_to_node(self, node_id):
        path = []
        current_id = node_id
        
        while current_id is not None:
            node = self.nodes.get(current_id)
            if node:
                path.append(node)
                current_id = node.parent_id
            else:
                break
        
        return list(reversed(path))
    
    def apply_quantum_collapse(self, node_id):
        """Simulates quantum collapse by selecting one path and increasing its probability"""
        node = self.get_node(node_id)
        if not node or not node.children:
            return
        
        # Get quantum probabilities for child nodes
        weights = quantum_path_generator(len(node.children), entanglement=self.entanglement_level)
        
        # Assign weights to children
        for i, child in enumerate(node.children):
            if i < len(weights):
                child_node = self.get_node(child.id)
                if child_node:
                    child_node.quantum_weight = weights[i]
                    # Update graph edge weight
                    if node_id in self.graph and child.id in self.graph[node_id]:
                        self.graph[node_id][child.id]['weight'] = weights[i]
    
    def quantum_interference(self, node_id1, node_id2, target_parent_id):
        """
        Creates a new node that represents quantum interference between two existing nodes
        """
        node1 = self.get_node(node_id1)
        node2 = self.get_node(node_id2)
        
        if not node1 or not node2 or target_parent_id not in self.nodes:
            return None
        
        # Create interference content by mixing the two nodes
        # In a real quantum system, this would be a superposition
        mixed_content = f"[QUANTUM INTERFERENCE]\n\n{node1.content}\n\nINTERFERES WITH\n\n{node2.content}"
        
        # Add the new interference node
        interference_node = StoryNode(
            mixed_content, 
            parent_id=target_parent_id,
            quantum_weight=0.5  # Interference has equal weight initially
        )
        interference_node.quantum_state = "interference"
        
        self.nodes[interference_node.id] = interference_node
        self.nodes[target_parent_id].add_child(interference_node)
        
        self.graph.add_node(
            interference_node.id, 
            content="[QUANTUM INTERFERENCE]"
        )
        self.graph.add_edge(target_parent_id, interference_node.id, weight=0.5)
        
        return interference_node.id
    
    def visualize(self):
        plt.figure(figsize=(14, 10))  # Larger figure size
        # Use a more readable layout algorithm
        pos = nx.kamada_kawai_layout(self.graph)  # or nx.fruchterman_reingold_layout
        
        # Increase node sizes and add more contrast
        node_sizes = [min(len(self.nodes[n].content), 1000) + 800 for n in self.graph.nodes()]
        
        # Use a better color scheme
        node_colors = []
        for n in self.graph.nodes():
            if self.nodes[n].quantum_state == "collapsed":
                node_colors.append("#4CAF50")  # Green
            elif self.nodes[n].quantum_state == "interference":
                node_colors.append("#9C27B0")  # Purple
            else:
                node_colors.append("#2196F3")  # Blue
        
        # Add node borders
        nx.draw_networkx_nodes(self.graph, pos, 
                            node_size=node_sizes,
                            node_color=node_colors,
                            edgecolors='white',
                            linewidths=1.5,
                            alpha=0.9)
        
        # Make edges more visible
        edge_weights = nx.get_edge_attributes(self.graph, 'weight')
        edge_widths = [max(w*3, 1.5) for w in edge_weights.values()]
        nx.draw_networkx_edges(self.graph, pos, width=edge_widths, alpha=0.8, 
                            edge_color='#555555', arrows=True, arrowsize=20)
        
        # Improve labels
        labels = {node: data.get('content', '')[:15] + "..." for node, data in self.graph.nodes(data=True)}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=11, font_weight='bold', font_color='black')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Add a title
        plt.title("Quantum Story Tree", fontsize=16, fontweight='bold')
        
        # Use a higher DPI for better quality
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
        buf.seek(0)
        img = Image.open(buf)
        return img
        
    def save(self, filename):
        data = {
            "root_id": self.root_id,
            "entanglement_level": self.entanglement_level,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.root_id = data["root_id"]
        self.entanglement_level = data.get("entanglement_level", 0.5)
        self.nodes = {}
        self.graph = nx.DiGraph()
        
        # First pass: create all nodes
        for nid, node_data in data["nodes"].items():
            node = StoryNode(
                node_data["content"], 
                node_id=nid, 
                parent_id=node_data.get("parent_id"),
                quantum_weight=node_data.get("quantum_weight", 1.0)
            )
            node.quantum_state = node_data.get("quantum_state", "superposition")
            
            self.nodes[nid] = node
            self.graph.add_node(nid, content=node_data["content"][:50] + "..." if len(node_data["content"]) > 50 else node_data["content"])
        
        # Second pass: connect nodes
        for nid, node_data in data["nodes"].items():
            parent_id = node_data.get("parent_id")
            if parent_id is not None:
                self.graph.add_edge(parent_id, nid, weight=node_data.get("quantum_weight", 1.0))
            
            for child_id in node_data.get("children", []):
                if child_id in self.nodes and child_id not in [child.id for child in self.nodes[nid].children]:
                    self.nodes[nid].children.append(self.nodes[child_id])

# Story Generator class with quantum features
class StoryGenerator:
    def __init__(self, gemini_client):
        self.client = gemini_client
        self.story_tree = StoryTree()
        self.current_node_id = None
        self.model = "gemini-1.5-pro"
        self.quantum_seed = int(time.time())  # Use current time as initial seed
    
    def set_entanglement(self, level):
        """Set the quantum entanglement level (0-1)"""
        self.story_tree.entanglement_level = max(0, min(1, level))
    
    def generate_initial_situation(self, genre, protagonist, antagonist, setting):
        prompt = f"""
        Generate an engaging opening situation for a {genre} story featuring:
        - Protagonist: {protagonist}
        - Antagonist: {antagonist}
        - Setting: {setting}
        
        The situation should be compelling and open-ended, allowing for multiple possible developments.
        Write only the opening paragraph (100-150 words) that sets the scene and introduces the conflict.
        Include quantum-inspired themes of uncertainty, possibility, or parallel realities.
        """
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 200,
        }
        
        model = self.client.GenerativeModel(model_name=self.model, generation_config=generation_config)
        response = model.generate_content(prompt)
        
        situation = response.text.strip()
        self.current_node_id = self.story_tree.add_root(situation)
        return situation
    
    def generate_possible_paths(self, current_situation, num_paths=3):
        # Update quantum seed for variety
        self.quantum_seed = (self.quantum_seed * 1103515245 + 12345) % (2**31)
        
        # Get quantum weights for the paths
        path_weights = quantum_path_generator(
            num_paths, 
            seed=self.quantum_seed,
            entanglement=self.story_tree.entanglement_level
        )
        
        # Generate paths with quantum-weighted emphasis
        paths = []
        for i in range(num_paths):
            try:
                weight = path_weights[i]
                quantum_emphasis = "strongly" if weight > 0.5 else "somewhat"
                
                prompt = f"""
                Based on the following situation in a story:
                
                "{current_situation}"
                
                Generate a {quantum_emphasis} compelling continuation that could follow. This path should
                have a quantum probability of approximately {weight:.2f}.
                
                Write a single paragraph (80-120 words) that continues the story in an interesting direction.
                Include some action, dialogue, or character development.
                
                For a path with probability {weight:.2f}, make the events feel {"more definite and likely" if weight > 0.5 else "more uncertain or surprising"}.
                """
                
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 200,
                }
                
                model = self.client.GenerativeModel(model_name=self.model, generation_config=generation_config)
                response = model.generate_content(prompt)
                paths.append(response.text.strip())
            except Exception as e:
                # Fallback content when API quota is exceeded
                paths.append(f"Path {i+1}: A quantum fluctuation has temporarily obscured this possibility. (API quota exceeded)")
        
        return paths, path_weights
    
    def add_paths(self, paths, weights):
        if not self.current_node_id:
            return []
        
        # Add each path as a child node with its quantum weight
        path_ids = []
        for i, (path, weight) in enumerate(zip(paths, weights)):
            path_id = self.story_tree.add_node(path, self.current_node_id, weight)
            path_ids.append(path_id)
        
        return path_ids
    
    def select_path(self, path_id):
        """Select a specific path and update current node"""
        node = self.story_tree.get_node(path_id)
        if node:
            self.current_node_id = path_id
            # Mark this node as collapsed (measured) in the quantum sense
            node.collapse()
            # Apply quantum collapse to influence future paths
            self.story_tree.apply_quantum_collapse(path_id)
            return node.content
        return None
    
    def get_full_story(self):
        """Retrieves the full story path from root to current node"""
        if self.current_node_id is None:
            return ""
            
        path = self.story_tree.get_path_to_node(self.current_node_id)
        story_sections = [node.content for node in path]
        return "\n\n".join(story_sections)
    
    def generate_custom_continuation(self, current_story, user_input):
        """Generate a continuation based on user input with quantum influence"""
        # Get quantum probability for this continuation
        prob = quantum_path_generator(1, seed=self.quantum_seed)[0]
        
        prompt = f"""
        Current story so far:
        
        {current_story}
        
        User wants the story to continue with this idea:
        {user_input}
        
        Write a single paragraph (100-150 words) that continues the story based on the user's idea.
        Make it coherent with what came before, but incorporate the user's direction.
        
        This continuation has a quantum probability of {prob:.2f}, so make it feel 
        {"more definite and impactful" if prob > 0.5 else "more uncertain or surprising"}.
        """
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 200,
        }
        
        model = self.client.GenerativeModel(model_name=self.model, generation_config=generation_config)
        response = model.generate_content(prompt)
        
        continuation = response.text.strip()
        new_node_id = self.story_tree.add_node(continuation, self.current_node_id, prob)
        self.current_node_id = new_node_id
        return continuation
    
    def generate_quantum_ending(self, current_story):
        """Generate a quantum-inspired ending that acknowledges multiple realities"""
        prompt = f"""
        Here is a story that needs a quantum-inspired conclusion:
        
        {current_story}
        
        Write a final paragraph (100-150 words) that provides a satisfying ending with quantum elements.
        The ending should:
        
        1. Resolve the main conflict while hinting at multiple possible outcomes
        2. Include quantum concepts like superposition, entanglement, or parallel realities
        3. Suggest that other versions of the story might exist in parallel universes
        4. Provide emotional closure while embracing quantum uncertainty
        """
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 200,
        }
        
        model = self.client.GenerativeModel(model_name=self.model, generation_config=generation_config)
        response = model.generate_content(prompt)
        
        ending = response.text.strip()
        end_node_id = self.story_tree.add_node(ending, self.current_node_id, 1.0)
        
        # Mark this node as collapsed (final)
        node = self.story_tree.get_node(end_node_id)
        if node:
            node.collapse()
            
        self.current_node_id = end_node_id
        return ending
    
    def get_story_title(self, full_story):
        """Generate a quantum-inspired title for the story"""
        prompt = f"""
        Based on the following quantum-inspired story, generate a compelling, creative title (5-10 words):
        
        {full_story}
        
        The title should:
        1. Capture the essence of the story
        2. Include a subtle reference to quantum concepts (uncertainty, superposition, entanglement, etc.)
        3. Be intriguing and memorable
        
        Return ONLY the title, nothing else.
        """
        
        generation_config = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 30,
        }
        
        model = self.client.GenerativeModel(model_name=self.model, generation_config=generation_config)
        response = model.generate_content(prompt)
        
        title = response.text.strip()
        return title
    
    def save_story(self, filename):
        """Saves the current story tree state"""
        self.story_tree.save(filename)
    
    def load_story(self, filename):
        """Loads a story tree from file"""
        self.story_tree.load(filename)
        # Set current node to the last node in the deepest path
        if self.story_tree.root_id is not None:
            self.current_node_id = self.story_tree.root_id

# Main Streamlit app
def main():
    # Initialize session states
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False
    
    if 'story_generator' not in st.session_state:
        st.session_state.story_generator = None
    
    if 'current_paths' not in st.session_state:
        st.session_state.current_paths = []
    
    if 'path_ids' not in st.session_state:
        st.session_state.path_ids = []
    
    if 'path_weights' not in st.session_state:
        st.session_state.path_weights = []
    
    if 'story_title' not in st.session_state:
        st.session_state.story_title = "Untitled Quantum Story"
    
    if 'story_completed' not in st.session_state:
        st.session_state.story_completed = False
    
    # App header
    st.markdown("<h1 class='main-header'>⚛️ NODE.AI: Quantum Decision Tree Story Writer</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align: center; font-size: 1.2rem; margin-bottom: 30px;'>
        Create branching narratives with quantum computing and AI assistance. 
        Experience stories that exist in superposition until you observe them.
        </p>
        """, 
        unsafe_allow_html=True
    )
    
    # Sidebar for API key and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Enter Gemini API Key", value="AIzaSyDPo1RF7Yplkj5HJTEyKYiXOwjTAQ7xqSM", type="password")
        
        if st.button("Set API Key") and api_key:
            try:
                client = initialize_gemini_client(api_key)
                st.session_state.story_generator = StoryGenerator(client)
                st.session_state.api_key_set = True
                st.success("API key set successfully!")
            except Exception as e:
                st.error(f"Error setting API key: {str(e)}")
        
        if st.session_state.api_key_set:
            st.success("✅ API key is set")
            
            # Quantum parameters
            st.subheader("🔬 Quantum Parameters")
            
            if st.session_state.story_generator:
                entanglement = st.slider(
                    "Quantum Entanglement", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.story_generator.story_tree.entanglement_level,
                    step=0.1,
                    help="Higher values create more interconnected story paths"
                )
                
                if st.button("Apply Quantum Settings"):
                    st.session_state.story_generator.set_entanglement(entanglement)
                    st.success("Quantum parameters updated!")
            
            # Story operations
            st.subheader("📚 Story Operations")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Story"):
                    if st.session_state.story_generator:
                        try:
                            filename = f"quantum_story_{int(time.time())}.json"
                            st.session_state.story_generator.save_story(filename)
                            st.success(f"Story saved as {filename}")
                        except Exception as e:
                            st.error(f"Error saving story: {str(e)}")
            
            with col2:
                saved_stories = [f for f in os.listdir() if f.startswith("quantum_story_") and f.endswith(".json")]
                if saved_stories:
                    selected_story = st.selectbox("Load Story", saved_stories)
                    if st.button("Load"):
                        try:
                            st.session_state.story_generator.load_story(selected_story)
                            st.success("Story loaded successfully!")
                            st.session_state.current_paths = []
                            st.session_state.path_ids = []
                            st.session_state.path_weights = []
                            st.session_state.story_completed = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading story: {str(e)}")
        
        # About section
        st.markdown("---")
        st.markdown(
            """
            ### About Quantum NODE.AI
            
            This tool uses quantum computing principles to create branching narratives that exist in superposition until observed.
            
            - ⚛️ Quantum-weighted story paths
            - 🔄 Paths exist in superposition until chosen
            - 🧠 Quantum interference between story possibilities
            - 📊 Visualize your quantum story structure
            
            Built with Streamlit, PennyLane quantum framework, and Gemini API
            """
        )
    
    # Main content area
    if not st.session_state.api_key_set:
        st.warning("Please set your Gemini API key in the sidebar to begin.")
        
        # Display quantum explanation
        st.markdown("### How Quantum Computing Enhances Storytelling")
        st.markdown("""
        This tool uses quantum computing principles to create more dynamic and unpredictable stories:
        
        1. **Superposition**: Story paths exist in multiple states simultaneously until observed
        2. **Entanglement**: Connecting story elements across different paths
        3. **Interference**: Creating new narrative possibilities by combining existing paths
        4. **Measurement**: When you choose a path, the story "collapses" into that reality
        
        The quantum algorithms influence the probability and nature of each story path, creating
        a more dynamic and unpredictable narrative experience.
        """)
        
    else:
        # Story creation form
        if not hasattr(st.session_state.story_generator, 'current_node_id') or st.session_state.story_generator.current_node_id is None:
            st.markdown("<h2 class='sub-header'>Create a New Quantum Story</h2>", unsafe_allow_html=True)
            
            with st.form("story_parameters"):
                col1, col2 = st.columns(2)
                
                with col1:
                    genre = st.selectbox(
                        "Genre", 
                        ["Quantum Fantasy", "Sci-Fi", "Mystery", "Adventure", "Romance", "Horror", "Historical"]
                    )
                    protagonist = st.text_input("Protagonist", "A quantum physicist")
                
                with col2:
                    antagonist = st.text_input("Antagonist", "A rogue AI")
                    setting = st.text_input("Setting", "A research facility between dimensions")
                
                submit_button = st.form_submit_button("Generate Story Beginning")
                
                if submit_button:
                    with st.spinner("Generating quantum story beginning..."):
                        situation = st.session_state.story_generator.generate_initial_situation(
                            genre, protagonist, antagonist, setting
                        )
                        st.session_state.current_node_id = st.session_state.story_generator.current_node_id
                        st.rerun()
        
        # Display current story and options
        else:
            # Get current node and full story
            current_node = st.session_state.story_generator.story_tree.get_node(
                st.session_state.story_generator.current_node_id
            )
            
            full_story = st.session_state.story_generator.get_full_story()
            
                       # Story title
            if 'story_title' not in st.session_state or st.session_state.story_title == "Untitled Quantum Story":
                if len(full_story) > 200:  # Only generate title if story has some content
                    with st.spinner("Generating quantum story title..."):
                        title = st.session_state.story_generator.get_story_title(full_story)
                        st.session_state.story_title = title
            
            st.markdown(f"<h1 class='story-title'>{st.session_state.story_title}</h1>", unsafe_allow_html=True)
            
            # Display story visualization
            if st.button("📊 Visualize Quantum Story Structure"):
                with st.spinner("Generating quantum story visualization..."):
                    img = st.session_state.story_generator.story_tree.visualize()
                    st.image(img, use_container_width=True)
                    st.markdown("""
                    **Visualization Legend:**
                    - **Blue nodes**: Story paths in quantum superposition
                    - **Green nodes**: "Collapsed" story paths (chosen by you)
                    - **Purple nodes**: Quantum interference between paths
                    - **Edge thickness**: Quantum probability weight
                    """)
            
            # Display current story
            st.markdown("<h2 class='sub-header'>Your Quantum Story</h2>", unsafe_allow_html=True)
            st.markdown(f"<div class='story-box'>{full_story}</div>", unsafe_allow_html=True)
            
            # If story is not completed, show options
            if not st.session_state.story_completed:
                # Generate new paths if needed
                if not st.session_state.current_paths:
                    with st.spinner("Generating quantum story paths..."):
                        paths, weights = st.session_state.story_generator.generate_possible_paths(current_node.content)
                        path_ids = st.session_state.story_generator.add_paths(paths, weights)
                        
                        st.session_state.current_paths = paths
                        st.session_state.path_ids = path_ids
                        st.session_state.path_weights = weights
                
                # Display path options
                st.markdown("<h3 class='sub-header'>Choose Your Path</h3>", unsafe_allow_html=True)
                
                for i, (path, path_id, weight) in enumerate(zip(
                    st.session_state.current_paths, 
                    st.session_state.path_ids,
                    st.session_state.path_weights
                )):
                    # Determine if this is a quantum path
                    is_quantum = weight > 0.4
                    
                    # Create path option with quantum badge if applicable
                    path_html = f"""
                    <div class='{"quantum-path" if is_quantum else "path-option"}'>
                        <p>{path}</p>
                        <p style='text-align: right; font-style: italic; margin-bottom: 0;'>
                            Quantum probability: {weight:.2f}
                            {f'<span class="quantum-badge">⚛️ Quantum</span>' if is_quantum else ''}
                        </p>
                    </div>
                    """
                    
                    st.markdown(path_html, unsafe_allow_html=True)
                    
                    if st.button(f"Choose Path {i+1}", key=f"path_{i}"):
                        with st.spinner("Quantum wavefunction collapsing..."):
                            st.session_state.story_generator.select_path(path_id)
                            st.session_state.current_paths = []
                            st.session_state.path_ids = []
                            st.session_state.path_weights = []
                            st.rerun()
                
                # Custom path option
                st.markdown("<h3 class='sub-header'>Or Create Your Own Path</h3>", unsafe_allow_html=True)
                
                custom_path = st.text_area("Describe what happens next", height=100, 
                                          placeholder="Enter your own idea for what happens next in the story...")
                
                if st.button("Add My Path") and custom_path:
                    with st.spinner("Integrating your quantum path..."):
                        continuation = st.session_state.story_generator.generate_custom_continuation(
                            full_story, custom_path
                        )
                        st.session_state.current_paths = []
                        st.session_state.path_ids = []
                        st.session_state.path_weights = []
                        st.rerun()
                
                # Quantum interference option
                if len(st.session_state.story_generator.story_tree.nodes) > 3:
                    st.markdown("<h3 class='sub-header'>⚛️ Create Quantum Interference</h3>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    Quantum interference allows you to combine two different story paths into a new reality.
                    Select two existing nodes to create an interference pattern.
                    """)
                    
                    # Get all nodes except current and its direct children
                    available_nodes = {
                        nid: node.content[:50] + "..." 
                        for nid, node in st.session_state.story_generator.story_tree.nodes.items()
                        if nid != st.session_state.story_generator.current_node_id and 
                           node.id not in [child.id for child in current_node.children]
                    }
                    
                    if len(available_nodes) >= 2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            node1 = st.selectbox("First Node", list(available_nodes.values()), 
                                                key="interference_node1")
                            node1_id = list(available_nodes.keys())[list(available_nodes.values()).index(node1)]
                        
                        with col2:
                            remaining_nodes = {k: v for k, v in available_nodes.items() if k != node1_id}
                            node2 = st.selectbox("Second Node", list(remaining_nodes.values()),
                                                key="interference_node2")
                            node2_id = list(remaining_nodes.keys())[list(remaining_nodes.values()).index(node2)]
                        
                        if st.button("Create Quantum Interference"):
                            with st.spinner("Creating quantum interference pattern..."):
                                interference_id = st.session_state.story_generator.story_tree.quantum_interference(
                                    node1_id, node2_id, st.session_state.story_generator.current_node_id
                                )
                                
                                if interference_id:
                                    st.session_state.story_generator.current_node_id = interference_id
                                    st.session_state.current_paths = []
                                    st.session_state.path_ids = []
                                    st.session_state.path_weights = []
                                    st.rerun()
                                else:
                                    st.error("Failed to create quantum interference. Please try different nodes.")
                
                # Option to end the story
                st.markdown("---")
                if st.button("🎭 Generate Quantum Ending"):
                    with st.spinner("Collapsing all quantum possibilities into a final ending..."):
                        ending = st.session_state.story_generator.generate_quantum_ending(full_story)
                        st.session_state.story_completed = True
                        st.rerun()
            
            # If story is completed, show ending options
            else:
                st.markdown("<h3 class='sub-header'>🎭 Your Quantum Story is Complete</h3>", unsafe_allow_html=True)
                
                if st.button("Start a New Story"):
                    # Reset everything
                    st.session_state.story_generator = StoryGenerator(initialize_gemini_client(api_key))
                    st.session_state.current_paths = []
                    st.session_state.path_ids = []
                    st.session_state.path_weights = []
                    st.session_state.story_title = "Untitled Quantum Story"
                    st.session_state.story_completed = False
                    st.rerun()
                
                # Export options
                export_format = st.selectbox("Export Format", ["Text", "PDF", "HTML"])
                
                if st.button("Export Story"):
                    try:
                        filename = f"{st.session_state.story_title.replace(' ', '_')}_{int(time.time())}"
                        
                        if export_format == "Text":
                            with open(f"{filename}.txt", "w") as f:
                                f.write(f"{st.session_state.story_title}\n\n")
                                f.write(full_story)
                            st.success(f"Story exported as {filename}.txt")
                        
                        elif export_format == "HTML":
                            with open(f"{filename}.html", "w") as f:
                                f.write(f"<html><head><title>{st.session_state.story_title}</title>")
                                f.write("<style>body{font-family:Arial;max-width:800px;margin:0 auto;padding:20px;}")
                                f.write("h1{color:#2563EB;} p{line-height:1.6;}</style></head><body>")
                                f.write(f"<h1>{st.session_state.story_title}</h1>")
                                f.write(f"<div>{full_story.replace(chr(10), '<br>')}</div>")
                                f.write("</body></html>")
                            st.success(f"Story exported as {filename}.html")
                        
                        elif export_format == "PDF":
                            st.warning("PDF export requires additional libraries. Please install reportlab.")
                            # PDF export would go here if reportlab is installed
                    
                    except Exception as e:
                        st.error(f"Error exporting story: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()